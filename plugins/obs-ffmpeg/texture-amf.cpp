#include <util/threading.h>
#include <opts-parser.h>
#include <obs-module.h>
#include <obs-avc.h>

#include <unordered_map>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <vector>
#include <mutex>
#include <deque>
#include <map>
#include <inttypes.h>

#include <AMF/components/VideoEncoderHEVC.h>
#include <AMF/components/VideoEncoderVCE.h>
#include <AMF/components/VideoEncoderAV1.h>
#include <AMF/core/Factory.h>
#include <AMF/core/Trace.h>

#ifdef _WIN32
#include <dxgi.h>
#include <d3d11.h>
#include <d3d11_1.h>

#include <util/windows/device-enum.h>
#include <util/windows/HRError.hpp>
#include <util/windows/ComPtr.hpp>
#endif

#ifdef __linux
#include <algorithm>
#include <GL/glcorearb.h>
#include <GL/glext.h>
#include <EGL/egl.h>
#include <vulkan/vulkan.h>
#include <AMF/core/VulkanAMF.h>
#endif

#include <util/platform.h>
#include <util/util.hpp>
#include <util/pipe.h>
#include <util/dstr.h>

using namespace amf;

/* ========================================================================= */
/* Junk                                                                      */

#define do_log(level, format, ...)                          \
	blog(level, "[%s: '%s'] " format, enc->encoder_str, \
	     obs_encoder_get_name(enc->encoder), ##__VA_ARGS__)

#define error(format, ...) do_log(LOG_ERROR, format, ##__VA_ARGS__)
#define warn(format, ...) do_log(LOG_WARNING, format, ##__VA_ARGS__)
#define info(format, ...) do_log(LOG_INFO, format, ##__VA_ARGS__)
#define debug(format, ...) do_log(LOG_DEBUG, format, ##__VA_ARGS__)

struct amf_error {
	const char *str;
	AMF_RESULT res;

	inline amf_error(const char *str, AMF_RESULT res) : str(str), res(res)
	{
	}
};

#define VK_CHECK(f)                                                      \
	{                                                                \
		VkResult res = (f);                                      \
		if (res != VK_SUCCESS) {                                 \
			blog(LOG_ERROR, "Vulkan error: " __FILE__ ":%d", \
			     __LINE__);                                  \
			throw "Vulkan error";                            \
		}                                                        \
	}

static VkFormat to_vk_format(AMF_SURFACE_FORMAT fmt)
{
	switch (fmt) {
	case AMF_SURFACE_NV12:
		return VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	case AMF_SURFACE_P010:
		return VK_FORMAT_G16_B16R16_2PLANE_420_UNORM;
	default:
		throw "Unsupported AMF_SURFACE_FORMAT";
	}
}

static VkFormat to_vk_format(enum gs_color_format fmt)
{
	switch (fmt) {
	case GS_R8:
		return VK_FORMAT_R8_UNORM;
	case GS_R16:
		return VK_FORMAT_R16_UNORM;
	case GS_R8G8:
		return VK_FORMAT_R8G8_UNORM;
	case GS_RG16:
		return VK_FORMAT_R16G16_UNORM;
	default:
		throw "Unsupported gs_color_format";
	}
}

static GLenum to_gl_format(enum gs_color_format fmt)
{
	switch (fmt) {
	case GS_R8:
		return GL_R8;
	case GS_R16:
		return GL_R16;
	case GS_R8G8:
		return GL_RG8;
	case GS_RG16:
		return GL_RG16;
	default:
		throw "Unsupported gs_color_format";
	}
}

struct handle_tex {
	uint32_t handle;
#ifdef _WIN32
	ComPtr<ID3D11Texture2D> tex;
	ComPtr<IDXGIKeyedMutex> km;
#else
	AMFVulkanSurface *surfaceVk = nullptr;
#endif
};

#ifdef __linux
struct gl_tex {
	GLuint glsem = 0;
	VkSemaphore sem = VK_NULL_HANDLE;
	GLuint glCopySem = 0;
	VkSemaphore copySem = VK_NULL_HANDLE;
	VkFence copyFence = VK_NULL_HANDLE;
	struct {
		uint32_t width = 0;
		uint32_t height = 0;
		VkImage image = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		GLuint glmem = 0;
		GLuint gltex = 0;
		GLuint fbo = 0;
	} planes[2];
};
#endif

struct adapter_caps {
	bool is_amd = false;
	bool supports_avc = false;
	bool supports_hevc = false;
	bool supports_av1 = false;
};

/* ------------------------------------------------------------------------- */

static std::map<uint32_t, adapter_caps> caps;
static bool h264_supported = false;
static AMFFactory *amf_factory = nullptr;
static AMFTrace *amf_trace = nullptr;
static void *amf_module = nullptr;
static uint64_t amf_version = 0;

/* ========================================================================= */
/* Main Implementation                                                       */

enum class amf_codec_type {
	AVC,
	HEVC,
	AV1,
};

struct amf_base {
	obs_encoder_t *encoder;
	const char *encoder_str;
	amf_codec_type codec;
	bool fallback;

	AMFContextPtr amf_context;
	AMFContext1Ptr amf_context1;
	AMFComponentPtr amf_encoder;
	AMFBufferPtr packet_data;
	AMFRate amf_frame_rate;
	AMFBufferPtr header;

	std::deque<AMFDataPtr> queued_packets;

	AMF_VIDEO_CONVERTER_COLOR_PROFILE_ENUM amf_color_profile;
	AMF_COLOR_TRANSFER_CHARACTERISTIC_ENUM amf_characteristic;
	AMF_COLOR_PRIMARIES_ENUM amf_primaries;
	AMF_SURFACE_FORMAT amf_format;

	amf_int64 max_throughput = 0;
	amf_int64 throughput = 0;
	int64_t dts_offset = 0;
	uint32_t cx;
	uint32_t cy;
	uint32_t linesize = 0;
	int fps_num;
	int fps_den;
	bool full_range;
	bool bframes_supported = false;
	bool first_update = true;

	inline amf_base(bool fallback) : fallback(fallback) {}
	virtual ~amf_base() = default;
	virtual void init() = 0;
};

using buf_t = std::vector<uint8_t>;

#ifdef _WIN32
using d3dtex_t = ComPtr<ID3D11Texture2D>;
#else
using d3dtex_t = handle_tex;
#endif

struct amf_texencode : amf_base, public AMFSurfaceObserver {
	volatile bool destroying = false;

	std::vector<handle_tex> input_textures;

	std::mutex textures_mutex;
	std::vector<d3dtex_t> available_textures;
	std::unordered_map<AMFSurface *, d3dtex_t> active_textures;

#ifdef _WIN32
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> context;
#else
	std::unique_ptr<AMFVulkanDevice> vk;
	VkQueue queue = VK_NULL_HANDLE;
	VkCommandPool cmdpool = VK_NULL_HANDLE;
	VkCommandBuffer cmdbuf = VK_NULL_HANDLE;
	struct gl_tex gltex = {};
	std::unordered_map<gs_texture *, GLuint> read_fbos;

	PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR;
	PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR;
	PFNGLGETERRORPROC glGetError;
	PFNGLCREATEMEMORYOBJECTSEXTPROC glCreateMemoryObjectsEXT;
	PFNGLDELETEMEMORYOBJECTSEXTPROC glDeleteMemoryObjectsEXT;
	PFNGLIMPORTMEMORYFDEXTPROC glImportMemoryFdEXT;
	PFNGLISMEMORYOBJECTEXTPROC glIsMemoryObjectEXT;
	PFNGLMEMORYOBJECTPARAMETERIVEXTPROC glMemoryObjectParameterivEXT;
	PFNGLGENTEXTURESPROC glGenTextures;
	PFNGLDELETETEXTURESPROC glDeleteTextures;
	PFNGLBINDTEXTUREPROC glBindTexture;
	PFNGLTEXPARAMETERIPROC glTexParameteri;
	PFNGLTEXSTORAGEMEM2DEXTPROC glTexStorageMem2DEXT;
	PFNGLGENSEMAPHORESEXTPROC glGenSemaphoresEXT;
	PFNGLDELETESEMAPHORESEXTPROC glDeleteSemaphoresEXT;
	PFNGLIMPORTSEMAPHOREFDEXTPROC glImportSemaphoreFdEXT;
	PFNGLISSEMAPHOREEXTPROC glIsSemaphoreEXT;
	PFNGLWAITSEMAPHOREEXTPROC glWaitSemaphoreEXT;
	PFNGLSIGNALSEMAPHOREEXTPROC glSignalSemaphoreEXT;
	PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
	PFNGLDELETEFRAMEBUFFERSPROC glDeleteFramebuffers;
	PFNGLBINDFRAMEBUFFERPROC glBindFramebuffer;
	PFNGLFRAMEBUFFERTEXTURE2DPROC glFramebufferTexture2D;
	PFNGLBLITFRAMEBUFFERPROC glBlitFramebuffer;
#endif

	inline amf_texencode() : amf_base(false) {}
	~amf_texencode()
	{
		os_atomic_set_bool(&destroying, true);
#ifdef __linux
		if (!vk)
			return;

		vkDeviceWaitIdle(vk->hDevice);
		vkFreeCommandBuffers(vk->hDevice, cmdpool, 1, &cmdbuf);
		vkDestroyCommandPool(vk->hDevice, cmdpool, nullptr);

		for (auto t : input_textures) {
			vkFreeMemory(vk->hDevice, t.surfaceVk->hMemory,
				     nullptr);
			vkDestroyImage(vk->hDevice, t.surfaceVk->hImage,
				       nullptr);
			delete t.surfaceVk;
		}

		obs_enter_graphics();

		for (int i = 0; i < 2; ++i) {
			auto p = gltex.planes[i];
			vkFreeMemory(vk->hDevice, p.memory, nullptr);
			vkDestroyImage(vk->hDevice, p.image, nullptr);
			this->glDeleteMemoryObjectsEXT(1, &p.glmem);
			this->glDeleteTextures(1, &p.gltex);
			this->glDeleteFramebuffers(1, &p.fbo);
		}
		vkDestroySemaphore(vk->hDevice, gltex.sem, nullptr);
		vkDestroySemaphore(vk->hDevice, gltex.copySem, nullptr);
		vkDestroyFence(vk->hDevice, gltex.copyFence, nullptr);
		this->glDeleteSemaphoresEXT(1, &gltex.glsem);
		this->glDeleteSemaphoresEXT(1, &gltex.glCopySem);

		for (auto f : read_fbos)
			this->glDeleteFramebuffers(1, &f.second);

		obs_leave_graphics();

		amf_encoder->Terminate();
		amf_context1->Terminate();
		amf_context->Terminate();

		vkDestroyDevice(vk->hDevice, nullptr);
		vkDestroyInstance(vk->hInstance, nullptr);
#endif
	}

	void AMF_STD_CALL OnSurfaceDataRelease(amf::AMFSurface *surf) override
	{
		if (os_atomic_load_bool(&destroying))
			return;

		std::scoped_lock lock(textures_mutex);

		auto it = active_textures.find(surf);
		if (it != active_textures.end()) {
			available_textures.push_back(it->second);
			active_textures.erase(it);
		}
	}

	void init() override
	{
#if defined(_WIN32)
		AMF_RESULT res = amf_context->InitDX11(device, AMF_DX11_1);
		if (res != AMF_OK)
			throw amf_error("InitDX11 failed", res);
#elif defined(__linux__)
		vk = std::make_unique<AMFVulkanDevice>();
		vk->cbSizeof = sizeof(AMFVulkanDevice);

		std::vector<const char *> instance_extensions = {
			VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
			VK_KHR_SURFACE_EXTENSION_NAME,
		};

		std::vector<const char *> device_extensions = {
			VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
			VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,
			VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
		};

		amf_size count = 0;
		amf_context1->GetVulkanDeviceExtensions(&count, nullptr);
		device_extensions.resize(device_extensions.size() + count);
		amf_context1->GetVulkanDeviceExtensions(
			&count,
			&device_extensions[device_extensions.size() - count]);

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "OBS";
		appInfo.apiVersion = VK_API_VERSION_1_2;

		VkInstanceCreateInfo instanceInfo = {};
		instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceInfo.pApplicationInfo = &appInfo;
		instanceInfo.enabledExtensionCount = instance_extensions.size();
		instanceInfo.ppEnabledExtensionNames =
			instance_extensions.data();
		VK_CHECK(vkCreateInstance(&instanceInfo, nullptr,
					  &vk->hInstance));

		uint32_t deviceCount = 0;
		VK_CHECK(vkEnumeratePhysicalDevices(vk->hInstance, &deviceCount,
						    nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		VK_CHECK(vkEnumeratePhysicalDevices(vk->hInstance, &deviceCount,
						    physicalDevices.data()));
		for (VkPhysicalDevice dev : physicalDevices) {
			VkPhysicalDeviceDriverProperties driverProps = {};
			driverProps.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;

			VkPhysicalDeviceProperties2 props = {};
			props.sType =
				VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			props.pNext = &driverProps;
			vkGetPhysicalDeviceProperties2(dev, &props);
			if (driverProps.driverID ==
			    VK_DRIVER_ID_AMD_PROPRIETARY) {
				vk->hPhysicalDevice = dev;
				break;
			}
		}
		if (!vk->hPhysicalDevice) {
			throw "Failed to find Vulkan device VK_DRIVER_ID_AMD_PROPRIETARY";
		}

		uint32_t deviceExtensionCount = 0;
		VK_CHECK(vkEnumerateDeviceExtensionProperties(
			vk->hPhysicalDevice, nullptr, &deviceExtensionCount,
			nullptr));
		std::vector<VkExtensionProperties> deviceExts(
			deviceExtensionCount);
		VK_CHECK(vkEnumerateDeviceExtensionProperties(
			vk->hPhysicalDevice, nullptr, &deviceExtensionCount,
			deviceExts.data()));
		std::vector<const char *> deviceExtensions;
		for (const char *name : device_extensions) {
			auto it = std::find_if(
				deviceExts.begin(), deviceExts.end(),
				[name](VkExtensionProperties e) {
					return strcmp(e.extensionName, name) ==
					       0;
				});
			if (it != deviceExts.end()) {
				deviceExtensions.push_back(name);
			}
		}

		float queuePriority = 1.0;
		std::vector<VkDeviceQueueCreateInfo> queueInfos;
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(
			vk->hPhysicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(
			queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(
			vk->hPhysicalDevice, &queueFamilyCount,
			queueFamilyProperties.data());
		for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
			VkDeviceQueueCreateInfo queueInfo = {};
			queueInfo.sType =
				VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueInfo.queueFamilyIndex = i;
			queueInfo.queueCount = 1;
			queueInfo.pQueuePriorities = &queuePriority;
			queueInfos.push_back(queueInfo);
		}

		VkDeviceCreateInfo deviceInfo = {};
		deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceInfo.queueCreateInfoCount = queueInfos.size();
		deviceInfo.pQueueCreateInfos = queueInfos.data();
		deviceInfo.enabledExtensionCount = deviceExtensions.size();
		deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
		VK_CHECK(vkCreateDevice(vk->hPhysicalDevice, &deviceInfo,
					nullptr, &vk->hDevice));

		AMF_RESULT res = amf_context1->InitVulkan(vk.get());
		if (res != AMF_OK)
			throw amf_error("InitVulkan failed", res);

		vkGetDeviceQueue(vk->hDevice, 0, 0, &queue);

		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = 0;
		cmdPoolInfo.flags =
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK(vkCreateCommandPool(vk->hDevice, &cmdPoolInfo, nullptr,
					     &cmdpool));

		VkCommandBufferAllocateInfo commandBufferInfo = {};
		commandBufferInfo.sType =
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferInfo.commandPool = cmdpool;
		commandBufferInfo.commandBufferCount = 1;
		VK_CHECK(vkAllocateCommandBuffers(vk->hDevice,
						  &commandBufferInfo, &cmdbuf));

#define GET_PROC_VK(x)                                 \
	x = reinterpret_cast<decltype(x)>(             \
		vkGetDeviceProcAddr(vk->hDevice, #x)); \
	if (!x)                                        \
		throw "Failed to resolve " #x;

#define GET_PROC_GL(x)                                            \
	x = reinterpret_cast<decltype(x)>(eglGetProcAddress(#x)); \
	if (!x)                                                   \
		throw "Failed to resolve " #x;

		GET_PROC_VK(vkGetMemoryFdKHR);
		GET_PROC_VK(vkGetSemaphoreFdKHR);
		GET_PROC_GL(glGetError);
		GET_PROC_GL(glCreateMemoryObjectsEXT);
		GET_PROC_GL(glDeleteMemoryObjectsEXT);
		GET_PROC_GL(glImportMemoryFdEXT);
		GET_PROC_GL(glIsMemoryObjectEXT);
		GET_PROC_GL(glMemoryObjectParameterivEXT);
		GET_PROC_GL(glGenTextures);
		GET_PROC_GL(glDeleteTextures);
		GET_PROC_GL(glBindTexture);
		GET_PROC_GL(glTexParameteri);
		GET_PROC_GL(glTexStorageMem2DEXT);
		GET_PROC_GL(glGenSemaphoresEXT);
		GET_PROC_GL(glDeleteSemaphoresEXT);
		GET_PROC_GL(glImportSemaphoreFdEXT);
		GET_PROC_GL(glIsSemaphoreEXT);
		GET_PROC_GL(glWaitSemaphoreEXT);
		GET_PROC_GL(glSignalSemaphoreEXT);
		GET_PROC_GL(glGenFramebuffers);
		GET_PROC_GL(glDeleteFramebuffers);
		GET_PROC_GL(glBindFramebuffer);
		GET_PROC_GL(glFramebufferTexture2D);
		GET_PROC_GL(glBlitFramebuffer);

#undef GET_PROC_VK
#undef GET_PROC_GL

#endif
	}
};

struct amf_fallback : amf_base, public AMFSurfaceObserver {
	volatile bool destroying = false;

	std::mutex buffers_mutex;
	std::vector<buf_t> available_buffers;
	std::unordered_map<AMFSurface *, buf_t> active_buffers;

	inline amf_fallback() : amf_base(true) {}
	~amf_fallback() { os_atomic_set_bool(&destroying, true); }

	void AMF_STD_CALL OnSurfaceDataRelease(amf::AMFSurface *surf) override
	{
		if (os_atomic_load_bool(&destroying))
			return;

		std::scoped_lock lock(buffers_mutex);

		auto it = active_buffers.find(surf);
		if (it != active_buffers.end()) {
			available_buffers.push_back(std::move(it->second));
			active_buffers.erase(it);
		}
	}

	void init() override
	{
#if defined(_WIN32)
		AMF_RESULT res = amf_context->InitDX11(nullptr, AMF_DX11_1);
		if (res != AMF_OK)
			throw amf_error("InitDX11 failed", res);
#elif defined(__linux__)
		AMF_RESULT res = amf_context1->InitVulkan(nullptr);
		if (res != AMF_OK)
			throw amf_error("InitVulkan failed", res);
#endif
	}
};

/* ------------------------------------------------------------------------- */
/* More garbage                                                              */

template<typename T>
static bool get_amf_property(amf_base *enc, const wchar_t *name, T *value)
{
	AMF_RESULT res = enc->amf_encoder->GetProperty(name, value);
	return res == AMF_OK;
}

template<typename T>
static void set_amf_property(amf_base *enc, const wchar_t *name, const T &value)
{
	AMF_RESULT res = enc->amf_encoder->SetProperty(name, value);
	if (res != AMF_OK)
		error("Failed to set property '%ls': %ls", name,
		      amf_trace->GetResultText(res));
}

#define set_avc_property(enc, name, value) \
	set_amf_property(enc, AMF_VIDEO_ENCODER_##name, value)
#define set_hevc_property(enc, name, value) \
	set_amf_property(enc, AMF_VIDEO_ENCODER_HEVC_##name, value)
#define set_av1_property(enc, name, value) \
	set_amf_property(enc, AMF_VIDEO_ENCODER_AV1_##name, value)

#define get_avc_property(enc, name, value) \
	get_amf_property(enc, AMF_VIDEO_ENCODER_##name, value)
#define get_hevc_property(enc, name, value) \
	get_amf_property(enc, AMF_VIDEO_ENCODER_HEVC_##name, value)
#define get_av1_property(enc, name, value) \
	get_amf_property(enc, AMF_VIDEO_ENCODER_AV1_##name, value)

#define get_opt_name(name)                                              \
	((enc->codec == amf_codec_type::AVC) ? AMF_VIDEO_ENCODER_##name \
	 : (enc->codec == amf_codec_type::HEVC)                         \
		 ? AMF_VIDEO_ENCODER_HEVC_##name                        \
		 : AMF_VIDEO_ENCODER_AV1_##name)
#define get_opt_name_enum(name)                                              \
	((enc->codec == amf_codec_type::AVC) ? (int)AMF_VIDEO_ENCODER_##name \
	 : (enc->codec == amf_codec_type::HEVC)                         \
		 ? (int)AMF_VIDEO_ENCODER_HEVC_##name                        \
		 : (int)AMF_VIDEO_ENCODER_AV1_##name)
#define set_opt(name, value) set_amf_property(enc, get_opt_name(name), value)
#define get_opt(name, value) get_amf_property(enc, get_opt_name(name), value)
#define set_avc_opt(name, value) set_avc_property(enc, name, value)
#define set_hevc_opt(name, value) set_hevc_property(enc, name, value)
#define set_av1_opt(name, value) set_av1_property(enc, name, value)
#define set_enum_opt(name, value) \
	set_amf_property(enc, get_opt_name(name), get_opt_name_enum(name##_##value))
#define set_avc_enum(name, value) \
	set_avc_property(enc, name, AMF_VIDEO_ENCODER_##name##_##value)
#define set_hevc_enum(name, value) \
	set_hevc_property(enc, name, AMF_VIDEO_ENCODER_HEVC_##name##_##value)
#define set_av1_enum(name, value) \
	set_av1_property(enc, name, AMF_VIDEO_ENCODER_AV1_##name##_##value)

/* ------------------------------------------------------------------------- */
/* Implementation                                                            */

#ifdef _WIN32
static HMODULE get_lib(const char *lib)
{
	HMODULE mod = GetModuleHandleA(lib);
	if (mod)
		return mod;

	return LoadLibraryA(lib);
}

#define AMD_VENDOR_ID 0x1002

typedef HRESULT(WINAPI *CREATEDXGIFACTORY1PROC)(REFIID, void **);

static bool amf_init_d3d11(amf_texencode *enc)
try {
	HMODULE dxgi = get_lib("DXGI.dll");
	HMODULE d3d11 = get_lib("D3D11.dll");
	CREATEDXGIFACTORY1PROC create_dxgi;
	PFN_D3D11_CREATE_DEVICE create_device;
	ComPtr<IDXGIFactory> factory;
	ComPtr<ID3D11Device> device;
	ComPtr<ID3D11DeviceContext> context;
	ComPtr<IDXGIAdapter> adapter;
	DXGI_ADAPTER_DESC desc;
	HRESULT hr;

	if (!dxgi || !d3d11)
		throw "Couldn't get D3D11/DXGI libraries? "
		      "That definitely shouldn't be possible.";

	create_dxgi = (CREATEDXGIFACTORY1PROC)GetProcAddress(
		dxgi, "CreateDXGIFactory1");
	create_device = (PFN_D3D11_CREATE_DEVICE)GetProcAddress(
		d3d11, "D3D11CreateDevice");

	if (!create_dxgi || !create_device)
		throw "Failed to load D3D11/DXGI procedures";

	hr = create_dxgi(__uuidof(IDXGIFactory2), (void **)&factory);
	if (FAILED(hr))
		throw HRError("CreateDXGIFactory1 failed", hr);

	obs_video_info ovi;
	obs_get_video_info(&ovi);

	hr = factory->EnumAdapters(ovi.adapter, &adapter);
	if (FAILED(hr))
		throw HRError("EnumAdapters failed", hr);

	adapter->GetDesc(&desc);
	if (desc.VendorId != AMD_VENDOR_ID)
		throw "Seems somehow AMF is trying to initialize "
		      "on a non-AMD adapter";

	hr = create_device(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
			   nullptr, 0, D3D11_SDK_VERSION, &device, nullptr,
			   &context);
	if (FAILED(hr))
		throw HRError("D3D11CreateDevice failed", hr);

	enc->device = device;
	enc->context = context;
	return true;

} catch (const HRError &err) {
	error("%s: %s: 0x%lX", __FUNCTION__, err.str, err.hr);
	return false;

} catch (const char *err) {
	error("%s: %s", __FUNCTION__, err);
	return false;
}

static void add_output_tex(amf_texencode *enc,
			   ComPtr<ID3D11Texture2D> &output_tex,
			   ID3D11Texture2D *from)
{
	ID3D11Device *device = enc->device;
	HRESULT hr;

	D3D11_TEXTURE2D_DESC desc;
	from->GetDesc(&desc);
	desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	desc.MiscFlags = 0;

	hr = device->CreateTexture2D(&desc, nullptr, &output_tex);
	if (FAILED(hr))
		throw HRError("Failed to create texture", hr);
}

static inline bool get_available_tex(amf_texencode *enc,
				     ComPtr<ID3D11Texture2D> &output_tex)
{
	std::scoped_lock lock(enc->textures_mutex);
	if (enc->available_textures.size()) {
		output_tex = enc->available_textures.back();
		enc->available_textures.pop_back();
		return true;
	}

	return false;
}

static inline void get_output_tex(amf_texencode *enc,
				  ComPtr<ID3D11Texture2D> &output_tex,
				  ID3D11Texture2D *from)
{
	if (!get_available_tex(enc, output_tex))
		add_output_tex(enc, output_tex, from);
}

static void get_tex_from_handle(amf_texencode *enc, uint32_t handle,
				IDXGIKeyedMutex **km_out,
				ID3D11Texture2D **tex_out)
{
	ID3D11Device *device = enc->device;
	ComPtr<ID3D11Texture2D> tex;
	HRESULT hr;

	for (size_t i = 0; i < enc->input_textures.size(); i++) {
		struct handle_tex &ht = enc->input_textures[i];
		if (ht.handle == handle) {
			ht.km.CopyTo(km_out);
			ht.tex.CopyTo(tex_out);
			return;
		}
	}

	hr = device->OpenSharedResource((HANDLE)(uintptr_t)handle,
					__uuidof(ID3D11Resource),
					(void **)&tex);
	if (FAILED(hr))
		throw HRError("OpenSharedResource failed", hr);

	ComQIPtr<IDXGIKeyedMutex> km(tex);
	if (!km)
		throw "QueryInterface(IDXGIKeyedMutex) failed";

	tex->SetEvictionPriority(DXGI_RESOURCE_PRIORITY_MAXIMUM);

	struct handle_tex new_ht = {handle, tex, km};
	enc->input_textures.push_back(std::move(new_ht));

	*km_out = km.Detach();
	*tex_out = tex.Detach();
}
#else
static uint32_t memoryTypeIndex(amf_texencode *enc,
				VkMemoryPropertyFlags properties,
				uint32_t typeBits)
{
	VkPhysicalDeviceMemoryProperties prop;
	vkGetPhysicalDeviceMemoryProperties(enc->vk->hPhysicalDevice, &prop);
	for (uint32_t i = 0; i < prop.memoryTypeCount; i++) {
		if ((prop.memoryTypes[i].propertyFlags & properties) ==
			    properties &&
		    typeBits & (1 << i)) {
			return i;
		}
	}
	return 0xFFFFFFFF;
}

static void cmd_buf_begin(amf_texencode *enc)
{
	VkCommandBufferBeginInfo commandBufferBegin = {};
	commandBufferBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	VK_CHECK(vkBeginCommandBuffer(enc->cmdbuf, &commandBufferBegin));
}

static void cmd_buf_submit(amf_texencode *enc, VkSemaphore *semaphore = nullptr,
			   VkFence *fence = nullptr)
{
	VK_CHECK(vkEndCommandBuffer(enc->cmdbuf));
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &enc->cmdbuf;
	submitInfo.signalSemaphoreCount = semaphore ? 1 : 0;
	submitInfo.pSignalSemaphores = semaphore;
	if (fence) {
		VK_CHECK(vkQueueSubmit(enc->queue, 1, &submitInfo, *fence));
		return;
	}
	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VkFence f;
	VK_CHECK(vkCreateFence(enc->vk->hDevice, &fenceInfo, nullptr, &f));
	VK_CHECK(vkQueueSubmit(enc->queue, 1, &submitInfo, f));
	VK_CHECK(vkWaitForFences(enc->vk->hDevice, 1, &f, VK_TRUE, UINT64_MAX));
	vkDestroyFence(enc->vk->hDevice, f, nullptr);
}

static void add_output_tex(amf_texencode *enc, handle_tex &output_tex,
			   encoder_texture *from)
{
	output_tex.surfaceVk = new AMFVulkanSurface;
	output_tex.surfaceVk->cbSizeof = sizeof(AMFVulkanSurface);
	output_tex.surfaceVk->pNext = nullptr;

	VkImageCreateInfo imageInfo = {};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = to_vk_format(enc->amf_format);
	imageInfo.extent.width = from->info.width;
	imageInfo.extent.height = from->info.height;
	imageInfo.extent.depth = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.mipLevels = 1;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
			  VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	imageInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VK_CHECK(vkCreateImage(enc->vk->hDevice, &imageInfo, nullptr,
			       &output_tex.surfaceVk->hImage));

	VkMemoryRequirements memoryReqs;
	vkGetImageMemoryRequirements(enc->vk->hDevice,
				     output_tex.surfaceVk->hImage, &memoryReqs);
	VkMemoryAllocateInfo memoryAllocInfo = {};
	memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocInfo.allocationSize = memoryReqs.size;
	memoryAllocInfo.memoryTypeIndex =
		memoryTypeIndex(enc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				memoryReqs.memoryTypeBits);
	VK_CHECK(vkAllocateMemory(enc->vk->hDevice, &memoryAllocInfo, nullptr,
				  &output_tex.surfaceVk->hMemory));
	VK_CHECK(vkBindImageMemory(enc->vk->hDevice,
				   output_tex.surfaceVk->hImage,
				   output_tex.surfaceVk->hMemory, 0));

	cmd_buf_begin(enc);
	VkImageMemoryBarrier imageBarrier = {};
	imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	imageBarrier.image = output_tex.surfaceVk->hImage;
	imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageBarrier.subresourceRange.layerCount = 1;
	imageBarrier.subresourceRange.levelCount = 1;
	imageBarrier.srcAccessMask = 0;
	imageBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
				     VK_ACCESS_MEMORY_WRITE_BIT;
	vkCmdPipelineBarrier(enc->cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
			     nullptr, 1, &imageBarrier);
	cmd_buf_submit(enc);

	output_tex.surfaceVk->iSize = memoryAllocInfo.allocationSize;
	output_tex.surfaceVk->eFormat = imageInfo.format;
	output_tex.surfaceVk->iWidth = imageInfo.extent.width;
	output_tex.surfaceVk->iHeight = imageInfo.extent.height;
	output_tex.surfaceVk->eCurrentLayout = imageInfo.initialLayout;
	output_tex.surfaceVk->eUsage = AMF_SURFACE_USAGE_DEFAULT;
	output_tex.surfaceVk->eAccess = AMF_MEMORY_CPU_LOCAL;
	output_tex.surfaceVk->Sync.cbSizeof = sizeof(AMFVulkanSync);
	output_tex.surfaceVk->Sync.pNext = nullptr;
	output_tex.surfaceVk->Sync.hSemaphore = nullptr;
	output_tex.surfaceVk->Sync.bSubmitted = true;
	output_tex.surfaceVk->Sync.hFence = nullptr;

	enc->input_textures.push_back(output_tex);
}

static inline void create_gl_tex(amf_texencode *enc, gl_tex &output_tex,
				 encoder_texture *from)
{
	if (output_tex.glsem)
		return;

	cmd_buf_begin(enc);
	for (int i = 0; i < 2; ++i) {
		obs_enter_graphics();
		auto gs_format = gs_texture_get_color_format(from->tex[i]);
		output_tex.planes[i].width = gs_texture_get_width(from->tex[i]);
		output_tex.planes[i].height =
			gs_texture_get_height(from->tex[i]);
		obs_leave_graphics();

		VkExternalMemoryImageCreateInfo extImageInfo = {};
		extImageInfo.sType =
			VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		extImageInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.pNext = &extImageInfo;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = to_vk_format(gs_format);
		imageInfo.extent.width = output_tex.planes[i].width;
		imageInfo.extent.height = output_tex.planes[i].height;
		imageInfo.extent.depth = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.mipLevels = 1;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
				  VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK(vkCreateImage(enc->vk->hDevice, &imageInfo, nullptr,
				       &output_tex.planes[i].image));

		VkMemoryRequirements memoryReqs;
		vkGetImageMemoryRequirements(enc->vk->hDevice,
					     output_tex.planes[i].image,
					     &memoryReqs);

		VkExportMemoryAllocateInfo expMemoryAllocInfo = {};
		expMemoryAllocInfo.sType =
			VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
		expMemoryAllocInfo.handleTypes =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

		VkMemoryDedicatedAllocateInfo dedMemoryAllocInfo = {};
		dedMemoryAllocInfo.sType =
			VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
		dedMemoryAllocInfo.image = output_tex.planes[i].image;
		dedMemoryAllocInfo.pNext = &expMemoryAllocInfo;

		VkMemoryAllocateInfo memoryAllocInfo = {};
		memoryAllocInfo.pNext = &dedMemoryAllocInfo;
		memoryAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocInfo.allocationSize = memoryReqs.size;
		memoryAllocInfo.memoryTypeIndex = memoryTypeIndex(
			enc, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			memoryReqs.memoryTypeBits);
		VK_CHECK(vkAllocateMemory(enc->vk->hDevice, &memoryAllocInfo,
					  nullptr,
					  &output_tex.planes[i].memory));
		VK_CHECK(vkBindImageMemory(enc->vk->hDevice,
					   output_tex.planes[i].image,
					   output_tex.planes[i].memory, 0));

		VkImageMemoryBarrier imageBarrier = {};
		imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		imageBarrier.image = output_tex.planes[i].image;
		imageBarrier.subresourceRange.aspectMask =
			VK_IMAGE_ASPECT_COLOR_BIT;
		imageBarrier.subresourceRange.layerCount = 1;
		imageBarrier.subresourceRange.levelCount = 1;
		imageBarrier.srcAccessMask = 0;
		imageBarrier.dstAccessMask = 0;
		vkCmdPipelineBarrier(enc->cmdbuf,
				     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
				     nullptr, 0, nullptr, 1, &imageBarrier);

		imageBarrier.oldLayout = imageBarrier.newLayout;
		imageBarrier.srcQueueFamilyIndex = 0;
		imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
		vkCmdPipelineBarrier(enc->cmdbuf,
				     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
				     nullptr, 0, nullptr, 1, &imageBarrier);

		// Import memory
		VkMemoryGetFdInfoKHR memFdInfo = {};
		memFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
		memFdInfo.memory = output_tex.planes[i].memory;
		memFdInfo.handleType =
			VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
		int fd = -1;
		VK_CHECK(enc->vkGetMemoryFdKHR(enc->vk->hDevice, &memFdInfo,
					       &fd));

		obs_enter_graphics();

		enc->glCreateMemoryObjectsEXT(1, &output_tex.planes[i].glmem);
		GLint dedicated = GL_TRUE;
		enc->glMemoryObjectParameterivEXT(
			output_tex.planes[i].glmem,
			GL_DEDICATED_MEMORY_OBJECT_EXT, &dedicated);
		enc->glImportMemoryFdEXT(output_tex.planes[i].glmem,
					 memoryAllocInfo.allocationSize,
					 GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);

		enc->glGenTextures(1, &output_tex.planes[i].gltex);
		enc->glBindTexture(GL_TEXTURE_2D, output_tex.planes[i].gltex);
		enc->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_TILING_EXT,
				     GL_OPTIMAL_TILING_EXT);
		enc->glTexStorageMem2DEXT(GL_TEXTURE_2D, 1,
					  to_gl_format(gs_format),
					  imageInfo.extent.width,
					  imageInfo.extent.height,
					  output_tex.planes[i].glmem, 0);

		enc->glGenFramebuffers(1, &output_tex.planes[i].fbo);
		enc->glBindFramebuffer(GL_FRAMEBUFFER,
				       output_tex.planes[i].fbo);
		enc->glFramebufferTexture2D(GL_FRAMEBUFFER,
					    GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
					    output_tex.planes[i].gltex, 0);
		enc->glBindFramebuffer(GL_FRAMEBUFFER, 0);

		bool import_ok =
			enc->glIsMemoryObjectEXT(output_tex.planes[i].glmem) &&
			enc->glGetError() == GL_NO_ERROR;

		obs_leave_graphics();

		if (!import_ok)
			throw "OpenGL texture import failed";
	}

	VkExportSemaphoreCreateInfo expSemInfo = {};
	expSemInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
	expSemInfo.handleTypes =
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

	VkSemaphoreCreateInfo semInfo = {};
	semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
	semInfo.pNext = &expSemInfo;
	VK_CHECK(vkCreateSemaphore(enc->vk->hDevice, &semInfo, nullptr,
				   &output_tex.sem));

	VK_CHECK(vkCreateSemaphore(enc->vk->hDevice, &semInfo, nullptr,
				   &output_tex.copySem));

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	VK_CHECK(vkCreateFence(enc->vk->hDevice, &fenceInfo, nullptr,
			       &output_tex.copyFence));

	cmd_buf_submit(enc, &output_tex.copySem, &output_tex.copyFence);

	// Import semaphores
	VkSemaphoreGetFdInfoKHR semFdInfo = {};
	semFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
	semFdInfo.semaphore = output_tex.sem;
	semFdInfo.handleType =
		VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
	int fd = -1;
	VK_CHECK(enc->vkGetSemaphoreFdKHR(enc->vk->hDevice, &semFdInfo, &fd));

	semFdInfo.semaphore = output_tex.copySem;
	int fdCopy = -1;
	VK_CHECK(enc->vkGetSemaphoreFdKHR(enc->vk->hDevice, &semFdInfo,
					  &fdCopy));

	obs_enter_graphics();

	enc->glGenSemaphoresEXT(1, &output_tex.glsem);
	enc->glGenSemaphoresEXT(1, &output_tex.glCopySem);
	enc->glImportSemaphoreFdEXT(output_tex.glsem,
				    GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
	enc->glImportSemaphoreFdEXT(output_tex.glCopySem,
				    GL_HANDLE_TYPE_OPAQUE_FD_EXT, fdCopy);

	bool import_ok = enc->glIsSemaphoreEXT(output_tex.glsem) &&
			 enc->glIsSemaphoreEXT(output_tex.glCopySem) &&
			 enc->glGetError() == GL_NO_ERROR;

	obs_leave_graphics();

	if (!import_ok)
		throw "OpenGL semaphore import failed";
}

static inline bool get_available_tex(amf_texencode *enc, handle_tex &output_tex)
{
	std::scoped_lock lock(enc->textures_mutex);
	if (enc->available_textures.size()) {
		output_tex = enc->available_textures.back();
		enc->available_textures.pop_back();
		return true;
	}

	return false;
}

static inline void get_output_tex(amf_texencode *enc, handle_tex &output_tex,
				  encoder_texture *from)
{
	if (!get_available_tex(enc, output_tex))
		add_output_tex(enc, output_tex, from);

	create_gl_tex(enc, enc->gltex, from);
}

static inline GLuint get_read_fbo(amf_texencode *enc, gs_texture *tex)
{
	auto it = enc->read_fbos.find(tex);
	if (it != enc->read_fbos.end()) {
		return it->second;
	}
	GLuint *tex_obj = static_cast<GLuint *>(gs_texture_get_obj(tex));
	GLuint fbo;
	enc->glGenFramebuffers(1, &fbo);
	enc->glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	enc->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
				    GL_TEXTURE_2D, *tex_obj, 0);
	enc->read_fbos.insert({tex, fbo});
	return fbo;
}
#endif

static constexpr amf_int64 macroblock_size = 16;

static inline void calc_throughput(amf_base *enc)
{
	amf_int64 mb_cx =
		((amf_int64)enc->cx + (macroblock_size - 1)) / macroblock_size;
	amf_int64 mb_cy =
		((amf_int64)enc->cy + (macroblock_size - 1)) / macroblock_size;
	amf_int64 mb_frame = mb_cx * mb_cy;

	enc->throughput =
		mb_frame * (amf_int64)enc->fps_num / (amf_int64)enc->fps_den;
}

static inline int get_avc_preset(amf_base *enc, const char *preset);
#if ENABLE_HEVC
static inline int get_hevc_preset(amf_base *enc, const char *preset);
#endif
static inline int get_av1_preset(amf_base *enc, const char *preset);

static inline int get_preset(amf_base *enc, const char *preset)
{
	if (enc->codec == amf_codec_type::AVC)
		return get_avc_preset(enc, preset);

#if ENABLE_HEVC
	else if (enc->codec == amf_codec_type::HEVC)
		return get_hevc_preset(enc, preset);

#endif
	else if (enc->codec == amf_codec_type::AV1)
		return get_av1_preset(enc, preset);

	return 0;
}

static inline void refresh_throughput_caps(amf_base *enc, const char *&preset)
{
	AMF_RESULT res = AMF_OK;
	AMFCapsPtr caps;

	set_opt(QUALITY_PRESET, get_preset(enc, preset));
	res = enc->amf_encoder->GetCaps(&caps);
	if (res == AMF_OK) {
		caps->GetProperty(get_opt_name(CAP_MAX_THROUGHPUT),
				  &enc->max_throughput);
	}
}

static inline void check_preset_compatibility(amf_base *enc,
					      const char *&preset)
{
	/* The throughput depends on the current preset and the other static
	 * encoder properties. If the throughput is lower than the max
	 * throughput, switch to a lower preset. */

	if (astrcmpi(preset, "highQuality") == 0) {
		if (!enc->max_throughput) {
			preset = "quality";
			set_opt(QUALITY_PRESET, get_preset(enc, preset));
		} else {
			if (enc->max_throughput < enc->throughput) {
				preset = "quality";
				refresh_throughput_caps(enc, preset);
			}
		}
	}

	if (astrcmpi(preset, "quality") == 0) {
		if (!enc->max_throughput) {
			preset = "balanced";
			set_opt(QUALITY_PRESET, get_preset(enc, preset));
		} else {
			if (enc->max_throughput < enc->throughput) {
				preset = "balanced";
				refresh_throughput_caps(enc, preset);
			}
		}
	}

	if (astrcmpi(preset, "balanced") == 0) {
		if (enc->max_throughput &&
		    enc->max_throughput < enc->throughput) {
			preset = "speed";
			refresh_throughput_caps(enc, preset);
		}
	}
}

static inline int64_t convert_to_amf_ts(amf_base *enc, int64_t ts)
{
	constexpr int64_t amf_timebase = AMF_SECOND;
	return ts * amf_timebase / (int64_t)enc->fps_den;
}

static inline int64_t convert_to_obs_ts(amf_base *enc, int64_t ts)
{
	constexpr int64_t amf_timebase = AMF_SECOND;
	return ts * (int64_t)enc->fps_den / amf_timebase;
}

static void convert_to_encoder_packet(amf_base *enc, AMFDataPtr &data,
				      encoder_packet *packet)
{
	if (!data)
		return;

	enc->packet_data = AMFBufferPtr(data);
	data->GetProperty(L"PTS", &packet->pts);

	const wchar_t *get_output_type = NULL;
	switch (enc->codec) {
	case amf_codec_type::AVC:
		get_output_type = AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE;
		break;
	case amf_codec_type::HEVC:
		get_output_type = AMF_VIDEO_ENCODER_HEVC_OUTPUT_DATA_TYPE;
		break;
	case amf_codec_type::AV1:
		get_output_type = AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE;
		break;
	}

	uint64_t type = 0;
	AMF_RESULT res = data->GetProperty(get_output_type, &type);
	if (res != AMF_OK)
		throw amf_error("Failed to GetProperty(): encoder output "
				"data type",
				res);

	if (enc->codec == amf_codec_type::AVC ||
	    enc->codec == amf_codec_type::HEVC) {
		switch (type) {
		case AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_IDR:
			packet->priority = OBS_NAL_PRIORITY_HIGHEST;
			break;
		case AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_I:
			packet->priority = OBS_NAL_PRIORITY_HIGH;
			break;
		case AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_P:
			packet->priority = OBS_NAL_PRIORITY_LOW;
			break;
		case AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_B:
			packet->priority = OBS_NAL_PRIORITY_DISPOSABLE;
			break;
		}
	} else if (enc->codec == amf_codec_type::AV1) {
		switch (type) {
		case AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_KEY:
			packet->priority = OBS_NAL_PRIORITY_HIGHEST;
			break;
		case AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_INTRA_ONLY:
			packet->priority = OBS_NAL_PRIORITY_HIGH;
			break;
		case AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_INTER:
			packet->priority = OBS_NAL_PRIORITY_LOW;
			break;
		case AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_SWITCH:
			packet->priority = OBS_NAL_PRIORITY_DISPOSABLE;
			break;
		case AMF_VIDEO_ENCODER_AV1_OUTPUT_FRAME_TYPE_SHOW_EXISTING:
			packet->priority = OBS_NAL_PRIORITY_DISPOSABLE;
			break;
		}
	}

	packet->data = (uint8_t *)enc->packet_data->GetNative();
	packet->size = enc->packet_data->GetSize();
	packet->type = OBS_ENCODER_VIDEO;
	packet->dts = convert_to_obs_ts(enc, data->GetPts());
	packet->keyframe = type == AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_IDR;

	if (enc->dts_offset)
		packet->dts -= enc->dts_offset;
}

#ifndef SEC_TO_NSEC
#define SEC_TO_NSEC 1000000000ULL
#endif

static void amf_encode_base(amf_base *enc, AMFSurface *amf_surf,
			    encoder_packet *packet, bool *received_packet)
{
	auto &queued_packets = enc->queued_packets;
	uint64_t ts_start = os_gettime_ns();
	AMF_RESULT res;

	*received_packet = false;

	bool waiting = true;
	while (waiting) {
		/* ----------------------------------- */
		/* submit frame                        */

		res = enc->amf_encoder->SubmitInput(amf_surf);

		if (res == AMF_OK || res == AMF_NEED_MORE_INPUT) {
			waiting = false;

		} else if (res == AMF_INPUT_FULL) {
			os_sleep_ms(1);

			uint64_t duration = os_gettime_ns() - ts_start;
			constexpr uint64_t timeout = 5 * SEC_TO_NSEC;

			if (duration >= timeout) {
				throw amf_error("SubmitInput timed out", res);
			}
		} else {
			throw amf_error("SubmitInput failed", res);
		}

		/* ----------------------------------- */
		/* query as many packets as possible   */

		AMFDataPtr new_packet;
		do {
			res = enc->amf_encoder->QueryOutput(&new_packet);
			if (new_packet)
				queued_packets.push_back(new_packet);

			if (res != AMF_REPEAT && res != AMF_OK) {
				throw amf_error("QueryOutput failed", res);
			}
		} while (!!new_packet);
	}

	/* ----------------------------------- */
	/* return a packet if available        */

	if (queued_packets.size()) {
		AMFDataPtr amf_out;

		amf_out = queued_packets.front();
		queued_packets.pop_front();

		*received_packet = true;
		convert_to_encoder_packet(enc, amf_out, packet);
	}
}

static bool amf_encode_tex(void *data, uint32_t handle, int64_t pts,
			   uint64_t lock_key, uint64_t *next_key,
			   encoder_packet *packet, bool *received_packet)
#ifdef _WIN32
try {
	amf_texencode *enc = (amf_texencode *)data;
	ID3D11DeviceContext *context = enc->context;
	ComPtr<ID3D11Texture2D> output_tex;
	ComPtr<ID3D11Texture2D> input_tex;
	ComPtr<IDXGIKeyedMutex> km;
	AMFSurfacePtr amf_surf;
	AMF_RESULT res;

	if (handle == GS_INVALID_HANDLE) {
		*next_key = lock_key;
		throw "Encode failed: bad texture handle";
	}

	/* ------------------------------------ */
	/* get the input tex                    */

	get_tex_from_handle(enc, handle, &km, &input_tex);

	/* ------------------------------------ */
	/* get an output tex                    */

	get_output_tex(enc, output_tex, input_tex);

	/* ------------------------------------ */
	/* copy to output tex                   */

	km->AcquireSync(lock_key, INFINITE);
	context->CopyResource((ID3D11Resource *)output_tex.Get(),
			      (ID3D11Resource *)input_tex.Get());
	context->Flush();
	km->ReleaseSync(*next_key);

	/* ------------------------------------ */
	/* map output tex to amf surface        */

	res = enc->amf_context->CreateSurfaceFromDX11Native(output_tex,
							    &amf_surf, enc);
	if (res != AMF_OK)
		throw amf_error("CreateSurfaceFromDX11Native failed", res);

	int64_t last_ts = convert_to_amf_ts(enc, pts - 1);
	int64_t cur_ts = convert_to_amf_ts(enc, pts);

	amf_surf->SetPts(cur_ts);
	amf_surf->SetProperty(L"PTS", pts);

	{
		std::scoped_lock lock(enc->textures_mutex);
		enc->active_textures[amf_surf.GetPtr()] = output_tex;
	}

	/* ------------------------------------ */
	/* do actual encode                     */

	amf_encode_base(enc, amf_surf, packet, received_packet);
	return true;

} catch (const char *err) {
	amf_texencode *enc = (amf_texencode *)data;
	error("%s: %s", __FUNCTION__, err);
	return false;

} catch (const amf_error &err) {
	amf_texencode *enc = (amf_texencode *)data;
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	*received_packet = false;
	return false;

} catch (const HRError &err) {
	amf_texencode *enc = (amf_texencode *)data;
	error("%s: %s: 0x%lX", __FUNCTION__, err.str, err.hr);
	*received_packet = false;
	return false;
}
#else
{
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(handle);
	UNUSED_PARAMETER(pts);
	UNUSED_PARAMETER(lock_key);
	UNUSED_PARAMETER(next_key);
	UNUSED_PARAMETER(packet);
	UNUSED_PARAMETER(received_packet);
	return false;
}
#endif

static bool amf_encode_tex2(void *data, encoder_texture *texture, int64_t pts,
			    uint64_t lock_key, uint64_t *next_key,
			    encoder_packet *packet, bool *received_packet)
try {
	UNUSED_PARAMETER(lock_key);
	UNUSED_PARAMETER(next_key);

	amf_texencode *enc = (amf_texencode *)data;
	handle_tex output_tex;
	AMFSurfacePtr amf_surf;
	AMF_RESULT res;

	if (!texture) {
		throw "Encode failed: bad texture handle";
	}

	/* ------------------------------------ */
	/* get an output tex                    */

	get_output_tex(enc, output_tex, texture);

	/* ------------------------------------ */
	/* copy to output tex                   */

	VK_CHECK(vkWaitForFences(enc->vk->hDevice, 1, &enc->gltex.copyFence,
				 VK_TRUE, UINT64_MAX));
	VK_CHECK(vkResetFences(enc->vk->hDevice, 1, &enc->gltex.copyFence));

	obs_enter_graphics();

	GLuint sem_tex[2];
	GLenum sem_layout[2];
	for (int i = 0; i < 2; ++i) {
		sem_tex[i] = enc->gltex.planes[i].gltex;
		sem_layout[i] = GL_LAYOUT_TRANSFER_SRC_EXT;
	}
	enc->glWaitSemaphoreEXT(enc->gltex.glCopySem, 0, 0, 2, sem_tex,
				sem_layout);
	for (int i = 0; i < 2; ++i) {
		GLuint read_fbo = get_read_fbo(enc, texture->tex[i]);
		enc->glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo);
		enc->glBindFramebuffer(GL_DRAW_FRAMEBUFFER,
				       enc->gltex.planes[i].fbo);
		enc->glBlitFramebuffer(0, 0, enc->gltex.planes[i].width,
				       enc->gltex.planes[i].height, 0, 0,
				       enc->gltex.planes[i].width,
				       enc->gltex.planes[i].height,
				       GL_COLOR_BUFFER_BIT, GL_NEAREST);
		enc->glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
		enc->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	}
	enc->glSignalSemaphoreEXT(enc->gltex.glsem, 0, 0, 2, sem_tex,
				  sem_layout);

	obs_leave_graphics();

	res = enc->amf_context1->CreateSurfaceFromVulkanNative(
		output_tex.surfaceVk, &amf_surf, enc);
	if (res != AMF_OK)
		throw amf_error("CreateSurfaceFromVulkanNative failed", res);

	/* ------------------------------------ */
	/* copy to submit tex                   */

	VkCommandBufferBeginInfo commandBufferBegin = {};
	commandBufferBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	VK_CHECK(vkBeginCommandBuffer(enc->cmdbuf, &commandBufferBegin));

	VkImageMemoryBarrier imageBarriers[2];
	imageBarriers[0] = {};
	imageBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageBarriers[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	imageBarriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	imageBarriers[0].image = enc->gltex.planes[0].image;
	imageBarriers[0].subresourceRange.aspectMask =
		VK_IMAGE_ASPECT_COLOR_BIT;
	imageBarriers[0].subresourceRange.layerCount = 1;
	imageBarriers[0].subresourceRange.levelCount = 1;
	imageBarriers[0].srcAccessMask = 0;
	imageBarriers[0].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	imageBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
	imageBarriers[0].dstQueueFamilyIndex = 0;
	imageBarriers[1] = {};
	imageBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	imageBarriers[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	imageBarriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	imageBarriers[1].image = enc->gltex.planes[1].image;
	imageBarriers[1].subresourceRange.aspectMask =
		VK_IMAGE_ASPECT_COLOR_BIT;
	imageBarriers[1].subresourceRange.layerCount = 1;
	imageBarriers[1].subresourceRange.levelCount = 1;
	imageBarriers[1].srcAccessMask = 0;
	imageBarriers[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	imageBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
	imageBarriers[1].dstQueueFamilyIndex = 0;
	vkCmdPipelineBarrier(enc->cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
			     nullptr, 2, imageBarriers);

	VkImageCopy imageCopy = {};
	imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageCopy.srcSubresource.mipLevel = 0;
	imageCopy.srcSubresource.baseArrayLayer = 0;
	imageCopy.srcSubresource.layerCount = 1;
	imageCopy.srcOffset.x = 0;
	imageCopy.srcOffset.y = 0;
	imageCopy.srcOffset.z = 0;
	imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
	imageCopy.dstSubresource.mipLevel = 0;
	imageCopy.dstSubresource.baseArrayLayer = 0;
	imageCopy.dstSubresource.layerCount = 1;
	imageCopy.dstOffset.x = 0;
	imageCopy.dstOffset.y = 0;
	imageCopy.dstOffset.z = 0;
	imageCopy.extent.width = enc->gltex.planes[0].width;
	imageCopy.extent.height = enc->gltex.planes[0].height;
	imageCopy.extent.depth = 1;
	vkCmdCopyImage(enc->cmdbuf, enc->gltex.planes[0].image,
		       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		       output_tex.surfaceVk->hImage, VK_IMAGE_LAYOUT_GENERAL, 1,
		       &imageCopy);

	imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
	imageCopy.extent.width = enc->gltex.planes[1].width;
	imageCopy.extent.height = enc->gltex.planes[1].height;
	vkCmdCopyImage(enc->cmdbuf, enc->gltex.planes[1].image,
		       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		       output_tex.surfaceVk->hImage, VK_IMAGE_LAYOUT_GENERAL, 1,
		       &imageCopy);

	imageBarriers[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	imageBarriers[0].dstAccessMask = 0;
	imageBarriers[0].srcQueueFamilyIndex = 0;
	imageBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
	imageBarriers[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	imageBarriers[1].dstAccessMask = 0;
	imageBarriers[1].srcQueueFamilyIndex = 0;
	imageBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
	vkCmdPipelineBarrier(enc->cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
			     VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
			     nullptr, 0, nullptr, 2, imageBarriers);

	VK_CHECK(vkEndCommandBuffer(enc->cmdbuf));

	VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &enc->cmdbuf;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &enc->gltex.sem;
	submitInfo.pWaitDstStageMask = &waitStage;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &enc->gltex.copySem;
	VK_CHECK(vkQueueSubmit(enc->queue, 1, &submitInfo,
			       enc->gltex.copyFence));

	output_tex.surfaceVk->Sync.hSemaphore = enc->gltex.copySem;
	output_tex.surfaceVk->Sync.bSubmitted = true;

	int64_t last_ts = convert_to_amf_ts(enc, pts - 1);
	int64_t cur_ts = convert_to_amf_ts(enc, pts);

	amf_surf->SetPts(cur_ts);
	amf_surf->SetProperty(L"PTS", pts);

	{
		std::scoped_lock lock(enc->textures_mutex);
		enc->active_textures[amf_surf.GetPtr()] = output_tex;
	}

	/* ------------------------------------ */
	/* do actual encode                     */

	amf_encode_base(enc, amf_surf, packet, received_packet);
	return true;

} catch (const char *err) {
	amf_texencode *enc = (amf_texencode *)data;
	error("%s: %s", __FUNCTION__, err);
	*received_packet = false;
	return false;

} catch (const amf_error &err) {
	amf_texencode *enc = (amf_texencode *)data;
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	*received_packet = false;
	return false;
}

static buf_t alloc_buf(amf_fallback *enc)
{
	buf_t buf;
	size_t size;

	if (enc->amf_format == AMF_SURFACE_NV12) {
		size = enc->linesize * enc->cy * 2;
	} else if (enc->amf_format == AMF_SURFACE_RGBA) {
		size = enc->linesize * enc->cy * 4;
	} else if (enc->amf_format == AMF_SURFACE_P010) {
		size = enc->linesize * enc->cy * 2 * 2;
	} else {
		throw "Invalid amf_format";
	}

	buf.resize(size);
	return buf;
}

static buf_t get_buf(amf_fallback *enc)
{
	std::scoped_lock lock(enc->buffers_mutex);
	buf_t buf;

	if (enc->available_buffers.size()) {
		buf = std::move(enc->available_buffers.back());
		enc->available_buffers.pop_back();
	} else {
		buf = alloc_buf(enc);
	}

	return buf;
}

static inline void copy_frame_data(amf_fallback *enc, buf_t &buf,
				   struct encoder_frame *frame)
{
	uint8_t *dst = &buf[0];

	if (enc->amf_format == AMF_SURFACE_NV12 ||
	    enc->amf_format == AMF_SURFACE_P010) {
		size_t size = enc->linesize * enc->cy;
		memcpy(&buf[0], frame->data[0], size);
		memcpy(&buf[size], frame->data[1], size / 2);

	} else if (enc->amf_format == AMF_SURFACE_RGBA) {
		memcpy(dst, frame->data[0], enc->linesize * enc->cy);
	}
}

static bool amf_encode_fallback(void *data, struct encoder_frame *frame,
				struct encoder_packet *packet,
				bool *received_packet)
try {
	amf_fallback *enc = (amf_fallback *)data;
	AMFSurfacePtr amf_surf;
	AMF_RESULT res;
	buf_t buf;

	if (!enc->linesize)
		enc->linesize = frame->linesize[0];

	buf = get_buf(enc);

	copy_frame_data(enc, buf, frame);

	res = enc->amf_context->CreateSurfaceFromHostNative(
		enc->amf_format, enc->cx, enc->cy, enc->linesize, 0, &buf[0],
		&amf_surf, enc);
	if (res != AMF_OK)
		throw amf_error("CreateSurfaceFromHostNative failed", res);

	int64_t last_ts = convert_to_amf_ts(enc, frame->pts - 1);
	int64_t cur_ts = convert_to_amf_ts(enc, frame->pts);

	amf_surf->SetPts(cur_ts);
	amf_surf->SetProperty(L"PTS", frame->pts);

	{
		std::scoped_lock lock(enc->buffers_mutex);
		enc->active_buffers[amf_surf.GetPtr()] = std::move(buf);
	}

	/* ------------------------------------ */
	/* do actual encode                     */

	amf_encode_base(enc, amf_surf, packet, received_packet);
	return true;

} catch (const amf_error &err) {
	amf_fallback *enc = (amf_fallback *)data;
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	*received_packet = false;
	return false;
} catch (const char *err) {
	amf_fallback *enc = (amf_fallback *)data;
	error("%s: %s", __FUNCTION__, err);
	*received_packet = false;
	return false;
}

static bool amf_extra_data(void *data, uint8_t **header, size_t *size)
{
	amf_base *enc = (amf_base *)data;
	if (!enc->header)
		return false;

	*header = (uint8_t *)enc->header->GetNative();
	*size = enc->header->GetSize();
	return true;
}

static void h264_video_info_fallback(void *, struct video_scale_info *info)
{
	switch (info->format) {
	case VIDEO_FORMAT_RGBA:
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
		info->format = VIDEO_FORMAT_RGBA;
		break;
	default:
		info->format = VIDEO_FORMAT_NV12;
		break;
	}
}

static void h265_video_info_fallback(void *, struct video_scale_info *info)
{
	switch (info->format) {
	case VIDEO_FORMAT_RGBA:
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
		info->format = VIDEO_FORMAT_RGBA;
		break;
	case VIDEO_FORMAT_I010:
	case VIDEO_FORMAT_P010:
		info->format = VIDEO_FORMAT_P010;
		break;
	default:
		info->format = VIDEO_FORMAT_NV12;
	}
}

static void av1_video_info_fallback(void *, struct video_scale_info *info)
{
	switch (info->format) {
	case VIDEO_FORMAT_RGBA:
	case VIDEO_FORMAT_BGRA:
	case VIDEO_FORMAT_BGRX:
		info->format = VIDEO_FORMAT_RGBA;
		break;
	case VIDEO_FORMAT_I010:
	case VIDEO_FORMAT_P010:
		info->format = VIDEO_FORMAT_P010;
		break;
	default:
		info->format = VIDEO_FORMAT_NV12;
	}
}

static bool amf_create_encoder(amf_base *enc)
try {
	AMF_RESULT res;

	/* ------------------------------------ */
	/* get video info                       */

	struct obs_video_info ovi;
	obs_get_video_info(&ovi);

	struct video_scale_info info;
	info.format = ovi.output_format;
	info.colorspace = ovi.colorspace;
	info.range = ovi.range;

	if (enc->fallback) {
		if (enc->codec == amf_codec_type::AVC)
			h264_video_info_fallback(NULL, &info);
		else if (enc->codec == amf_codec_type::HEVC)
			h265_video_info_fallback(NULL, &info);
		else
			av1_video_info_fallback(NULL, &info);
	}

	enc->cx = obs_encoder_get_width(enc->encoder);
	enc->cy = obs_encoder_get_height(enc->encoder);
	enc->amf_frame_rate = AMFConstructRate(ovi.fps_num, ovi.fps_den);
	enc->fps_num = (int)ovi.fps_num;
	enc->fps_den = (int)ovi.fps_den;
	enc->full_range = info.range == VIDEO_RANGE_FULL;

	switch (info.colorspace) {
	case VIDEO_CS_601:
		enc->amf_color_profile =
			enc->full_range
				? AMF_VIDEO_CONVERTER_COLOR_PROFILE_FULL_601
				: AMF_VIDEO_CONVERTER_COLOR_PROFILE_601;
		enc->amf_primaries = AMF_COLOR_PRIMARIES_SMPTE170M;
		enc->amf_characteristic =
			AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE170M;
		break;
	case VIDEO_CS_DEFAULT:
	case VIDEO_CS_709:
		enc->amf_color_profile =
			enc->full_range
				? AMF_VIDEO_CONVERTER_COLOR_PROFILE_FULL_709
				: AMF_VIDEO_CONVERTER_COLOR_PROFILE_709;
		enc->amf_primaries = AMF_COLOR_PRIMARIES_BT709;
		enc->amf_characteristic =
			AMF_COLOR_TRANSFER_CHARACTERISTIC_BT709;
		break;
	case VIDEO_CS_SRGB:
		enc->amf_color_profile =
			enc->full_range
				? AMF_VIDEO_CONVERTER_COLOR_PROFILE_FULL_709
				: AMF_VIDEO_CONVERTER_COLOR_PROFILE_709;
		enc->amf_primaries = AMF_COLOR_PRIMARIES_BT709;
		enc->amf_characteristic =
			AMF_COLOR_TRANSFER_CHARACTERISTIC_IEC61966_2_1;
		break;
	case VIDEO_CS_2100_HLG:
		enc->amf_color_profile =
			enc->full_range
				? AMF_VIDEO_CONVERTER_COLOR_PROFILE_FULL_2020
				: AMF_VIDEO_CONVERTER_COLOR_PROFILE_2020;
		enc->amf_primaries = AMF_COLOR_PRIMARIES_BT2020;
		enc->amf_characteristic =
			AMF_COLOR_TRANSFER_CHARACTERISTIC_ARIB_STD_B67;
		break;
	case VIDEO_CS_2100_PQ:
		enc->amf_color_profile =
			enc->full_range
				? AMF_VIDEO_CONVERTER_COLOR_PROFILE_FULL_2020
				: AMF_VIDEO_CONVERTER_COLOR_PROFILE_2020;
		enc->amf_primaries = AMF_COLOR_PRIMARIES_BT2020;
		enc->amf_characteristic =
			AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE2084;
		break;
	}

	switch (info.format) {
	case VIDEO_FORMAT_NV12:
		enc->amf_format = AMF_SURFACE_NV12;
		break;
	case VIDEO_FORMAT_P010:
		enc->amf_format = AMF_SURFACE_P010;
		break;
	case VIDEO_FORMAT_RGBA:
		enc->amf_format = AMF_SURFACE_RGBA;
		break;
	}

	/* ------------------------------------ */
	/* create encoder                       */

	res = amf_factory->CreateContext(&enc->amf_context);
	if (res != AMF_OK)
		throw amf_error("CreateContext failed", res);

	enc->amf_context1 = AMFContext1Ptr(enc->amf_context);

	enc->init();

	const wchar_t *codec = nullptr;
	switch (enc->codec) {
	case (amf_codec_type::AVC):
		codec = AMFVideoEncoderVCE_AVC;
		break;
	case (amf_codec_type::HEVC):
		codec = AMFVideoEncoder_HEVC;
		break;
	case (amf_codec_type::AV1):
		codec = AMFVideoEncoder_AV1;
		break;
	default:
		codec = AMFVideoEncoder_HEVC;
	}
	res = amf_factory->CreateComponent(enc->amf_context, codec,
					   &enc->amf_encoder);
	if (res != AMF_OK)
		throw amf_error("CreateComponent failed", res);

	calc_throughput(enc);
	return true;

} catch (const amf_error &err) {
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	return false;
}

static void amf_destroy(void *data)
{
	amf_base *enc = (amf_base *)data;
	delete enc;
}

static void check_texture_encode_capability(obs_encoder_t *encoder,
					    amf_codec_type codec)
{
	obs_video_info ovi;
	obs_get_video_info(&ovi);
	bool avc = amf_codec_type::AVC == codec;
	bool hevc = amf_codec_type::HEVC == codec;
	bool av1 = amf_codec_type::AV1 == codec;

	if (obs_encoder_scaling_enabled(encoder))
		throw "Encoder scaling is active";

	if (hevc || av1) {
		if (!obs_nv12_tex_active() && !obs_p010_tex_active())
			throw "NV12/P010 textures aren't active";
	} else if (!obs_nv12_tex_active()) {
		throw "NV12 textures aren't active";
	}

	video_t *video = obs_encoder_video(encoder);
	const struct video_output_info *voi = video_output_get_info(video);
	switch (voi->format) {
	case VIDEO_FORMAT_I010:
	case VIDEO_FORMAT_P010:
		break;
	default:
		switch (voi->colorspace) {
		case VIDEO_CS_2100_PQ:
		case VIDEO_CS_2100_HLG:
			throw "OBS does not support 8-bit output of Rec. 2100";
		}
	}

	if ((avc && !caps[ovi.adapter].supports_avc) ||
	    (hevc && !caps[ovi.adapter].supports_hevc) ||
	    (av1 && !caps[ovi.adapter].supports_av1))
		throw "Wrong adapter";
}

#include "texture-amf-opts.hpp"

static void amf_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "bitrate", 2500);
	obs_data_set_default_int(settings, "cqp", 20);
	obs_data_set_default_string(settings, "rate_control", "CBR");
	obs_data_set_default_string(settings, "preset", "quality");
	obs_data_set_default_string(settings, "profile", "high");
}

static bool rate_control_modified(obs_properties_t *ppts, obs_property_t *p,
				  obs_data_t *settings)
{
	const char *rc = obs_data_get_string(settings, "rate_control");
	bool cqp = astrcmpi(rc, "CQP") == 0;
	bool qvbr = astrcmpi(rc, "QVBR") == 0;

	p = obs_properties_get(ppts, "bitrate");
	obs_property_set_visible(p, !cqp && !qvbr);
	p = obs_properties_get(ppts, "cqp");
	obs_property_set_visible(p, cqp || qvbr);
	return true;
}

static obs_properties_t *amf_properties_internal(amf_codec_type codec)
{
	obs_properties_t *props = obs_properties_create();
	obs_property_t *p;

	p = obs_properties_add_list(props, "rate_control",
				    obs_module_text("RateControl"),
				    OBS_COMBO_TYPE_LIST,
				    OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p, "CBR", "CBR");
	obs_property_list_add_string(p, "CQP", "CQP");
	obs_property_list_add_string(p, "VBR", "VBR");
	obs_property_list_add_string(p, "VBR_LAT", "VBR_LAT");
	obs_property_list_add_string(p, "QVBR", "QVBR");
	obs_property_list_add_string(p, "HQVBR", "HQVBR");
	obs_property_list_add_string(p, "HQCBR", "HQCBR");

	obs_property_set_modified_callback(p, rate_control_modified);

	p = obs_properties_add_int(props, "bitrate", obs_module_text("Bitrate"),
				   50, 100000, 50);
	obs_property_int_set_suffix(p, " Kbps");

	obs_properties_add_int(props, "cqp", obs_module_text("NVENC.CQLevel"),
			       0, codec == amf_codec_type::AV1 ? 63 : 51, 1);

	p = obs_properties_add_int(props, "keyint_sec",
				   obs_module_text("KeyframeIntervalSec"), 0,
				   10, 1);
	obs_property_int_set_suffix(p, " s");

	p = obs_properties_add_list(props, "preset", obs_module_text("Preset"),
				    OBS_COMBO_TYPE_LIST,
				    OBS_COMBO_FORMAT_STRING);

#define add_preset(val) \
	obs_property_list_add_string(p, obs_module_text("AMF.Preset." val), val)
	if (amf_codec_type::AV1 == codec) {
		add_preset("highQuality");
	}
	add_preset("quality");
	add_preset("balanced");
	add_preset("speed");
#undef add_preset

	if (amf_codec_type::AVC == codec || amf_codec_type::AV1 == codec) {
		p = obs_properties_add_list(props, "profile",
					    obs_module_text("Profile"),
					    OBS_COMBO_TYPE_LIST,
					    OBS_COMBO_FORMAT_STRING);

#define add_profile(val) obs_property_list_add_string(p, val, val)
		if (amf_codec_type::AVC == codec)
			add_profile("high");
		add_profile("main");
		if (amf_codec_type::AVC == codec)
			add_profile("baseline");
#undef add_profile
	}

	if (amf_codec_type::AVC == codec) {
		obs_properties_add_int(props, "bf", obs_module_text("BFrames"),
				       0, 5, 1);
	}

	p = obs_properties_add_text(props, "ffmpeg_opts",
				    obs_module_text("AMFOpts"),
				    OBS_TEXT_DEFAULT);
	obs_property_set_long_description(p,
					  obs_module_text("AMFOpts.ToolTip"));

	return props;
}

static obs_properties_t *amf_avc_properties(void *unused)
{
	UNUSED_PARAMETER(unused);
	return amf_properties_internal(amf_codec_type::AVC);
}

static obs_properties_t *amf_hevc_properties(void *unused)
{
	UNUSED_PARAMETER(unused);
	return amf_properties_internal(amf_codec_type::HEVC);
}

static obs_properties_t *amf_av1_properties(void *unused)
{
	UNUSED_PARAMETER(unused);
	return amf_properties_internal(amf_codec_type::AV1);
}

/* ========================================================================= */
/* AVC Implementation                                                        */

static const char *amf_avc_get_name(void *)
{
	return "AMD HW H.264 (AVC)";
}

static inline int get_avc_preset(amf_base *enc, const char *preset)
{
	UNUSED_PARAMETER(enc);
	if (astrcmpi(preset, "quality") == 0)
		return AMF_VIDEO_ENCODER_QUALITY_PRESET_QUALITY;
	else if (astrcmpi(preset, "speed") == 0)
		return AMF_VIDEO_ENCODER_QUALITY_PRESET_SPEED;

	return AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED;
}

static inline int get_avc_rate_control(const char *rc_str)
{
	if (astrcmpi(rc_str, "cqp") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP;
	else if (astrcmpi(rc_str, "cbr") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR;
	else if (astrcmpi(rc_str, "vbr") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR;
	else if (astrcmpi(rc_str, "vbr_lat") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR;
	else if (astrcmpi(rc_str, "qvbr") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR;
	else if (astrcmpi(rc_str, "hqvbr") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_HIGH_QUALITY_VBR;
	else if (astrcmpi(rc_str, "hqcbr") == 0)
		return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_HIGH_QUALITY_CBR;

	return AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR;
}

static inline int get_avc_profile(obs_data_t *settings)
{
	const char *profile = obs_data_get_string(settings, "profile");

	if (astrcmpi(profile, "baseline") == 0)
		return AMF_VIDEO_ENCODER_PROFILE_BASELINE;
	else if (astrcmpi(profile, "main") == 0)
		return AMF_VIDEO_ENCODER_PROFILE_MAIN;
	else if (astrcmpi(profile, "constrained_baseline") == 0)
		return AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_BASELINE;
	else if (astrcmpi(profile, "constrained_high") == 0)
		return AMF_VIDEO_ENCODER_PROFILE_CONSTRAINED_HIGH;

	return AMF_VIDEO_ENCODER_PROFILE_HIGH;
}

static void amf_avc_update_data(amf_base *enc, int rc, int64_t bitrate,
				int64_t qp)
{
	if (rc != AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP &&
	    rc != AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_QUALITY_VBR) {
		set_avc_property(enc, TARGET_BITRATE, bitrate);
		set_avc_property(enc, PEAK_BITRATE, bitrate);
		set_avc_property(enc, VBV_BUFFER_SIZE, bitrate);

		if (rc == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR) {
			set_avc_property(enc, FILLER_DATA_ENABLE, true);
		}
	} else {
		set_avc_property(enc, QP_I, qp);
		set_avc_property(enc, QP_P, qp);
		set_avc_property(enc, QP_B, qp);
		set_avc_property(enc, QVBR_QUALITY_LEVEL, qp);
	}
}

static bool amf_avc_update(void *data, obs_data_t *settings)
try {
	amf_base *enc = (amf_base *)data;

	if (enc->first_update) {
		enc->first_update = false;
		return true;
	}

	int64_t bitrate = obs_data_get_int(settings, "bitrate");
	int64_t qp = obs_data_get_int(settings, "cqp");
	const char *rc_str = obs_data_get_string(settings, "rate_control");
	int rc = get_avc_rate_control(rc_str);
	AMF_RESULT res;

	amf_avc_update_data(enc, rc, bitrate * 1000, qp);

	res = enc->amf_encoder->ReInit(enc->cx, enc->cy);
	if (res != AMF_OK)
		throw amf_error("AMFComponent::Init failed", res);

	return true;

} catch (const amf_error &err) {
	amf_base *enc = (amf_base *)data;
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	return false;
}

static bool amf_avc_init(void *data, obs_data_t *settings)
{
	amf_base *enc = (amf_base *)data;

	int64_t bitrate = obs_data_get_int(settings, "bitrate");
	int64_t qp = obs_data_get_int(settings, "cqp");
	const char *preset = obs_data_get_string(settings, "preset");
	const char *profile = obs_data_get_string(settings, "profile");
	const char *rc_str = obs_data_get_string(settings, "rate_control");
	int64_t bf = obs_data_get_int(settings, "bf");

	if (enc->bframes_supported) {
		set_avc_property(enc, MAX_CONSECUTIVE_BPICTURES, 3);
		set_avc_property(enc, B_PIC_PATTERN, bf);

	} else if (bf != 0) {
		warn("B-Frames set to %" PRId64 " but b-frames are not "
		     "supported by this device",
		     bf);
		bf = 0;
	}

	int rc = get_avc_rate_control(rc_str);

	set_avc_property(enc, RATE_CONTROL_METHOD, rc);
	if (rc != AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTANT_QP)
		set_avc_property(enc, ENABLE_VBAQ, true);

	amf_avc_update_data(enc, rc, bitrate * 1000, qp);

	set_avc_property(enc, ENFORCE_HRD, true);
	set_avc_property(enc, HIGH_MOTION_QUALITY_BOOST_ENABLE, false);

	int keyint_sec = (int)obs_data_get_int(settings, "keyint_sec");
	int gop_size = (keyint_sec) ? keyint_sec * enc->fps_num / enc->fps_den
				    : 250;

	set_avc_property(enc, IDR_PERIOD, gop_size);

	bool repeat_headers = obs_data_get_bool(settings, "repeat_headers");
	if (repeat_headers)
		set_avc_property(enc, HEADER_INSERTION_SPACING, gop_size);

	set_avc_property(enc, DE_BLOCKING_FILTER, true);

	check_preset_compatibility(enc, preset);

	const char *ffmpeg_opts = obs_data_get_string(settings, "ffmpeg_opts");
	if (ffmpeg_opts && *ffmpeg_opts) {
		struct obs_options opts = obs_parse_options(ffmpeg_opts);
		for (size_t i = 0; i < opts.count; i++) {
			amf_apply_opt(enc, &opts.options[i]);
		}
		obs_free_options(opts);
	}

	if (!ffmpeg_opts || !*ffmpeg_opts)
		ffmpeg_opts = "(none)";

	info("settings:\n"
	     "\trate_control: %s\n"
	     "\tbitrate:      %" PRId64 "\n"
	     "\tcqp:          %" PRId64 "\n"
	     "\tkeyint:       %d\n"
	     "\tpreset:       %s\n"
	     "\tprofile:      %s\n"
	     "\tb-frames:     %" PRId64 "\n"
	     "\twidth:        %d\n"
	     "\theight:       %d\n"
	     "\tparams:       %s",
	     rc_str, bitrate, qp, gop_size, preset, profile, bf, enc->cx,
	     enc->cy, ffmpeg_opts);

	return true;
}

static void amf_avc_create_internal(amf_base *enc, obs_data_t *settings)
{
	AMF_RESULT res;
	AMFVariant p;

	enc->codec = amf_codec_type::AVC;

	if (!amf_create_encoder(enc))
		throw "Failed to create encoder";

	AMFCapsPtr caps;
	res = enc->amf_encoder->GetCaps(&caps);
	if (res == AMF_OK) {
		caps->GetProperty(AMF_VIDEO_ENCODER_CAP_BFRAMES,
				  &enc->bframes_supported);
		caps->GetProperty(AMF_VIDEO_ENCODER_CAP_MAX_THROUGHPUT,
				  &enc->max_throughput);
	}

	const char *preset = obs_data_get_string(settings, "preset");

	set_avc_property(enc, FRAMESIZE, AMFConstructSize(enc->cx, enc->cy));
	set_avc_property(enc, USAGE, AMF_VIDEO_ENCODER_USAGE_TRANSCODING);
	set_avc_property(enc, QUALITY_PRESET, get_avc_preset(enc, preset));
	set_avc_property(enc, PROFILE, get_avc_profile(settings));
	set_avc_property(enc, LOWLATENCY_MODE, false);
	set_avc_property(enc, CABAC_ENABLE, AMF_VIDEO_ENCODER_UNDEFINED);
	set_avc_property(enc, PREENCODE_ENABLE, true);
	set_avc_property(enc, OUTPUT_COLOR_PROFILE, enc->amf_color_profile);
	set_avc_property(enc, OUTPUT_TRANSFER_CHARACTERISTIC,
			 enc->amf_characteristic);
	set_avc_property(enc, OUTPUT_COLOR_PRIMARIES, enc->amf_primaries);
	set_avc_property(enc, FULL_RANGE_COLOR, enc->full_range);

	amf_avc_init(enc, settings);

	res = enc->amf_encoder->Init(enc->amf_format, enc->cx, enc->cy);
	if (res != AMF_OK)
		throw amf_error("AMFComponent::Init failed", res);

	set_avc_property(enc, FRAMERATE, enc->amf_frame_rate);

	res = enc->amf_encoder->GetProperty(AMF_VIDEO_ENCODER_EXTRADATA, &p);
	if (res == AMF_OK && p.type == AMF_VARIANT_INTERFACE)
		enc->header = AMFBufferPtr(p.pInterface);

	if (enc->bframes_supported) {
		amf_int64 b_frames = 0;
		amf_int64 b_max = 0;

		if (get_avc_property(enc, B_PIC_PATTERN, &b_frames) &&
		    get_avc_property(enc, MAX_CONSECUTIVE_BPICTURES, &b_max))
			enc->dts_offset = b_frames + 1;
		else
			enc->dts_offset = 0;
	}
}

static void *amf_avc_create_texencode(obs_data_t *settings,
				      obs_encoder_t *encoder)
try {
	check_texture_encode_capability(encoder, amf_codec_type::AVC);

	std::unique_ptr<amf_texencode> enc = std::make_unique<amf_texencode>();
	enc->encoder = encoder;
	enc->encoder_str = "texture-amf-h264";

#ifdef _WIN32
	if (!amf_init_d3d11(enc.get()))
		throw "Failed to create D3D11";
#endif

	amf_avc_create_internal(enc.get(), settings);
	return enc.release();

} catch (const amf_error &err) {
	blog(LOG_ERROR, "[texture-amf-h264] %s: %s: %ls", __FUNCTION__, err.str,
	     amf_trace->GetResultText(err.res));
	return obs_encoder_create_rerouted(encoder, "h264_fallback_amf");

} catch (const char *err) {
	blog(LOG_ERROR, "[texture-amf-h264] %s: %s", __FUNCTION__, err);
	return obs_encoder_create_rerouted(encoder, "h264_fallback_amf");
}

static void *amf_avc_create_fallback(obs_data_t *settings,
				     obs_encoder_t *encoder)
try {
	std::unique_ptr<amf_fallback> enc = std::make_unique<amf_fallback>();
	enc->encoder = encoder;
	enc->encoder_str = "fallback-amf-h264";

	video_t *video = obs_encoder_video(encoder);
	const struct video_output_info *voi = video_output_get_info(video);
	switch (voi->format) {
	case VIDEO_FORMAT_I010:
	case VIDEO_FORMAT_P010: {
		const char *const text =
			obs_module_text("AMF.10bitUnsupportedAvc");
		obs_encoder_set_last_error(encoder, text);
		throw text;
	}
	default:
		switch (voi->colorspace) {
		case VIDEO_CS_2100_PQ:
		case VIDEO_CS_2100_HLG: {
			const char *const text =
				obs_module_text("AMF.8bitUnsupportedHdr");
			obs_encoder_set_last_error(encoder, text);
			throw text;
		}
		}
	}

	amf_avc_create_internal(enc.get(), settings);
	return enc.release();

} catch (const amf_error &err) {
	blog(LOG_ERROR, "[fallback-amf-h264] %s: %s: %ls", __FUNCTION__,
	     err.str, amf_trace->GetResultText(err.res));
	return nullptr;

} catch (const char *err) {
	blog(LOG_ERROR, "[fallback-amf-h264] %s: %s", __FUNCTION__, err);
	return nullptr;
}

static void register_avc()
{
	struct obs_encoder_info amf_encoder_info = {};
	amf_encoder_info.id = "h264_texture_amf";
	amf_encoder_info.type = OBS_ENCODER_VIDEO;
	amf_encoder_info.codec = "h264";
	amf_encoder_info.get_name = amf_avc_get_name;
	amf_encoder_info.create = amf_avc_create_texencode;
	amf_encoder_info.destroy = amf_destroy;
	/* FIXME: Figure out why encoder does not survive reconfiguration
	amf_encoder_info.update = amf_avc_update; */
	amf_encoder_info.encode_texture = amf_encode_tex;
	amf_encoder_info.encode_texture2 = amf_encode_tex2;
	amf_encoder_info.get_defaults = amf_defaults;
	amf_encoder_info.get_properties = amf_avc_properties;
	amf_encoder_info.get_extra_data = amf_extra_data;
	amf_encoder_info.caps = OBS_ENCODER_CAP_PASS_TEXTURE;

	obs_register_encoder(&amf_encoder_info);

	amf_encoder_info.id = "h264_fallback_amf";
	amf_encoder_info.caps = OBS_ENCODER_CAP_INTERNAL |
				OBS_ENCODER_CAP_DYN_BITRATE;
	amf_encoder_info.encode_texture = nullptr;
	amf_encoder_info.encode_texture2 = nullptr;
	amf_encoder_info.create = amf_avc_create_fallback;
	amf_encoder_info.encode = amf_encode_fallback;
	amf_encoder_info.get_video_info = h264_video_info_fallback;

	obs_register_encoder(&amf_encoder_info);
}

/* ========================================================================= */
/* HEVC Implementation                                                       */

#if ENABLE_HEVC

static const char *amf_hevc_get_name(void *)
{
	return "AMD HW H.265 (HEVC)";
}

static inline int get_hevc_preset(amf_base *enc, const char *preset)
{
	UNUSED_PARAMETER(enc);
	if (astrcmpi(preset, "balanced") == 0)
		return AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_BALANCED;
	else if (astrcmpi(preset, "speed") == 0)
		return AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_SPEED;

	return AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_QUALITY;
}

static inline int get_hevc_rate_control(const char *rc_str)
{
	if (astrcmpi(rc_str, "cqp") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP;
	else if (astrcmpi(rc_str, "vbr_lat") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR;
	else if (astrcmpi(rc_str, "vbr") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR;
	else if (astrcmpi(rc_str, "cbr") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR;
	else if (astrcmpi(rc_str, "qvbr") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_QUALITY_VBR;
	else if (astrcmpi(rc_str, "hqvbr") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_HIGH_QUALITY_VBR;
	else if (astrcmpi(rc_str, "hqcbr") == 0)
		return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_HIGH_QUALITY_CBR;

	return AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR;
}

static void amf_hevc_update_data(amf_base *enc, int rc, int64_t bitrate,
				 int64_t qp)
{
	if (rc != AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP &&
	    rc != AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_QUALITY_VBR) {
		set_hevc_property(enc, TARGET_BITRATE, bitrate);
		set_hevc_property(enc, PEAK_BITRATE, bitrate);
		set_hevc_property(enc, VBV_BUFFER_SIZE, bitrate);

		if (rc == AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR) {
			set_hevc_property(enc, FILLER_DATA_ENABLE, true);
		}
	} else {
		set_hevc_property(enc, QP_I, qp);
		set_hevc_property(enc, QP_P, qp);
		set_hevc_property(enc, QVBR_QUALITY_LEVEL, qp);
	}
}

static bool amf_hevc_update(void *data, obs_data_t *settings)
try {
	amf_base *enc = (amf_base *)data;

	if (enc->first_update) {
		enc->first_update = false;
		return true;
	}

	int64_t bitrate = obs_data_get_int(settings, "bitrate");
	int64_t qp = obs_data_get_int(settings, "cqp");
	const char *rc_str = obs_data_get_string(settings, "rate_control");
	int rc = get_hevc_rate_control(rc_str);
	AMF_RESULT res;

	amf_hevc_update_data(enc, rc, bitrate * 1000, qp);

	res = enc->amf_encoder->ReInit(enc->cx, enc->cy);
	if (res != AMF_OK)
		throw amf_error("AMFComponent::Init failed", res);

	return true;

} catch (const amf_error &err) {
	amf_base *enc = (amf_base *)data;
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	return false;
}

static bool amf_hevc_init(void *data, obs_data_t *settings)
{
	amf_base *enc = (amf_base *)data;

	int64_t bitrate = obs_data_get_int(settings, "bitrate");
	int64_t qp = obs_data_get_int(settings, "cqp");
	const char *preset = obs_data_get_string(settings, "preset");
	const char *profile = obs_data_get_string(settings, "profile");
	const char *rc_str = obs_data_get_string(settings, "rate_control");
	int rc = get_hevc_rate_control(rc_str);

	set_hevc_property(enc, RATE_CONTROL_METHOD, rc);
	if (rc != AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CONSTANT_QP)
		set_hevc_property(enc, ENABLE_VBAQ, true);

	amf_hevc_update_data(enc, rc, bitrate * 1000, qp);

	set_hevc_property(enc, ENFORCE_HRD, true);
	set_hevc_property(enc, HIGH_MOTION_QUALITY_BOOST_ENABLE, false);

	int keyint_sec = (int)obs_data_get_int(settings, "keyint_sec");
	int gop_size = (keyint_sec) ? keyint_sec * enc->fps_num / enc->fps_den
				    : 250;

	set_hevc_property(enc, GOP_SIZE, gop_size);

	check_preset_compatibility(enc, preset);

	const char *ffmpeg_opts = obs_data_get_string(settings, "ffmpeg_opts");
	if (ffmpeg_opts && *ffmpeg_opts) {
		struct obs_options opts = obs_parse_options(ffmpeg_opts);
		for (size_t i = 0; i < opts.count; i++) {
			amf_apply_opt(enc, &opts.options[i]);
		}
		obs_free_options(opts);
	}

	if (!ffmpeg_opts || !*ffmpeg_opts)
		ffmpeg_opts = "(none)";

	info("settings:\n"
	     "\trate_control: %s\n"
	     "\tbitrate:      %" PRId64 "\n"
	     "\tcqp:          %" PRId64 "\n"
	     "\tkeyint:       %d\n"
	     "\tpreset:       %s\n"
	     "\tprofile:      %s\n"
	     "\twidth:        %d\n"
	     "\theight:       %d\n"
	     "\tparams:       %s",
	     rc_str, bitrate, qp, gop_size, preset, profile, enc->cx, enc->cy,
	     ffmpeg_opts);

	return true;
}

static inline bool is_hlg(amf_base *enc)
{
	return enc->amf_characteristic ==
	       AMF_COLOR_TRANSFER_CHARACTERISTIC_ARIB_STD_B67;
}

static inline bool is_pq(amf_base *enc)
{
	return enc->amf_characteristic ==
	       AMF_COLOR_TRANSFER_CHARACTERISTIC_SMPTE2084;
}

constexpr amf_uint16 amf_hdr_primary(uint32_t num, uint32_t den)
{
	return (amf_uint16)(num * 50000 / den);
}

constexpr amf_uint32 lum_mul = 10000;

constexpr amf_uint32 amf_make_lum(amf_uint32 val)
{
	return val * lum_mul;
}

static void amf_hevc_create_internal(amf_base *enc, obs_data_t *settings)
{
	AMF_RESULT res;
	AMFVariant p;

	enc->codec = amf_codec_type::HEVC;

	if (!amf_create_encoder(enc))
		throw "Failed to create encoder";

	AMFCapsPtr caps;
	res = enc->amf_encoder->GetCaps(&caps);
	if (res == AMF_OK) {
		caps->GetProperty(AMF_VIDEO_ENCODER_HEVC_CAP_MAX_THROUGHPUT,
				  &enc->max_throughput);
	}

	const bool is10bit = enc->amf_format == AMF_SURFACE_P010;
	const bool pq = is_pq(enc);
	const bool hlg = is_hlg(enc);
	const bool is_hdr = pq || hlg;
	const char *preset = obs_data_get_string(settings, "preset");

	set_hevc_property(enc, FRAMESIZE, AMFConstructSize(enc->cx, enc->cy));
	set_hevc_property(enc, USAGE, AMF_VIDEO_ENCODER_USAGE_TRANSCODING);
	set_hevc_property(enc, QUALITY_PRESET, get_hevc_preset(enc, preset));
	set_hevc_property(enc, COLOR_BIT_DEPTH,
			  is10bit ? AMF_COLOR_BIT_DEPTH_10
				  : AMF_COLOR_BIT_DEPTH_8);
	set_hevc_property(enc, PROFILE,
			  is10bit ? AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10
				  : AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN);
	set_hevc_property(enc, LOWLATENCY_MODE, false);
	set_hevc_property(enc, OUTPUT_COLOR_PROFILE, enc->amf_color_profile);
	set_hevc_property(enc, OUTPUT_TRANSFER_CHARACTERISTIC,
			  enc->amf_characteristic);
	set_hevc_property(enc, OUTPUT_COLOR_PRIMARIES, enc->amf_primaries);
	set_hevc_property(enc, NOMINAL_RANGE, enc->full_range);

	if (is_hdr) {
		const int hdr_nominal_peak_level =
			pq ? (int)obs_get_video_hdr_nominal_peak_level()
			   : (hlg ? 1000 : 0);

		AMFBufferPtr buf;
		enc->amf_context->AllocBuffer(AMF_MEMORY_HOST,
					      sizeof(AMFHDRMetadata), &buf);
		AMFHDRMetadata *md = (AMFHDRMetadata *)buf->GetNative();
		md->redPrimary[0] = amf_hdr_primary(17, 25);
		md->redPrimary[1] = amf_hdr_primary(8, 25);
		md->greenPrimary[0] = amf_hdr_primary(53, 200);
		md->greenPrimary[1] = amf_hdr_primary(69, 100);
		md->bluePrimary[0] = amf_hdr_primary(3, 20);
		md->bluePrimary[1] = amf_hdr_primary(3, 50);
		md->whitePoint[0] = amf_hdr_primary(3127, 10000);
		md->whitePoint[1] = amf_hdr_primary(329, 1000);
		md->minMasteringLuminance = 0;
		md->maxMasteringLuminance =
			amf_make_lum(hdr_nominal_peak_level);
		md->maxContentLightLevel = hdr_nominal_peak_level;
		md->maxFrameAverageLightLevel = hdr_nominal_peak_level;
		set_hevc_property(enc, INPUT_HDR_METADATA, buf);
	}

	amf_hevc_init(enc, settings);

	res = enc->amf_encoder->Init(enc->amf_format, enc->cx, enc->cy);
	if (res != AMF_OK)
		throw amf_error("AMFComponent::Init failed", res);

	set_hevc_property(enc, FRAMERATE, enc->amf_frame_rate);

	res = enc->amf_encoder->GetProperty(AMF_VIDEO_ENCODER_HEVC_EXTRADATA,
					    &p);
	if (res == AMF_OK && p.type == AMF_VARIANT_INTERFACE)
		enc->header = AMFBufferPtr(p.pInterface);
}

static void *amf_hevc_create_texencode(obs_data_t *settings,
				       obs_encoder_t *encoder)
try {
	check_texture_encode_capability(encoder, amf_codec_type::HEVC);

	std::unique_ptr<amf_texencode> enc = std::make_unique<amf_texencode>();
	enc->encoder = encoder;
	enc->encoder_str = "texture-amf-h265";

#ifdef _WIN32
	if (!amf_init_d3d11(enc.get()))
		throw "Failed to create D3D11";
#endif

	amf_hevc_create_internal(enc.get(), settings);
	return enc.release();

} catch (const amf_error &err) {
	blog(LOG_ERROR, "[texture-amf-h265] %s: %s: %ls", __FUNCTION__, err.str,
	     amf_trace->GetResultText(err.res));
	return obs_encoder_create_rerouted(encoder, "h265_fallback_amf");

} catch (const char *err) {
	blog(LOG_ERROR, "[texture-amf-h265] %s: %s", __FUNCTION__, err);
	return obs_encoder_create_rerouted(encoder, "h265_fallback_amf");
}

static void *amf_hevc_create_fallback(obs_data_t *settings,
				      obs_encoder_t *encoder)
try {
	std::unique_ptr<amf_fallback> enc = std::make_unique<amf_fallback>();
	enc->encoder = encoder;
	enc->encoder_str = "fallback-amf-h265";

	video_t *video = obs_encoder_video(encoder);
	const struct video_output_info *voi = video_output_get_info(video);
	switch (voi->format) {
	case VIDEO_FORMAT_I010:
	case VIDEO_FORMAT_P010:
		break;
	default:
		switch (voi->colorspace) {
		case VIDEO_CS_2100_PQ:
		case VIDEO_CS_2100_HLG: {
			const char *const text =
				obs_module_text("AMF.8bitUnsupportedHdr");
			obs_encoder_set_last_error(encoder, text);
			throw text;
		}
		}
	}

	amf_hevc_create_internal(enc.get(), settings);
	return enc.release();

} catch (const amf_error &err) {
	blog(LOG_ERROR, "[fallback-amf-h265] %s: %s: %ls", __FUNCTION__,
	     err.str, amf_trace->GetResultText(err.res));
	return nullptr;

} catch (const char *err) {
	blog(LOG_ERROR, "[fallback-amf-h265] %s: %s", __FUNCTION__, err);
	return nullptr;
}

static void register_hevc()
{
	struct obs_encoder_info amf_encoder_info = {};
	amf_encoder_info.id = "h265_texture_amf";
	amf_encoder_info.type = OBS_ENCODER_VIDEO;
	amf_encoder_info.codec = "hevc";
	amf_encoder_info.get_name = amf_hevc_get_name;
	amf_encoder_info.create = amf_hevc_create_texencode;
	amf_encoder_info.destroy = amf_destroy;
	/* FIXME: Figure out why encoder does not survive reconfiguration
	amf_encoder_info.update = amf_hevc_update; */
	amf_encoder_info.encode_texture = amf_encode_tex;
	amf_encoder_info.encode_texture2 = amf_encode_tex2;
	amf_encoder_info.get_defaults = amf_defaults;
	amf_encoder_info.get_properties = amf_hevc_properties;
	amf_encoder_info.get_extra_data = amf_extra_data;
	amf_encoder_info.caps = OBS_ENCODER_CAP_PASS_TEXTURE;

	obs_register_encoder(&amf_encoder_info);

	amf_encoder_info.id = "h265_fallback_amf";
	amf_encoder_info.caps = OBS_ENCODER_CAP_INTERNAL |
				OBS_ENCODER_CAP_DYN_BITRATE;
	amf_encoder_info.encode_texture = nullptr;
	amf_encoder_info.encode_texture2 = nullptr;
	amf_encoder_info.create = amf_hevc_create_fallback;
	amf_encoder_info.encode = amf_encode_fallback;
	amf_encoder_info.get_video_info = h265_video_info_fallback;

	obs_register_encoder(&amf_encoder_info);
}

#endif //ENABLE_HEVC

/* ========================================================================= */
/* AV1 Implementation                                                        */

static const char *amf_av1_get_name(void *)
{
	return "AMD HW AV1";
}

static inline int get_av1_preset(amf_base *enc, const char *preset)
{
	UNUSED_PARAMETER(enc);
	if (astrcmpi(preset, "highquality") == 0)
		return AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_HIGH_QUALITY;
	else if (astrcmpi(preset, "quality") == 0)
		return AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_QUALITY;
	else if (astrcmpi(preset, "balanced") == 0)
		return AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_BALANCED;
	else if (astrcmpi(preset, "speed") == 0)
		return AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_SPEED;

	return AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_BALANCED;
}

static inline int get_av1_rate_control(const char *rc_str)
{
	if (astrcmpi(rc_str, "cqp") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CONSTANT_QP;
	else if (astrcmpi(rc_str, "vbr_lat") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR;
	else if (astrcmpi(rc_str, "vbr") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR;
	else if (astrcmpi(rc_str, "cbr") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR;
	else if (astrcmpi(rc_str, "qvbr") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_QUALITY_VBR;
	else if (astrcmpi(rc_str, "hqvbr") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_HIGH_QUALITY_VBR;
	else if (astrcmpi(rc_str, "hqcbr") == 0)
		return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_HIGH_QUALITY_CBR;

	return AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR;
}

static inline int get_av1_profile(obs_data_t *settings)
{
	const char *profile = obs_data_get_string(settings, "profile");

	if (astrcmpi(profile, "main") == 0)
		return AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN;

	return AMF_VIDEO_ENCODER_AV1_PROFILE_MAIN;
}

static void amf_av1_update_data(amf_base *enc, int rc, int64_t bitrate,
				int64_t cq_value)
{
	if (rc != AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CONSTANT_QP &&
	    rc != AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_QUALITY_VBR) {
		set_av1_property(enc, TARGET_BITRATE, bitrate);
		set_av1_property(enc, PEAK_BITRATE, bitrate);
		set_av1_property(enc, VBV_BUFFER_SIZE, bitrate);

		if (rc == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR) {
			set_av1_property(enc, FILLER_DATA, true);
		} else if (
			rc == AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR ||
			rc == AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_HIGH_QUALITY_VBR) {
			set_av1_property(enc, PEAK_BITRATE, bitrate * 1.5);
		}
	} else {
		int64_t qp = cq_value * 4;
		set_av1_property(enc, QVBR_QUALITY_LEVEL, qp / 4);
		set_av1_property(enc, Q_INDEX_INTRA, qp);
		set_av1_property(enc, Q_INDEX_INTER, qp);
	}
}

static bool amf_av1_update(void *data, obs_data_t *settings)
try {
	amf_base *enc = (amf_base *)data;

	if (enc->first_update) {
		enc->first_update = false;
		return true;
	}

	int64_t bitrate = obs_data_get_int(settings, "bitrate");
	int64_t cq_level = obs_data_get_int(settings, "cqp");
	const char *rc_str = obs_data_get_string(settings, "rate_control");
	int rc = get_av1_rate_control(rc_str);

	amf_av1_update_data(enc, rc, bitrate * 1000, cq_level);

	AMF_RESULT res = enc->amf_encoder->ReInit(enc->cx, enc->cy);
	if (res != AMF_OK)
		throw amf_error("AMFComponent::Init failed", res);

	return true;

} catch (const amf_error &err) {
	amf_base *enc = (amf_base *)data;
	error("%s: %s: %ls", __FUNCTION__, err.str,
	      amf_trace->GetResultText(err.res));
	return false;
}

static bool amf_av1_init(void *data, obs_data_t *settings)
{
	amf_base *enc = (amf_base *)data;

	int64_t bitrate = obs_data_get_int(settings, "bitrate");
	int64_t qp = obs_data_get_int(settings, "cqp");
	const char *preset = obs_data_get_string(settings, "preset");
	const char *profile = obs_data_get_string(settings, "profile");
	const char *rc_str = obs_data_get_string(settings, "rate_control");

	int rc = get_av1_rate_control(rc_str);
	set_av1_property(enc, RATE_CONTROL_METHOD, rc);

	amf_av1_update_data(enc, rc, bitrate * 1000, qp);

	set_av1_property(enc, ENFORCE_HRD, true);

	int keyint_sec = (int)obs_data_get_int(settings, "keyint_sec");
	int gop_size = (keyint_sec) ? keyint_sec * enc->fps_num / enc->fps_den
				    : 250;
	set_av1_property(enc, GOP_SIZE, gop_size);

	const char *ffmpeg_opts = obs_data_get_string(settings, "ffmpeg_opts");
	if (ffmpeg_opts && *ffmpeg_opts) {
		struct obs_options opts = obs_parse_options(ffmpeg_opts);
		for (size_t i = 0; i < opts.count; i++) {
			amf_apply_opt(enc, &opts.options[i]);
		}
		obs_free_options(opts);
	}

	check_preset_compatibility(enc, preset);

	if (!ffmpeg_opts || !*ffmpeg_opts)
		ffmpeg_opts = "(none)";

	info("settings:\n"
	     "\trate_control: %s\n"
	     "\tbitrate:      %" PRId64 "\n"
	     "\tcqp:          %" PRId64 "\n"
	     "\tkeyint:       %d\n"
	     "\tpreset:       %s\n"
	     "\tprofile:      %s\n"
	     "\twidth:        %d\n"
	     "\theight:       %d\n"
	     "\tparams:       %s",
	     rc_str, bitrate, qp, gop_size, preset, profile, enc->cx, enc->cy,
	     ffmpeg_opts);

	return true;
}

static void amf_av1_create_internal(amf_base *enc, obs_data_t *settings)
{
	enc->codec = amf_codec_type::AV1;

	if (!amf_create_encoder(enc))
		throw "Failed to create encoder";

	AMFCapsPtr caps;
	AMF_RESULT res = enc->amf_encoder->GetCaps(&caps);
	if (res == AMF_OK) {
		caps->GetProperty(AMF_VIDEO_ENCODER_AV1_CAP_MAX_THROUGHPUT,
				  &enc->max_throughput);
	}

	const bool is10bit = enc->amf_format == AMF_SURFACE_P010;
	const char *preset = obs_data_get_string(settings, "preset");

	set_av1_property(enc, FRAMESIZE, AMFConstructSize(enc->cx, enc->cy));
	set_av1_property(enc, USAGE, AMF_VIDEO_ENCODER_USAGE_TRANSCODING);
	set_av1_property(enc, ALIGNMENT_MODE,
			 AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_NO_RESTRICTIONS);
	set_av1_property(enc, QUALITY_PRESET, get_av1_preset(enc, preset));
	set_av1_property(enc, COLOR_BIT_DEPTH,
			 is10bit ? AMF_COLOR_BIT_DEPTH_10
				 : AMF_COLOR_BIT_DEPTH_8);
	set_av1_property(enc, PROFILE, get_av1_profile(settings));
	set_av1_property(enc, ENCODING_LATENCY_MODE,
			 AMF_VIDEO_ENCODER_AV1_ENCODING_LATENCY_MODE_NONE);
	// set_av1_property(enc, RATE_CONTROL_PREENCODE, true);
	set_av1_property(enc, OUTPUT_COLOR_PROFILE, enc->amf_color_profile);
	set_av1_property(enc, OUTPUT_TRANSFER_CHARACTERISTIC,
			 enc->amf_characteristic);
	set_av1_property(enc, OUTPUT_COLOR_PRIMARIES, enc->amf_primaries);

	amf_av1_init(enc, settings);

	res = enc->amf_encoder->Init(enc->amf_format, enc->cx, enc->cy);
	if (res != AMF_OK)
		throw amf_error("AMFComponent::Init failed", res);

	set_av1_property(enc, FRAMERATE, enc->amf_frame_rate);

	AMFVariant p;
	res = enc->amf_encoder->GetProperty(AMF_VIDEO_ENCODER_AV1_EXTRA_DATA,
					    &p);
	if (res == AMF_OK && p.type == AMF_VARIANT_INTERFACE)
		enc->header = AMFBufferPtr(p.pInterface);
}

static void *amf_av1_create_texencode(obs_data_t *settings,
				      obs_encoder_t *encoder)
try {
	check_texture_encode_capability(encoder, amf_codec_type::AV1);

	std::unique_ptr<amf_texencode> enc = std::make_unique<amf_texencode>();
	enc->encoder = encoder;
	enc->encoder_str = "texture-amf-av1";

#ifdef _WIN32
	if (!amf_init_d3d11(enc.get()))
		throw "Failed to create D3D11";
#endif

	amf_av1_create_internal(enc.get(), settings);
	return enc.release();

} catch (const amf_error &err) {
	blog(LOG_ERROR, "[texture-amf-av1] %s: %s: %ls", __FUNCTION__, err.str,
	     amf_trace->GetResultText(err.res));
	return obs_encoder_create_rerouted(encoder, "av1_fallback_amf");

} catch (const char *err) {
	blog(LOG_ERROR, "[texture-amf-av1] %s: %s", __FUNCTION__, err);
	return obs_encoder_create_rerouted(encoder, "av1_fallback_amf");
}

static void *amf_av1_create_fallback(obs_data_t *settings,
				     obs_encoder_t *encoder)
try {
	std::unique_ptr<amf_fallback> enc = std::make_unique<amf_fallback>();
	enc->encoder = encoder;
	enc->encoder_str = "fallback-amf-av1";

	video_t *video = obs_encoder_video(encoder);
	const struct video_output_info *voi = video_output_get_info(video);
	switch (voi->format) {
	case VIDEO_FORMAT_I010:
	case VIDEO_FORMAT_P010: {
		break;
	}
	default:
		switch (voi->colorspace) {
		case VIDEO_CS_2100_PQ:
		case VIDEO_CS_2100_HLG: {
			const char *const text =
				obs_module_text("AMF.8bitUnsupportedHdr");
			obs_encoder_set_last_error(encoder, text);
			throw text;
		}
		}
	}

	amf_av1_create_internal(enc.get(), settings);
	return enc.release();

} catch (const amf_error &err) {
	blog(LOG_ERROR, "[fallback-amf-av1] %s: %s: %ls", __FUNCTION__, err.str,
	     amf_trace->GetResultText(err.res));
	return nullptr;

} catch (const char *err) {
	blog(LOG_ERROR, "[fallback-amf-av1] %s: %s", __FUNCTION__, err);
	return nullptr;
}

static void amf_av1_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "bitrate", 2500);
	obs_data_set_default_int(settings, "cqp", 20);
	obs_data_set_default_string(settings, "rate_control", "CBR");
	obs_data_set_default_string(settings, "preset", "quality");
	obs_data_set_default_string(settings, "profile", "high");
}

static void register_av1()
{
	struct obs_encoder_info amf_encoder_info = {};
	amf_encoder_info.id = "av1_texture_amf";
	amf_encoder_info.type = OBS_ENCODER_VIDEO;
	amf_encoder_info.codec = "av1";
	amf_encoder_info.get_name = amf_av1_get_name;
	amf_encoder_info.create = amf_av1_create_texencode;
	amf_encoder_info.destroy = amf_destroy;
	/* FIXME: Figure out why encoder does not survive reconfiguration
	amf_encoder_info.update = amf_av1_update; */
	amf_encoder_info.encode_texture = amf_encode_tex;
	amf_encoder_info.encode_texture2 = amf_encode_tex2;
	amf_encoder_info.get_defaults = amf_av1_defaults;
	amf_encoder_info.get_properties = amf_av1_properties;
	amf_encoder_info.get_extra_data = amf_extra_data;
	amf_encoder_info.caps = OBS_ENCODER_CAP_PASS_TEXTURE;

	obs_register_encoder(&amf_encoder_info);

	amf_encoder_info.id = "av1_fallback_amf";
	amf_encoder_info.caps = OBS_ENCODER_CAP_INTERNAL |
				OBS_ENCODER_CAP_DYN_BITRATE;
	amf_encoder_info.encode_texture = nullptr;
	amf_encoder_info.encode_texture2 = nullptr;
	amf_encoder_info.create = amf_av1_create_fallback;
	amf_encoder_info.encode = amf_encode_fallback;
	amf_encoder_info.get_video_info = av1_video_info_fallback;

	obs_register_encoder(&amf_encoder_info);
}

/* ========================================================================= */
/* Global Stuff                                                              */

static bool enum_luids(void *param, uint32_t idx, uint64_t luid)
{
	std::stringstream &cmd = *(std::stringstream *)param;
	cmd << " " << std::hex << luid;
	UNUSED_PARAMETER(idx);
	return true;
}

#ifdef _WIN32
#define OBS_AMF_TEST "obs-amf-test.exe"
#else
#define OBS_AMF_TEST "obs-amf-test"
#endif

extern "C" void amf_load(void)
try {
	AMF_RESULT res;
#ifdef _WIN32
	HMODULE amf_module_test;

	/* Check if the DLL is present before running the more expensive */
	/* obs-amf-test.exe, but load it as data so it can't crash us    */
	amf_module_test =
		LoadLibraryExW(AMF_DLL_NAME, nullptr, LOAD_LIBRARY_AS_DATAFILE);
	if (!amf_module_test)
		throw "No AMF library";
	FreeLibrary(amf_module_test);
#else
	void *amf_module_test = os_dlopen(AMF_DLL_NAMEA);
	if (!amf_module_test)
		throw "No AMF library";
	os_dlclose(amf_module_test);
#endif

	/* ----------------------------------- */
	/* Check for supported codecs          */

	BPtr<char> test_exe = os_get_executable_path_ptr(OBS_AMF_TEST);
	std::stringstream cmd;
	std::string caps_str;

	cmd << test_exe;
#ifdef _WIN32
	enum_graphics_device_luids(enum_luids, &cmd);
#endif

	os_process_pipe_t *pp = os_process_pipe_create(cmd.str().c_str(), "r");
	if (!pp)
		throw "Failed to launch the AMF test process I guess";

	for (;;) {
		char data[2048];
		size_t len =
			os_process_pipe_read(pp, (uint8_t *)data, sizeof(data));
		if (!len)
			break;

		caps_str.append(data, len);
	}

	os_process_pipe_destroy(pp);

	if (caps_str.empty())
		throw "Seems the AMF test subprocess crashed. "
		      "Better there than here I guess. "
		      "Let's just skip loading AMF then I suppose.";

	ConfigFile config;
	if (config.OpenString(caps_str.c_str()) != 0)
		throw "Failed to open config string";

	const char *error = config_get_string(config, "error", "string");
	if (error)
		throw std::string(error);

	uint32_t adapter_count = (uint32_t)config_num_sections(config);
	bool avc_supported = false;
	bool hevc_supported = false;
	bool av1_supported = false;

	for (uint32_t i = 0; i < adapter_count; i++) {
		std::string section = std::to_string(i);
		adapter_caps &info = caps[i];

		info.is_amd =
			config_get_bool(config, section.c_str(), "is_amd");
		info.supports_avc = config_get_bool(config, section.c_str(),
						    "supports_avc");
		info.supports_hevc = config_get_bool(config, section.c_str(),
						     "supports_hevc");
		info.supports_av1 = config_get_bool(config, section.c_str(),
						    "supports_av1");

		avc_supported |= info.supports_avc;
		hevc_supported |= info.supports_hevc;
		av1_supported |= info.supports_av1;
	}

	if (!avc_supported && !hevc_supported && !av1_supported)
		throw "Neither AVC, HEVC, nor AV1 are supported by any devices";

	/* ----------------------------------- */
	/* Init AMF                            */

	amf_module = os_dlopen(AMF_DLL_NAMEA);
	if (!amf_module)
		throw "AMF library failed to load";

	AMFInit_Fn init =
		(AMFInit_Fn)os_dlsym(amf_module, AMF_INIT_FUNCTION_NAME);
	if (!init)
		throw "Failed to get AMFInit address";

	res = init(AMF_FULL_VERSION, &amf_factory);
	if (res != AMF_OK)
		throw amf_error("AMFInit failed", res);

	res = amf_factory->GetTrace(&amf_trace);
	if (res != AMF_OK)
		throw amf_error("GetTrace failed", res);

	AMFQueryVersion_Fn get_ver = (AMFQueryVersion_Fn)os_dlsym(
		amf_module, AMF_QUERY_VERSION_FUNCTION_NAME);
	if (!get_ver)
		throw "Failed to get AMFQueryVersion address";

	res = get_ver(&amf_version);
	if (res != AMF_OK)
		throw amf_error("AMFQueryVersion failed", res);

#ifndef DEBUG_AMF_STUFF
	amf_trace->EnableWriter(AMF_TRACE_WRITER_DEBUG_OUTPUT, false);
	amf_trace->EnableWriter(AMF_TRACE_WRITER_CONSOLE, false);
#endif

	/* ----------------------------------- */
	/* Register encoders                   */

	if (avc_supported)
		register_avc();
#if ENABLE_HEVC
	if (hevc_supported)
		register_hevc();
#endif
	if (av1_supported)
		register_av1();

} catch (const std::string &str) {
	/* doing debug here because string exceptions indicate the user is
	 * probably not using AMD */
	blog(LOG_DEBUG, "%s: %s", __FUNCTION__, str.c_str());

} catch (const char *str) {
	/* doing debug here because string exceptions indicate the user is
	 * probably not using AMD */
	blog(LOG_DEBUG, "%s: %s", __FUNCTION__, str);

} catch (const amf_error &err) {
	/* doing an error here because it means at least the library has loaded
	 * successfully, so they probably have AMD at this point */
	blog(LOG_ERROR, "%s: %s: 0x%uX", __FUNCTION__, err.str,
	     (uint32_t)err.res);
}

extern "C" void amf_unload(void)
{
	if (amf_module && amf_trace) {
		amf_trace->TraceFlush();
		amf_trace->UnregisterWriter(L"obs_amf_trace_writer");
	}
}
