# `.\pytorch\aten\src\ATen\DLConvertor.cpp`

```
// 包含 ATen 库中的 DLConvertor.h 和 Functions.h 头文件
#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>

// 使用 std 命名空间
using namespace std;
// ATen 库的命名空间
namespace at {

// 获取 Tensor 对应的 DLDataType 结构
DLDataType getDLDataType(const Tensor& t) {
  // 创建 DLDataType 结构并初始化 lanes 为 1
  DLDataType dtype;
  dtype.lanes = 1;
  // 计算 Tensor 元素大小并赋值给 bits 字段
  dtype.bits = t.element_size() * 8;
  
  // 根据 Tensor 的标量类型进行分支处理
  switch (t.scalar_type()) {
    // 对于无符号整数类型，设置 DLDataType 的 code 为 kDLUInt
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
    case ScalarType::Byte:
    case ScalarType::UInt16:
    case ScalarType::UInt32:
    case ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    
    // 对于有符号 char 类型，设置 DLDataType 的 code 为 kDLInt
    case ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    
    // 对于浮点数类型（双精度和单精度），设置 DLDataType 的 code 为 kDLFloat
    case ScalarType::Double:
    case ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    
    // 对于整数类型（int、long、short），设置 DLDataType 的 code 为 kDLInt
    case ScalarType::Int:
    case ScalarType::Long:
    case ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    
    // 对于半精度浮点数和布尔类型，设置相应的 DLDataType 的 code
    case ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    
    // 对于复数类型，设置 DLDataType 的 code 为 kDLComplex
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    
    // 对于其他特殊类型，根据情况设置相应的错误检查并抛出异常
    case ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case ScalarType::Float8_e5m2:
    case ScalarType::Float8_e5m2fnuz:
    case ScalarType::Float8_e4m3fn:
    case ScalarType::Float8_e4m3fnuz:
      TORCH_CHECK(false, "float8 types are not supported by dlpack");
      break;
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      TORCH_CHECK(false, "QUInt/QInt types are not supported by dlpack");
      break;
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      TORCH_CHECK(false, "Bit types are not supported by dlpack");
      break;
    case ScalarType::Undefined:
      TORCH_CHECK(false, "Undefined is not a valid ScalarType");
    case ScalarType::NumOptions:
      TORCH_CHECK(false, "NumOptions is not a valid ScalarType");
  }
  // 返回构建好的 DLDataType 结构
  return dtype;
}

// 获取 Tensor 对应的 DLDevice 结构
static DLDevice getDLDevice(const Tensor& tensor, c10::DeviceIndex device_id) {
  // 创建 DLDevice 结构并将 device_id 转换为 int32_t 类型
  DLDevice ctx;
  ctx.device_id = static_cast<int32_t>(device_id);
  
  // 根据 Tensor 的设备类型进行分支处理
  switch (tensor.device().type()) {
    // 对于 CPU 设备，设置 DLDevice 的 device_type 为 kDLCPU
    case DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    
    // 对于 CUDA 设备，设置 DLDevice 的 device_type 为 kDLCUDA
    case DeviceType::CUDA:
      ctx.device_type = DLDeviceType::kDLCUDA;
      break;
    
    // 其他设备类型暂不支持，抛出相应的异常
    default:
      TORCH_CHECK(false, "Unsupported device type for DLDevice conversion");
      break;
  }
  // 返回构建好的 DLDevice 结构
  return ctx;
}
#ifdef USE_ROCM
      // 如果启用了 ROCM，则将上下文设备类型设置为 DLDeviceType::kDLROCM，
      // 这样 PyTorch 将会将其识别为 CUDA 设备，而其他所有设备应该看到 HIP 设备
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      // 如果未启用 ROCM，则将上下文设备类型设置为 DLDeviceType::kDLCUDA
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      // 跳出 switch 语句
      break;
    case DeviceType::OPENCL:
      // 如果设备类型是 OPENCL，则将上下文设备类型设置为 DLDeviceType::kDLOpenCL
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case DeviceType::HIP:
      // 如果设备类型是 HIP，则将上下文设备类型设置为 DLDeviceType::kDLROCM
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case DeviceType::XPU:
      // 如果设备类型是 XPU，则将上下文设备类型设置为 DLDeviceType::kDLOneAPI，
      // 并从张量的设备索引获取设备 ID
      ctx.device_type = DLDeviceType::kDLOneAPI;
      ctx.device_id =
          at::detail::getXPUHooks().getGlobalIdxFromDevice(tensor.device());
      break;
    default:
      // 如果是不支持的设备类型，则抛出错误并提示不支持该设备类型
      TORCH_CHECK(false, "Cannot pack tensors on " + tensor.device().str());
  }
  // 返回更新后的上下文对象
  return ctx;
}

static Device getATenDevice(const DLDevice& ctx, void* data) {
  // 根据上下文中的设备类型，返回对应的 ATen 设备对象
  switch (ctx.device_type) {
    case DLDeviceType::kDLCPU:
      // 如果设备类型是 CPU，则返回 CPU 设备对象
      return at::Device(DeviceType::CPU);
#ifndef USE_ROCM
    // 如果编译环境不是 ROCM，不支持 CUDA
    case DLDeviceType::kDLCUDA:
      // 如果设备类型是 CUDA，则返回 CUDA 设备对象，并传入设备 ID
      return at::Device(DeviceType::CUDA, ctx.device_id);
#endif
    case DLDeviceType::kDLOpenCL:
      // 如果设备类型是 OpenCL，则返回 OpenCL 设备对象，并传入设备 ID
      return at::Device(DeviceType::OPENCL, ctx.device_id);
    case DLDeviceType::kDLROCM:
#ifdef USE_ROCM
      // 如果启用了 ROCM，将 DLDeviceType::kDLROCM 视为 CUDA 设备，返回 CUDA 设备对象
      return at::Device(DeviceType::CUDA, ctx.device_id);
#else
      // 如果未启用 ROCM，则将 DLDeviceType::kDLROCM 视为 HIP 设备，返回 HIP 设备对象
      return at::Device(DeviceType::HIP, ctx.device_id);
#endif
    case DLDeviceType::kDLOneAPI:
      // 如果设备类型是 OneAPI，则从数据指针获取设备对象
      return at::detail::getXPUHooks().getDeviceFromPtr(data);
    default:
      // 如果是不支持的设备类型，则抛出错误并提示不支持该设备类型
      TORCH_CHECK(
          false, "Unsupported device_type: ", std::to_string(ctx.device_type));
  }
}

ScalarType toScalarType(const DLDataType& dtype) {
  // 将 DLDataType 转换为 ScalarType 类型
  ScalarType stype = ScalarType::Undefined;
  // 检查数据类型的 lanes 是否为 1，ATen 不支持 lanes 不等于 1 的情况
  TORCH_CHECK(dtype.lanes == 1, "ATen does not support lanes != 1");
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      // 根据 kDLUInt 数据类型的位数，选择对应的 ScalarType 类型
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Byte;
          break;
        case 16:
          stype = ScalarType::UInt16;
          break;
        case 32:
          stype = ScalarType::UInt32;
          break;
        case 64:
          stype = ScalarType::UInt64;
          break;
        default:
          // 不支持的 kUInt 位数时抛出错误
          TORCH_CHECK(
              false, "Unsupported kUInt bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLInt:
      // 根据 kDLInt 数据类型的位数，选择对应的 ScalarType 类型
      switch (dtype.bits) {
        case 8:
          stype = ScalarType::Char;
          break;
        case 16:
          stype = ScalarType::Short;
          break;
        case 32:
          stype = ScalarType::Int;
          break;
        case 64:
          stype = ScalarType::Long;
          break;
        default:
          // 不支持的 kInt 位数时抛出错误
          TORCH_CHECK(
              false, "Unsupported kInt bits ", std::to_string(dtype.bits));
      }
      break;
    // 根据 DLDataTypeCode 类型进行不同的处理
    switch (dtype.code) {
        // 如果是浮点类型
        case DLDataTypeCode::kDLFloat:
          // 根据位数选择对应的 Torch 的标量类型
          switch (dtype.bits) {
            // 如果是 16 位
            case 16:
              stype = ScalarType::Half;
              break;
            // 如果是 32 位
            case 32:
              stype = ScalarType::Float;
              break;
            // 如果是 64 位
            case 64:
              stype = ScalarType::Double;
              break;
            // 如果位数不支持，抛出错误信息
            default:
              TORCH_CHECK(
                  false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
          }
          break;
        // 如果是 Bfloat 类型
        case DLDataTypeCode::kDLBfloat:
          // 如果是 16 位
          switch (dtype.bits) {
            case 16:
              stype = ScalarType::BFloat16;
              break;
            // 如果位数不支持，抛出错误信息
            default:
              TORCH_CHECK(
                  false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
          }
          break;
        // 如果是复数类型
        case DLDataTypeCode::kDLComplex:
          // 根据位数选择对应的复数类型
          switch (dtype.bits) {
            // 如果是 32 位
            case 32:
              stype = ScalarType::ComplexHalf;
              break;
            // 如果是 64 位
            case 64:
              stype = ScalarType::ComplexFloat;
              break;
            // 如果是 128 位
            case 128:
              stype = ScalarType::ComplexDouble;
              break;
            // 如果位数不支持，抛出错误信息
            default:
              TORCH_CHECK(
                  false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
          }
          break;
        // 如果是布尔类型
        case DLDataTypeCode::kDLBool:
          // 如果是 8 位
          switch (dtype.bits) {
            case 8:
              stype = ScalarType::Bool;
              break;
            // 如果位数不支持，抛出错误信息
            default:
              TORCH_CHECK(
                  false, "Unsupported kDLBool bits ", std::to_string(dtype.bits));
          }
          break;
        // 如果是未知的类型代码，抛出错误信息
        default:
          TORCH_CHECK(false, "Unsupported code ", std::to_string(dtype.code));
    }
    // 返回选择的 Torch 标量类型
    return stype;
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
// 定义结构体 ATenDLMTensor，包含 ATen 的 Tensor 和 DLManagedTensor
struct ATenDLMTensor {
  Tensor handle;               // ATen 的 Tensor 对象
  DLManagedTensor tensor;      // DLManagedTensor 对象
};

// 定义用于释放 DLManagedTensor 内存的函数
static void deleter(DLManagedTensor* arg) {
  delete static_cast<ATenDLMTensor*>(arg->manager_ctx);  // 释放管理上下文中的 ATenDLMTensor 对象
}

// 将 ATen 的 Tensor 转换为 DLpack 的 DLManagedTensor，返回指向 DLManagedTensor 的指针
DLManagedTensor* toDLPack(const Tensor& src) {
  // 创建一个新的视图 Tensor，可能会对步长进行归一化
  // gh-83069
  auto shape = src.sizes();                         // 获取 Tensor 的形状
  auto strides = src.strides().vec();               // 获取 Tensor 的步长信息
  for (int i = 0; i < src.dim(); i++) {             // 遍历 Tensor 的每一个维度
    if (shape[i] < 2) {                             // 如果当前维度的大小小于 2
      strides[i] = 1;                               // 将该维度的步长设置为 1
    }
  }

  auto view = src.as_strided(shape, strides, src.storage_offset());  // 创建 Tensor 的视图
  ATenDLMTensor* atDLMTensor(new ATenDLMTensor);     // 创建新的 ATenDLMTensor 对象
  atDLMTensor->handle = view;                        // 将视图 Tensor 赋给 ATenDLMTensor 的 handle 成员
  atDLMTensor->tensor.manager_ctx = atDLMTensor;     // 设置 DLManagedTensor 的管理上下文为 ATenDLMTensor 对象
  atDLMTensor->tensor.deleter = &deleter;            // 设置 DLManagedTensor 的删除函数为 deleter
  atDLMTensor->tensor.dl_tensor.data = view.data_ptr();  // 设置 DLManagedTensor 的数据指针为视图 Tensor 的数据指针
  c10::DeviceIndex device_id = 0;                    // 初始化设备 ID
  if (src.is_cuda()) {                              // 如果 Tensor 存储在 GPU 上
    device_id = src.get_device();                    // 获取 Tensor 的设备 ID
  }
  atDLMTensor->tensor.dl_tensor.device = getDLDevice(src, device_id);  // 获取 DLpack 的设备信息
  atDLMTensor->tensor.dl_tensor.ndim = src.dim();    // 设置 DLpack 的维度信息
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);  // 获取 DLpack 的数据类型信息
  atDLMTensor->tensor.dl_tensor.shape = view.sizes().data();  // 设置 DLpack 的形状信息
  atDLMTensor->tensor.dl_tensor.strides = view.strides().data();  // 设置 DLpack 的步长信息
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;     // 设置 DLpack 的字节偏移量为 0
  return &(atDLMTensor->tensor);                     // 返回 DLManagedTensor 的指针
}

// 根据 DLManagedTensor 创建对应的 ATen Tensor
Tensor fromDLPack(DLManagedTensor* src) {
  auto deleter = [src](void* self) {                 // 创建 lambda 表达式作为删除器
    if (src->deleter) {                              // 如果存在删除器
      src->deleter(src);                             // 调用删除器释放 DLManagedTensor
    }
  };
  return fromDLPack(src, std::move(deleter));        // 调用重载的 fromDLPack 函数
}

// 根据 DLManagedTensor 创建对应的 ATen Tensor，带有自定义的删除器
Tensor fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter) {
  Device device = getATenDevice(src->dl_tensor.device, src->dl_tensor.data);  // 获取 ATen 的设备信息
  ScalarType stype = toScalarType(src->dl_tensor.dtype);  // 获取 ATen 的数据类型
  if (!src->dl_tensor.strides) {                     // 如果 DLpack 的步长信息不存在
    return at::from_blob(
        src->dl_tensor.data,                         // 数据指针
        IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),  // 形状信息
        std::move(deleter),                          // 移动自定义删除器
        at::device(device).dtype(stype),             // 设备和数据类型信息
        {device});                                   // 设备列表
  }
  return at::from_blob(
      src->dl_tensor.data,                           // 数据指针
      IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),  // 形状信息
      IntArrayRef(src->dl_tensor.strides, src->dl_tensor.ndim),  // 步长信息
      deleter,                                       // 自定义删除器
      at::device(device).dtype(stype),               // 设备和数据类型信息
      {device});                                     // 设备列表
}
} // namespace at
```