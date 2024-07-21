# `.\pytorch\android\pytorch_android\src\main\cpp\pytorch_jni_common.cpp`

```py
// C++ 头文件包含，包括断言、输入输出流、智能指针、字符串等标准库
#include <cassert>
#include <iostream>
#include <memory>
#include <string>

// 包含 PyTorch C10 库中的特定头文件，如内存格式、范围等
#include <c10/core/MemoryFormat.h>
#include <c10/util/irange.h>

// 包含 Facebook JNI 库的 ByteBuffer 和 fbjni 头文件
#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

// 包含自定义的 PyTorch JNI 公共头文件
#include "pytorch_jni_common.h"

// 如果在 Android 平台上且未定义 USE_PTHREADPOOL，则定义 USE_PTHREADPOOL
#if defined(__ANDROID__)
#ifndef USE_PTHREADPOOL
#define USE_PTHREADPOOL
#endif /* USE_PTHREADPOOL */

// 包含 pthreadpool-cpp 头文件，用于 Android 平台的线程池管理
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#endif

// 定义命名空间 pytorch_jni
namespace pytorch_jni {

// 将 JNI 设备代码转换为 PyTorch 的设备类型
c10::DeviceType deviceJniCodeToDeviceType(jint deviceJniCode) {
  if (deviceJniCode == kDeviceCPU) {
    return at::kCPU;
  } else if (deviceJniCode == kDeviceVulkan) {
    return at::kVulkan;
  }

  // 抛出 Java 异常，表示未知设备类型
  facebook::jni::throwNewJavaException(
      facebook::jni::gJavaLangIllegalArgumentException, "Unknown device");
}

// 初始化静态变量 is_initialized_
bool Trace::is_initialized_ = false;

// 在开启了 TRACE_ENABLED 并且在 Android 平台上，声明 ATrace 的函数指针
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
Trace::fp_ATrace_beginSection Trace::ATrace_beginSection;
Trace::fp_ATrace_endSection Trace::ATrace_endSection;
#endif

// 初始化 Trace 类，主要是加载 Android 平台上的 libandroid.so 动态库，并获取 ATrace 函数的地址
void Trace::init() {
#if defined(TRACE_ENABLED) && defined(__ANDROID__)
  void* lib = dlopen("libandroid.so", RTLD_NOW || RTLD_LOCAL);
  if (lib != NULL) {
    Trace::ATrace_beginSection = reinterpret_cast<fp_ATrace_beginSection>(
        dlsym(lib, "ATrace_beginSection"));
    Trace::ATrace_endSection =
        reinterpret_cast<fp_ATrace_endSection>(dlsym(lib, "ATrace_endSection"));
  }
#endif
}

// 定义各种张量数据类型的常量，与 DType.java 必须保持同步
// 这些常量不应该被序列化，因为它们可能在不同版本间发生变化
constexpr static int kTensorDTypeUInt8 = 1;
constexpr static int kTensorDTypeInt8 = 2;
constexpr static int kTensorDTypeInt32 = 3;
constexpr static int kTensorDTypeFloat32 = 4;
constexpr static int kTensorDTypeInt64 = 5;
constexpr static int kTensorDTypeFloat64 = 6;

// 定义张量的内存格式常量
constexpr static int kTensorMemoryFormatContiguous = 1;
constexpr static int kTensorMemoryFormatChannelsLast = 2;
constexpr static int kTensorMemoryFormatChannelsLast3d = 3;

// 定义 Java 中 HashMap 的模板结构体 JHashMap
template <typename K = jobject, typename V = jobject>
struct JHashMap
    : facebook::jni::JavaClass<JHashMap<K, V>, facebook::jni::JMap<K, V>> {
  constexpr static auto kJavaDescriptor = "Ljava/util/HashMap;";

  using Super =
      facebook::jni::JavaClass<JHashMap<K, V>, facebook::jni::JMap<K, V>>;

  // 创建一个新的 JHashMap 实例
  static facebook::jni::local_ref<JHashMap<K, V>> create() {
    return Super::newInstance();
  }

  // 向 HashMap 中插入键值对
  void put(
      facebook::jni::alias_ref<facebook::jni::JObject::javaobject> key,
      facebook::jni::alias_ref<facebook::jni::JObject::javaobject> value) {
    static auto putMethod =
        Super::javaClassStatic()
            ->template getMethod<facebook::jni::alias_ref<
                facebook::jni::JObject::javaobject>(
                facebook::jni::alias_ref<facebook::jni::JObject::javaobject>,
                facebook::jni::alias_ref<facebook::jni::JObject::javaobject>)>(
                "put");
    putMethod(Super::self(), key, value);
  }
};

// 定义创建新的 ATensor 实例的静态函数
static at::Tensor newAtTensor(
    facebook::jni::alias_ref<facebook::jni::JBuffer> jbuffer,
    // 获取 Java 传入的 jlongArray 对象的别名，表示张量的形状
    facebook::jni::alias_ref<jlongArray> jshape,
    // 表示张量的数据类型
    jint jdtype,
    // 表示张量的内存布局格式
    jint jmemoryFormat) {
  // 获取张量的维度（rank）
  const auto rank = jshape->size();
  // 获取张量形状的数组
  const auto shapeArr = jshape->getRegion(0, rank);
  // 将形状数组转换为 C++ 标准库中的 vector
  std::vector<int64_t> shapeVec{};
  shapeVec.reserve(rank);
  // 计算张量的元素总数
  auto numel = 1;
  for (const auto i : c10::irange(rank)) {
    shapeVec.push_back(shapeArr[i]);
    numel *= shapeArr[i];
  }
  // 获取当前 JNI 环境
  JNIEnv* jni = facebook::jni::Environment::current();
  // 定义张量数据类型的元信息
  caffe2::TypeMeta typeMeta{};
  // 定义数据元素大小（字节数）
  int dataElementSizeBytes = 0;
  // 根据不同的数据类型设置元素大小和类型元信息
  if (kTensorDTypeFloat32 == jdtype) {
    dataElementSizeBytes = 4;
    typeMeta = caffe2::TypeMeta::Make<float>();
  } else if (kTensorDTypeInt32 == jdtype) {
    dataElementSizeBytes = 4;
    typeMeta = caffe2::TypeMeta::Make<int32_t>();
  } else if (kTensorDTypeInt8 == jdtype) {
    dataElementSizeBytes = 1;
    typeMeta = caffe2::TypeMeta::Make<int8_t>();
  } else if (kTensorDTypeUInt8 == jdtype) {
    dataElementSizeBytes = 1;
    typeMeta = caffe2::TypeMeta::Make<uint8_t>();
  } else if (kTensorDTypeFloat64 == jdtype) {
    dataElementSizeBytes = 8;
    typeMeta = caffe2::TypeMeta::Make<double>();
  } else if (kTensorDTypeInt64 == jdtype) {
    dataElementSizeBytes = 8;
    typeMeta = caffe2::TypeMeta::Make<int64_t>();
  } else {
    // 如果数据类型未知，则抛出 Java 异常
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unknown Tensor jdtype %d",
        jdtype);
  }
  // 获取数据缓冲区的容量
  const auto dataCapacity = jni->GetDirectBufferCapacity(jbuffer.get());
  // 检查数据缓冲区容量是否与张量元素数目一致，如果不一致则抛出异常
  if (dataCapacity != numel) {
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Tensor dimensions(elements number:%d, element byte size:%d, total "
        "bytes:%d) inconsistent with buffer capacity(%d)",
        numel,
        dataElementSizeBytes,
        numel * dataElementSizeBytes,
        dataCapacity);
  }

  // 根据内存布局格式创建 Torch 张量
  if (jmemoryFormat == kTensorMemoryFormatChannelsLast) {
    // 对于通道最后的二维内存布局，使用对应的步长信息创建张量
    auto sizes = torch::IntArrayRef(shapeVec);
    return torch::from_blob(
        jni->GetDirectBufferAddress(jbuffer.get()),
        sizes,
        torch::IntArrayRef(c10::get_channels_last_strides_2d(sizes)),
        at::TensorOptions(typeMeta).memory_format(
            at::MemoryFormat::ChannelsLast));
  } else if (jmemoryFormat == kTensorMemoryFormatChannelsLast3d) {
    // 对于通道最后的三维内存布局，使用对应的步长信息创建张量
    auto sizes = torch::IntArrayRef(shapeVec);
    return torch::from_blob(
        jni->GetDirectBufferAddress(jbuffer.get()),
        sizes,
        torch::IntArrayRef(c10::get_channels_last_strides_3d(sizes)),
        at::TensorOptions(typeMeta).memory_format(
            at::MemoryFormat::ChannelsLast3d));
  }
  // 默认情况下，根据给定的形状创建 Torch 张量
  return torch::from_blob(
      jni->GetDirectBufferAddress(jbuffer.get()),
      torch::IntArrayRef(shapeVec),
      at::TensorOptions(typeMeta));
}
}

class TensorHybrid : public facebook::jni::HybridClass<TensorHybrid> {
 public:
  constexpr static const char* kJavaDescriptor = "Lorg/pytorch/Tensor;";

  explicit TensorHybrid(at::Tensor tensor) : tensor_(tensor) {}

  static facebook::jni::local_ref<TensorHybrid::jhybriddata> initHybrid(
      facebook::jni::alias_ref<TensorHybrid::javaobject> jTensorThis) {
    // 获取静态成员变量和方法句柄，这些变量和方法用于处理 Java 侧对 Tensor 对象的操作
    static auto cls = TensorHybrid::javaClassStatic();
    static const auto jMethodDTypeCode = cls->getMethod<jint()>("dtypeJniCode");
    static const auto jMethodMemoryFormatCode =
        cls->getMethod<jint()>("memoryFormatJniCode");
    static const auto jFieldShape = cls->getField<jlongArray>("shape");
    static const auto jMethodGetDataBuffer = cls->getMethod<
        facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
        "getRawDataBuffer");

    // 调用 newAtTensor 函数创建一个新的 ATen Tensor 对象，并返回其 C++ 的 HybridData 实例
    at::Tensor tensor = newAtTensor(
        jMethodGetDataBuffer(jTensorThis),
        jTensorThis->getFieldValue(jFieldShape),
        jMethodDTypeCode(jTensorThis),
        jMethodMemoryFormatCode(jTensorThis));
    return makeCxxInstance(std::move(tensor));
  }

  static facebook::jni::local_ref<TensorHybrid::javaobject>
  newJTensorFromAtTensor(const at::Tensor& input_tensor) {
    // Java 封装当前仅支持连续的张量。

    int jmemoryFormat = 0;
    at::Tensor tensor{};
    // 根据不同的内存格式检查输入张量是否连续，如果不是，则转换为连续张量
    if (input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      tensor = input_tensor;
      jmemoryFormat = kTensorMemoryFormatChannelsLast;
    } else if (input_tensor.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      tensor = input_tensor;
      jmemoryFormat = kTensorMemoryFormatChannelsLast3d;
    } else {
      tensor = input_tensor.contiguous();
      jmemoryFormat = kTensorMemoryFormatContiguous;
    }

    // 确定张量的标量类型，并映射到 Java 侧的整数类型代码
    const auto scalarType = tensor.scalar_type();
    int jdtype = 0;
    if (at::kFloat == scalarType) {
      jdtype = kTensorDTypeFloat32;
    } else if (at::kInt == scalarType) {
      jdtype = kTensorDTypeInt32;
    } else if (at::kByte == scalarType) {
      jdtype = kTensorDTypeUInt8;
    } else if (at::kChar == scalarType) {
      jdtype = kTensorDTypeInt8;
    } else if (at::kLong == scalarType) {
      jdtype = kTensorDTypeInt64;
    } else if (at::kDouble == scalarType) {
      jdtype = kTensorDTypeFloat64;
    } else {
      // 如果张量的标量类型不支持，抛出 Java 异常
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "at::Tensor scalar type %s is not supported on java side",
          c10::toString(scalarType));
    }

    // 获取张量的形状并转换为 jlongArray
    const auto& tensorShape = tensor.sizes();
    std::vector<jlong> tensorShapeVec;
    for (const auto& s : tensorShape) {
      tensorShapeVec.push_back(s);
    }
    facebook::jni::local_ref<jlongArray> jTensorShape =
        facebook::jni::make_long_array(tensorShapeVec.size());
    jTensorShape->setRegion(0, tensorShapeVec.size(), tensorShapeVec.data());

    // 获取 TensorHybrid 类的静态成员变量和方法句柄
    static auto cls = TensorHybrid::javaClassStatic();
    // 使用 tensor 对象的数据指针和字节大小创建一个 JByteBuffer 对象，用于 JNI 调用
    facebook::jni::local_ref<facebook::jni::JByteBuffer> jTensorBuffer =
        facebook::jni::JByteBuffer::wrapBytes(
            (uint8_t*)tensor.data_ptr(), tensor.nbytes());
    
    // 设置 JByteBuffer 对象的字节顺序为本地字节顺序
    jTensorBuffer->order(facebook::jni::JByteOrder::nativeOrder());

    // 获取静态方法 nativeNewTensor 的引用，用于创建新的 TensorHybrid 对象
    static const auto jMethodNewTensor =
        cls->getStaticMethod<facebook::jni::local_ref<TensorHybrid::javaobject>(
            facebook::jni::alias_ref<facebook::jni::JByteBuffer>,
            facebook::jni::alias_ref<jlongArray>,
            jint,
            jint,
            facebook::jni::alias_ref<jhybriddata>)>("nativeNewTensor");
    
    // 调用 nativeNewTensor 方法创建新的 TensorHybrid 对象并返回
    return jMethodNewTensor(
        cls,
        jTensorBuffer,
        jTensorShape,
        jdtype,
        jmemoryFormat,
        makeCxxInstance(tensor));
  }

  // 根据给定的 JTensor 对象创建一个新的 ATen Tensor 对象
  static at::Tensor newAtTensorFromJTensor(
      facebook::jni::alias_ref<TensorHybrid::javaobject> jtensor) {
    static auto cls = TensorHybrid::javaClassStatic();
    
    // 获取 JTensor 对象的 dtypeJniCode 方法并调用获取 dtype
    static const auto dtypeMethod = cls->getMethod<jint()>("dtypeJniCode");
    jint jdtype = dtypeMethod(jtensor);

    // 获取 JTensor 对象的 memoryFormatJniCode 方法并调用获取 memory format
    static const auto memoryFormatMethod =
        cls->getMethod<jint()>("memoryFormatJniCode");
    jint jmemoryFormat = memoryFormatMethod(jtensor);

    // 获取 JTensor 对象的 shape 字段的值作为 jlongArray
    static const auto shapeField = cls->getField<jlongArray>("shape");
    auto jshape = jtensor->getFieldValue(shapeField);

    // 调用 JTensor 对象的 getRawDataBuffer 方法获取数据缓冲区对象 JBuffer
    static auto dataBufferMethod = cls->getMethod<
        facebook::jni::local_ref<facebook::jni::JBuffer::javaobject>()>(
        "getRawDataBuffer");
    facebook::jni::local_ref<facebook::jni::JBuffer> jbuffer =
        dataBufferMethod(jtensor);
    
    // 使用 JBuffer 对象、shape、dtype 和 memory format 创建新的 ATen Tensor 对象
    return newAtTensor(jbuffer, jshape, jdtype, jmemoryFormat);
  }

  // 返回当前对象持有的 ATen Tensor 对象
  at::Tensor tensor() const {
    return tensor_;
  }
// 创建一个静态方法对象，用于从字典中创建包含字符串键的 JIValue 对象
static auto jMethodDictStringKey =
    JIValue::javaClassStatic()
        ->getStaticMethod<facebook::jni::local_ref<JIValue>(
            facebook::jni::alias_ref<facebook::jni::JMap<
                facebook::jni::alias_ref<facebook::jni::JString::javaobject>,
                facebook::jni::alias_ref<JIValue::javaobject>>>)>(
            "dictStringKeyFrom");

// 创建一个空的 JHashMap 对象，用于存储从 C++ 字典转换而来的 Java 字符串键和 JIValue 对象
auto jmap = JHashMap<
    facebook::jni::alias_ref<facebook::jni::JString::javaobject>,
    facebook::jni::alias_ref<JIValue::javaobject>>::create();

// 遍历输入的 C++ 字典 dict，将其中的每对键值对转换为 Java 字符串键和对应的 JIValue 对象，并放入 jmap 中
for (auto& pair : dict) {
  jmap->put(
      facebook::jni::make_jstring(pair.key().toStringRef()),
      JIValue::newJIValueFromAtIValue(pair.value()));
}

// 调用之前创建的 jMethodDictStringKey 方法，将静态 Java 类和 jmap 作为参数，返回创建的 JIValue 对象
return jMethodDictStringKey(JIValue::javaClassStatic(), jmap);



// 创建一个静态方法对象，用于从字典中创建包含长整型键的 JIValue 对象
static auto jMethodDictLongKey =
    JIValue::javaClassStatic()
        ->getStaticMethod<facebook::jni::local_ref<JIValue>(
            facebook::jni::alias_ref<facebook::jni::JMap<
                facebook::jni::alias_ref<facebook::jni::JLong::javaobject>,
                facebook::jni::alias_ref<JIValue::javaobject>>>)>(
            "dictLongKeyFrom");

// 创建一个空的 JHashMap 对象，用于存储从 C++ 字典 dict 转换而来的 Java 长整型键和 JIValue 对象
auto jmap = JHashMap<
    facebook::jni::alias_ref<facebook::jni::JLong::javaobject>,
    facebook::jni::alias_ref<JIValue::javaobject>>::create();

// 遍历输入的 C++ 字典 dict，将其中的每对键值对转换为 Java 长整型键和对应的 JIValue 对象，并放入 jmap 中
for (auto& pair : dict) {
  jmap->put(
      facebook::jni::JLong::valueOf(pair.key().toInt()),
      JIValue::newJIValueFromAtIValue(pair.value()));
}

// 调用之前创建的 jMethodDictLongKey 方法，将静态 Java 类和 jmap 作为参数，返回创建的 JIValue 对象
return jMethodDictLongKey(JIValue::javaClassStatic(), jmap);



// 创建一个 JIValue 对象，根据输入的 at::IValue 类型 ivalue 进行转换
// 如果 ivalue 是 None 类型，则返回一个表示空值的 JIValue 对象
if (ivalue.isNone()) {
  static auto jMethodOptionalNull =
      JIValue::javaClassStatic()
          ->getStaticMethod<facebook::jni::local_ref<JIValue>()>(
              "optionalNull");
  return jMethodOptionalNull(JIValue::javaClassStatic());
}
// 如果 ivalue 是 Tensor 类型，则将其转换为对应的 JIValue 对象
else if (ivalue.isTensor()) {
  static auto jMethodTensor =
      JIValue::javaClassStatic()
          ->getStaticMethod<facebook::jni::local_ref<JIValue>(
              facebook::jni::local_ref<TensorHybrid::javaobject>)>("from");
  const auto& tensor = ivalue.toTensor();
  return jMethodTensor(
      JIValue::javaClassStatic(),
      TensorHybrid::newJTensorFromAtTensor(tensor));
}
// 如果 ivalue 是 Bool 类型，则将其转换为对应的 JIValue 对象
else if (ivalue.isBool()) {
  static auto jMethodBool =
      JIValue::javaClassStatic()
          ->getStaticMethod<facebook::jni::local_ref<JIValue>(jboolean)>(
              "from");
  return jMethodBool(JIValue::javaClassStatic(), ivalue.toBool());
}
// 如果 ivalue 是 Int 类型，则将其转换为对应的 JIValue 对象
else if (ivalue.isInt()) {
    // 如果输入值是整数类型
    static auto jMethodInt =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jlong)>("from");
    // 调用 JNI 方法将整数值转换为 JIValue 对象并返回
    return jMethodInt(JIValue::javaClassStatic(), ivalue.toInt());
  } else if (ivalue.isDouble()) {
    // 如果输入值是双精度浮点数类型
    static auto jMethodDouble =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(jdouble)>(
                "from");
    // 调用 JNI 方法将双精度浮点数值转换为 JIValue 对象并返回
    return jMethodDouble(JIValue::javaClassStatic(), ivalue.toDouble());
  } else if (ivalue.isString()) {
    // 如果输入值是字符串类型
    static auto jMethodString =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JString::javaobject>)>(
                "from");
    // 调用 JNI 方法将字符串转换为 JIValue 对象并返回
    return jMethodString(
        JIValue::javaClassStatic(),
        facebook::jni::make_jstring(ivalue.toStringRef()));
  } else if (ivalue.isTuple()) {
    // 如果输入值是元组类型
    auto elementsVec = ivalue.toTupleRef().elements();
    static auto jMethodTupleArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JArrayClass<
                    JIValue::javaobject>::javaobject>)>("tupleFrom");
    // 创建一个 JNI 数组来存储元组的元素
    auto jElementsArray =
        facebook::jni::JArrayClass<JIValue::javaobject>::newArray(
            elementsVec.size());
    auto index = 0;
    // 将元组的每个元素转换为 JIValue 对象存入 JNI 数组
    for (const auto& e : elementsVec) {
      (*jElementsArray)[index++] = JIValue::newJIValueFromAtIValue(e);
    }
    // 调用 JNI 方法将 JNI 数组转换为 JIValue 对象并返回
    return jMethodTupleArr(JIValue::javaClassStatic(), jElementsArray);
  } else if (ivalue.isBoolList()) {
    // 如果输入值是布尔列表类型
    auto list = ivalue.toBoolList();
    static auto jMethodBoolListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<jbooleanArray>)>("listFrom");
    // 创建 JNI 布尔数组来存储布尔列表的值
    size_t n = list.size();
    auto jArray = facebook::jni::make_boolean_array(n);
    auto jArrayPinned = jArray->pin();
    auto index = 0;
    // 将布尔列表的每个值复制到 JNI 布尔数组中
    for (const auto& e : list) {
      jArrayPinned[index++] = e;
    }
    // 调用 JNI 方法将 JNI 布尔数组转换为 JIValue 对象并返回
    return jMethodBoolListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isIntList()) {
    // 如果输入值是整数列表类型
    auto list = ivalue.toIntList();
    static auto jMethodLongListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<jlongArray>)>("listFrom");
    // 创建 JNI 长整型数组来存储整数列表的值
    size_t n = list.size();
    auto jArray = facebook::jni::make_long_array(n);
    auto jArrayPinned = jArray->pin();
    auto index = 0;
    // 将整数列表的每个值复制到 JNI 长整型数组中
    for (const auto& e : list) {
      jArrayPinned[index++] = e;
    }
    // 调用 JNI 方法将 JNI 长整型数组转换为 JIValue 对象并返回
    return jMethodLongListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isDoubleList()) {
    // 如果输入值是双精度浮点数列表类型
    auto list = ivalue.toDoubleList();
    static auto jMethoDoubleListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<jdoubleArray>)>("listFrom");
    // 创建 JNI 双精度浮点数数组来存储双精度浮点数列表的值
    size_t n = list.size();
    auto jArray = facebook::jni::make_double_array(n);
    auto jArrayPinned = jArray->pin();
    auto index = 0;
    // 将双精度浮点数列表的每个值复制到 JNI 双精度浮点数数组中
    for (const auto& e : list) {
      jArrayPinned[index++] = e;
    }
    // 调用 JNI 方法将 JNI 双精度浮点数数组转换为 JIValue 对象并返回
    return jMethoDoubleListArr(JIValue::javaClassStatic(), jArray);
    // 获取列表的大小
    size_t n = list.size();
    // 创建一个双精度数组对象，大小为 n
    auto jArray = facebook::jni::make_double_array(n);
    // 获取双精度数组的内存引用
    auto jArrayPinned = jArray->pin();
    // 初始化索引为 0
    auto index = 0;
    // 遍历列表中的每个元素 e
    for (const auto& e : list) {
      // 将列表中的每个元素复制到双精度数组中
      jArrayPinned[index++] = e;
    }
    // 调用特定方法 jMethoDoubleListArr，传入 JIValue 的静态 Java 类和双精度数组对象 jArray
    return jMethoDoubleListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isTensorList()) {
    // 如果 ivalue 是张量列表
    auto list = ivalue.toTensorList();
    // 获取处理张量列表的静态 Java 方法 jMethodTensorListArr
    static auto jMethodTensorListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JArrayClass<
                    TensorHybrid::javaobject>::javaobject>)>("listFrom");
    // 创建一个张量对象数组 jArray，大小为 list 的大小
    auto jArray =
        facebook::jni::JArrayClass<TensorHybrid::javaobject>::newArray(
            list.size());
    // 初始化索引为 0
    auto index = 0;
    // 遍历张量列表中的每个元素 e
    for (const auto& e : list) {
      // 将每个张量对象转换为对应的 Java 对象，并存储到 jArray 中
      (*jArray)[index++] = TensorHybrid::newJTensorFromAtTensor(e);
    }
    // 调用 jMethodTensorListArr 方法，传入 JIValue 的静态 Java 类和张量数组 jArray
    return jMethodTensorListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isList()) {
    // 如果 ivalue 是列表
    auto list = ivalue.toList();
    // 获取处理列表的静态 Java 方法 jMethodListArr
    static auto jMethodListArr =
        JIValue::javaClassStatic()
            ->getStaticMethod<facebook::jni::local_ref<JIValue>(
                facebook::jni::alias_ref<facebook::jni::JArrayClass<
                    JIValue::javaobject>::javaobject>)>("listFrom");
    // 创建一个 JIValue 对象数组 jArray，大小为 list 的大小
    auto jArray =
        facebook::jni::JArrayClass<JIValue::javaobject>::newArray(list.size());
    // 初始化索引为 0
    auto index = 0;
    // 遍历列表中的每个元素 e
    for (const auto& e : list) {
      // 将每个 IValue 对象转换为对应的 Java 对象，并存储到 jArray 中
      (*jArray)[index++] = JIValue::newJIValueFromAtIValue(e);
    }
    // 调用 jMethodListArr 方法，传入 JIValue 的静态 Java 类和对象数组 jArray
    return jMethodListArr(JIValue::javaClassStatic(), jArray);
  } else if (ivalue.isGenericDict()) {
    // 如果 ivalue 是通用字典
    auto dict = ivalue.toGenericDict();
    // 获取字典键的类型
    const auto keyType = dict.keyType();

    // 检查键类型是否存在，否则抛出异常
    if (!keyType) {
      facebook::jni::throwNewJavaException(
          facebook::jni::gJavaLangIllegalArgumentException,
          "Unknown IValue-Dict key type");
    }

    // 根据键类型执行不同的操作
    if (*keyType == *c10::StringType::get()) {
      // 如果键类型是字符串，调用 stringDictCallback 处理字典，并返回结果
      return stringDictCallback(std::move(dict));
    } else if (*keyType == *c10::IntType::get()) {
      // 如果键类型是整数，调用 intDictCallback 处理字典，并返回结果
      return intDictCallback(std::move(dict));
    }

    // 若键类型不是字符串或整数，则抛出异常
    facebook::jni::throwNewJavaException(
        facebook::jni::gJavaLangIllegalArgumentException,
        "Unsupported IValue-Dict key type: %s",
        keyType->str().c_str());
  }

  // 如果 ivalue 的类型不被支持，则抛出相应的异常
  facebook::jni::throwNewJavaException(
      facebook::jni::gJavaLangIllegalArgumentException,
      "Unsupported IValue type %s",
      ivalue.tagKind().c_str());
}

at::IValue JIValue::JIValueToAtIValue(
    facebook::jni::alias_ref<JIValue> jivalue) {
  // 创建跟踪对象 _s，用于记录函数调用路径
  Trace _s{"jni::JIValue::JIValueToAtIValue"};
  // 获取静态常量 typeCodeField，表示 JNI 类中的类型码字段 mTypeCode
  static const auto typeCodeField =
      JIValue::javaClassStatic()->getField<jint>("mTypeCode");
  // 获取 jivalue 对象中的 mTypeCode 字段值
  const auto typeCode = jivalue->getFieldValue(typeCodeField);
  // 根据类型码判断 JIValue 类型并转换为对应的 at::IValue 类型
  if (JIValue::kTypeCodeNull == typeCode) {
    // 如果类型码表示空值，则返回空的 at::IValue
    return at::IValue{};
  } else if (JIValue::kTypeCodeTensor == typeCode) {
    // 如果类型码表示张量，则调用 toTensor 方法转换为 at::IValue
    static const auto jMethodGetTensor =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::alias_ref<TensorHybrid::javaobject>()>(
                "toTensor");
    return TensorHybrid::newAtTensorFromJTensor(jMethodGetTensor(jivalue));
  } else if (JIValue::kTypeCodeBool == typeCode) {
    // 如果类型码表示布尔值，则调用 toBool 方法获取布尔值并转换为 at::IValue
    static const auto jMethodGetBool =
        JIValue::javaClassStatic()->getMethod<jboolean()>("toBool");
    // 明确将 jboolean 转换为 bool 类型，因为 jboolean 实际上是 uint8_t
    bool b = jMethodGetBool(jivalue);
    return at::IValue{b};
  } else if (JIValue::kTypeCodeLong == typeCode) {
    // 如果类型码表示长整型，则调用 toLong 方法获取长整型并转换为 at::IValue
    static const auto jMethodGetLong =
        JIValue::javaClassStatic()->getMethod<jlong()>("toLong");
    return at::IValue{(int64_t)jMethodGetLong(jivalue)};
  } else if (JIValue::kTypeCodeDouble == typeCode) {
    // 如果类型码表示双精度浮点数，则调用 toDouble 方法获取浮点数并转换为 at::IValue
    static const auto jMethodGetDouble =
        JIValue::javaClassStatic()->getMethod<jdouble()>("toDouble");
    return at::IValue{jMethodGetDouble(jivalue)};
  } else if (JIValue::kTypeCodeString == typeCode) {
    // 如果类型码表示字符串，则调用 toStr 方法获取字符串并转换为 at::IValue
    static const auto jMethodGetString =
        JIValue::javaClassStatic()->getMethod<jstring()>("toStr");
    return at::IValue{jMethodGetString(jivalue)->toStdString()};
  } else if (JIValue::kTypeCodeTuple == typeCode) {
    // 如果类型码表示元组，则调用 toTuple 方法获取元组并转换为 at::IValue
    static const auto jMethodGetTuple =
        JIValue::javaClassStatic()
            ->getMethod<
                facebook::jni::JArrayClass<JIValue::javaobject>::javaobject()>(
                "toTuple");
    auto jarray = jMethodGetTuple(jivalue);
    size_t n = jarray->size();

    std::vector<at::IValue> elements;
    elements.reserve(n);
    for (const auto i : c10::irange(n)) {
      auto jivalue_element = jarray->getElement(i);
      auto element = JIValue::JIValueToAtIValue(jivalue_element);
      elements.push_back(std::move(element));
    }
    // 创建并返回包含元组元素的 at::IValue
    return c10::ivalue::Tuple::create(std::move(elements));
  } else if (JIValue::kTypeCodeBoolList == typeCode) {
    // 如果类型码表示布尔值列表，则调用 toBoolList 方法获取列表并转换为 at::IValue
    static const auto jMethodGetBoolList =
        JIValue::javaClassStatic()->getMethod<jbooleanArray()>("toBoolList");
    auto jArray = jMethodGetBoolList(jivalue);
    auto jArrayPinned = jArray->pin();
    size_t n = jArrayPinned.size();
    c10::List<bool> list{};
    list.reserve(n);
    for (const auto i : c10::irange(n)) {
      list.push_back(jArrayPinned[i]);
    }
    // 创建并返回包含布尔值列表的 at::IValue
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeLongList == typeCode) {
    // 如果类型码表示长整型列表，则调用 toLongList 方法获取列表并转换为 at::IValue
    static const auto jMethodGetLongList =
        JIValue::javaClassStatic()->getMethod<jlongArray()>("toLongList");
    // 获取长整型列表的方法，并获取其对应的 Java 对象数组
    auto jArray = jMethodGetLongList(jivalue);
    // 将 Java 对象数组固定在内存中，以确保数据在 C++ 环境下的稳定性
    auto jArrayPinned = jArray->pin();
    // 获取数组的长度
    size_t n = jArrayPinned.size();
    // 创建一个 C++ 的长整型列表，预留空间以容纳 Java 数组的所有元素
    c10::List<int64_t> list{};
    list.reserve(n);
    // 将固定在内存中的 Java 数组的元素逐个添加到 C++ 列表中
    for (const auto i : c10::irange(n)) {
      list.push_back(jArrayPinned[i]);
    }
    // 将 C++ 列表包装成 PyTorch 的 IValue 类型并返回
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeDoubleList == typeCode) {
    // 获取双精度浮点数列表的方法，并获取其对应的 Java 对象数组
    static const auto jMethodGetDoubleList =
        JIValue::javaClassStatic()->getMethod<jdoubleArray()>("toDoubleList");
    auto jArray = jMethodGetDoubleList(jivalue);
    // 将 Java 对象数组固定在内存中
    auto jArrayPinned = jArray->pin();
    // 获取数组的长度
    size_t n = jArrayPinned.size();
    // 创建一个 C++ 的双精度浮点数列表，预留空间以容纳 Java 数组的所有元素
    c10::List<double> list{};
    list.reserve(n);
    // 将固定在内存中的 Java 数组的元素逐个添加到 C++ 列表中
    for (const auto i : c10::irange(n)) {
      list.push_back(jArrayPinned[i]);
    }
    // 将 C++ 列表包装成 PyTorch 的 IValue 类型并返回
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeTensorList == typeCode) {
    // 获取张量列表的方法，并获取其对应的 Java 对象数组
    static const auto jMethodGetTensorList =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::JArrayClass<
                TensorHybrid::javaobject>::javaobject()>("toTensorList");
    auto jArray = jMethodGetTensorList(jivalue);
    // 获取数组的长度
    size_t n = jArray->size();
    // 创建一个 C++ 的张量列表，预留空间以容纳 Java 数组的所有元素
    c10::List<at::Tensor> list{};
    list.reserve(n);
    // 将 Java 数组的每个元素转换为 C++ 的张量并添加到列表中
    for (const auto i : c10::irange(n)) {
      list.push_back(
          TensorHybrid::newAtTensorFromJTensor(jArray->getElement(i)));
    }
    // 将 C++ 列表包装成 PyTorch 的 IValue 类型并返回
    return at::IValue{std::move(list)};
  } else if (JIValue::kTypeCodeList == typeCode) {
    // 获取通用列表的方法，并获取其对应的 Java 对象数组
    static const auto jMethodGetList =
        JIValue::javaClassStatic()
            ->getMethod<
                facebook::jni::JArrayClass<JIValue::javaobject>::javaobject()>(
                "toList");
    auto jarray = jMethodGetList(jivalue);
    // 获取数组的长度
    size_t n = jarray->size();
    // 如果列表为空，返回一个空的通用列表
    if (n == 0) {
      return at::IValue{c10::impl::GenericList(c10::TensorType::get())};
    }

    // 获取列表的第一个元素，并将其转换为 C++ 的 IValue 类型
    auto jivalue_first_element = jarray->getElement(0);
    auto first_element = JIValue::JIValueToAtIValue(jivalue_first_element);
    // 创建一个具有相同类型的 C++ 通用列表，并将第一个元素添加进去
    c10::impl::GenericList list{c10::unshapedType(first_element.type())};
    list.reserve(n);
    list.push_back(first_element);
    // 将列表的剩余元素逐个转换为 C++ 的 IValue 类型并添加进通用列表中
    for (const auto i : c10::irange(1, n)) {
      auto jivalue_element = jarray->getElement(i);
      auto element = JIValue::JIValueToAtIValue(jivalue_element);
      list.push_back(element);
    }
    // 将 C++ 通用列表包装成 PyTorch 的 IValue 类型并返回
    return at::IValue{list};
  } else if (JIValue::kTypeCodeDictStringKey == typeCode) {
    // 获取字符串键字典的方法，并获取其对应的 Java 对象映射
    static const auto jMethodGetDictStringKey =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::JMap<jstring, JIValue::javaobject>::
                            javaobject()>("toDictStringKey");
    auto jmap = jMethodGetDictStringKey(jivalue);
    auto it = jmap->begin();
    // 如果字典为空，返回一个空的通用字典
    if (it == jmap->end()) {
      return at::IValue{c10::impl::GenericDict(
          c10::StringType::get(), c10::TensorType::get())};
    }

    // 获取字典的第一个条目值，并将其转换为 C++ 的 IValue 类型
    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    // 创建一个具有相同值类型的 C++ 通用字典
    c10::impl::GenericDict dict{
        c10::StringType::get(), c10::unshapedType(firstEntryValue.type())};
    // 将字典的每个条目值逐个转换为 C++ 的 IValue 类型并添加进通用字典中
    for (auto& entry : *jmap) {
      auto element = JIValue::JIValueToAtIValue(entry.second);
      dict.insert_or_assign(entry.first->toStdString(), element);
    }
    // 将 C++ 通用字典包装成 PyTorch 的 IValue 类型并返回
    return at::IValue{dict};
    // 将第一个键值对插入字典中，键为字符串类型的第一个键的字符串表示，值为第一个条目的转换后的 IValue
    dict.insert(it->first->toStdString(), firstEntryValue);
    // 迭代器移动到下一个键值对
    it++;
    // 遍历剩余的 jmap 中的键值对
    for (; it != jmap->end(); it++) {
      // 将键值对插入字典中，键为字符串类型的键的字符串表示，值为将 JIValue 转换为 AtIValue 后的结果
      dict.insert(
          it->first->toStdString(), JIValue::JIValueToAtIValue(it->second));
    }
    // 返回一个包含所有键值对的 c10::impl::GenericDict 类型的 at::IValue 对象
    return at::IValue{dict};
  } else if (JIValue::kTypeCodeDictLongKey == typeCode) {
    // 获取静态方法 toDictLongKey 并应用到 jivalue 上，返回一个 jmap
    static const auto jMethodGetDictLongKey =
        JIValue::javaClassStatic()
            ->getMethod<facebook::jni::JMap<
                facebook::jni::JLong::javaobject,
                JIValue::javaobject>::javaobject()>("toDictLongKey");
    auto jmap = jMethodGetDictLongKey(jivalue);
    auto it = jmap->begin();
    // 若 jmap 为空，则返回一个空的 c10::impl::GenericDict 类型的 at::IValue 对象
    if (it == jmap->end()) {
      return at::IValue{
          c10::impl::GenericDict(c10::IntType::get(), c10::TensorType::get())};
    }

    // 将第一个键值对插入字典中，键为 long 类型的第一个键的 long 值，值为第一个条目的转换后的 IValue
    auto firstEntryValue = JIValue::JIValueToAtIValue(it->second);
    c10::impl::GenericDict dict{
        c10::IntType::get(), c10::unshapedType(firstEntryValue.type())};
    dict.insert((int64_t)it->first->longValue(), firstEntryValue);
    // 迭代器移动到下一个键值对
    it++;
    // 遍历剩余的 jmap 中的键值对
    for (; it != jmap->end(); it++) {
      // 将键值对插入字典中，键为 long 类型的键的 long 值，值为将 JIValue 转换为 AtIValue 后的结果
      dict.insert(
          (int64_t)it->first->longValue(),
          JIValue::JIValueToAtIValue(it->second));
    }
    // 返回一个包含所有键值对的 c10::impl::GenericDict 类型的 at::IValue 对象
    return at::IValue{dict};
  }

  // 抛出一个 Java 异常，表示未知的 IValue 类型码 typeCode
  facebook::jni::throwNewJavaException(
      facebook::jni::gJavaLangIllegalArgumentException,
      "Unknown IValue typeCode %d",
      typeCode);
}

#if defined(__ANDROID__)
// 定义一个名为 PyTorchAndroidJni 的 C++ 类，继承自 JavaClass<PyTorchAndroidJni>
class PyTorchAndroidJni : public facebook::jni::JavaClass<PyTorchAndroidJni> {
 public:
  // 定义 Java 类描述符
  constexpr static auto kJavaDescriptor = "Lorg/pytorch/PyTorchAndroid;";

  // 注册 JNI 方法
  static void registerNatives() {
    // 调用 JavaClass<PyTorchAndroidJni> 的静态方法注册本地方法
    javaClassStatic()->registerNatives({
        // 创建本地方法对象，指定方法名 "nativeSetNumThreads" 和对应的 C++ 实现函数 PyTorchAndroidJni::setNumThreads
        makeNativeMethod(
            "nativeSetNumThreads", PyTorchAndroidJni::setNumThreads),
    });
  }

  // 设置线程数的 JNI 方法
  static void setNumThreads(facebook::jni::alias_ref<jclass>, jint numThreads) {
    // 调用 caffe2::pthreadpool() 的方法设置线程数
    caffe2::pthreadpool()->set_thread_count(numThreads);
  }
};
#endif

// 通用的 JNI 方法注册函数
void common_registerNatives() {
  // 定义静态变量 once，确保代码只执行一次
  static const int once = []() {
#if defined(__ANDROID__)
    // 如果是 Android 平台，调用 pytorch_jni::PyTorchAndroidJni 的注册方法
    pytorch_jni::PyTorchAndroidJni::registerNatives();
#endif
    return 0;
  }();
  // 防止编译器警告未使用变量 once
  ((void)once);
}

} // namespace pytorch_jni
```