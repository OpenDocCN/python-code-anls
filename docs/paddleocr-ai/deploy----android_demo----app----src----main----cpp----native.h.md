# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\native.h`

```py
// 包含头文件 common.h
// 包含 Android Bitmap 相关头文件
// 包含 JNI 相关头文件
// 包含 OpenCV 相关头文件
// 包含 string 和 vector 头文件
#pragma once

#include "common.h"
#include <android/bitmap.h>
#include <jni.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 将 JNI 的 jstring 转换为 C++ 的 std::string
inline std::string jstring_to_cpp_string(JNIEnv *env, jstring jstr) {
  // 在 Java 中，一个 Unicode 字符将使用 2 个字节（utf16）进行编码。
  // 因此 jstring 将包含 utf16 字符。C++ 中的 std::string 本质上是一个字节的字符串，而不是字符，
  // 所以如果我们想要将 jstring 从 JNI 传递到 C++，我们需要将 utf16 转换为字节。
  if (!jstr) {
    return "";
  }
  // 获取 jstring 的类对象
  const jclass stringClass = env->GetObjectClass(jstr);
  // 获取 getBytes 方法的 ID
  const jmethodID getBytes =
      env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
  // 调用 getBytes 方法获取字节数组
  const jbyteArray stringJbytes = (jbyteArray)env->CallObjectMethod(
      jstr, getBytes, env->NewStringUTF("UTF-8"));

  // 获取字节数组的长度
  size_t length = (size_t)env->GetArrayLength(stringJbytes);
  // 获取字节数组的指针
  jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

  // 将字节数组转换为 std::string
  std::string ret = std::string(reinterpret_cast<char *>(pBytes), length);
  // 释放字节数组的指针
  env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

  // 删除本地引用
  env->DeleteLocalRef(stringJbytes);
  env->DeleteLocalRef(stringClass);
  return ret;
}

// 将 C++ 的 std::string 转换为 JNI 的 jstring
inline jstring cpp_string_to_jstring(JNIEnv *env, std::string str) {
  // 获取字符串的指针
  auto *data = str.c_str();
  // 获取 java.lang.String 类的类对象
  jclass strClass = env->FindClass("java/lang/String");
  // 获取 java.lang.String 类的构造方法 ID
  jmethodID strClassInitMethodID =
      env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");

  // 创建字节数组并设置数据
  jbyteArray bytes = env->NewByteArray(strlen(data));
  env->SetByteArrayRegion(bytes, 0, strlen(data),
                          reinterpret_cast<const jbyte *>(data));

  // 创建字符串编码
  jstring encoding = env->NewStringUTF("UTF-8");
  // 创建 jstring 对象
  jstring res = (jstring)(
      env->NewObject(strClass, strClassInitMethodID, bytes, encoding));

  // 删除本地引用
  env->DeleteLocalRef(strClass);
  env->DeleteLocalRef(encoding);
  env->DeleteLocalRef(bytes);

  return res;
}
// 将 C++ 中的 float 数组转换为 Java 中的 jfloatArray
inline jfloatArray cpp_array_to_jfloatarray(JNIEnv *env, const float *buf,
                                            int64_t len) {
  // 如果数组长度为 0，则返回一个长度为 0 的 jfloatArray
  if (len == 0) {
    return env->NewFloatArray(0);
  }
  // 创建一个长度为 len 的 jfloatArray
  jfloatArray result = env->NewFloatArray(len);
  // 将 buf 数组中的数据复制到 jfloatArray 中
  env->SetFloatArrayRegion(result, 0, len, buf);
  return result;
}

// 将 C++ 中的 int 数组转换为 Java 中的 jintArray
inline jintArray cpp_array_to_jintarray(JNIEnv *env, const int *buf,
                                        int64_t len) {
  // 创建一个长度为 len 的 jintArray
  jintArray result = env->NewIntArray(len);
  // 将 buf 数组中的数据复制到 jintArray 中
  env->SetIntArrayRegion(result, 0, len, buf);
  return result;
}

// 将 C++ 中的 int8_t 数组转换为 Java 中的 jbyteArray
inline jbyteArray cpp_array_to_jbytearray(JNIEnv *env, const int8_t *buf,
                                          int64_t len) {
  // 创建一个长度为 len 的 jbyteArray
  jbyteArray result = env->NewByteArray(len);
  // 将 buf 数组中的数据复制到 jbyteArray 中
  env->SetByteArrayRegion(result, 0, len, buf);
  return result;
}

// 将 C++ 中的 std::vector<int64_t> 转换为 Java 中的 jlongArray
inline jlongArray int64_vector_to_jlongarray(JNIEnv *env,
                                             const std::vector<int64_t> &vec) {
  // 创建一个长度为 vec.size() 的 jlongArray
  jlongArray result = env->NewLongArray(vec.size());
  // 创建一个 buf 数组，用于存储 vec 中的数据
  jlong *buf = new jlong[vec.size()];
  // 将 vec 中的数据复制到 buf 数组中
  for (size_t i = 0; i < vec.size(); ++i) {
    buf[i] = (jlong)vec[i];
  }
  // 将 buf 数组中的数据复制到 jlongArray 中
  env->SetLongArrayRegion(result, 0, vec.size(), buf);
  delete[] buf;
  return result;
}

// 将 Java 中的 jlongArray 转换为 C++ 中的 std::vector<int64_t>
inline std::vector<int64_t> jlongarray_to_int64_vector(JNIEnv *env,
                                                       jlongArray data) {
  // 获取 jlongArray 的长度
  int data_size = env->GetArrayLength(data);
  // 获取 jlongArray 的数据指针
  jlong *data_ptr = env->GetLongArrayElements(data, nullptr);
  // 将 jlongArray 中的数据转换为 std::vector<int64_t>
  std::vector<int64_t> data_vec(data_ptr, data_ptr + data_size);
  // 释放 jlongArray 的数据指针
  env->ReleaseLongArrayElements(data, data_ptr, 0);
  return data_vec;
}

// 将 Java 中的 jfloatArray 转换为 C++ 中的 std::vector<float>
inline std::vector<float> jfloatarray_to_float_vector(JNIEnv *env,
                                                      jfloatArray data) {
  // 获取 jfloatArray 的长度
  int data_size = env->GetArrayLength(data);
  // 获取 jfloatArray 的数据指针
  jfloat *data_ptr = env->GetFloatArrayElements(data, nullptr);
  // 将 jfloatArray 中的数据转换为 std::vector<float>
  std::vector<float> data_vec(data_ptr, data_ptr + data_size);
  // 释放 jfloatArray 的数据指针
  env->ReleaseFloatArrayElements(data, data_ptr, 0);
  return data_vec;
}
// 将 Android 的 Bitmap 对象转换为 OpenCV 的 Mat 对象
inline cv::Mat bitmap_to_cv_mat(JNIEnv *env, jobject bitmap) {
  // 定义 AndroidBitmapInfo 结构体，用于存储 Bitmap 对象的信息
  AndroidBitmapInfo info;
  // 获取 Bitmap 对象的信息，并将结果存储在 info 中
  int result = AndroidBitmap_getInfo(env, bitmap, &info);
  // 如果获取信息失败，则记录错误信息并返回空的 Mat 对象
  if (result != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOGE("AndroidBitmap_getInfo failed, result: %d", result);
    return cv::Mat{};
  }
  // 如果 Bitmap 对象的格式不是 RGBA_8888，则记录错误信息并返回空的 Mat 对象
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("Bitmap format is not RGBA_8888 !");
    return cv::Mat{};
  }
  // 定义指向 Bitmap 数据的指针
  unsigned char *srcData = NULL;
  // 锁定 Bitmap 数据，将其指针赋值给 srcData
  AndroidBitmap_lockPixels(env, bitmap, (void **)&srcData);
  // 创建一个高度为 info.height，宽度为 info.width，通道数为 4 的 Mat 对象
  cv::Mat mat = cv::Mat::zeros(info.height, info.width, CV_8UC4);
  // 将 Bitmap 数据拷贝到 Mat 对象中
  memcpy(mat.data, srcData, info.height * info.width * 4);
  // 解锁 Bitmap 数据
  AndroidBitmap_unlockPixels(env, bitmap);
  // 将 RGBA 格式的 Mat 对象转换为 BGR 格式
  cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
  /**
  // 将 Mat 对象保存为图片文件，如果保存失败则记录错误信息
  if (!cv::imwrite("/sdcard/1/copy.jpg", mat)){
      LOGE("Write image failed " );
  }
   */
  // 返回转换后的 Mat 对象
  return mat;
}
```