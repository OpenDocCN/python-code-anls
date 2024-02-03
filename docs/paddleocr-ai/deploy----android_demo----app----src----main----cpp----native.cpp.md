# `.\PaddleOCR\deploy\android_demo\app\src\main\cpp\native.cpp`

```py
// 包含必要的头文件
#include "native.h"
#include "ocr_ppredictor.h"
#include <algorithm>
#include <paddle_api.h>
#include <string>

// 将字符串转换为对应的 CPU 模式
static paddle::lite_api::PowerMode str_to_cpu_mode(const std::string &cpu_mode);

// JNI 入口函数，初始化 OCR 模型预测器
extern "C" JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_init(
    JNIEnv *env, jobject thiz, jstring j_det_model_path,
    jstring j_rec_model_path, jstring j_cls_model_path, jint j_use_opencl, jint j_thread_num,
    jstring j_cpu_mode) {
  // 将 Java 字符串转换为 C++ 字符串
  std::string det_model_path = jstring_to_cpp_string(env, j_det_model_path);
  std::string rec_model_path = jstring_to_cpp_string(env, j_rec_model_path);
  std::string cls_model_path = jstring_to_cpp_string(env, j_cls_model_path);
  int thread_num = j_thread_num;
  std::string cpu_mode = jstring_to_cpp_string(env, j_cpu_mode);
  // 初始化 OCR 配置
  ppredictor::OCR_Config conf;
  conf.use_opencl = j_use_opencl;
  conf.thread_num = thread_num;
  conf.mode = str_to_cpu_mode(cpu_mode);
  // 创建 OCR 预测器对象
  ppredictor::OCR_PPredictor *orc_predictor =
      new ppredictor::OCR_PPredictor{conf};
  // 从文件初始化 OCR 模型
  orc_predictor->init_from_file(det_model_path, rec_model_path, cls_model_path);
  // 将 OCR 预测器对象转换为 jlong 类型返回
  return reinterpret_cast<jlong>(orc_predictor);
}

/**
 * "LITE_POWER_HIGH" convert to paddle::lite_api::LITE_POWER_HIGH
 * @param cpu_mode
 * @return
 */
static paddle::lite_api::PowerMode
// 将字符串映射到对应的 CPU 模式
str_to_cpu_mode(const std::string &cpu_mode) {
  // 静态映射表，将字符串映射到对应的 CPU 模式枚举值
  static std::map<std::string, paddle::lite_api::PowerMode> cpu_mode_map{
      {"LITE_POWER_HIGH", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_LOW", paddle::lite_api::LITE_POWER_HIGH},
      {"LITE_POWER_FULL", paddle::lite_api::LITE_POWER_FULL},
      {"LITE_POWER_NO_BIND", paddle::lite_api::LITE_POWER_NO_BIND},
      {"LITE_POWER_RAND_HIGH", paddle::lite_api::LITE_POWER_RAND_HIGH},
      {"LITE_POWER_RAND_LOW", paddle::lite_api::LITE_POWER_RAND_LOW}};
  // 将输入字符串转换为大写形式
  std::string upper_key;
  std::transform(cpu_mode.cbegin(), cpu_mode.cend(), upper_key.begin(),
                 ::toupper);
  // 在映射表中查找对应的 CPU 模式
  auto index = cpu_mode_map.find(upper_key.c_str());
  // 如果未找到对应的 CPU 模式，则返回默认值 LITE_POWER_HIGH
  if (index == cpu_mode_map.end()) {
    LOGE("cpu_mode not found %s", upper_key.c_str());
    return paddle::lite_api::LITE_POWER_HIGH;
  } else {
    // 返回找到的 CPU 模式
    return index->second;
  }
}

// JNI 接口函数，用于执行 OCR 模型的推理
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_forward(
    JNIEnv *env, jobject thiz, jlong java_pointer, jobject original_image,jint j_max_size_len, jint j_run_det, jint j_run_cls, jint j_run_rec) {
  LOGI("begin to run native forward");
  // 检查 JAVA 指针是否为空
  if (java_pointer == 0) {
    LOGE("JAVA pointer is NULL");
    return cpp_array_to_jfloatarray(env, nullptr, 0);
  }

  // 将原始图像转换为 OpenCV Mat 对象
  cv::Mat origin = bitmap_to_cv_mat(env, original_image);
  // 如果转换失败，打印错误信息
  if (origin.size == 0) {
    LOGE("origin bitmap cannot convert to CV Mat");
  // 返回一个空的 jfloatarray
  return cpp_array_to_jfloatarray(env, nullptr, 0);
}

// 将传入的参数赋值给对应的变量
int max_size_len = j_max_size_len;
int run_det = j_run_det;
int run_cls = j_run_cls;
int run_rec = j_run_rec;

// 将 java_pointer 转换为 OCR_PPredictor 指针类型
ppredictor::OCR_PPredictor *ppredictor = (ppredictor::OCR_PPredictor *)java_pointer;

// 创建一个存储 int64_t 类型数据的向量 dims_arr
std::vector<int64_t> dims_arr;

// 调用 OCR_PPredictor 对象的 infer_ocr 方法进行 OCR 推理
std::vector<ppredictor::OCRPredictResult> results = ppredictor->infer_ocr(origin, max_size_len, run_det, run_cls, run_rec);
LOGI("infer_ocr finished with boxes %ld", results.size());

// 将 std::vector<ppredictor::OCRPredictResult> 序列化成 float 数组，用于传输到 Java 层再反序列化
std::vector<float> float_arr;
for (const ppredictor::OCRPredictResult &r : results) {
  // 添加 OCR 结果中的点数、单词索引、分数
  float_arr.push_back(r.points.size());
  float_arr.push_back(r.word_index.size());
  float_arr.push_back(r.score);
  // 添加检测点
  for (const std::vector<int> &point : r.points) {
    float_arr.push_back(point.at(0));
    float_arr.push_back(point.at(1));
  }
  // 添加识别单词索引
  for (int index : r.word_index) {
    float_arr.push_back(index);
  }
  // 添加分类结果
  float_arr.push_back(r.cls_label);
  float_arr.push_back(r.cls_score);
}

// 将 float 数组转换为 jfloatarray 并返回
return cpp_array_to_jfloatarray(env, float_arr.data(), float_arr.size());
}

// 释放 OCR 模型预测器对象
extern "C" JNIEXPORT void JNICALL
Java_com_baidu_paddle_lite_demo_ocr_OCRPredictorNative_release(
    JNIEnv *env, jobject thiz, jlong java_pointer) {
  // 如果 JAVA 指针为空，打印错误信息并返回
  if (java_pointer == 0) {
    LOGE("JAVA pointer is NULL");
    return;
  }
  // 将 JAVA 指针转换为 OCR 模型预测器指针
  ppredictor::OCR_PPredictor *ppredictor =
      (ppredictor::OCR_PPredictor *)java_pointer;
  // 释放 OCR 模型预测器对象
  delete ppredictor;
}
```