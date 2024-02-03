# `.\PaddleOCR\deploy\cpp_infer\src\paddleocr.cpp`

```py
// 包含版权声明和许可信息
// 根据 Apache 许可证 2.0 版本使用此文件
// 如果不遵守许可证，不得使用此文件
// 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
// 根据适用法律或书面同意，分发的软件基于“原样”分发
// 没有任何明示或暗示的保证或条件，无论是明示还是暗示
// 请查看许可证以获取特定语言的权限和限制

// 包含头文件
#include <include/args.h>
#include <include/paddleocr.h>

#include "auto_log/autolog.h"

// 命名空间 PaddleOCR
namespace PaddleOCR {

// PPOCR 类的构造函数
PPOCR::PPOCR() {
  // 如果启用了检测器
  if (FLAGS_det) {
    // 创建 DBDetector 对象
    this->detector_ = new DBDetector(
        FLAGS_det_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_limit_type,
        FLAGS_limit_side_len, FLAGS_det_db_thresh, FLAGS_det_db_box_thresh,
        FLAGS_det_db_unclip_ratio, FLAGS_det_db_score_mode, FLAGS_use_dilation,
        FLAGS_use_tensorrt, FLAGS_precision);
  }

  // 如果启用了分类器和角度分类器
  if (FLAGS_cls && FLAGS_use_angle_cls) {
    // 创建 Classifier 对象
    this->classifier_ = new Classifier(
        FLAGS_cls_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_cls_thresh,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_cls_batch_num);
  }

  // 如果启用了识别器
  if (FLAGS_rec) {
    // 创建 CRNNRecognizer 对象
    this->recognizer_ = new CRNNRecognizer(
        FLAGS_rec_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_rec_char_dict_path,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_rec_batch_num,
        FLAGS_rec_img_h, FLAGS_rec_img_w);
  }
};

// OCR 方法，对图像列表进行 OCR 处理
std::vector<std::vector<OCRPredictResult>>
PPOCR::ocr(std::vector<cv::Mat> img_list, bool det, bool rec, bool cls) {
  // 存储 OCR 结果的二维向量
  std::vector<std::vector<OCRPredictResult>> ocr_results;

  // 如果不需要检测
  if (!det) {
    // 创建一个存储 OCR 预测结果的向量
    std::vector<OCRPredictResult> ocr_result;
    // 调整 OCR 结果向量的大小为图像列表的大小
    ocr_result.resize(img_list.size());
    // 如果需要分类并且分类器不为空
    if (cls && this->classifier_ != nullptr) {
      // 对图像列表进行分类
      this->cls(img_list, ocr_result);
      // 遍历每个图像的分类结果
      for (int i = 0; i < img_list.size(); i++) {
        // 如果分类标签为奇数且分类得分高于分类阈值
        if (ocr_result[i].cls_label % 2 == 1 &&
            ocr_result[i].cls_score > this->classifier_->cls_thresh) {
          // 对图像进行顺时针旋转
          cv::rotate(img_list[i], img_list[i], 1);
        }
      }
    }
    // 如果需要识别文本
    if (rec) {
      // 对图像列表进行文本识别
      this->rec(img_list, ocr_result);
    }
    // 遍历 OCR 结果向量
    for (int i = 0; i < ocr_result.size(); ++i) {
      // 创建临时的 OCR 预测结果向量
      std::vector<OCRPredictResult> ocr_result_tmp;
      // 将当前 OCR 预测结果添加到临时向量中
      ocr_result_tmp.push_back(ocr_result[i]);
      // 将临时 OCR 预测结果向量添加到最终结果中
      ocr_results.push_back(ocr_result_tmp);
    }
  } else {
    // 对图像列表中的每个图像进行 OCR
    for (int i = 0; i < img_list.size(); ++i) {
      // 调用 OCR 函数进行文本识别，并将结果添加到最终结果中
      std::vector<OCRPredictResult> ocr_result =
          this->ocr(img_list[i], true, rec, cls);
      ocr_results.push_back(ocr_result);
    }
  }
  // 返回最终的 OCR 结果
  return ocr_results;
// 返回OCR结果的函数，包括检测、识别和分类
std::vector<OCRPredictResult> PPOCR::ocr(cv::Mat img, bool det, bool rec, bool cls) {

  // 存储OCR结果的向量
  std::vector<OCRPredictResult> ocr_result;
  
  // 检测
  this->det(img, ocr_result);
  
  // 裁剪图像
  std::vector<cv::Mat> img_list;
  for (int j = 0; j < ocr_result.size(); j++) {
    cv::Mat crop_img;
    crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
    img_list.push_back(crop_img);
  }
  
  // 分类
  if (cls && this->classifier_ != nullptr) {
    this->cls(img_list, ocr_result);
    for (int i = 0; i < img_list.size(); i++) {
      if (ocr_result[i].cls_label % 2 == 1 &&
          ocr_result[i].cls_score > this->classifier_->cls_thresh) {
        cv::rotate(img_list[i], img_list[i], 1);
      }
    }
  }
  
  // 识别
  if (rec) {
    this->rec(img_list, ocr_result);
  }
  
  // 返回OCR结果
  return ocr_result;
}

// 检测函数，用于检测文本框
void PPOCR::det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results) {
  std::vector<std::vector<std::vector<int>>> boxes;
  std::vector<double> det_times;

  // 运行检测器，获取文本框和检测时间
  this->detector_->Run(img, boxes, det_times);

  // 将检测结果存入OCR结果向量
  for (int i = 0; i < boxes.size(); i++) {
    OCRPredictResult res;
    res.box = boxes[i];
    ocr_results.push_back(res);
  }
  
  // 对文本框进行排序，从上到下，从左到右
  Utility::sorted_boxes(ocr_results);
  this->time_info_det[0] += det_times[0];
  this->time_info_det[1] += det_times[1];
  this->time_info_det[2] += det_times[2];
}

// 识别函数，用于识别文本
void PPOCR::rec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results) {
  std::vector<std::string> rec_texts(img_list.size(), "");
  std::vector<float> rec_text_scores(img_list.size(), 0);
  std::vector<double> rec_times;
  
  // 运行识别器，获取识别文本和识别时间
  this->recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);
  
  // 输出识别结果
  for (int i = 0; i < rec_texts.size(); i++) {
    ocr_results[i].text = rec_texts[i];
  }
}
    # 将第 i 个 OCR 结果的得分设置为第 i 个识别文本得分
    ocr_results[i].score = rec_text_scores[i];
  }
  # 更新总识别时间信息：第 0 个元素为总识别时间，第 1 个元素为文本检测时间，第 2 个元素为文本识别时间
  this->time_info_rec[0] += rec_times[0];
  this->time_info_rec[1] += rec_times[1];
  this->time_info_rec[2] += rec_times[2];
// 对输入的图像列表进行分类，更新 OCR 结果中的分类标签和得分
void PPOCR::cls(std::vector<cv::Mat> img_list,
                std::vector<OCRPredictResult> &ocr_results) {
  // 初始化分类标签和得分的向量
  std::vector<int> cls_labels(img_list.size(), 0);
  std::vector<float> cls_scores(img_list.size(), 0);
  std::vector<double> cls_times;
  // 运行分类器，获取分类结果和时间信息
  this->classifier_->Run(img_list, cls_labels, cls_scores, cls_times);
  // 输出分类结果到 OCR 结果中
  for (int i = 0; i < cls_labels.size(); i++) {
    ocr_results[i].cls_label = cls_labels[i];
    ocr_results[i].cls_score = cls_scores[i];
  }
  // 更新分类时间信息
  this->time_info_cls[0] += cls_times[0];
  this->time_info_cls[1] += cls_times[1];
  this->time_info_cls[2] += cls_times[2];
}

// 重置计时器
void PPOCR::reset_timer() {
  // 将检测、识别和分类的时间信息重置为零
  this->time_info_det = {0, 0, 0};
  this->time_info_rec = {0, 0, 0};
  this->time_info_cls = {0, 0, 0};
}

// 记录性能日志
void PPOCR::benchmark_log(int img_num) {
  // 如果检测时间、识别时间或分类时间大于零，则记录性能日志
  if (this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2] > 0) {
    AutoLogger autolog_det("ocr_det", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads, 1, "dynamic",
                           FLAGS_precision, this->time_info_det, img_num);
    autolog_det.report();
  }
  if (this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2] > 0) {
    AutoLogger autolog_rec("ocr_rec", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_rec_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_rec, img_num);
    autolog_rec.report();
  }
  if (this->time_info_cls[0] + this->time_info_cls[1] + this->time_info_cls[2] > 0) {
    AutoLogger autolog_cls("ocr_cls", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_cls_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_cls, img_num);
    autolog_cls.report();
  }
}

// 析构函数
PPOCR::~PPOCR() {
  // 如果检测器对象不为空，则释放内存
  if (this->detector_ != nullptr) {
    // 如果存在人脸检测器对象，则删除该对象
    delete this->detector_;
  }
  // 如果存在分类器对象，则删除该对象
  if (this->classifier_ != nullptr) {
    delete this->classifier_;
  }
  // 如果存在人脸识别器对象，则删除该对象
  if (this->recognizer_ != nullptr) {
    delete this->recognizer_;
  }
};

} // namespace PaddleOCR
```