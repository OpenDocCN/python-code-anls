# `.\PaddleOCR\deploy\cpp_infer\include\paddleocr.h`

```py
// 声明代码版权信息和许可证信息
// 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
// 根据 Apache 许可证 2.0 版本许可；
// 除非符合许可证的规定，否则不得使用此文件。
// 您可以在以下网址获取许可证的副本：
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则根据许可证分发的软件
// 均基于“按原样”分发，不附带任何明示或暗示的担保或条件。
// 请查看许可证以获取特定语言的权限和限制。

// 包含 OCR 分类、检测和识别的头文件
#include <include/ocr_cls.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>

// 命名空间 PaddleOCR
namespace PaddleOCR {

// PPOCR 类声明
class PPOCR {
public:
  // 构造函数
  explicit PPOCR();
  // 析构函数
  ~PPOCR();

  // OCR 方法，接受图像列表和三个布尔值参数
  std::vector<std::vector<OCRPredictResult>> ocr(std::vector<cv::Mat> img_list,
                                                 bool det = true,
                                                 bool rec = true,
                                                 bool cls = true);
  // OCR 方法，接受单个图像和三个布尔值参数
  std::vector<OCRPredictResult> ocr(cv::Mat img, bool det = true,
                                    bool rec = true, bool cls = true);

  // 重置计时器
  void reset_timer();
  // 记录基准日志，接受图像数量参数
  void benchmark_log(int img_num);

protected:
  // 检测时间信息数组
  std::vector<double> time_info_det = {0, 0, 0};
  // 识别时间信息数组
  std::vector<double> time_info_rec = {0, 0, 0};
  // 分类时间信息数组
  std::vector<double> time_info_cls = {0, 0, 0};

  // 检测方法，接受图像和 OCR 结果数组参数
  void det(cv::Mat img, std::vector<OCRPredictResult> &ocr_results);
  // 识别方法，接受图像列表和 OCR 结果数组参数
  void rec(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results);
  // 分类方法，接受图像列表和 OCR 结果数组参数
  void cls(std::vector<cv::Mat> img_list,
           std::vector<OCRPredictResult> &ocr_results);

private:
  // 检测器指针
  DBDetector *detector_ = nullptr;
  // 分类器指针
  Classifier *classifier_ = nullptr;
  // CRNN 识别器指针
  CRNNRecognizer *recognizer_ = nullptr;
};

} // 命名空间 PaddleOCR
```