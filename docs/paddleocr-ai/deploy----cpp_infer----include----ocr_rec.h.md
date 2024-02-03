# `.\PaddleOCR\deploy\cpp_infer\include\ocr_rec.h`

```py
// 版权声明，告知代码版权归属
// 根据 Apache 许可证 2.0 版本规定使用该文件
// 获取许可证的链接
// 根据适用法律或书面同意，根据许可证分发的软件基于“原样”分发，没有任何明示或暗示的保证或条件
// 查看许可证以获取特定语言的权限和限制

// 包含必要的头文件
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <include/ocr_cls.h>
#include <include/utility.h>

// 命名空间定义
namespace PaddleOCR {

// CRNNRecognizer 类的构造函数，初始化各个参数
class CRNNRecognizer {
public:
  explicit CRNNRecognizer(const std::string &model_dir, const bool &use_gpu,
                          const int &gpu_id, const int &gpu_mem,
                          const int &cpu_math_library_num_threads,
                          const bool &use_mkldnn, const std::string &label_path,
                          const bool &use_tensorrt,
                          const std::string &precision,
                          const int &rec_batch_num, const int &rec_img_h,
                          const int &rec_img_w) {
    // 初始化 GPU 相关参数
    this->use_gpu_ = use_gpu;
    this->gpu_id_ = gpu_id;
    this->gpu_mem_ = gpu_mem;
    // 初始化 CPU 线程数
    this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
    // 初始化是否使用 MKL-DNN
    this->use_mkldnn_ = use_mkldnn;
    // 初始化是否使用 TensorRT
    this->use_tensorrt_ = use_tensorrt;
    // 初始化精度
    this->precision_ = precision;
    // 初始化识别批次数
    this->rec_batch_num_ = rec_batch_num;
    // 初始化识别图像高度
    this->rec_img_h_ = rec_img_h;
    // 初始化识别图像宽度
    this->rec_img_w_ = rec_img_w;
    // 初始化识别图像形状
    std::vector<int> rec_image_shape = {3, rec_img_h, rec_img_w};
    this->rec_image_shape_ = rec_image_shape;

    // 读取标签文件内容
    this->label_list_ = Utility::ReadDict(label_path);
    // 在标签列表中插入空白字符，用于 CTC
    this->label_list_.insert(this->label_list_.begin(), "#");
    // 将一个空字符串添加到标签列表中
    this->label_list_.push_back(" ");

    // 调用LoadModel函数加载Paddle推理模型
    LoadModel(model_dir);
  }

  // 加载Paddle推理模型
  void LoadModel(const std::string &model_dir);

  // 运行推理模型，对输入的图像列表进行推理，返回识别文本、文本得分和推理时间
  void Run(std::vector<cv::Mat> img_list, std::vector<std::string> &rec_texts,
           std::vector<float> &rec_text_scores, std::vector<double> &times);
// 私有成员变量，用于存储预测器指针
private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  // 是否使用 GPU，默认为 false
  bool use_gpu_ = false;
  // GPU 设备 ID，默认为 0
  int gpu_id_ = 0;
  // GPU 内存大小，默认为 4000
  int gpu_mem_ = 4000;
  // CPU 数学库线程数，默认为 4
  int cpu_math_library_num_threads_ = 4;
  // 是否使用 MKL-DNN，默认为 false
  bool use_mkldnn_ = false;

  // 标签列表
  std::vector<std::string> label_list_;

  // 均值，用于数据预处理，默认为 {0.5, 0.5, 0.5}
  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  // 缩放比例，用于数据预处理，默认为 {2.0, 2.0, 2.0}
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  // 是否进行缩放，默认为 true
  bool is_scale_ = true;
  // 是否使用 TensorRT，默认为 false
  bool use_tensorrt_ = false;
  // 精度设置，默认为 "fp32"
  std::string precision_ = "fp32";
  // 推理批次数，默认为 6
  int rec_batch_num_ = 6;
  // 图像高度，默认为 32
  int rec_img_h_ = 32;
  // 图像宽度，默认为 320
  int rec_img_w_ = 320;
  // 图像形状，包含通道数、高度和宽度，默认为 {3, 32, 320}
  std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
  
  // 预处理操作对象
  // 图像大小调整操作
  CrnnResizeImg resize_op_;
  // 归一化操作
  Normalize normalize_op_;
  // 批次维度置换操作
  PermuteBatch permute_op_;

}; // class CrnnRecognizer

} // namespace PaddleOCR
```