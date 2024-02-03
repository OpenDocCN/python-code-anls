# `.\PaddleOCR\deploy\cpp_infer\include\postprocess_op.h`

```py
// 版权声明，告知代码版权归属于 PaddlePaddle Authors
// 根据 Apache 许可证 2.0 版本授权，限制使用条件和责任限制
// 获取许可证的链接
// 根据适用法律或书面同意，分发的软件基于“原样”基础分发
// 没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制

#pragma once

// 包含头文件 clipper.h 和 utility.h
#include "include/clipper.h"
#include "include/utility.h"

// 命名空间 PaddleOCR
namespace PaddleOCR {

// DBPostProcessor 类定义
class DBPostProcessor {
public:
  // 计算轮廓面积
  void GetContourArea(const std::vector<std::vector<float>> &box,
                      float unclip_ratio, float &distance);

  // 解除边界框的裁剪
  cv::RotatedRect UnClip(std::vector<std::vector<float>> box,
                         const float &unclip_ratio);

  // 将矩阵转换为向量
  float **Mat2Vec(cv::Mat mat);

  // 顺时针排序点
  std::vector<std::vector<int>>
  OrderPointsClockwise(std::vector<std::vector<int>> pts);

  // 获取最小边界框
  std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box,
                                               float &ssid);

  // 快速计算边界框得分
  float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);

  // 多边形得分累积
  float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

  // 从位图获取边界框
  std::vector<std::vector<std::vector<int>>>
  BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                  const float &box_thresh, const float &det_db_unclip_ratio,
                  const std::string &det_db_score_mode);

  // 过滤标签检测结果
  std::vector<std::vector<std::vector<int>>>
  FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes,
                  float ratio_h, float ratio_w, cv::Mat srcimg);
// 定义私有成员函数，用于比较两个整数向量的大小
static bool XsortInt(std::vector<int> a, std::vector<int> b);

// 定义私有成员函数，用于比较两个浮点数向量的大小
static bool XsortFp32(std::vector<float> a, std::vector<float> b);

// 将 OpenCV 的 Mat 对象转换为二维浮点数向量
std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

// 返回两个整数中的最大值
inline int _max(int a, int b) { return a >= b ? a : b; }

// 返回两个整数中的最小值
inline int _min(int a, int b) { return a >= b ? b : a; }

// 对输入值进行范围限制，确保在指定范围内
template <class T> inline T clamp(T x, T min, T max) {
  if (x > max)
    return max;
  if (x < min)
    return min;
  return x;
}

// 对输入值进行范围限制，确保在指定范围内
inline float clampf(float x, float min, float max) {
  if (x > max)
    return max;
  if (x < min)
    return min;
  return x;
}

// 初始化 TablePostProcessor 类的成员变量
void init(std::string label_path, bool merge_no_span_structure = true);

// 对输入数据进行处理，生成结果
void Run(std::vector<float> &loc_preds, std::vector<float> &structure_probs,
         std::vector<float> &rec_scores, std::vector<int> &loc_preds_shape,
         std::vector<int> &structure_probs_shape,
         std::vector<std::vector<std::string>> &rec_html_tag_batch,
         std::vector<std::vector<std::vector<int>>> &rec_boxes_batch,
         std::vector<int> &width_list, std::vector<int> &height_list);

// 初始化 PicodetPostProcessor 类的成员变量
void init(std::string label_path, const double score_threshold = 0.4,
          const double nms_threshold = 0.5,
          const std::vector<int> &fpn_stride = {8, 16, 32, 64});

// 对输入数据进行处理，生成结果
void Run(std::vector<StructurePredictResult> &results,
         std::vector<std::vector<float>> outs, std::vector<int> ori_shape,
         std::vector<int> resize_shape, int eg_max);

// 存储用于处理的特定步长值
std::vector<int> fpn_stride_ = {8, 16, 32, 64};
# 定义私有成员函数，将检测结果转换为边界框
StructurePredictResult disPred2Bbox(std::vector<float> bbox_pred, int label,
                                    float score, int x, int y, int stride,
                                    std::vector<int> im_shape, int reg_max);

# 定义非极大值抑制函数，用于去除重叠边界框
void nms(std::vector<StructurePredictResult> &input_boxes,
         float nms_threshold);

# 存储标签列表的字符串向量
std::vector<std::string> label_list_;

# 设置置信度阈值为0.4
double score_threshold_ = 0.4;

# 设置非极大值抑制阈值为0.5
double nms_threshold_ = 0.5;

# 设置类别数量为5
int num_class_ = 5;
};

} // namespace PaddleOCR
```