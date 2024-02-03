# `.\PaddleOCR\deploy\cpp_infer\include\paddlestructure.h`

```
// 版权声明，告知代码版权归属于 PaddlePaddle 作者
// 根据 Apache 许可证 2.0 版本使用此文件
// 只有在遵守许可证的情况下才能使用此文件
// 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
// 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的
// 没有任何明示或暗示的担保或条件，包括但不限于特定用途的适用性
// 请查看许可证以获取有关权限和限制的详细信息

#pragma once

#include <include/paddleocr.h>
#include <include/structure_layout.h>
#include <include/structure_table.h>

namespace PaddleOCR {

// PaddleStructure 类继承自 PPOCR 类
class PaddleStructure : public PPOCR {
public:
  // 构造函数
  explicit PaddleStructure();
  // 析构函数
  ~PaddleStructure();

  // 结构化识别函数，返回结构化预测结果
  std::vector<StructurePredictResult> structure(cv::Mat img,
                                                bool layout = false,
                                                bool table = true,
                                                bool ocr = false);

  // 重置计时器
  void reset_timer();
  // 记录基准日志
  void benchmark_log(int img_num);

private:
  // 存储表格识别时间信息
  std::vector<double> time_info_table = {0, 0, 0};
  // 存储布局识别时间信息
  std::vector<double> time_info_layout = {0, 0, 0};

  // 表格识别模型指针
  StructureTableRecognizer *table_model_ = nullptr;
  // 布局识别模型指针
  StructureLayoutRecognizer *layout_model_ = nullptr;

  // 布局识别函数
  void layout(cv::Mat img,
              std::vector<StructurePredictResult> &structure_result);

  // 表格识别函数
  void table(cv::Mat img, StructurePredictResult &structure_result);

  // 重建表格函数
  std::string rebuild_table(std::vector<std::string> rec_html_tags,
                            std::vector<std::vector<int>> rec_boxes,
                            std::vector<OCRPredictResult> &ocr_result);

  // 计算两个框之间的距离
  float dis(std::vector<int> &box1, std::vector<int> &box2);

  // 比较距离的静态函数
  static bool comparison_dis(const std::vector<float> &dis1,
                             const std::vector<float> &dis2) {
    # 如果第一个元组的第二个元素小于第二个元组的第二个元素，则返回True
    if (dis1[1] < dis2[1]) {
      return true;
    # 如果第一个元组的第二个元素等于第二个元组的第二个元素，则比较第一个元组的第一个元素和第二个元组的第一个元素
    } else if (dis1[1] == dis2[1]) {
      return dis1[0] < dis2[0];
    # 如果第一个元组的第二个元素大于第二个元组的第二个元素，则返回False
    } else {
      return false;
    }
  }
};

} // namespace PaddleOCR
```