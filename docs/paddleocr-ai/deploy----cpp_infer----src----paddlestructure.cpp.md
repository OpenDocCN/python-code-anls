# `.\PaddleOCR\deploy\cpp_infer\src\paddlestructure.cpp`

```
// 版权声明，告知代码版权归属及使用许可
// 引入必要的头文件和命名空间
#include <include/args.h>
#include <include/paddlestructure.h>

#include "auto_log/autolog.h"

// 命名空间声明
namespace PaddleOCR {

// PaddleStructure 类的构造函数
PaddleStructure::PaddleStructure() {
  // 根据 FLAGS_layout 标志判断是否需要创建 StructureLayoutRecognizer 对象
  if (FLAGS_layout) {
    // 创建 StructureLayoutRecognizer 对象
    this->layout_model_ = new StructureLayoutRecognizer(
        FLAGS_layout_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_layout_dict_path,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_layout_score_threshold,
        FLAGS_layout_nms_threshold);
  }
  // 根据 FLAGS_table 标志判断是否需要创建 StructureTableRecognizer 对象
  if (FLAGS_table) {
    // 创建 StructureTableRecognizer 对象
    this->table_model_ = new StructureTableRecognizer(
        FLAGS_table_model_dir, FLAGS_use_gpu, FLAGS_gpu_id, FLAGS_gpu_mem,
        FLAGS_cpu_threads, FLAGS_enable_mkldnn, FLAGS_table_char_dict_path,
        FLAGS_use_tensorrt, FLAGS_precision, FLAGS_table_batch_num,
        FLAGS_table_max_len, FLAGS_merge_no_span_structure);
  }
};

// PaddleStructure 类的结构识别方法
std::vector<StructurePredictResult>
PaddleStructure::structure(cv::Mat srcimg, bool layout, bool table, bool ocr) {
  // 复制输入图像
  cv::Mat img;
  srcimg.copyTo(img);

  // 存储结构预测结果的向量
  std::vector<StructurePredictResult> structure_results;

  // 根据 layout 标志判断是否进行布局识别
  if (layout) {
    // 调用 layout 方法进行布局识别
    this->layout(img, structure_results);
  } else {
    // 创建一个结构预测结果对象
    StructurePredictResult res;
    // 设置结果类型为表格
    res.type = "table";
    // 初始化结果框的坐标
    res.box = std::vector<float>(4, 0.0);
    res.box[2] = img.cols;
    res.box[3] = img.rows;
    // 将结果添加到结构化结果向量中
    structure_results.push_back(res);
  }
  // 定义感兴趣区域图像
  cv::Mat roi_img;
  // 遍历结构化结果向量
  for (int i = 0; i < structure_results.size(); i++) {
    // 裁剪图像
    roi_img = Utility::crop_image(img, structure_results[i].box);
    // 如果是表格并且需要处理表格
    if (structure_results[i].type == "table" && table) {
      // 处理表格
      this->table(roi_img, structure_results[i]);
    } 
    // 如果需要进行 OCR
    else if (ocr) {
      // 对感兴趣区域图像进行 OCR
      structure_results[i].text_res = this->ocr(roi_img, true, true, false);
    }
  }

  // 返回处理后的结构化结果向量
  return structure_results;
};

// 定义 PaddleStructure 类的 layout 方法，用于处理布局识别
void PaddleStructure::layout(
    cv::Mat img, std::vector<StructurePredictResult> &structure_result) {
  // 存储布局识别的时间信息
  std::vector<double> layout_times;
  // 运行布局识别模型，获取结果和时间信息
  this->layout_model_->Run(img, structure_result, layout_times);

  // 更新布局识别时间信息
  this->time_info_layout[0] += layout_times[0];
  this->time_info_layout[1] += layout_times[1];
  this->time_info_layout[2] += layout_times[2];
}

// 定义 PaddleStructure 类的 table 方法，用于处理表格识别
void PaddleStructure::table(cv::Mat img,
                            StructurePredictResult &structure_result) {
  // 预测结构
  std::vector<std::vector<std::string>> structure_html_tags;
  std::vector<float> structure_scores(1, 0);
  std::vector<std::vector<std::vector<int>>> structure_boxes;
  std::vector<double> structure_times;
  std::vector<cv::Mat> img_list;
  img_list.push_back(img);

  // 运行表格识别模型，获取结果和时间信息
  this->table_model_->Run(img_list, structure_html_tags, structure_scores,
                          structure_boxes, structure_times);

  // 更新表格识别时间信息
  this->time_info_table[0] += structure_times[0];
  this->time_info_table[1] += structure_times[1];
  this->time_info_table[2] += structure_times[2];

  // 存储 OCR 识别结果
  std::vector<OCRPredictResult> ocr_result;
  std::string html;
  int expand_pixel = 3;

  // 遍历图像列表
  for (int i = 0; i < img_list.size(); i++) {
    // 检测文本区域
    this->det(img_list[i], ocr_result);
    // 裁剪图像
    std::vector<cv::Mat> rec_img_list;
    std::vector<int> ocr_box;
    for (int j = 0; j < ocr_result.size(); j++) {
      ocr_box = Utility::xyxyxyxy2xyxy(ocr_result[j].box);
      ocr_box[0] = std::max(0, ocr_box[0] - expand_pixel);
      ocr_box[1] = std::max(0, ocr_box[1] - expand_pixel),
      ocr_box[2] = std::min(img_list[i].cols, ocr_box[2] + expand_pixel);
      ocr_box[3] = std::min(img_list[i].rows, ocr_box[3] + expand_pixel);

      cv::Mat crop_img = Utility::crop_image(img_list[i], ocr_box);
      rec_img_list.push_back(crop_img);
    }
    // 文本识别
    this->rec(rec_img_list, ocr_result);
    // 重建表格
    # 使用 rebuild_table 函数重建表格，传入结构化 HTML 标签、结构框和 OCR 结果作为参数
    html = this->rebuild_table(structure_html_tags[i], structure_boxes[i], ocr_result);
    # 将重建的 HTML 存储到结构化结果中
    structure_result.html = html;
    # 将结构框存储到结构化结果中
    structure_result.cell_box = structure_boxes[i];
    # 将 HTML 得分存储到结构化结果中
    structure_result.html_score = structure_scores[i];
};

std::string
PaddleStructure::rebuild_table(std::vector<std::string> structure_html_tags,
                               std::vector<std::vector<int>> structure_boxes,
                               std::vector<OCRPredictResult> &ocr_result) {
  // 匹配同一单元格中的文本
  std::vector<std::vector<std::string>> matched(structure_boxes.size(),
                                                std::vector<std::string>());

  std::vector<int> ocr_box;
  std::vector<int> structure_box;
  for (int i = 0; i < ocr_result.size(); i++) {
    // 将 OCR 结果的边界框转换为 xyxy 格式
    ocr_box = Utility::xyxyxyxy2xyxy(ocr_result[i].box);
    ocr_box[0] -= 1;
    ocr_box[1] -= 1;
    ocr_box[2] += 1;
    ocr_box[3] += 1;
    // 初始化距离列表
    std::vector<std::vector<float>> dis_list(structure_boxes.size(),
                                             std::vector<float>(3, 100000.0));
    for (int j = 0; j < structure_boxes.size(); j++) {
      if (structure_boxes[j].size() == 8) {
        // 将结构框的边界框转换为 xyxy 格式
        structure_box = Utility::xyxyxyxy2xyxy(structure_boxes[j]);
      } else {
        structure_box = structure_boxes[j];
      }
      // 计算 OCR 边界框与结构框之间的距离
      dis_list[j][0] = this->dis(ocr_box, structure_box);
      // 计算 OCR 边界框与结构框之间的 IoU
      dis_list[j][1] = 1 - Utility::iou(ocr_box, structure_box);
      dis_list[j][2] = j;
    }
    // 找到最小距离的索引
    std::sort(dis_list.begin(), dis_list.end(),
              PaddleStructure::comparison_dis);
    matched[dis_list[0][2]].push_back(ocr_result[i].text);
  }

  // 获取预测的 HTML
  std::string html_str = "";
  int td_tag_idx = 0;
  for (int i = 0; i < structure_html_tags.size(); i++) {
    // 如果当前结构标签中包含 "</td>"，则执行以下操作
    if (structure_html_tags[i].find("</td>") != std::string::npos) {
      // 如果当前结构标签中包含 "<td></td>"，则执行以下操作
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        // 在 html 字符串中添加 "<td>"
        html_str += "<td>";
      }
      // 如果匹配到的 td 标签数量大于 0，则执行以下操作
      if (matched[td_tag_idx].size() > 0) {
        // 初始化一个布尔变量 b_with 为 false
        bool b_with = false;
        // 如果匹配到的 td 标签中第一个包含 "<b>"，且匹配到的 td 标签数量大于 1，则执行以下操作
        if (matched[td_tag_idx][0].find("<b>") != std::string::npos &&
            matched[td_tag_idx].size() > 1) {
          // 将 b_with 设置为 true，并在 html 字符串中添加 "<b>"
          b_with = true;
          html_str += "<b>";
        }
        // 遍历匹配到的 td 标签
        for (int j = 0; j < matched[td_tag_idx].size(); j++) {
          // 获取当前 td 标签的内容
          std::string content = matched[td_tag_idx][j];
          // 如果匹配到的 td 标签数量大于 1，则执行以下操作
          if (matched[td_tag_idx].size() > 1) {
            // 去除空格，<b> 和 </b>
            if (content.length() > 0 && content.at(0) == ' ') {
              content = content.substr(0);
            }
            if (content.length() > 2 && content.substr(0, 3) == "<b>") {
              content = content.substr(3);
            }
            if (content.length() > 4 &&
                content.substr(content.length() - 4) == "</b>") {
              content = content.substr(0, content.length() - 4);
            }
            if (content.empty()) {
              continue;
            }
            // 添加空格
            if (j != matched[td_tag_idx].size() - 1 &&
                content.at(content.length() - 1) != ' ') {
              content += ' ';
            }
          }
          // 在 html 字符串中添加当前 td 标签的内容
          html_str += content;
        }
        // 如果 b_with 为 true，则在 html 字符串中添加 "</b>"
        if (b_with) {
          html_str += "</b>";
        }
      }
      // 如果当前结构标签中包含 "<td></td>"，则在 html 字符串中添加 "</td>"，否则添加当前结构标签
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "</td>";
      } else {
        html_str += structure_html_tags[i];
      }
      // td 标签索引加一
      td_tag_idx += 1;
    } else {
      // 如果当前结构标签不包含 "</td>"，则直接添加当前结构标签到 html 字符串
      html_str += structure_html_tags[i];
    }
  }
  // 返回最终的 html 字符串
  return html_str;
// 计算两个矩形框之间的距离
float PaddleStructure::dis(std::vector<int> &box1, std::vector<int> &box2) {
  // 提取第一个矩形框的坐标信息
  int x1_1 = box1[0];
  int y1_1 = box1[1];
  int x2_1 = box1[2];
  int y2_1 = box1[3];

  // 提取第二个矩形框的坐标信息
  int x1_2 = box2[0];
  int y1_2 = box2[1];
  int x2_2 = box2[2];
  int y2_2 = box2[3];

  // 计算两个矩形框之间的距离
  float dis =
      abs(x1_2 - x1_1) + abs(y1_2 - y1_1) + abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  float dis_2 = abs(x1_2 - x1_1) + abs(y1_2 - y1_1);
  float dis_3 = abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  // 返回距离并加上最小的两个距离
  return dis + std::min(dis_2, dis_3);
}

// 重置计时器
void PaddleStructure::reset_timer() {
  // 重置检测、识别、分类、表格、布局的时间信息
  this->time_info_det = {0, 0, 0};
  this->time_info_rec = {0, 0, 0};
  this->time_info_cls = {0, 0, 0};
  this->time_info_table = {0, 0, 0};
  this->time_info_layout = {0, 0, 0};
}

// 记录性能日志
void PaddleStructure::benchmark_log(int img_num) {
  // 如果检测时间信息大于0，则记录检测性能日志
  if (this->time_info_det[0] + this->time_info_det[1] + this->time_info_det[2] >
      0) {
    AutoLogger autolog_det("ocr_det", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads, 1, "dynamic",
                           FLAGS_precision, this->time_info_det, img_num);
    autolog_det.report();
  }
  // 如果识别时间信息大于0，则记录识别性能日志
  if (this->time_info_rec[0] + this->time_info_rec[1] + this->time_info_rec[2] >
      0) {
    AutoLogger autolog_rec("ocr_rec", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_rec_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_rec, img_num);
    autolog_rec.report();
  }
  // 如果分类时间信息大于0，则记录分类性能日志
  if (this->time_info_cls[0] + this->time_info_cls[1] + this->time_info_cls[2] >
      0) {
    AutoLogger autolog_cls("ocr_cls", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_cls_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_cls, img_num);
  // 调用report方法，用于输出日志信息
  autolog_cls.report();
}
// 如果时间信息表中的时间总和大于0
if (this->time_info_table[0] + this->time_info_table[1] +
        this->time_info_table[2] >
    0) {
  // 创建AutoLogger对象autolog_table，用于记录表格相关的日志信息
  AutoLogger autolog_table("table", FLAGS_use_gpu, FLAGS_use_tensorrt,
                           FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                           FLAGS_cls_batch_num, "dynamic", FLAGS_precision,
                           this->time_info_table, img_num);
  // 调用report方法，用于输出表格相关的日志信息
  autolog_table.report();
}
// 如果布局信息表中的时间总和大于0
if (this->time_info_layout[0] + this->time_info_layout[1] +
        this->time_info_layout[2] >
    0) {
  // 创建AutoLogger对象autolog_layout，用于记录布局相关的日志信息
  AutoLogger autolog_layout("layout", FLAGS_use_gpu, FLAGS_use_tensorrt,
                            FLAGS_enable_mkldnn, FLAGS_cpu_threads,
                            FLAGS_cls_batch_num, "dynamic", FLAGS_precision,
                            this->time_info_layout, img_num);
  // 调用report方法，用于输出布局相关的日志信息
  autolog_layout.report();
}
}

// PaddleStructure 类的析构函数
PaddleStructure::~PaddleStructure() {
  // 检查 table_model_ 是否为空指针
  if (this->table_model_ != nullptr) {
    // 如果不为空，释放内存
    delete this->table_model_;
  }
};

// 结束 PaddleOCR 命名空间
} // namespace PaddleOCR
```