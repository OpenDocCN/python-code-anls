# `.\PaddleOCR\deploy\cpp_infer\src\main.cpp`

```
// 版权声明和许可信息
// 2020年PaddlePaddle作者保留所有权利。
// 根据Apache许可证2.0版（“许可证”）获得许可;
// 除非符合许可证的规定，否则您不得使用此文件。
// 您可以在以下网址获取许可证的副本：
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则软件
// 根据“原样”分发，不提供任何形式的担保或条件，
// 无论是明示的还是暗示的。
// 请查看许可证以获取特定语言的权限和
// 许可证下的限制。
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>

using namespace PaddleOCR;

// 检查参数设置
void check_params() {
  // 如果进行文本检测
  if (FLAGS_det) {
    // 检查文本检测模型目录和输入图像目录是否为空
    if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty()) {
      // 提示正确的使用方式
      std::cout << "Usage[det]: ./ppocr "
                   "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      // 退出程序
      exit(1);
    }
  }
  // 如果进行文本识别
  if (FLAGS_rec) {
    // 提示使用PP-OCRv3时，默认的rec_image_shape参数为'3, 48, 320'，
    // 如果使用PP-OCRv2或更早版本的识别模型，请设置--rec_image_shape='3,32,320'
    std::cout
        << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
           "if you are using recognition model with PP-OCRv2 or an older "
           "version, "
           "please set --rec_image_shape='3,32,320"
        << std::endl;
    // 检查文本识别模型目录和输入图像目录是否为空
    if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      // 提示正确的使用方式
      std::cout << "Usage[rec]: ./ppocr "
                   "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      // 退出程序
      exit(1);
    }
  }
  // 如果进行文本分类并且使用角度分类
  if (FLAGS_cls && FLAGS_use_angle_cls) {
    // 检查是否缺少分类模型目录或输入图像目录，如果缺少则输出用法信息并退出程序
    if (FLAGS_cls_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[cls]: ./ppocr "
                << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  // 如果需要进行表格识别
  if (FLAGS_table) {
    // 检查是否缺少表格模型目录、检测模型目录、识别模型目录或输入图像目录，如果缺少则输出用法信息并退出程序
    if (FLAGS_table_model_dir.empty() || FLAGS_det_model_dir.empty() ||
        FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[table]: ./ppocr "
                << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  // 如果需要进行布局分析
  if (FLAGS_layout) {
    // 检查是否缺少布局模型目录或输入图像目录，如果缺少则输出用法信息并退出程序
    if (FLAGS_layout_model_dir.empty() || FLAGS_image_dir.empty()) {
      std::cout << "Usage[layout]: ./ppocr "
                << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
      exit(1);
    }
  }
  // 检查精度参数是否为 'fp32'、'fp16' 或 'int8'，如果不是则输出提示信息并退出程序
  if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
      FLAGS_precision != "int8") {
    std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
              << std::endl;
    exit(1);
  }
}

// 对给定的图像列表进行 OCR（光学字符识别）处理
void ocr(std::vector<cv::String> &cv_all_img_names) {
  // 创建 PPOCR 对象
  PPOCR ocr = PPOCR();

  // 如果设置了 benchmark 标志，则重置计时器
  if (FLAGS_benchmark) {
    ocr.reset_timer();
  }

  // 初始化图像列表和图像名称列表
  std::vector<cv::Mat> img_list;
  std::vector<cv::String> img_names;
  // 遍历所有图像名称
  for (int i = 0; i < cv_all_img_names.size(); ++i) {
    // 读取图像
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    // 如果图像数据为空，则输出错误信息并继续下一张图像
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }
    // 将图像和图像名称添加到对应列表中
    img_list.push_back(img);
    img_names.push_back(cv_all_img_names[i]);
  }

  // 对图像列表进行 OCR 处理，获取识别结果
  std::vector<std::vector<OCRPredictResult>> ocr_results =
      ocr.ocr(img_list, FLAGS_det, FLAGS_rec, FLAGS_cls);

  // 遍历图像名称列表
  for (int i = 0; i < img_names.size(); ++i) {
    // 输出预测的图像名称
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    // 打印 OCR 结果
    Utility::print_result(ocr_results[i]);
    // 如果设置了可视化和检测标志，则可视化边界框
    if (FLAGS_visualize && FLAGS_det) {
      std::string file_name = Utility::basename(img_names[i]);
      cv::Mat srcimg = img_list[i];
      Utility::VisualizeBboxes(srcimg, ocr_results[i],
                               FLAGS_output + "/" + file_name);
    }
  }
  // 如果设置了 benchmark 标志，则记录处理时间
  if (FLAGS_benchmark) {
    ocr.benchmark_log(cv_all_img_names.size());
  }
}

// 对给定的图像列表进行结构化处理
void structure(std::vector<cv::String> &cv_all_img_names) {
  // 创建 PaddleStructure 对象
  PaddleOCR::PaddleStructure engine = PaddleOCR::PaddleStructure();

  // 如果设置了 benchmark 标志，则重置计时器
  if (FLAGS_benchmark) {
    engine.reset_timer();
  }

  // 遍历所有图像名称
  for (int i = 0; i < cv_all_img_names.size(); i++) {
    // 输出预测的图像名称
    std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
    // 读取图像
    cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
    // 如果图像数据为空，则输出错误信息并继续下一张图像
    if (!img.data) {
      std::cerr << "[ERROR] image read failed! image path: "
                << cv_all_img_names[i] << std::endl;
      continue;
    }

    // 对图像进行结构化处理，获取结果
    std::vector<StructurePredictResult> structure_results = engine.structure(
        img, FLAGS_layout, FLAGS_table, FLAGS_det && FLAGS_rec);
    // 遍历结构化结果列表
    for (int j = 0; j < structure_results.size(); j++) {
      // 输出索引和类型信息
      std::cout << j << "\ttype: " << structure_results[j].type
                << ", region: [";
      // 输出区域坐标和置信度
      std::cout << structure_results[j].box[0] << ","
                << structure_results[j].box[1] << ","
                << structure_results[j].box[2] << ","
                << structure_results[j].box[3] << "], score: ";
      std::cout << structure_results[j].confidence << ", res: ";

      // 如果类型为表格
      if (structure_results[j].type == "table") {
        // 输出 HTML 内容
        std::cout << structure_results[j].html << std::endl;
        // 如果表格有单元格坐标且需要可视化
        if (structure_results[j].cell_box.size() > 0 && FLAGS_visualize) {
          // 获取文件名
          std::string file_name = Utility::basename(cv_all_img_names[i]);

          // 可视化边界框
          Utility::VisualizeBboxes(img, structure_results[j],
                                   FLAGS_output + "/" + std::to_string(j) +
                                       "_" + file_name);
        }
      } else {
        // 输出 OCR 结果数量
        std::cout << "count of ocr result is : "
                  << structure_results[j].text_res.size() << std::endl;
        // 如果有 OCR 结果
        if (structure_results[j].text_res.size() > 0) {
          // 输出开始打印 OCR 结果
          std::cout << "********** print ocr result "
                    << "**********" << std::endl;
          // 打印 OCR 结果
          Utility::print_result(structure_results[j].text_res);
          // 输出结束打印 OCR 结果
          std::cout << "********** end print ocr result "
                    << "**********" << std::endl;
        }
      }
    }
  }
  // 如果需要进行基准测试
  if (FLAGS_benchmark) {
    // 记录基准测试日志
    engine.benchmark_log(cv_all_img_names.size());
  }
}

int main(int argc, char **argv) {
  // 解析命令行参数
  google::ParseCommandLineFlags(&argc, &argv, true);
  // 检查参数是否有效
  check_params();

  // 检查指定的图像路径是否存在
  if (!Utility::PathExists(FLAGS_image_dir)) {
    std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
              << std::endl;
    // 退出程序并返回错误码1
    exit(1);
  }

  // 获取指定目录下所有图像文件名
  std::vector<cv::String> cv_all_img_names;
  cv::glob(FLAGS_image_dir, cv_all_img_names);
  // 打印总图像数量
  std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

  // 如果指定的输出路径不存在，则创建
  if (!Utility::PathExists(FLAGS_output)) {
    Utility::CreateDir(FLAGS_output);
  }
  // 根据不同的类型参数执行不同的操作
  if (FLAGS_type == "ocr") {
    ocr(cv_all_img_names);
  } else if (FLAGS_type == "structure") {
    structure(cv_all_img_names);
  } else {
    // 打印提示信息，说明只支持 'ocr' 和 'structure' 两种类型
    std::cout << "only value in ['ocr','structure'] is supported" << std::endl;
  }
}
```