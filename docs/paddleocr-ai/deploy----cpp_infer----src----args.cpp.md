# `.\PaddleOCR\deploy\cpp_infer\src\args.cpp`

```
// 版权声明，版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证 2.0 版本授权
// 除非符合许可证的规定，否则不得使用此文件
// 可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则软件
// 根据许可证以“原样”分发
// 没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和限制

#include <gflags/gflags.h>

// 常见参数
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");  // 是否使用 GPU 进行推断
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");  // 是否使用 TensorRT
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");  // GPU 设备 ID
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");  // GPU 内存大小
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");  // CPU 线程数
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");  // 是否在 CPU 上使用 MKL-DNN
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");  // 精度为 fp32/fp16/int8 中的一个
DEFINE_bool(benchmark, false, "Whether use benchmark.");  // 是否进行基准测试
DEFINE_string(output, "./output/", "Save benchmark log path.");  // 保存基准测试日志的路径
DEFINE_string(image_dir, "", "Dir of input image.");  // 输入图像的目录
DEFINE_string(
    type, "ocr",
    "Perform ocr or structure, the value is selected in ['ocr','structure'].");  // 执行 OCR 或结构化，值为 ['ocr','structure'] 中的一个
// 检测相关
DEFINE_string(det_model_dir, "", "Path of det inference model.");  // 检测推断模型的路径
DEFINE_string(limit_type, "max", "limit_type of input image.");  // 输入图像的限制类型
DEFINE_int32(limit_side_len, 960, "limit_side_len of input image.");  // 输入图像的限制边长
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");  // det_db_thresh 的阈值
DEFINE_double(det_db_box_thresh, 0.6, "Threshold of det_db_box_thresh.");  // det_db_box_thresh 的阈值
DEFINE_double(det_db_unclip_ratio, 1.5, "Threshold of det_db_unclip_ratio.");  // det_db_unclip_ratio 的阈值
DEFINE_bool(use_dilation, false, "Whether use the dilation on output map.");  // 是否在输出地图上使用膨胀
DEFINE_string(det_db_score_mode, "slow", "Whether use polygon score.");  // 是否使用多边形分数
// 定义一个布尔类型的命令行参数 visualize，用于控制是否显示检测结果
DEFINE_bool(visualize, true, "Whether show the detection results.");

// 分类相关
// 定义一个布尔类型的命令行参数 use_angle_cls，用于控制是否使用角度分类
DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
// 定义一个字符串类型的命令行参数 cls_model_dir，用于指定分类推断模型的路径
DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
// 定义一个双精度浮点数类型的命令行参数 cls_thresh，用于指定分类的阈值
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");
// 定义一个整数类型的命令行参数 cls_batch_num，用于指定分类的批次数
DEFINE_int32(cls_batch_num, 1, "cls_batch_num.");

// 识别相关
// 定义一个字符串类型的命令行参数 rec_model_dir，用于指定识别推断模型的路径
DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
// 定义一个整数类型的命令行参数 rec_batch_num，用于指定识别的批次数
DEFINE_int32(rec_batch_num, 6, "rec_batch_num.");
// 定义一个字符串类型的命令行参数 rec_char_dict_path，用于指定字符字典的路径
DEFINE_string(rec_char_dict_path, "../../ppocr/utils/ppocr_keys_v1.txt",
              "Path of dictionary.");
// 定义一个整数类型的命令行参数 rec_img_h，用于指定识别图像的高度
DEFINE_int32(rec_img_h, 48, "rec image height");
// 定义一个整数类型的命令行参数 rec_img_w，用于指定识别图像的宽度
DEFINE_int32(rec_img_w, 320, "rec image width");

// 布局模型相关
// 定义一个字符串类型的命令行参数 layout_model_dir，用于指定表格布局推断模型的路径
DEFINE_string(layout_model_dir, "", "Path of table layout inference model.");
// 定义一个字符串类型的命令行参数 layout_dict_path，用于指定布局字典的路径
DEFINE_string(layout_dict_path,
              "../../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt",
              "Path of dictionary.");
// 定义一个双精度浮点数类型的命令行参数 layout_score_threshold，用于指定布局的分数阈值
DEFINE_double(layout_score_threshold, 0.5, "Threshold of score.");
// 定义一个双精度浮点数类型的命令行参数 layout_nms_threshold，用于指定布局的非极大值抑制阈值
DEFINE_double(layout_nms_threshold, 0.5, "Threshold of nms.");

// 结构模型相关
// 定义一个字符串类型的命令行参数 table_model_dir，用于指定表格结构推断模型的路径
DEFINE_string(table_model_dir, "", "Path of table struture inference model.");
// 定义一个整数类型的命令行参数 table_max_len，用于指定输入图像的最大长度
DEFINE_int32(table_max_len, 488, "max len size of input image.");
// 定义一个整数类型的命令行参数 table_batch_num，用于指定表格的批次数
DEFINE_int32(table_batch_num, 1, "table_batch_num.");
// 定义一个布尔类型的命令行参数 merge_no_span_structure，用于控制是否合并 <td> 和 </td> 为 <td></td>
DEFINE_bool(merge_no_span_structure, true,
            "Whether merge <td> and </td> to <td></td>");
// 定义一个字符串类型的命令行参数 table_char_dict_path，用于指定表格结构字典的路径
DEFINE_string(table_char_dict_path,
              "../../ppocr/utils/dict/table_structure_dict_ch.txt",
              "Path of dictionary.");

// OCR 前向相关
// 定义一个布尔类型的命令行参数 det，用于控制是否在前向过程中使用检测
DEFINE_bool(det, true, "Whether use det in forward.");
// 定义一个布尔类型的命令行参数 rec，用于控制是否在前向过程中使用识别
DEFINE_bool(rec, true, "Whether use rec in forward.");
// 定义一个布尔类型的命令行参数 cls，用于控制是否在前向过程中使用分类
DEFINE_bool(cls, false, "Whether use cls in forward.");
// 定义一个布尔类型的命令行参数 table，用于控制是否在前向过程中使用表格结构
DEFINE_bool(table, false, "Whether use table structure in forward.");
// 定义一个布尔类型的命令行参数 layout，用于控制是否在前向过程中使用布局分析
DEFINE_bool(layout, false, "Whether use layout analysis in forward.");
```