# `.\PaddleOCR\deploy\cpp_infer\include\args.h`

```
// 版权声明，版权归 PaddlePaddle 作者所有
//
// 根据 Apache 许可证，版本 2.0 进行许可;
// 除非符合许可证的规定，否则不得使用此文件
// 您可以在以下网址获取许可证的副本
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// 除非适用法律要求或书面同意，否则软件
// 根据许可证“按原样”分发，
// 没有任何明示或暗示的保证或条件
// 请查看许可证以获取特定语言的权限和
// 许可证下的限制

#pragma once

#include <gflags/gflags.h>

// 声明通用参数
DECLARE_bool(use_gpu);
DECLARE_bool(use_tensorrt);
DECLARE_int32(gpu_id);
DECLARE_int32(gpu_mem);
DECLARE_int32(cpu_threads);
DECLARE_bool(enable_mkldnn);
DECLARE_string(precision);
DECLARE_bool(benchmark);
DECLARE_string(output);
DECLARE_string(image_dir);
DECLARE_string(type);
// 检测相关
DECLARE_string(det_model_dir);
DECLARE_string(limit_type);
DECLARE_int32(limit_side_len);
DECLARE_double(det_db_thresh);
DECLARE_double(det_db_box_thresh);
DECLARE_double(det_db_unclip_ratio);
DECLARE_bool(use_dilation);
DECLARE_string(det_db_score_mode);
DECLARE_bool(visualize);
// 分类相关
DECLARE_bool(use_angle_cls);
DECLARE_string(cls_model_dir);
DECLARE_double(cls_thresh);
DECLARE_int32(cls_batch_num;
// 识别相关
DECLARE_string(rec_model_dir);
DECLARE_int32(rec_batch_num);
DECLARE_string(rec_char_dict_path);
DECLARE_int32(rec_img_h);
DECLARE_int32(rec_img_w;
// 布局模型相关
DECLARE_string(layout_model_dir);
DECLARE_string(layout_dict_path);
DECLARE_double(layout_score_threshold);
DECLARE_double(layout_nms_threshold);
// 结构模型相关
DECLARE_string(table_model_dir);
DECLARE_int32(table_max_len);
DECLARE_int32(table_batch_num);
DECLARE_string(table_char_dict_path);
DECLARE_bool(merge_no_span_structure);
// 前向相关
DECLARE_bool(det);
# 声明一个名为rec的布尔类型变量
DECLARE_bool(rec);
# 声明一个名为cls的布尔类型变量
DECLARE_bool(cls);
# 声明一个名为table的布尔类型变量
DECLARE_bool(table);
# 声明一个名为layout的布尔类型变量
DECLARE_bool(layout);
```