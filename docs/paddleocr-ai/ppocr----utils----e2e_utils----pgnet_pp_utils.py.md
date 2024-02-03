# `.\PaddleOCR\ppocr\utils\e2e_utils\pgnet_pp_utils.py`

```
# 版权声明和许可信息
# 从未来导入绝对路径
# 导入 paddle 库
# 导入操作系统库
# 导入系统库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..'))
# 导入 extract_textpoint_slow 模块
from extract_textpoint_slow import *
# 导入 extract_textpoint_fast 模块中的 generate_pivot_list_fast 和 restore_poly 函数
from extract_textpoint_fast import generate_pivot_list_fast, restore_poly

# 定义 PGNet_PostProcess 类
class PGNet_PostProcess(object):
    # 初始化方法，接收参数包括字符字典路径、有效集合、分数阈值、输出字典、形状列表和点聚合模式
    def __init__(self,
                 character_dict_path,
                 valid_set,
                 score_thresh,
                 outs_dict,
                 shape_list,
                 point_gather_mode=None):
        # 获取字符字典
        self.Lexicon_Table = get_dict(character_dict_path)
        # 设置有效集合
        self.valid_set = valid_set
        # 设置分数阈值
        self.score_thresh = score_thresh
        # 设置输出字典
        self.outs_dict = outs_dict
        # 设置形状列表
        self.shape_list = shape_list
        # 设置点聚合模式
        self.point_gather_mode = point_gather_mode
    # 快速后处理函数，用于处理模型输出结果
    def pg_postprocess_fast(self):
        # 获取模型输出中的得分、边界、字符、方向信息
        p_score = self.outs_dict['f_score']
        p_border = self.outs_dict['f_border']
        p_char = self.outs_dict['f_char']
        p_direction = self.outs_dict['f_direction']
        
        # 如果得分是 paddle.Tensor 类型，则转换为 numpy 数组
        if isinstance(p_score, paddle.Tensor):
            p_score = p_score[0].numpy()
            p_border = p_border[0].numpy()
            p_direction = p_direction[0].numpy()
            p_char = p_char[0].numpy()
        else:
            p_score = p_score[0]
            p_border = p_border[0]
            p_direction = p_direction[0]
            p_char = p_char[0]

        # 获取输入图像的高度、宽度以及缩放比例
        src_h, src_w, ratio_h, ratio_w = self.shape_list[0]
        
        # 生成基准点列表和序列字符串
        instance_yxs_list, seq_strs = generate_pivot_list_fast(
            p_score,
            p_char,
            p_direction,
            self.Lexicon_Table,
            score_thresh=self.score_thresh,
            point_gather_mode=self.point_gather_mode)
        
        # 恢复多边形列表和保留字符串列表
        poly_list, keep_str_list = restore_poly(instance_yxs_list, seq_strs,
                                                p_border, ratio_w, ratio_h,
                                                src_w, src_h, self.valid_set)
        
        # 构建返回数据字典，包含多边形列表和保留字符串列表
        data = {
            'points': poly_list,
            'texts': keep_str_list,
        }
        
        # 返回数据字典
        return data
```