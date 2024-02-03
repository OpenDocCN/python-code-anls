# `.\PaddleOCR\ppocr\modeling\heads\proposal_local_graph.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
# 代码来源于：
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/modules/proposal_local_graph.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入必要的库
import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from lanms import merge_quadrangle_n9 as la_nms

# 导入自定义的 RoIAlignRotated 操作
from ppocr.ext_op import RoIAlignRotated
# 导入本地图模块
from .local_graph import (euclidean_distance_matrix, feature_embedding,
                          normalize_adjacent_matrix)

# 定义填充孔洞的函数
def fill_hole(input_mask):
    # 获取输入 mask 的高度和宽度
    h, w = input_mask.shape
    # 创建一个比输入 mask 大 2 像素的画布
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    # 将输入 mask 复制到画布中间
    canvas[1:h + 1, 1:w + 1] = input_mask.copy()

    # 创建一个比输入 mask 大 4 像素的 mask
    mask = np.zeros((h + 4, w + 4), np.uint8)

    # 使用洪泛填充算法填充孔洞
    cv2.floodFill(canvas, mask, (0, 0), 1)
    # 将填充后的画布还原为布尔类型的 mask
    canvas = canvas[1:h + 1, 1:w + 1].astype(np.bool_)

    # 返回填充后的 mask
    return ~canvas | input_mask

# 定义提议局部图类
class ProposalLocalGraphs:
    # 初始化函数，接受多个参数用于设置模型的各项参数
    def __init__(self, k_at_hops, num_adjacent_linkages, node_geo_feat_len,
                     pooling_scale, pooling_output_size, nms_thr, min_width,
                     max_width, comp_shrink_ratio, comp_w_h_ratio, comp_score_thr,
                     text_region_thr, center_region_thr, center_region_area_thr):
    
            # 确保 k_at_hops 参数长度为2
            assert len(k_at_hops) == 2
            # 确保 k_at_hops 参数为元组类型
            assert isinstance(k_at_hops, tuple)
            # 确保 num_adjacent_linkages 参数为整数类型
            assert isinstance(num_adjacent_linkages, int)
            # 确保 node_geo_feat_len 参数为整数类型
            assert isinstance(node_geo_feat_len, int)
            # 确保 pooling_scale 参数为浮点数类型
            assert isinstance(pooling_scale, float)
            # 确保 pooling_output_size 参数为元组类型
            assert isinstance(pooling_output_size, tuple)
            # 确保 nms_thr 参数为浮点数类型
            assert isinstance(nms_thr, float)
            # 确保 min_width 参数为浮点数类型
            assert isinstance(min_width, float)
            # 确保 max_width 参数为浮点数类型
            assert isinstance(max_width, float)
            # 确保 comp_shrink_ratio 参数为浮点数类型
            assert isinstance(comp_shrink_ratio, float)
            # 确保 comp_w_h_ratio 参数为浮点数类型
            assert isinstance(comp_w_h_ratio, float)
            # 确保 comp_score_thr 参数为浮点数类型
            assert isinstance(comp_score_thr, float)
            # 确保 text_region_thr 参数为浮点数类型
            assert isinstance(text_region_thr, float)
            # 确保 center_region_thr 参数为浮点数类型
            assert isinstance(center_region_thr, float)
            # 确保 center_region_area_thr 参数为整数类型
            assert isinstance(center_region_area_thr, int)
    
            # 将参数赋值给对象的属性
            self.k_at_hops = k_at_hops
            self.active_connection = num_adjacent_linkages
            self.local_graph_depth = len(self.k_at_hops)
            self.node_geo_feat_dim = node_geo_feat_len
            self.pooling = RoIAlignRotated(pooling_output_size, pooling_scale)
            self.nms_thr = nms_thr
            self.min_width = min_width
            self.max_width = max_width
            self.comp_shrink_ratio = comp_shrink_ratio
            self.comp_w_h_ratio = comp_w_h_ratio
            self.comp_score_thr = comp_score_thr
            self.text_region_thr = text_region_thr
            self.center_region_thr = center_region_thr
            self.center_region_area_thr = center_region_area_thr
```