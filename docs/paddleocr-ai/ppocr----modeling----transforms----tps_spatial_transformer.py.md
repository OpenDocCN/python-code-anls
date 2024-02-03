# `.\PaddleOCR\ppocr\modeling\transforms\tps_spatial_transformer.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用代码来源
# https://github.com/ayumiymk/aster.pytorch/blob/master/lib/models/tps_spatial_transformer.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
from paddle import nn, ParamAttr
from paddle.nn import functional as F
import numpy as np
import itertools

# 定义 grid_sample 函数，用于对输入进行网格采样
def grid_sample(input, grid, canvas=None):
    # 设置输入的梯度为可计算
    input.stop_gradient = False

    # 检查是否为 fp16 类型
    is_fp16 = False
    if grid.dtype != paddle.float32:
        data_type = grid.dtype
        input = input.cast(paddle.float32)
        grid = grid.cast(paddle.float32)
        is_fp16 = True
    # 使用 paddle.nn.functional 中的 grid_sample 函数对输入进行网格采样
    output = F.grid_sample(input, grid)
    if is_fp16:
        output = output.cast(data_type)
        grid = grid.cast(data_type)

    # 如果没有给定 canvas，则直接返回输出
    if canvas is None:
        return output
    else:
        # 创建与输入形状相同的全为 1 的张量作为输入掩码
        input_mask = paddle.ones(shape=input.shape)
        if is_fp16:
            input_mask = input_mask.cast(paddle.float32)
            grid = grid.cast(paddle.float32)
        # 使用 grid_sample 函数对输入掩码进行网格采样
        output_mask = F.grid_sample(input_mask, grid)
        if is_fp16:
            output_mask = output_mask.cast(data_type)
        # 根据输出掩码对输出和 canvas 进行加权求和
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

# 计算部分表示，根据输入点和控制点计算
def compute_partial_repr(input_points, control_points):
    N = input_points.shape[0]
    # 获取控制点的数量
    M = control_points.shape[0]
    # 计算输入点与控制点之间的差值
    pairwise_diff = paddle.reshape(
        input_points, shape=[N, 1, 2]) - paddle.reshape(
            control_points, shape=[1, M, 2])
    # 计算差值的平方
    pairwise_diff_square = pairwise_diff * pairwise_diff
    # 计算欧氏距离的平方
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :,
                                                                         1]
    # 计算表示矩阵
    repr_matrix = 0.5 * pairwise_dist * paddle.log(pairwise_dist)
    # 修复数值误差，将所有 NaN 替换为 0
    mask = np.array(repr_matrix != repr_matrix)
    repr_matrix[mask] = 0
    # 返回表示矩阵
    return repr_matrix
# 根据给定的控制点数量和边距构建输出控制点
def build_output_control_points(num_control_points, margins):
    # 解包边距
    margin_x, margin_y = margins
    # 计算每一侧的控制点数量
    num_ctrl_pts_per_side = num_control_points // 2
    # 在 x 轴上均匀分布控制点
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_ctrl_pts_per_side)
    # 在顶部创建控制点
    ctrl_pts_y_top = np.ones(num_ctrl_pts_per_side) * margin_y
    # 在底部创建控制点
    ctrl_pts_y_bottom = np.ones(num_ctrl_pts_per_side) * (1.0 - margin_y)
    # 组合顶部控制点坐标
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    # 组合底部控制点坐标
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    # 连接顶部和底部控制点数组
    output_ctrl_pts_arr = np.concatenate(
        [ctrl_pts_top, ctrl_pts_bottom], axis=0)
    # 转换为 PaddlePaddle 的张量
    output_ctrl_pts = paddle.to_tensor(output_ctrl_pts_arr)
    return output_ctrl_pts

# 定义 TPSSpatialTransformer 类
class TPSSpatialTransformer(nn.Layer):
    # 前向传播函数
    def forward(self, input, source_control_points):
        # 断言控制点张量的维度为3
        assert source_control_points.ndimension() == 3
        # 断言控制点张量的第二维度与控制点数量相等
        assert source_control_points.shape[1] == self.num_control_points
        # 断言控制点张量的第三维度为2
        assert source_control_points.shape[2] == 2
        # 获取批量大小
        batch_size = paddle.shape(source_control_points)[0]

        # 扩展填充矩阵
        padding_matrix = paddle.expand(
            self.padding_matrix, shape=[batch_size, 3, 2])
        # 拼接控制点张量和填充矩阵
        Y = paddle.concat([
            source_control_points.astype(padding_matrix.dtype), padding_matrix
        ], 1)
        # 计算映射矩阵
        mapping_matrix = paddle.matmul(self.inverse_kernel, Y)
        # 计算源坐标
        source_coordinate = paddle.matmul(self.target_coordinate_repr,
                                          mapping_matrix)

        # 重塑为网格
        grid = paddle.reshape(
            source_coordinate,
            shape=[-1, self.target_height, self.target_width, 2])
        # 将网格限制在[0, 1]范围内
        grid = paddle.clip(grid, 0, 1)
        # 输入到 grid_sample 的输入是归一化的[-1, 1]，但我们得到的是[0, 1]
        grid = 2.0 * grid - 1.0
        # 使用 grid_sample 函数进行采样
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate
```