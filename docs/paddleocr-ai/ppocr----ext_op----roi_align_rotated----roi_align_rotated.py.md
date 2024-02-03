# `.\PaddleOCR\ppocr\ext_op\roi_align_rotated\roi_align_rotated.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用代码来源
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/ops/roi_align_rotated.py

# 导入 PaddlePaddle 模块
import paddle
import paddle.nn as nn
# 从 PaddlePaddle 的 C++ 扩展中加载自定义操作
from paddle.utils.cpp_extension import load
# 加载自定义 JIT 操作
custom_ops = load(
    name="custom_jit_ops",
    sources=[
        "ppocr/ext_op/roi_align_rotated/roi_align_rotated.cc",
        "ppocr/ext_op/roi_align_rotated/roi_align_rotated.cu"
    ])

# 获取自定义操作中的 roi_align_rotated 函数
roi_align_rotated = custom_ops.roi_align_rotated

# 定义 RoIAlignRotated 类，用于处理旋转的 RoI 对齐池化
class RoIAlignRotated(nn.Layer):
    """RoI align pooling layer for rotated proposals.

    """

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 aligned=True,
                 clockwise=False):
        super(RoIAlignRotated, self).__init__()

        # 如果 out_size 是整数，则设置输出高度和宽度为相同值
        if isinstance(out_size, int):
            self.out_h = out_size
            self.out_w = out_size
        # 如果 out_size 是元组，则设置输出高度和宽度为元组中的值
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            self.out_h, self.out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')

        # 设置空间尺度、采样数量、对齐方式和旋转方向
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.aligned = aligned
        self.clockwise = clockwise
    # 定义一个前向传播函数，接受特征和感兴趣区域作为输入
    def forward(self, feats, rois):
        # 调用 roi_align_rotated 函数，对特征和感兴趣区域进行旋转对齐操作，得到输出
        output = roi_align_rotated(feats, rois, self.out_h, self.out_w,
                                   self.spatial_scale, self.sample_num,
                                   self.aligned, self.clockwise)
        # 返回处理后的输出
        return output
```