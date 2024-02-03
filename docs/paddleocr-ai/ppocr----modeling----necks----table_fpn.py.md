# `.\PaddleOCR\ppocr\modeling\necks\table_fpn.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件，无论是明示还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr

# 定义 TableFPN 类，继承自 nn.Layer
class TableFPN(nn.Layer):
    # 前向传播函数
    def forward(self, x):
        # 将输入 x 拆分为 c2, c3, c4, c5 四个部分
        c2, c3, c4, c5 = x

        # 对 c5 进行卷积操作得到 in5
        in5 = self.in5_conv(c5)
        # 对 c4 进行卷积操作得到 in4
        in4 = self.in4_conv(c4)
        # 对 c3 进行卷积操作得到 in3
        in3 = self.in3_conv(c3)
        # 对 c2 进行卷积操作得到 in2
        in2 = self.in2_conv(c2)

        # 将 in5 上采样到和 in4 相同大小，然后与 in4 相加得到 out4
        out4 = in4 + F.upsample(
            in5, size=in4.shape[2:4], mode="nearest", align_mode=1)  # 1/16
        # 将 out4 上采样到和 in3 相同大小，然后与 in3 相加得到 out3
        out3 = in3 + F.upsample(
            out4, size=in3.shape[2:4], mode="nearest", align_mode=1)  # 1/8
        # 将 out3 上采样到和 in2 相同大小，然后与 in2 相加得到 out2
        out2 = in2 + F.upsample(
            out3, size=in2.shape[2:4], mode="nearest", align_mode=1)  # 1/4

        # 将 out4 上采样到和 in5 相同大小得到 p4
        p4 = F.upsample(out4, size=in5.shape[2:4], mode="nearest", align_mode=1)
        # 将 out3 上采样到和 in5 相同大小得到 p3
        p3 = F.upsample(out3, size=in5.shape[2:4], mode="nearest", align_mode=1)
        # 将 out2 上采样到和 in5 相同大小得到 p2
        p2 = F.upsample(out2, size=in5.shape[2:4], mode="nearest", align_mode=1)
        # 将 in5, p4, p3, p2 沿着 axis=1 进行拼接得到 fuse
        fuse = paddle.concat([in5, p4, p3, p2], axis=1)
        # 对 fuse 进行卷积操作并乘以 0.005 得到 fuse_conv
        fuse_conv = self.fuse_conv(fuse) * 0.005
        # 返回 c5 与 fuse_conv 相加的结果
        return [c5 + fuse_conv]
```