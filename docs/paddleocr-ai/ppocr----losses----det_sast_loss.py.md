# `.\PaddleOCR\ppocr\losses\det_sast_loss.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
from .det_basic_loss import DiceLoss
import numpy as np

# 定义 SASTLoss 类，继承自 nn.Layer
class SASTLoss(nn.Layer):
    """
    """

    # 初始化方法
    def __init__(self, eps=1e-6, **kwargs):
        # 调用父类的初始化方法
        super(SASTLoss, self).__init__()
        # 创建 DiceLoss 对象，传入 eps 参数
        self.dice_loss = DiceLoss(eps=eps)
```