# `.\PaddleOCR\ppocr\optimizer\regularizer.py`

```py
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
# 请查看许可证以获取特定语言的权限和限制

# 导入未来的绝对导入、除法、打印函数和 Unicode 字符串
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 导入 Paddle 模块
import paddle

# 定义 L1 正则化类
class L1Decay(object):
    """
    L1 Weight Decay Regularization, which encourages the weights to be sparse.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    # 初始化函数
    def __init__(self, factor=0.0):
        super(L1Decay, self).__init__()
        # 设置正则化系数
        self.coeff = factor

    # 调用函数
    def __call__(self):
        # 创建 L1 正则化对象
        reg = paddle.regularizer.L1Decay(self.coeff)
        return reg

# 定义 L2 正则化类
class L2Decay(object):
    """
    L2 Weight Decay Regularization, which helps to prevent the model over-fitting.
    Args:
        factor(float): regularization coeff. Default:0.0.
    """

    # 初始化函数
    def __init__(self, factor=0.0):
        super(L2Decay, self).__init__()
        # 设置正则化系数
        self.coeff = float(factor)

    # 调用函数
    def __call__(self):
        return self.coeff
```