# `.\PaddleOCR\ppocr\postprocess\pg_postprocess.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何形式的担保或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入必要的模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# 获取当前文件所在目录
__dir__ = os.path.dirname(__file__)
# 将当前目录添加到系统路径中
sys.path.append(__dir__)
# 将当前目录的上一级目录添加到系统路径中
sys.path.append(os.path.join(__dir__, '..'))
# 导入 PGNet_PostProcess 类
from ppocr.utils.e2e_utils.pgnet_pp_utils import PGNet_PostProcess

# 定义 PGPostProcess 类
class PGPostProcess(object):
    """
    The post process for PGNet.
    """

    # 初始化方法
    def __init__(self,
                 character_dict_path,
                 valid_set,
                 score_thresh,
                 mode,
                 point_gather_mode=None,
                 **kwargs):
        # 字符字典路径
        self.character_dict_path = character_dict_path
        # 验证集
        self.valid_set = valid_set
        # 分数阈值
        self.score_thresh = score_thresh
        # 模式
        self.mode = mode
        # 点聚合模式
        self.point_gather_mode = point_gather_mode

        # 判断是否为 Python 3.5 版本
        self.is_python35 = False
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            self.is_python35 = True
    # 定义一个类方法，接受输出字典和形状列表作为参数
    def __call__(self, outs_dict, shape_list):
        # 创建一个 PGNet_PostProcess 对象，传入字符字典路径、有效集合、分数阈值、输出字典、形状列表和点聚合模式
        post = PGNet_PostProcess(
            self.character_dict_path,
            self.valid_set,
            self.score_thresh,
            outs_dict,
            shape_list,
            point_gather_mode=self.point_gather_mode)
        # 如果模式为'fast'，则调用快速后处理方法
        if self.mode == 'fast':
            data = post.pg_postprocess_fast()
        # 否则调用慢速后处理方法
        else:
            data = post.pg_postprocess_slow()
        # 返回处理后的数据
        return data
```