# `.\PaddleOCR\deploy\hubserving\structure_layout\params.py`

```py
# 版权声明和许可信息
# 本代码版权归 PaddlePaddle 作者所有。保留所有权利。
# 根据 Apache 许可证 2.0 版本授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 导入未来的绝对导入、除法和打印功能
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 定义一个 Config 类
class Config(object):
    pass

# 读取参数函数
def read_params():
    # 创建一个 Config 实例
    cfg = Config()

    # 布局分析的参数
    cfg.layout_model_dir = './inference/picodet_lcnet_x1_0_fgd_layout_infer/'
    cfg.layout_dict_path = './ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
    cfg.layout_score_threshold = 0.5
    cfg.layout_nms_threshold = 0.5

    # 返回配置对象
    return cfg
```