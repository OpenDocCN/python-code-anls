# `.\PaddleOCR\ppocr\metrics\__init__.py`

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
# 请查看许可证以获取有关权限和限制的具体语言

# 导入未来的绝对导入、除法、打印函数和 Unicode 字符串
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 导入 copy 模块
import copy

# 导入所有模块
__all__ = ["build_metric"]

# 从不同的模块中导入不同的类
from .det_metric import DetMetric, DetFCEMetric
from .rec_metric import RecMetric, CNTMetric, CANMetric
from .cls_metric import ClsMetric
from .e2e_metric import E2EMetric
from .distillation_metric import DistillationMetric
from .table_metric import TableMetric
from .kie_metric import KIEMetric
from .vqa_token_ser_metric import VQASerTokenMetric
from .vqa_token_re_metric import VQAReTokenMetric
from .sr_metric import SRMetric
from .ct_metric import CTMetric

# 构建度量函数
def build_metric(config):
    # 支持的度量函数列表
    support_dict = [
        "DetMetric", "DetFCEMetric", "RecMetric", "ClsMetric", "E2EMetric",
        "DistillationMetric", "TableMetric", 'KIEMetric', 'VQASerTokenMetric',
        'VQAReTokenMetric', 'SRMetric', 'CTMetric', 'CNTMetric', 'CANMetric'
    ]

    # 深拷贝配置
    config = copy.deepcopy(config)
    # 获取模块名
    module_name = config.pop("name")
    # 断言模块名在支持的列表中，否则抛出异常
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict))
    # 根据模块名创建模块类实例
    module_class = eval(module_name)(**config)
    # 返回模块类实例
    return module_class
```