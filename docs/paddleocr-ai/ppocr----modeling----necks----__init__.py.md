# `.\PaddleOCR\ppocr\modeling\necks\__init__.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发在“按原样”基础上，
# 没有任何明示或暗示的保证或条件
# 有关特定语言的权限和限制，请参阅许可证

# 导入所有模块
__all__ = ['build_neck']

# 构建 neck 模块
def build_neck(config):
    # 导入各种 neck 模块
    from .db_fpn import DBFPN, RSEFPN, LKPAN
    from .east_fpn import EASTFPN
    from .sast_fpn import SASTFPN
    from .rnn import SequenceEncoder
    from .pg_fpn import PGFPN
    from .table_fpn import TableFPN
    from .fpn import FPN
    from .fce_fpn import FCEFPN
    from .pren_fpn import PRENFPN
    from .csp_pan import CSPPAN
    from .ct_fpn import CTFPN
    from .fpn_unet import FPN_UNet
    from .rf_adaptor import RFAdaptor

    # 支持的 neck 模块列表
    support_dict = [
        'FPN', 'FCEFPN', 'LKPAN', 'DBFPN', 'RSEFPN', 'EASTFPN', 'SASTFPN',
        'SequenceEncoder', 'PGFPN', 'TableFPN', 'PRENFPN', 'CSPPAN', 'CTFPN',
        'RFAdaptor', 'FPN_UNet'
    ]

    # 弹出配置中的模块名
    module_name = config.pop('name')
    # 断言模块名在支持的模块列表中
    assert module_name in support_dict, Exception('neck only support {}'.format(
        support_dict))

    # 根据模块名创建模块类实例
    module_class = eval(module_name)(**config)
    return module_class
```