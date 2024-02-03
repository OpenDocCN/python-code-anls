# `.\PaddleOCR\ppocr\modeling\transforms\__init__.py`

```
# 版权声明和许可证信息
# 本代码版权归 PaddlePaddle 作者所有
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

__all__ = ['build_transform']

# 构建变换函数，根据配置参数构建不同的变换模块
def build_transform(config):
    # 导入 TPS 变换模块
    from .tps import TPS
    # 导入 STN_ON 变换模块
    from .stn import STN_ON
    # 导入 TSRN 变换模块
    from .tsrn import TSRN
    # 导入 TBSRN 变换模块
    from .tbsrn import TBSRN
    # 导入 GA_SPIN 变换模块
    from .gaspin_transformer import GA_SPIN_Transformer as GA_SPIN

    # 支持的变换模块列表
    support_dict = ['TPS', 'STN_ON', 'GA_SPIN', 'TSRN', 'TBSRN']

    # 弹出配置参数中的模块名
    module_name = config.pop('name')
    # 断言模块名在支持的模块列表中，否则抛出异常
    assert module_name in support_dict, Exception(
        'transform only support {}'.format(support_dict))
    # 根据模块名动态创建对应的模块类实例
    module_class = eval(module_name)(**config)
    # 返回创建的模块类实例
    return module_class
```