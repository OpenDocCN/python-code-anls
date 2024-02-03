# `.\PaddleOCR\ppocr\postprocess\__init__.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入未来的绝对导入、除法、打印函数和 Unicode 字符串
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 导入 copy 模块
import copy

# 定义模块可导出的内容
__all__ = ['build_post_process']

# 导入其他模块中的类
from .db_postprocess import DBPostProcess, DistillationDBPostProcess
from .east_postprocess import EASTPostProcess
from .sast_postprocess import SASTPostProcess
from .fce_postprocess import FCEPostProcess
from .rec_postprocess import CTCLabelDecode, AttnLabelDecode, SRNLabelDecode, \
    DistillationCTCLabelDecode, NRTRLabelDecode, SARLabelDecode, \
    SEEDLabelDecode, PRENLabelDecode, ViTSTRLabelDecode, ABINetLabelDecode, \
    SPINLabelDecode, VLLabelDecode, RFLLabelDecode, SATRNLabelDecode
from .cls_postprocess import ClsPostProcess
from .pg_postprocess import PGPostProcess
from .vqa_token_ser_layoutlm_postprocess import VQASerTokenLayoutLMPostProcess, DistillationSerPostProcess
from .vqa_token_re_layoutlm_postprocess import VQAReTokenLayoutLMPostProcess, DistillationRePostProcess
from .table_postprocess import TableMasterLabelDecode, TableLabelDecode
from .picodet_postprocess import PicoDetPostProcess
from .ct_postprocess import CTPostProcess
from .drrg_postprocess import DRRGPostprocess
from .rec_postprocess import CANLabelDecode

# 定义构建后处理过程的函数，接受配置和全局配置参数
def build_post_process(config, global_config=None):
    # 定义一个支持的后处理模块列表
    support_dict = [
        'DBPostProcess', 'EASTPostProcess', 'SASTPostProcess', 'FCEPostProcess',
        'CTCLabelDecode', 'AttnLabelDecode', 'ClsPostProcess', 'SRNLabelDecode',
        'PGPostProcess', 'DistillationCTCLabelDecode', 'TableLabelDecode',
        'DistillationDBPostProcess', 'NRTRLabelDecode', 'SARLabelDecode',
        'SEEDLabelDecode', 'VQASerTokenLayoutLMPostProcess',
        'VQAReTokenLayoutLMPostProcess', 'PRENLabelDecode',
        'DistillationSARLabelDecode', 'ViTSTRLabelDecode', 'ABINetLabelDecode',
        'TableMasterLabelDecode', 'SPINLabelDecode',
        'DistillationSerPostProcess', 'DistillationRePostProcess',
        'VLLabelDecode', 'PicoDetPostProcess', 'CTPostProcess',
        'RFLLabelDecode', 'DRRGPostprocess', 'CANLabelDecode',
        'SATRNLabelDecode'
    ]
    
    # 如果配置中的模块名为 'PSEPostProcess'，则导入对应的模块并添加到支持列表中
    if config['name'] == 'PSEPostProcess':
        from .pse_postprocess import PSEPostProcess
        support_dict.append('PSEPostProcess')
    
    # 复制配置信息，准备修改
    config = copy.deepcopy(config)
    # 获取模块名并从配置中移除
    module_name = config.pop('name')
    # 如果模块名为 "None"，则返回空
    if module_name == "None":
        return
    # 如果有全局配置信息，则更新当前配置信息
    if global_config is not None:
        config.update(global_config)
    # 检查模块名是否在支持列表中，如果不在则抛出异常
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    # 根据模块名动态创建对应的类实例
    module_class = eval(module_name)(**config)
    # 返回创建的模块实例
    return module_class
```