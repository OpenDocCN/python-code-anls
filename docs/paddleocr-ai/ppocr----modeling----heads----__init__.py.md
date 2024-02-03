# `.\PaddleOCR\ppocr\modeling\heads\__init__.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 基于“按原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

__all__ = ['build_head']

# 构建头部模块
def build_head(config):
    # 导入检测头部模块
    from .det_db_head import DBHead, PFHeadLocal
    from .det_east_head import EASTHead
    from .det_sast_head import SASTHead
    from .det_pse_head import PSEHead
    from .det_fce_head import FCEHead
    from .e2e_pg_head import PGHead
    from .det_ct_head import CT_Head
    # 导入识别头部模块
    from .rec_ctc_head import CTCHead
    from .rec_att_head import AttentionHead
    from .rec_srn_head import SRNHead
    from .rec_nrtr_head import Transformer
    from .rec_sar_head import SARHead
    from .rec_aster_head import AsterHead
    from .rec_pren_head import PRENHead
    from .rec_multi_head import MultiHead
    from .rec_spin_att_head import SPINAttentionHead
    from .rec_abinet_head import ABINetHead
    from .rec_robustscanner_head import RobustScannerHead
    from .rec_visionlan_head import VLHead
    from .rec_rfl_head import RFLHead
    from .rec_can_head import CANHead
    from .rec_satrn_head import SATRNHead

    # 导入分类头部模块
    from .cls_head import ClsHead

    # 导入知识抽取头部模块
    from .kie_sdmgr_head import SDMGRHead

    # 导入表格注意力头部模块
    from .table_att_head import TableAttentionHead, SLAHead
    # 导入表格主模块头部模块
    from .table_master_head import TableMasterHead
    # 定义一个支持的头部模块列表
    support_dict = [
        'DBHead', 'PSEHead', 'FCEHead', 'EASTHead', 'SASTHead', 'CTCHead',
        'ClsHead', 'AttentionHead', 'SRNHead', 'PGHead', 'Transformer',
        'TableAttentionHead', 'SARHead', 'AsterHead', 'SDMGRHead', 'PRENHead',
        'MultiHead', 'ABINetHead', 'TableMasterHead', 'SPINAttentionHead',
        'VLHead', 'SLAHead', 'RobustScannerHead', 'CT_Head', 'RFLHead',
        'DRRGHead', 'CANHead', 'SATRNHead', 'PFHeadLocal'
    ]

    # 如果配置中的模块名为 'DRRGHead'，则导入 'DRRGHead' 模块并添加到支持列表中
    if config['name'] == 'DRRGHead':
        from .det_drrg_head import DRRGHead
        support_dict.append('DRRGHead')

    # 从配置中弹出模块名
    module_name = config.pop('name')
    # 断言弹出的模块名在支持列表中，否则抛出异常
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    # 根据模块名动态创建模块类实例
    module_class = eval(module_name)(**config)
    # 返回创建的模块类实例
    return module_class
```