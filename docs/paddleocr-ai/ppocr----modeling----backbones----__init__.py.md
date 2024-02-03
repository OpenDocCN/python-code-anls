# `.\PaddleOCR\ppocr\modeling\backbones\__init__.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证以“原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

__all__ = ["build_backbone"]

# 构建骨干网络
def build_backbone(config, model_type):
    # 如果模型类型是 "det" 或 "table"
    if model_type == "det" or model_type == "table":
        # 导入检测模型的 MobileNetV3
        from .det_mobilenet_v3 import MobileNetV3
        # 导入检测模型的 ResNet
        from .det_resnet import ResNet
        # 导入检测模型的 ResNet_vd
        from .det_resnet_vd import ResNet_vd
        # 导入检测模型的 ResNet_SAST
        from .det_resnet_vd_sast import ResNet_SAST
        # 导入检测模型的 PPLCNet
        from .det_pp_lcnet import PPLCNet
        # 导入识别模型的 PPLCNetV3
        from .rec_lcnetv3 import PPLCNetV3
        # 导入识别模型的 PPHGNet_small
        from .rec_hgnet import PPHGNet_small
        # 支持的模型字典
        support_dict = [
            "MobileNetV3", "ResNet", "ResNet_vd", "ResNet_SAST", "PPLCNet",
            "PPLCNetV3", "PPHGNet_small"
        ]
        # 如果模型类型是 "table"
        if model_type == "table":
            # 导入表格检测模型的 TableResNetExtra
            from .table_master_resnet import TableResNetExtra
            # 添加到支持的模型字典中
            support_dict.append('TableResNetExtra')
    # 如果模型类型是"rec"或"cls"，则导入相关模型类
    elif model_type == "rec" or model_type == "cls":
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_resnet_vd import ResNet
        from .rec_resnet_fpn import ResNetFPN
        from .rec_mv1_enhance import MobileNetV1Enhance
        from .rec_nrtr_mtb import MTB
        from .rec_resnet_31 import ResNet31
        from .rec_resnet_32 import ResNet32
        from .rec_resnet_45 import ResNet45
        from .rec_resnet_aster import ResNet_ASTER
        from .rec_micronet import MicroNet
        from .rec_efficientb3_pren import EfficientNetb3_PREN
        from .rec_svtrnet import SVTRNet
        from .rec_vitstr import ViTSTR
        from .rec_resnet_rfl import ResNetRFL
        from .rec_densenet import DenseNet
        from .rec_shallow_cnn import ShallowCNN
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_hgnet import PPHGNet_small
        # 定义支持的模型类列表
        support_dict = [
            'MobileNetV1Enhance', 'MobileNetV3', 'ResNet', 'ResNetFPN', 'MTB',
            'ResNet31', 'ResNet45', 'ResNet_ASTER', 'MicroNet',
            'EfficientNetb3_PREN', 'SVTRNet', 'ViTSTR', 'ResNet32', 'ResNetRFL',
            'DenseNet', 'ShallowCNN', 'PPLCNetV3', 'PPHGNet_small'
        ]
    # 如果模型类型是"e2e"，则导入相关模型类
    elif model_type == 'e2e':
        from .e2e_resnet_vd_pg import ResNet
        # 定义支持的模型类列表
        support_dict = ['ResNet']
    # 如果模型类型是"kie"，则导入相关模型类
    elif model_type == 'kie':
        from .kie_unet_sdmgr import Kie_backbone
        from .vqa_layoutlm import LayoutLMForSer, LayoutLMv2ForSer, LayoutLMv2ForRe, LayoutXLMForSer, LayoutXLMForRe
        # 定义支持的模型类列表
        support_dict = [
            'Kie_backbone', 'LayoutLMForSer', 'LayoutLMv2ForSer',
            'LayoutLMv2ForRe', 'LayoutXLMForSer', 'LayoutXLMForRe'
        ]
    # 如果模型类型是"table"，则导入相关模型类
    elif model_type == 'table':
        from .table_resnet_vd import ResNet
        from .table_mobilenet_v3 import MobileNetV3
        # 定义支持的模型类列表
        support_dict = ['ResNet', 'MobileNetV3']
    else:
        # 如果模型类型不在以上几种情况中，则抛出未实现错误
        raise NotImplementedError

    # 从配置中弹出模型名称并赋值给module_name
    module_name = config.pop('name')
    # 检查模块名称是否在支持的字典中，如果不在则抛出异常
    assert module_name in support_dict, Exception(
        "when model typs is {}, backbone only support {}".format(model_type,
                                                                 support_dict))
    # 使用 eval 函数根据模块名称创建对应的类实例
    module_class = eval(module_name)(**config)
    # 返回创建的模块类实例
    return module_class
```