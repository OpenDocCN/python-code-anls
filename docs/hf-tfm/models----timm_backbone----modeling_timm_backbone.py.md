# `.\models\timm_backbone\modeling_timm_backbone.py`

```
# 引入必要的模块和函数
from typing import Optional, Tuple, Union  # 导入类型提示所需的模块

import torch  # 导入 PyTorch 库

from ...modeling_outputs import BackboneOutput  # 导入模型输出的背景输出
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的实用工具
from ...utils import is_timm_available, is_torch_available, requires_backends  # 导入用于检查库是否可用的工具函数
from ...utils.backbone_utils import BackboneMixin  # 导入背景混合类
from .configuration_timm_backbone import TimmBackboneConfig  # 导入 Timm 模型的配置类

# 检查是否安装了 timm 库
if is_timm_available():
    import timm  # 如果可用，导入 timm 库

# 检查是否安装了 torch 库
if is_torch_available():
    from torch import Tensor  # 如果可用，从 torch 库导入 Tensor 类型

class TimmBackbone(PreTrainedModel, BackboneMixin):
    """
    Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
    other models in the library keeping the same API.
    """
    
    main_input_name = "pixel_values"  # 定义主要输入名称为 "pixel_values"
    supports_gradient_checkpointing = False  # 不支持梯度检查点
    config_class = TimmBackboneConfig  # 指定配置类为 TimmBackboneConfig
    # 初始化方法，接受配置和其他关键字参数
    def __init__(self, config, **kwargs):
        # 要求使用"timm"库作为后端
        requires_backends(self, "timm")
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置信息保存在实例中
        self.config = config

        # 如果配置中未设置backbone，则抛出数值错误
        if config.backbone is None:
            raise ValueError("backbone is not set in the config. Please set it to a timm model name.")

        # 如果配置中指定的backbone不在timm支持的模型列表中，则抛出数值错误
        if config.backbone not in timm.list_models():
            raise ValueError(f"backbone {config.backbone} is not supported by timm.")

        # 如果配置中存在out_features参数（不支持），则抛出数值错误，建议使用out_indices
        if hasattr(config, "out_features") and config.out_features is not None:
            raise ValueError("out_features is not supported by TimmBackbone. Please use out_indices instead.")

        # 获取配置中的use_pretrained_backbone参数，如果未设置，则抛出数值错误
        pretrained = getattr(config, "use_pretrained_backbone", None)
        if pretrained is None:
            raise ValueError("use_pretrained_backbone is not set in the config. Please set it to True or False.")

        # 默认情况下，仅使用最后一层。这与transformers模型的默认行为匹配
        # 如果配置中存在out_indices参数，则使用该参数；否则默认使用最后一层（-1）
        out_indices = config.out_indices if getattr(config, "out_indices", None) is not None else (-1,)

        # 使用timm库创建指定的模型
        self._backbone = timm.create_model(
            config.backbone,
            pretrained=pretrained,
            features_only=config.features_only,
            in_chans=config.num_channels,
            out_indices=out_indices,
            **kwargs,
        )

        # 如果配置中设置了freeze_batch_norm_2d参数为True，则冻结模型中所有的BatchNorm2d和SyncBatchNorm层
        if getattr(config, "freeze_batch_norm_2d", False):
            self.freeze_batch_norm_2d()

        # _backbone的return_layers属性用于控制模型调用时的输出
        self._return_layers = self._backbone.return_layers
        # _backbone的feature_info.info属性包含所有层的信息，将其转换为字典形式保存在_all_layers中
        self._all_layers = {layer["module"]: str(i) for i, layer in enumerate(self._backbone.feature_info.info)}
        
        # 调用父类的_init_backbone方法，初始化backbone模型
        super()._init_backbone(config)
    # 通过预训练模型名或路径创建一个新的实例
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 要求类依赖的后端模块是 "vision" 和 "timm"
        requires_backends(cls, ["vision", "timm"])
        # 从特定位置导入 TimmBackboneConfig 类
        from ...models.timm_backbone import TimmBackboneConfig

        # 从关键字参数中弹出配置对象，若不存在则使用 TimmBackboneConfig 的默认配置
        config = kwargs.pop("config", TimmBackboneConfig())

        # 确定是否使用 timm 的 backbone，默认为 True；若不是，则抛出 ValueError
        use_timm = kwargs.pop("use_timm_backbone", True)
        if not use_timm:
            raise ValueError("use_timm_backbone must be True for timm backbones")

        # 从关键字参数中获取或使用 TimmBackboneConfig 中的默认值来设置通道数
        num_channels = kwargs.pop("num_channels", config.num_channels)
        # 从关键字参数中获取或使用 TimmBackboneConfig 中的默认值来设置仅提取特征的标志
        features_only = kwargs.pop("features_only", config.features_only)
        # 从关键字参数中获取或使用 TimmBackboneConfig 中的默认值来设置是否使用预训练的 backbone
        use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        # 从关键字参数中获取或使用 TimmBackboneConfig 中的默认值来设置输出的索引列表
        out_indices = kwargs.pop("out_indices", config.out_indices)

        # 使用给定的参数创建一个 TimmBackboneConfig 对象
        config = TimmBackboneConfig(
            backbone=pretrained_model_name_or_path,
            num_channels=num_channels,
            features_only=features_only,
            use_pretrained_backbone=use_pretrained_backbone,
            out_indices=out_indices,
        )
        # 调用父类的 _from_config 方法，传递配置对象和其它关键字参数
        return super()._from_config(config, **kwargs)

    # 冻结模型中所有 2D 批归一化层的参数
    def freeze_batch_norm_2d(self):
        timm.layers.freeze_batch_norm_2d(self._backbone)

    # 解冻模型中所有 2D 批归一化层的参数
    def unfreeze_batch_norm_2d(self):
        timm.layers.unfreeze_batch_norm_2d(self._backbone)

    # 空的初始化权重函数，确保类在库中的兼容性
    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibility of the class in the library.
        """
        pass

    # 前向传播函数，接收像素值作为输入，并可以选择返回注意力、隐藏状态或以字典形式返回结果
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        # 如果 return_dict 为 None，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_hidden_states 为 None，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_attentions 为 None，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 如果要输出注意力机制信息，则抛出 ValueError，因为 timm 模型暂不支持注意力输出
        if output_attentions:
            raise ValueError("Cannot output attentions for timm backbones at the moment")

        # 如果要输出隐藏状态信息
        if output_hidden_states:
            # 修改返回的层级以包括所有的 backbone 阶段
            self._backbone.return_layers = self._all_layers
            # 使用 backbone 模型提取特征，返回隐藏状态
            hidden_states = self._backbone(pixel_values, **kwargs)
            # 恢复返回的层级设置
            self._backbone.return_layers = self._return_layers
            # 从隐藏状态中提取指定索引的特征图
            feature_maps = tuple(hidden_states[i] for i in self.out_indices)
        else:
            # 直接使用 backbone 模型提取特征，不返回隐藏状态
            feature_maps = self._backbone(pixel_values, **kwargs)
            hidden_states = None

        # 将特征图转换为元组
        feature_maps = tuple(feature_maps)
        # 如果隐藏状态不为 None，则将其转换为元组；否则设置为 None
        hidden_states = tuple(hidden_states) if hidden_states is not None else None

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 构造输出元组，包含特征图
            output = (feature_maps,)
            # 如果需要输出隐藏状态，则将隐藏状态也添加到输出中
            if output_hidden_states:
                output = output + (hidden_states,)
            # 返回构造的输出元组
            return output

        # 如果需要返回字典形式的输出，则构造 BackboneOutput 对象并返回
        return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states, attentions=None)
```