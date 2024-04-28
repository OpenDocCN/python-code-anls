# `.\transformers\models\timm_backbone\modeling_timm_backbone.py`

```
# 设置文件编码格式为 utf-8
# 版权声明和许可证信息

# 引入类型提示模块中的类型
from typing import Optional, Tuple, Union

# 引入 pytorch 模块
import torch

# 引入 HuggingFace 库中的模型输出和模型工具类
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import is_timm_available, is_torch_available, requires_backends

# 引入 timm 模型的配置类
from .configuration_timm_backbone import TimmBackboneConfig

# 如果 timm 模型可用
if is_timm_available():
    # 则引入 timm 模型

# 如果 torch 模块可用
if is_torch_available():
    # 则引入 Tensor 类型

# 定义 TimmBackbone 类，继承自 PreTrainedModel 和 BackboneMixin
class TimmBackbone(PreTrainedModel, BackboneMixin):
    """
    用于将 timm 模型包装成可以与库中其他模型互换使用的类。这保持了相同的 API。
    """

    # 设置主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False
    # 设置配置类为 TimmBackboneConfig
    config_class = TimmBackboneConfig
    # 初始化方法，接受配置对象和额外的关键字参数
    def __init__(self, config, **kwargs):
        # 检查所需后端是否存在，这里是指检查是否存在名为"timm"的后端
        requires_backends(self, "timm")
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的配置对象保存到实例属性中
        self.config = config

        # 如果配置中未设置backbone，则抛出值错误异常
        if config.backbone is None:
            raise ValueError("backbone is not set in the config. Please set it to a timm model name.")

        # 如果配置中指定的backbone不在timm支持的模型列表中，则抛出值错误异常
        if config.backbone not in timm.list_models():
            raise ValueError(f"backbone {config.backbone} is not supported by timm.")

        # 如果配置中存在out_features属性且不为None，则抛出值错误异常
        if hasattr(config, "out_features") and config.out_features is not None:
            raise ValueError("out_features is not supported by TimmBackbone. Please use out_indices instead.")

        # 获取配置中的use_pretrained_backbone属性，若未设置则抛出值错误异常
        pretrained = getattr(config, "use_pretrained_backbone", None)
        if pretrained is None:
            raise ValueError("use_pretrained_backbone is not set in the config. Please set it to True or False.")

        # 默认情况下，只取最后一层作为输出层。这与transformers模型的默认设置相匹配。
        # 如果配置中设置了out_indices属性，则使用该属性值作为输出索引；否则，默认输出最后一层
        out_indices = config.out_indices if getattr(config, "out_indices", None) is not None else (-1,)

        # 使用timm库创建模型
        self._backbone = timm.create_model(
            config.backbone,
            pretrained=pretrained,
            # 对于transformer架构，当前不支持该参数
            features_only=config.features_only,
            # 输入通道数
            in_chans=config.num_channels,
            # 输出索引
            out_indices=out_indices,
            **kwargs,
        )

        # 如果配置中存在freeze_batch_norm_2d属性且为True，则调用freeze_batch_norm_2d方法
        if getattr(config, "freeze_batch_norm_2d", False):
            self.freeze_batch_norm_2d()

        # 控制模型调用时的输出，如果output_hidden_states为True，则修改return_layers以包含所有层
        self._return_layers = self._backbone.return_layers
        # 存储所有层及其索引的字典
        self._all_layers = {layer["module"]: str(i) for i, layer in enumerate(self._backbone.feature_info.info)}
        # 调用父类的_init_backbone方法
        super()._init_backbone(config)

    # 类方法
    @classmethod
    # 从预训练模型名称或路径创建实例，接受额外的模型参数和关键字参数
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 确保后端支持视觉和timm库
        requires_backends(cls, ["vision", "timm"])
        # 导入TimmBackboneConfig配置类
        from ...models.timm_backbone import TimmBackboneConfig

        # 从关键字参数中弹出配置，使用默认值为TimmBackboneConfig的实例
        config = kwargs.pop("config", TimmBackboneConfig())

        # 从关键字参数中弹出是否使用timm背骨的标志，如果不使用则引发值错误
        use_timm = kwargs.pop("use_timm_backbone", True)
        if not use_timm:
            raise ValueError("use_timm_backbone must be True for timm backbones")

        # 从关键字参数中弹出通道数量，特征模式，预训练背骨使用标志和输出索引
        num_channels = kwargs.pop("num_channels", config.num_channels)
        features_only = kwargs.pop("features_only", config.features_only)
        use_pretrained_backbone = kwargs.pop("use_pretrained_backbone", config.use_pretrained_backbone)
        out_indices = kwargs.pop("out_indices", config.out_indices)

        # 根据弹出的参数创建TimmBackboneConfig配置类实例
        config = TimmBackboneConfig(
            backbone=pretrained_model_name_or_path,
            num_channels=num_channels,
            features_only=features_only,
            use_pretrained_backbone=use_pretrained_backbone,
            out_indices=out_indices,
        )
        # 返回调用超类_from_config方法得到的结果
        return super()._from_config(config, **kwargs)

    # 冻结二维批量归一化
    def freeze_batch_norm_2d(self):
        # 调用timm.layers中的freeze_batch_norm_2d方法传入背骨
        timm.layers.freeze_batch_norm_2d(self._backbone)

    # 解冻二维批量归一化
    def unfreeze_batch_norm_2d(self):
        # 调用timm.layers中的unfreeze_batch_norm_2d方法传入背骨
        timm.layers.unfreeze_batch_norm_2d(self._backbone)

    # 初始化权重函数，确保类在库中的兼容性
    def _init_weights(self, module):
        """
        Empty init weights function to ensure compatibility of the class in the library.
        """
        # 空的初始化权重函数，以确保类在库中的兼容性
        pass

    # 前向传播函数，接受像素值，输出是否关注，隐藏状态，返回字典等关键字参数
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    # 根据输入参数确定是否返回字典形式结果，如果不指定则根据配置决定是否使用返回字典形式
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # 根据输入参数确定是否返回隐藏层状态，如果不指定则根据配置决定是否输出隐藏层状态
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 根据输入参数确定是否返回注意力权重，如果不指定则根据配置决定是否输出注意力权重
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    
            if output_attentions:
                # 如果需要输出注意力权重，则抛出异常，目前不支持 timm backbones 输出注意力权重
                raise ValueError("Cannot output attentions for timm backbones at the moment")
    
            if output_hidden_states:
                # 如果需要输出隐藏层状态，修改返回的层以包括骨干网络的所有阶段
                self._backbone.return_layers = self._all_layers
                # 使用骨干网络处理输入像素值参数
                hidden_states = self._backbone(pixel_values, **kwargs)
                # 恢复返回的层
                self._backbone.return_layers = self._return_layers
                # 根据索引选择特征图
                feature_maps = tuple(hidden_states[i] for i in self.out_indices)
            else:
                # 如果不输出隐藏层状态，直接使用骨干网络处理输入像素值参数
                feature_maps = self._backbone(pixel_values, **kwargs)
                hidden_states = None
    
            # 转换为元组形式
            feature_maps = tuple(feature_maps)
            hidden_states = tuple(hidden_states) if hidden_states is not None else None
    
            if not return_dict:
                # 如果不需要返回字典形式结果
                output = (feature_maps,)
                if output_hidden_states:
                    # 如果输出隐藏层状态，将隐藏层状态添加到输出中
                    output = output + (hidden_states,)
                return output
    
            # 返回字典形式结果
            return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states, attentions=None)
```