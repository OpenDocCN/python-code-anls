# `.\utils\backbone_utils.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2023 年由 HuggingFace Inc. 团队编写
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，不提供任何形式的担保或条件
# 有关详细信息，请参阅许可证。
#
# 导入需要的模块和库
import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union

# 定义枚举类型 BackboneType，包含 TIMM 和 TRANSFORMERS 两种类型
class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"

# 定义函数 verify_out_features_out_indices，用于验证给定的 out_features 和 out_indices 是否对应给定的 stage_names
def verify_out_features_out_indices(
    out_features: Optional[Iterable[str]], out_indices: Optional[Iterable[int]], stage_names: Optional[Iterable[str]]
):
    """
    Verify that out_indices and out_features are valid for the given stage_names.
    """
    # 如果 stage_names 为 None，则抛出 ValueError 异常
    if stage_names is None:
        raise ValueError("Stage_names must be set for transformers backbones")

    # 如果 out_features 不为 None
    if out_features is not None:
        # 如果 out_features 不是列表类型，则抛出 ValueError 异常
        if not isinstance(out_features, (list,)):
            raise ValueError(f"out_features must be a list got {type(out_features)}")
        # 如果 out_features 中的任何特征不在 stage_names 中，则抛出 ValueError 异常
        if any(feat not in stage_names for feat in out_features):
            raise ValueError(f"out_features must be a subset of stage_names: {stage_names} got {out_features}")
        # 如果 out_features 中存在重复的特征，则抛出 ValueError 异常
        if len(out_features) != len(set(out_features)):
            raise ValueError(f"out_features must not contain any duplicates, got {out_features}")
        # 如果 out_features 不是按照 stage_names 中的顺序排列，则抛出 ValueError 异常
        if out_features != (sorted_feats := [feat for feat in stage_names if feat in out_features]):
            raise ValueError(
                f"out_features must be in the same order as stage_names, expected {sorted_feats} got {out_features}"
            )
    # 如果给定了 out_indices 参数
    if out_indices is not None:
        # 检查 out_indices 是否为列表或元组，否则引发异常
        if not isinstance(out_indices, (list, tuple)):
            raise ValueError(f"out_indices must be a list or tuple, got {type(out_indices)}")
        
        # 将负索引转换为其对应的正索引值：[-1,] -> [len(stage_names) - 1,]
        positive_indices = tuple(idx % len(stage_names) if idx < 0 else idx for idx in out_indices)
        
        # 检查所有正索引是否在有效范围内
        if any(idx for idx in positive_indices if idx not in range(len(stage_names))):
            raise ValueError(f"out_indices must be valid indices for stage_names {stage_names}, got {out_indices}")
        
        # 检查正索引列表是否含有重复值
        if len(positive_indices) != len(set(positive_indices)):
            msg = f"out_indices must not contain any duplicates, got {out_indices}"
            msg += f" (equivalent to {positive_indices}))" if positive_indices != out_indices else ""
            raise ValueError(msg)
        
        # 检查正索引是否按照 stage_names 中的顺序排列
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = tuple(idx for _, idx in sorted(zip(positive_indices, out_indices), key=lambda x: x[0]))
            raise ValueError(
                f"out_indices must be in the same order as stage_names, expected {sorted_negative} got {out_indices}"
            )

    # 如果同时给定了 out_features 和 out_indices 参数
    if out_features is not None and out_indices is not None:
        # 检查 out_features 和 out_indices 的长度是否一致
        if len(out_features) != len(out_indices):
            raise ValueError("out_features and out_indices should have the same length if both are set")
        
        # 检查 out_features 是否与 out_indices 对应到 stage_names 中的同一阶段
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError("out_features and out_indices should correspond to the same stages if both are set")
# 定义函数 _align_output_features_output_indices，用于根据 stage_names 对齐给定的 out_features 和 out_indices
def _align_output_features_output_indices(
    out_features: Optional[List[str]],
    out_indices: Optional[Union[List[int], Tuple[int]]],
    stage_names: List[str],
):
    """
    Finds the corresponding `out_features` and `out_indices` for the given `stage_names`.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: input `out_indices` and `out_features` are returned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    # 如果 out_indices 和 out_features 都未设置，则将它们设为最后一个 stage_names 的值
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    # 如果 out_indices 未设置但 out_features 设置了，则根据 out_features 设置 out_indices
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    # 如果 out_features 未设置但 out_indices 设置了，则根据 out_indices 设置 out_features
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    
    # 返回经过对齐处理后的 out_features 和 out_indices
    return out_features, out_indices


# 定义函数 get_aligned_output_features_output_indices，用于获取经过对齐处理后的 out_features 和 out_indices
def get_aligned_output_features_output_indices(
    out_features: Optional[List[str]],
    out_indices: Optional[Union[List[int], Tuple[int]]],
    stage_names: List[str],
) -> Tuple[List[str], List[int]]:
    """
    Get the `out_features` and `out_indices` so that they are aligned.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: they are verified to be aligned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    # 首先验证 out_features 和 out_indices 是否有效
    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    # 调用 _align_output_features_output_indices 函数，获取对齐后的 out_features 和 out_indices
    output_features, output_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # 验证对齐的输出特征和输出索引是否有效
    verify_out_features_out_indices(out_features=output_features, out_indices=output_indices, stage_names=stage_names)
    # 返回验证后的输出特征和输出索引
    return output_features, output_indices
class BackboneMixin:
    backbone_type: Optional[BackboneType] = None

    def _init_timm_backbone(self, config) -> None:
        """
        Initialize the backbone model from timm The backbone must already be loaded to self._backbone
        """
        # 检查 self._backbone 是否已经被加载，若未加载则抛出数值错误
        if getattr(self, "_backbone", None) is None:
            raise ValueError("self._backbone must be set before calling _init_timm_backbone")

        # 根据 self._backbone 的特征信息，获取阶段名称和特征数量
        self.stage_names = [stage["module"] for stage in self._backbone.feature_info.info]
        self.num_features = [stage["num_chs"] for stage in self._backbone.feature_info.info]
        out_indices = self._backbone.feature_info.out_indices
        out_features = self._backbone.feature_info.module_name()

        # 验证输出特征和输出索引的有效性
        verify_out_features_out_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self._out_features, self._out_indices = out_features, out_indices

    def _init_transformers_backbone(self, config) -> None:
        """
        Initialize the backbone model from transformers
        """
        # 获取配置中的阶段名称、输出特征和输出索引
        stage_names = getattr(config, "stage_names")
        out_features = getattr(config, "out_features", None)
        out_indices = getattr(config, "out_indices", None)

        # 根据配置中的信息，对齐输出特征和输出索引
        self.stage_names = stage_names
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=stage_names
        )
        # 每个阶段的通道数量，这在 transformer backbone 模型初始化中设置
        self.num_features = None

    def _init_backbone(self, config) -> None:
        """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
        # 将配置保存到 self.config
        self.config = config

        # 检查是否使用 timm backbone，如果是，则调用 _init_timm_backbone 方法初始化
        self.use_timm_backbone = getattr(config, "use_timm_backbone", False)
        self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS

        # 根据选择的 backbone 类型初始化对应的方法
        if self.backbone_type == BackboneType.TIMM:
            self._init_timm_backbone(config)
        elif self.backbone_type == BackboneType.TRANSFORMERS:
            self._init_transformers_backbone(config)
        else:
            raise ValueError(f"backbone_type {self.backbone_type} not supported.")

    @property
    def out_features(self):
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        设置 out_features 属性。这还会更新 out_indices 属性，使其与新的 out_features 匹配。
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        """
        获取 out_indices 属性的值。
        """
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        设置 out_indices 属性。这还会更新 out_features 属性，使其与新的 out_indices 匹配。
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    @property
    def out_feature_channels(self):
        """
        返回每个阶段的通道数。即使该阶段不在 out_features 列表中，当前的骨干网络也会输出通道数。
        """
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self):
        """
        返回与 out_features 列表中每个名称对应的通道数列表。
        """
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
        """
        使用前向方法的参数签名过滤 kwargs，并调用 self(*args, **filtered_kwargs)。
        """
        signature = dict(inspect.signature(self.forward).parameters)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
        return self(*args, **filtered_kwargs)

    def forward(
        self,
        pixel_values,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        抛出 NotImplementedError，要求派生类实现此方法。
        """
        raise NotImplementedError("This method should be implemented by the derived class.")

    def to_dict(self):
        """
        将此实例序列化为 Python 字典。重写自 `PretrainedConfig` 的默认 `to_dict()` 方法以包括 `out_features` 和 `out_indices` 属性。
        """
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output
class BackboneConfigMixin:
    """
    A Mixin to support handling the `out_features` and `out_indices` attributes for the backbone configurations.
    """

    @property
    def out_features(self):
        """
        Getter for `out_features` attribute.
        """
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        Setter for `out_features` attribute. Updates `out_indices` accordingly.
        
        Args:
            out_features (List[str]): List of output feature names.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        """
        Getter for `out_indices` attribute.
        """
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        Setter for `out_indices` attribute. Updates `out_features` accordingly.
        
        Args:
            out_indices (Union[Tuple[int], List[int]]): Tuple or list of output indices.
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.
        Overrides the default `to_dict()` from `PretrainedConfig` to include `out_features` and `out_indices`.
        
        Returns:
            dict: Serialized dictionary representation of the instance.
        """
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output


def load_backbone(config):
    """
    Loads the backbone model from a config object.
    
    If the config is from the backbone model itself, then we return a backbone model with randomly initialized
    weights.
    
    If the config is from the parent model of the backbone model itself, then we load the pretrained backbone weights
    if specified.
    
    Args:
        config: Configuration object that may contain `backbone_config`, `use_timm_backbone`, `use_pretrained_backbone`,
                `backbone`, and `backbone_kwargs` attributes.
    
    Raises:
        ValueError: If both `backbone_kwargs` and `backbone_config` are specified, or if both `backbone_config` and
                    `backbone` with `use_pretrained_backbone` are specified.
    """
    from transformers import AutoBackbone, AutoConfig

    backbone_config = getattr(config, "backbone_config", None)
    use_timm_backbone = getattr(config, "use_timm_backbone", None)
    use_pretrained_backbone = getattr(config, "use_pretrained_backbone", None)
    backbone_checkpoint = getattr(config, "backbone", None)
    backbone_kwargs = getattr(config, "backbone_kwargs", None)

    backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs

    if backbone_kwargs and backbone_config is not None:
        raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

    if backbone_config is not None and backbone_checkpoint is not None and use_pretrained_backbone is not None:
        raise ValueError("Cannot specify both config.backbone_config and config.backbone")
    # 如果以下任一参数被设置，则传入的配置来自具有主干模型的模型。
    if (
        backbone_config is None
        and use_timm_backbone is None
        and backbone_checkpoint is None
        and backbone_checkpoint is None
    ):
        # 返回根据配置创建的自动主干模型
        return AutoBackbone.from_config(config=config, **backbone_kwargs)

    # 父模型中具有主干的配置
    if use_timm_backbone:
        if backbone_checkpoint is None:
            # 如果 use_timm_backbone 为 True，但未设置 backbone_checkpoint，则抛出数值错误
            raise ValueError("config.backbone must be set if use_timm_backbone is True")
        # 由于 timm 主干最初添加到模型时，需要传入 use_pretrained_backbone
        # 以确定是否加载预训练权重。
        backbone = AutoBackbone.from_pretrained(
            backbone_checkpoint,
            use_timm_backbone=use_timm_backbone,
            use_pretrained_backbone=use_pretrained_backbone,
            **backbone_kwargs,
        )
    elif use_pretrained_backbone:
        if backbone_checkpoint is None:
            # 如果 use_pretrained_backbone 为 True，但未设置 backbone_checkpoint，则抛出数值错误
            raise ValueError("config.backbone must be set if use_pretrained_backbone is True")
        # 根据预训练的主干模型创建自动主干模型
        backbone = AutoBackbone.from_pretrained(backbone_checkpoint, **backbone_kwargs)
    else:
        if backbone_config is None and backbone_checkpoint is None:
            # 如果没有设置 backbone_config 和 backbone_checkpoint，则抛出数值错误
            raise ValueError("Either config.backbone_config or config.backbone must be set")
        if backbone_config is None:
            # 如果未设置 backbone_config，则从预训练模型加载配置
            backbone_config = AutoConfig.from_pretrained(backbone_checkpoint, **backbone_kwargs)
        # 根据配置创建自动主干模型
        backbone = AutoBackbone.from_config(config=backbone_config)
    # 返回所选的主干模型
    return backbone
```