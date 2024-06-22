# `.\transformers\utils\backbone_utils.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

""" 用于由主干和其组件使用的工具集合。"""

# 导入必要的库
import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union

# 定义枚举类型 BackboneType
class BackboneType(enum.Enum):
    TIMM = "timm"
    TRANSFORMERS = "transformers"

# 验证 out_features 和 out_indices 对于给定的 stage_names 是否有效
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
        # 如果 out_features 中有任何元素不在 stage_names 中，则抛出 ValueError 异常
        if any(feat not in stage_names for feat in out_features):
            raise ValueError(f"out_features must be a subset of stage_names: {stage_names} got {out_features}")
        # 如果 out_features 中有重复元素，则抛出 ValueError 异常
        if len(out_features) != len(set(out_features)):
            raise ValueError(f"out_features must not contain any duplicates, got {out_features}")
        # 如果 out_features 不按照 stage_names 的顺序排列，则抛出 ValueError 异常
        if out_features != (sorted_feats := [feat for feat in stage_names if feat in out_features]):
            raise ValueError(
                f"out_features must be in the same order as stage_names, expected {sorted_feats} got {out_features}"
            )
    # 检查是否提供了输出索引
    if out_indices is not None:
        # 检查输出索引是否为列表或元组类型，如果不是则引发数值错误
        if not isinstance(out_indices, (list, tuple)):
            raise ValueError(f"out_indices must be a list or tuple, got {type(out_indices)}")
        # 将负索引转换为其正数等价索引：[-1,] -> [len(stage_names) - 1,]
        positive_indices = tuple(idx % len(stage_names) if idx < 0 else idx for idx in out_indices)
        # 检查正数索引是否在有效范围内，如果不是则引发数值错误
        if any(idx for idx in positive_indices if idx not in range(len(stage_names))):
            raise ValueError(f"out_indices must be valid indices for stage_names {stage_names}, got {out_indices}")
        # 检查正数索引是否包含重复值，如果包含则引发数值错误
        if len(positive_indices) != len(set(positive_indices)):
            msg = f"out_indices must not contain any duplicates, got {out_indices}"
            msg += f"(equivalent to {positive_indices}))" if positive_indices != out_indices else ""
            raise ValueError(msg)
        # 检查正数索引是否按照 stage_names 的顺序排列，如果不是则引发数值错误
        if positive_indices != tuple(sorted(positive_indices)):
            sorted_negative = tuple(idx for _, idx in sorted(zip(positive_indices, out_indices), key=lambda x: x[0]))
            raise ValueError(
                f"out_indices must be in the same order as stage_names, expected {sorted_negative} got {out_indices}"
            )

    # 如果同时提供了输出特征和输出索引
    if out_features is not None and out_indices is not None:
        # 检查输出特征和输出索引的长度是否相等，如果不相等则引发数值错误
        if len(out_features) != len(out_indices):
            raise ValueError("out_features and out_indices should have the same length if both are set")
        # 检查输出特征和输出索引是否对应相同的阶段，如果不对应则引发数值错误
        if out_features != [stage_names[idx] for idx in out_indices]:
            raise ValueError("out_features and out_indices should correspond to the same stages if both are set")
def _align_output_features_output_indices(
    out_features: Optional[List[str]],  # 定义可选参数 out_features，用于存储要输出的特征名称列表
    out_indices: Optional[Union[List[int], Tuple[int]]],  # 定义可选参数 out_indices，用于存储要输出的特征索引列表
    stage_names: List[str],  # 定义参数 stage_names，用于存储骨干网络的阶段名称列表
):
    """
    找到给定 stage_names 对应的 out_features 和 out_indices。

    逻辑如下：
        - out_features 未设置，out_indices 设置：将 out_features 设置为与 out_indices 对应的 out_features。
        - out_indices 未设置，out_features 设置：将 out_indices 设置为与 out_features 对应的 out_indices。
        - out_indices 和 out_features 都未设置：将 out_indices 和 out_features 设置为最后一个阶段。
        - out_indices 和 out_features 都设置：返回输入的 out_indices 和 out_features。

    Args:
        out_features (`List[str]`): 骨干网络要输出的特征名称列表。
        out_indices (`List[int]` or `Tuple[int]`): 骨干网络要输出的特征索引列表。
        stage_names (`List[str]`): 骨干网络的阶段名称列表。
    """
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]  # 如果 out_indices 和 out_features 都未设置，则将 out_indices 设置为最后一个阶段的索引
        out_features = [stage_names[-1]]  # 如果 out_indices 和 out_features 都未设置，则将 out_features 设置为最后一个阶段的名称
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]  # 如果 out_indices 未设置，out_features 设置，则将 out_indices 设置为 out_features 对应的索引
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]  # 如果 out_features 未设置，out_indices 设置，则将 out_features 设置为 out_indices 对应的名称
    return out_features, out_indices  # 返回对齐后的 out_features 和 out_indices


def get_aligned_output_features_output_indices(
    out_features: Optional[List[str]],  # 定义可选参数 out_features，用于存储要输出的特征名称列表
    out_indices: Optional[Union[List[int], Tuple[int]]],  # 定义可选参数 out_indices，用于存储要输出的特征索引列表
    stage_names: List[str],  # 定义参数 stage_names，用于存储骨干网络的阶段名称列表
) -> Tuple[List[str], List[int]]:
    """
    获取对齐后的 out_features 和 out_indices。

    逻辑如下：
        - out_features 未设置，out_indices 设置：将 out_features 设置为与 out_indices 对应的 out_features。
        - out_indices 未设置，out_features 设置：将 out_indices 设置为与 out_features 对应的 out_indices。
        - out_indices 和 out_features 都未设置：将 out_indices 和 out_features 设置为最后一个阶段。
        - out_indices 和 out_features 都设置：验证它们是否对齐。

    Args:
        out_features (`List[str]`): 骨干网络要输出的特征名称列表。
        out_indices (`List[int]` or `Tuple[int]`): 骨干网络要输出的特征索引列表。
        stage_names (`List[str]`): 骨干网络的阶段名称列表。
    """
    # 首先验证 out_features 和 out_indices 是否有效
    verify_out_features_out_indices(out_features=out_features, out_indices=out_indices, stage_names=stage_names)
    output_features, output_indices = _align_output_features_output_indices(
        out_features=out_features, out_indices=out_indices, stage_names=stage_names
    )
    # 验证对齐的输出特征和输出索引是否有效
    verify_out_features_out_indices(out_features=output_features, out_indices=output_indices, stage_names=stage_names)
    # 返回输出特征和输出索引
    return output_features, output_indices
class BackboneMixin:
    backbone_type: Optional[BackboneType] = None

    def _init_timm_backbone(self, config) -> None:
        """
        Initialize the backbone model from timm The backbone must already be loaded to self._backbone
        """
        # 检查 self._backbone 是否已经被设置
        if getattr(self, "_backbone", None) is None:
            raise ValueError("self._backbone must be set before calling _init_timm_backbone")

        # 获取每个阶段的名称和通道数
        self.stage_names = [stage["module"] for stage in self._backbone.feature_info.info]
        self.num_features = [stage["num_chs"] for stage in self._backbone.feature_info.info]
        out_indices = self._backbone.feature_info.out_indices
        out_features = self._backbone.feature_info.module_name()

        # 验证输出索引和输出特征是否有效
        verify_out_features_out_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self._out_features, self._out_indices = out_features, out_indices

    def _init_transformers_backbone(self, config) -> None:
        stage_names = getattr(config, "stage_names")
        out_features = getattr(config, "out_features", None)
        out_indices = getattr(config, "out_indices", None)

        self.stage_names = stage_names
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=stage_names
        )
        # 每个阶段的通道数。这在变换器骨干模型初始化中设置
        self.num_features = None

    def _init_backbone(self, config) -> None:
        """
        Method to initialize the backbone. This method is called by the constructor of the base class after the
        pretrained model weights have been loaded.
        """
        # 设置配置
        self.config = config

        # 检查是否使用 timm 骨干
        self.use_timm_backbone = getattr(config, "use_timm_backbone", False)
        self.backbone_type = BackboneType.TIMM if self.use_timm_backbone else BackboneType.TRANSFORMERS

        # 根据骨干类型初始化骨干
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
        设置 out_features 属性。这也会更新 out_indices 属性以匹配新的 out_features。
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        设置 out_indices 属性。这也会更新 out_features 属性以匹配新的 out_indices。
        """
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    @property
    def out_feature_channels(self):
        # 当前的骨干网络将为每个阶段输出通道数，即使该阶段不在 out_features 列表中。
        return {stage: self.num_features[i] for i, stage in enumerate(self.stage_names)}

    @property
    def channels(self):
        return [self.out_feature_channels[name] for name in self.out_features]

    def forward_with_filtered_kwargs(self, *args, **kwargs):
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
        raise NotImplementedError("This method should be implemented by the derived class.")

    def to_dict(self):
        """
        将此实例序列化为 Python 字典。覆盖默认的 `to_dict()` 方法，以包括 `out_features` 和 `out_indices` 属性。
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
        # 返回_out_features属性的值
        return self._out_features

    @out_features.setter
    def out_features(self, out_features: List[str]):
        """
        Set the out_features attribute. This will also update the out_indices attribute to match the new out_features.
        """
        # 设置out_features属性，并更新out_indices属性以匹配新的out_features
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=None, stage_names=self.stage_names
        )

    @property
    def out_indices(self):
        # 返回_out_indices属性的值
        return self._out_indices

    @out_indices.setter
    def out_indices(self, out_indices: Union[Tuple[int], List[int]]):
        """
        Set the out_indices attribute. This will also update the out_features attribute to match the new out_indices.
        """
        # 设置out_indices属性，并更新out_features属性以匹配新的out_indices
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=None, out_indices=out_indices, stage_names=self.stage_names
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default `to_dict()` from `PretrainedConfig` to
        include the `out_features` and `out_indices` attributes.
        """
        # 将实例序列化为Python字典。覆盖`PretrainedConfig`中默认的`to_dict()`，包括`out_features`和`out_indices`属性
        output = super().to_dict()
        output["out_features"] = output.pop("_out_features")
        output["out_indices"] = output.pop("_out_indices")
        return output
```