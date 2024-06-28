# `.\onnx\utils.py`

```py
# 导入所需的模块和类型声明
from ctypes import c_float, sizeof
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

# 如果是类型检查，导入相关的预处理模块
if TYPE_CHECKING:
    from .. import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # tests_ignore

# 枚举类型，定义参数的数据格式
class ParameterFormat(Enum):
    Float = c_float

    @property
    def size(self) -> int:
        """
        返回该数据类型所需的字节数

        Returns:
            Integer > 0
        """
        return sizeof(self.value)

# 计算有效轴维度的函数
def compute_effective_axis_dimension(dimension: int, fixed_dimension: int, num_token_to_add: int = 0) -> int:
    """

    Args:
        dimension: 当前轴的维度
        fixed_dimension: 固定的轴维度
        num_token_to_add: 需要添加的标记数量

    Returns:
        计算后的有效轴维度
    """
    # 如果维度 <= 0，使用固定维度
    if dimension <= 0:
        dimension = fixed_dimension

    dimension -= num_token_to_add
    return dimension

# 计算序列化参数大小的函数
def compute_serialized_parameters_size(num_parameters: int, dtype: ParameterFormat) -> int:
    """
    计算在给定存储格式中序列化模型时所有参数占用的大小

    Args:
        num_parameters: 需要保存的参数数量
        dtype: 每个参数保存的数据格式

    Returns:
        所有参数保存时占用的字节数
    """
    return num_parameters * dtype.size

# 获取预处理器的函数
def get_preprocessor(model_name: str) -> Optional[Union["AutoTokenizer", "AutoFeatureExtractor", "AutoProcessor"]]:
    """
    获取适用于 `model_name` 的预处理器（分词器、特征提取器或处理器）。

    Args:
        model_name (`str`): 模型名称，用于加载预处理器。

    Returns:
        `Optional[Union[AutoTokenizer, AutoFeatureExtractor, AutoProcessor]]`:
            如果找到处理器，则返回处理器。如果存在分词器或特征提取器，则返回分词器或特征提取器。如果同时存在分词器和特征提取器，则会引发错误。如果找不到预处理器，则返回 `None`。
    """
    # 避免循环导入问题，仅在此处导入
    from .. import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # tests_ignore

    try:
        return AutoProcessor.from_pretrained(model_name)
    # 处理可能发生的异常：ValueError, OSError, KeyError
    except (ValueError, OSError, KeyError):
        # 初始化 tokenizer 和 feature_extractor 变量为 None
        tokenizer, feature_extractor = None, None
        
        # 尝试根据模型名称加载 AutoTokenizer，可能会抛出 OSError 或 KeyError 异常
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except (OSError, KeyError):
            pass
        
        # 尝试根据模型名称加载 AutoFeatureExtractor，可能会抛出 OSError 或 KeyError 异常
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        except (OSError, KeyError):
            pass

        # 检查 tokenizer 和 feature_extractor 是否都不为 None
        if tokenizer is not None and feature_extractor is not None:
            # 如果两者都不为 None，则抛出 ValueError 异常，指示找到了同时存在的 tokenizer 和 feature extractor
            raise ValueError(
                f"Couldn't auto-detect preprocessor for {model_name}. Found both a tokenizer and a feature extractor."
            )
        elif tokenizer is None and feature_extractor is None:
            # 如果两者都为 None，则返回 None，表示未能找到有效的预处理器
            return None
        elif tokenizer is not None:
            # 如果只有 tokenizer 不为 None，则返回 tokenizer
            return tokenizer
        else:
            # 如果只有 feature_extractor 不为 None，则返回 feature_extractor
            return feature_extractor
```