# `.\transformers\onnx\utils.py`

```
# 导入所需模块和类
from ctypes import c_float, sizeof
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

# 如果 TYPE_CHECKING 为真，则导入测试所需的模块和类，否则忽略
if TYPE_CHECKING:
    from .. import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # tests_ignore

# 定义枚举类 ParameterFormat，表示参数的数据格式
class ParameterFormat(Enum):
    Float = c_float

    @property
    def size(self) -> int:
        """
        Number of byte required for this data type

        Returns:
            Integer > 0
        """
        # 返回当前数据类型所占用的字节数
        return sizeof(self.value)

# 定义函数 compute_effective_axis_dimension，计算有效的轴维度
def compute_effective_axis_dimension(dimension: int, fixed_dimension: int, num_token_to_add: int = 0) -> int:
    """

    Args:
        dimension: 当前轴的维度
        fixed_dimension: 固定的轴维度
        num_token_to_add: 要添加的标记数量

    Returns:
        有效的轴维度

    """
    # 如果维度小于等于 0，则使用固定的轴维度
    if dimension <= 0:
        dimension = fixed_dimension

    # 减去要添加的标记数量
    dimension -= num_token_to_add
    return dimension

# 定义函数 compute_serialized_parameters_size，计算序列化参数的大小
def compute_serialized_parameters_size(num_parameters: int, dtype: ParameterFormat) -> int:
    """
    Compute the size taken by all the parameters in the given the storage format when serializing the model

    Args:
        num_parameters: Number of parameters to be saved
        dtype: The data format each parameter will be saved

    Returns:
        Size (in byte) taken to save all the parameters
    """
    # 计算保存所有参数所需的大小（字节数）
    return num_parameters * dtype.size

# 定义函数 get_preprocessor，获取给定模型名称的预处理器
def get_preprocessor(model_name: str) -> Optional[Union["AutoTokenizer", "AutoFeatureExtractor", "AutoProcessor"]]:
    """
    Gets a preprocessor (tokenizer, feature extractor or processor) that is available for `model_name`.

    Args:
        model_name (`str`): Name of the model for which a preprocessor are loaded.

    Returns:
        `Optional[Union[AutoTokenizer, AutoFeatureExtractor, AutoProcessor]]`:
            If a processor is found, it is returned. Otherwise, if a tokenizer or a feature extractor exists, it is
            returned. If both a tokenizer and a feature extractor exist, an error is raised. The function returns
            `None` if no preprocessor is found.
    """
    # 避免循环导入问题，仅在此处导入所需模块和类
    from .. import AutoFeatureExtractor, AutoProcessor, AutoTokenizer  # tests_ignore

    try:
        # 从预训练模型名称加载预处理器
        return AutoProcessor.from_pretrained(model_name)
    except (ValueError, OSError, KeyError):
    # 捕获可能发生的 Value Error、Os Error、Key Error 异常
        tokenizer, feature_extractor = None, None
        # 初始化 tokenizer 和 feature_extractor 变量为 None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # 尝试根据给定的模型名称加载自动化的令牌化器
        except (OSError, KeyError):
            pass
            # 如果加载过程中发生 OSError 或 KeyError 异常，则跳过错误
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            # 尝试根据给定的模型名称加载自动化的特征提取器
        except (OSError, KeyError):
            pass
            # 如果加载过程中发生 OSError 或 KeyError 异常，则跳过错误

        if tokenizer is not None and feature_extractor is not None:
        # 如果 tokenizer 和 feature_extractor 都不为空
            raise ValueError(
                f"Couldn't auto-detect preprocessor for {model_name}. Found both a tokenizer and a feature extractor."
            )
            # 抛出 Value Error 异常，提示检测到了 tokenizer 和 feature_extractor，无法自动检测预处理器
        elif tokenizer is None and feature_extractor is None:
        # 如果 tokenizer 和 feature_extractor 都为空
            return None
            # 返回 None
        elif tokenizer is not None:
        # 如果 tokenizer 不为空
            return tokenizer
            # 返回 tokenizer
        else:
            return feature_extractor
            # 返回 feature_extractor
```