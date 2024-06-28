# `.\modeling_flax_utils.py`

```
# coding=utf-8
# 代码文件声明使用 UTF-8 编码

# 导入标准库和第三方库
import gc  # 垃圾回收模块
import json  # JSON 数据格式处理模块
import os  # 系统操作模块
import re  # 正则表达式模块
import warnings  # 警告模块
from functools import partial  # 偏函数功能
from pickle import UnpicklingError  # 反序列化错误异常
from typing import Any, Dict, Optional, Set, Tuple, Union  # 类型提示模块

# 导入 Flax 和 JAX 库
import flax.linen as nn  # Flax 的线性模块
import jax  # JAX 数值计算库
import jax.numpy as jnp  # JAX 的 NumPy 接口
import msgpack.exceptions  # MsgPack 序列化异常模块
from flax.core.frozen_dict import FrozenDict, unfreeze  # 冻结字典和解冻功能
from flax.serialization import from_bytes, to_bytes  # 对象序列化和反序列化
from flax.traverse_util import flatten_dict, unflatten_dict  # 字典扁平化和反扁平化
from jax.random import PRNGKey  # JAX 随机数生成模块

# 导入本地的配置和工具函数
from .configuration_utils import PretrainedConfig  # 预训练模型配置类
from .dynamic_module_utils import custom_object_save  # 自定义对象保存函数
from .generation import FlaxGenerationMixin, GenerationConfig  # 生成相关模块
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict  # 加载 PyTorch 检查点到 Flax 状态字典
from .utils import (
    FLAX_WEIGHTS_INDEX_NAME, FLAX_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME,  # 各种常量定义
    WEIGHTS_INDEX_NAME, WEIGHTS_NAME,  # 权重文件名和索引名
    PushToHubMixin,  # 推送到 Hub 的混合类
    add_code_sample_docstrings,  # 添加代码示例文档字符串
    add_start_docstrings_to_model_forward,  # 添加模型前向方法的文档字符串
    cached_file,  # 缓存文件函数
    copy_func,  # 复制函数对象
    download_url,  # 下载 URL 资源函数
    has_file,  # 检查文件是否存在函数
    is_offline_mode,  # 检查是否处于离线模式函数
    is_remote_url,  # 检查 URL 是否远程地址函数
    logging,  # 日志模块
    replace_return_docstrings,  # 替换返回值的文档字符串函数
)
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files  # Hub 工具函数
from .utils.import_utils import is_safetensors_available  # 检查是否安装安全张量库


if is_safetensors_available():
    from safetensors import safe_open  # 安全打开文件函数
    from safetensors.flax import load_file as safe_load_file  # 安全加载文件函数
    from safetensors.flax import save_file as safe_save_file  # 安全保存文件函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def quick_gelu(x):
    """
    快速 GELU 激活函数的定义，使用 JAX 实现
    """
    return x * jax.nn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),  # 使用 Flax 提供的精确 GELU 激活函数
    "relu": nn.relu,  # 使用 Flax 提供的 ReLU 激活函数
    "silu": nn.swish,  # 使用 Flax 提供的 SiLU（Swish）激活函数
    "swish": nn.swish,  # 使用 Flax 提供的 Swish 激活函数
    "gelu_new": partial(nn.gelu, approximate=True),  # 使用 Flax 提供的近似 GELU 激活函数
    "quick_gelu": quick_gelu,  # 使用定义的快速 GELU 激活函数
    "gelu_pytorch_tanh": partial(nn.gelu, approximate=True),  # 使用 Flax 提供的近似 GELU 激活函数
}


def dtype_byte_size(dtype):
    """
    根据数据类型 `dtype` 返回一个参数占用的字节数。例如：
    ```py
    >>> dtype_byte_size(np.float32)
    4
    ```
    """
    if dtype == bool:
        return 1 / 8  # 布尔类型占用 1 位，即 1/8 字节
    bit_search = re.search(r"[^\d](\d+)$", dtype.name)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")  # 若数据类型不合法，抛出异常
    bit_size = int(bit_search.groups()[0])  # 获取数据类型的位数大小
    return bit_size // 8  # 返回字节数


def flax_shard_checkpoint(params, max_shard_size="10GB"):
    """
    将参数 `params` 拆分为多个小的检查点文件，以便于存储和传输。
    """
    # 将模型状态字典拆分为子检查点，使得每个子检查点的最终大小不超过给定的大小限制。
    # 子检查点的确定是通过按照状态字典的键的顺序迭代进行的，因此不会优化使每个子检查点尽可能接近传递的最大大小。
    # 例如，如果限制是10GB，并且我们有大小为[6GB, 6GB, 2GB, 6GB, 2GB, 2GB]的权重，则它们将被分割为[6GB]、[6+2GB]、[6+2+2GB]，而不是[6+2+2GB]、[6+2GB]、[6GB]。
    # <Tip warning={true}>
    # 如果模型中的某个权重大于`max_shard_size`，它将单独存在于其自己的子检查点中，其大小将大于`max_shard_size`。
    # </Tip>
    
    Args:
        params (`Union[Dict, FrozenDict]`): 模型参数的`PyTree`表示。
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            每个子检查点的最大大小。如果表示为字符串，则需要是数字后跟单位（例如`"5MB"`）。
    """
    # 将`max_shard_size`转换为整数表示
    max_shard_size = convert_file_size_to_int(max_shard_size)
    
    # 初始化用于存储分片状态字典的列表
    sharded_state_dicts = []
    # 当前分块的字典
    current_block = {}
    # 当前分块的大小
    current_block_size = 0
    # 总大小
    total_size = 0
    
    # 将参数展平为键值对
    weights = flatten_dict(params, sep="/")
    for item in weights:
        # 计算权重项的大小
        weight_size = weights[item].size * dtype_byte_size(weights[item].dtype)
    
        # 如果当前分块加上当前权重项的大小超过了最大分块大小，进行分块
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0
    
        # 将权重项添加到当前分块中
        current_block[item] = weights[item]
        current_block_size += weight_size
        total_size += weight_size
    
    # 添加最后一个分块
    sharded_state_dicts.append(current_block)
    
    # 如果只有一个分片，直接返回
    if len(sharded_state_dicts) == 1:
        return {FLAX_WEIGHTS_NAME: sharded_state_dicts[0]}, None
    
    # 否则，构建权重映射和分片文件名
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = FLAX_WEIGHTS_NAME.replace(".msgpack", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.msgpack")
        shards[shard_file] = shard
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file
    
    # 添加元数据
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index
# FlaxPreTrainedModel 类，继承自 PushToHubMixin 和 FlaxGenerationMixin
class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
    # 所有模型的基类。
    r"""
    Base class for all models.

    [`FlaxPreTrainedModel`] takes care of storing the configuration of the models and handles methods for loading,
    downloading and saving models.

    Class attributes (overridden by derived classes):

        - **config_class** ([`PretrainedConfig`]) -- A subclass of [`PretrainedConfig`] to use as configuration class
          for this model architecture.
        - **base_model_prefix** (`str`) -- A string indicating the attribute associated to the base model in derived
          classes of the same architecture adding modules on top of the base model.
        - **main_input_name** (`str`) -- The name of the principal input to the model (often `input_ids` for NLP
          models, `pixel_values` for vision models and `input_values` for speech models).
    """

    # 模型配置类，默认为 None
    config_class = None
    # 基模型前缀，默认为空字符串
    base_model_prefix = ""
    # 主要输入名称，默认为 "input_ids"
    main_input_name = "input_ids"
    # 自动类
    _auto_class = None
    # 缺失的键集合
    _missing_keys = set()

    # 模型初始化方法
    def __init__(
        self,
        config: PretrainedConfig,
        module: nn.Module,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        # 如果 config 为 None，则抛出 ValueError
        if config is None:
            raise ValueError("config cannot be None")

        # 如果 module 为 None，则抛出 ValueError
        if module is None:
            raise ValueError("module cannot be None")

        # 下面的属性用于在派生类中作为类型化属性暴露，因此为私有属性。
        # 存储配置对象
        self._config = config
        # 存储模块对象
        self._module = module

        # 下面的属性为每个派生类通用的公共属性。
        # 初始化随机数生成器的 key
        self.key = PRNGKey(seed)
        # 数据类型，默认为 jnp.float32
        self.dtype = dtype
        # 输入形状，默认为 (1, 1)
        self.input_shape = input_shape
        # 生成配置对象，基于模型配置生成，如果可以生成的话
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

        # 标志模型是否已初始化
        self._is_initialized = _do_init

        # 如果 _do_init 为 True，则随机初始化参数
        if _do_init:
            # 随机初始化模型参数
            random_params = self.init_weights(self.key, input_shape)
            # 计算参数的形状树
            params_shape_tree = jax.eval_shape(lambda params: params, random_params)
        else:
            # 如果 _do_init 为 False，则部分初始化模型参数
            init_fn = partial(self.init_weights, input_shape=input_shape)
            params_shape_tree = jax.eval_shape(init_fn, self.key)

            # 日志记录，提示模型权重未初始化
            logger.info(
                "Model weights are not initialized as `_do_init` is set to `False`. "
                f"Make sure to call `{self.__class__.__name__}.init_weights` manually to initialize the weights."
            )

        # 存储参数形状树
        self._params_shape_tree = params_shape_tree

        # 将必需参数保存为集合
        self._required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())

        # 如果 _do_init 为 True，则设置模型参数
        if _do_init:
            self.params = random_params
    # 定义一个抽象方法，用于初始化模型的权重。子类必须实现这个方法。
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> Dict:
        raise NotImplementedError(f"init method has to be implemented for {self}")

    # 定义一个抽象方法，用于启用梯度检查点功能。子类必须实现这个方法。
    def enable_gradient_checkpointing(self):
        raise NotImplementedError(f"gradient checkpointing method has to be implemented for {self}")

    # 类方法，用于根据给定的配置和其他参数创建类的实例。
    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    # 返回字符串标识，指示这是一个 Flax 模型。
    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a Flax model.
        """
        return "flax"

    # 返回模型的配置信息。
    @property
    def config(self) -> PretrainedConfig:
        return self._config

    # 返回模型的内部模块。
    @property
    def module(self) -> nn.Module:
        return self._module

    # 返回模型的参数，可以是普通字典或者冻结字典。
    @property
    def params(self) -> Union[Dict, FrozenDict]:
        if not self._is_initialized:
            raise ValueError(
                "`params` cannot be accessed from model when the model is created with `_do_init=False`. "
                "You must call `init_weights` manually and store the params outside of the model and "
                "pass it explicitly where needed."
            )
        return self._params

    # 返回模型所需的参数集合。
    @property
    def required_params(self) -> Set:
        return self._required_params

    # 返回模型参数的形状树。
    @property
    def params_shape_tree(self) -> Dict:
        return self._params_shape_tree

    # 设置模型的参数，如果模型未初始化则抛出异常。
    @params.setter
    def params(self, params: Union[Dict, FrozenDict]):
        # 如果模型未初始化，则不设置参数。
        if not self._is_initialized:
            raise ValueError(
                "`params` cannot be set from model when the model is created with `_do_init=False`. "
                "You store the params outside of the model."
            )

        # 如果参数是冻结字典，则解冻成普通字典。
        if isinstance(params, FrozenDict):
            params = unfreeze(params)
        
        # 检查参数是否包含所有必需的参数键。
        param_keys = set(flatten_dict(params).keys())
        if len(self.required_params - param_keys) > 0:
            raise ValueError(
                "Some parameters are missing. Make sure that `params` include the following "
                f"parameters {self.required_params - param_keys}"
            )
        
        # 设置模型的参数。
        self._params = params
    def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any = None) -> Any:
        """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """

        # 从 https://github.com/deepmind/jmp/blob/3a8318abc3292be38582794dbf7b094e6583b192/jmp/_src/policy.py#L27 中借用
        # 定义条件转换函数，用于将参数中的浮点值转换为指定的 dtype
        def conditional_cast(param):
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                param = param.astype(dtype)
            return param

        # 如果 mask 为 None，则直接对 params 应用 tree_map 转换
        if mask is None:
            return jax.tree_util.tree_map(conditional_cast, params)

        # 将 params 展平为字典
        flat_params = flatten_dict(params)
        # 将 mask 也展平并获取其结构
        flat_mask, _ = jax.tree_util.tree_flatten(mask)

        # 遍历展平后的 mask 和 params 的键值对，并根据 mask 的值进行条件转换
        for masked, key in zip(flat_mask, sorted(flat_params.keys())):
            if masked:
                flat_params[key] = conditional_cast(flat_params[key])

        # 返回转换后的 params 的非展平版本
        return unflatten_dict(flat_params)

    def to_bf16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `params` to `jax.numpy.bfloat16`. This returns a new `params` tree and does not cast
        the `params` in place.

        This method can be used on TPU to explicitly convert the model parameters to bfloat16 precision to do full
        half-precision training or to save weights in bfloat16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip.

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # load model
        >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> model.params = model.to_bf16(model.params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
        >>> flat_params = traverse_util.flatten_dict(model.params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> model.params = model.to_bf16(model.params, mask)
        ```
        """
        return self._cast_floating_to(params, jnp.bfloat16, mask)
    def to_fp32(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `parmas` to `jax.numpy.float32`. This method can be used to explicitly convert the
        model parameters to fp32 precision. This returns a new `params` tree and does not cast the `params` in place.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # Download model and configuration from huggingface.co
        >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
        >>> # By default, the model params will be in fp32, to illustrate the use of this method,
        >>> # we'll first cast to fp16 and back to fp32
        >>> model.params = model.to_f16(model.params)
        >>> # now cast back to fp32
        >>> model.params = model.to_fp32(model.params)
        ```
        
        # 使用 jax 库中的 numpy 模块将浮点型参数 `params` 转换为单精度浮点数（float32）
        return self._cast_floating_to(params, jnp.float32, mask)
    # 将浮点数参数 `params` 转换为 `jax.numpy.float16` 类型。返回一个新的 `params` 树，不会原地修改 `params`。
    #
    # 在 GPU 上可以使用此方法显式地将模型参数转换为 float16 精度，以进行全半精度训练，或者将权重保存为 float16 以节省内存并提高速度。
    #
    # 参数:
    #     params (`Union[Dict, FrozenDict]`):
    #         模型参数的 PyTree 结构。
    #     mask (`Union[Dict, FrozenDict]`, 可选):
    #         与 `params` 结构相同的 PyTree。叶子节点应为布尔值，`True` 表示要转换的参数，`False` 表示要跳过的参数。
    #
    # 示例:
    #
    # ```python
    # >>> from transformers import FlaxBertModel
    # >>>
    # >>> # 加载模型
    # >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
    # >>>
    # >>> # 默认情况下，模型参数将是 fp32 类型，要将其转换为 float16 类型
    # >>> model.params = model.to_fp16(model.params)
    # >>>
    # >>> # 如果不想转换某些参数（例如层归一化的偏置和缩放）
    # >>> # 则按以下方式传递 mask
    # >>> from flax import traverse_util
    # >>>
    # >>> model = FlaxBertModel.from_pretrained("google-bert/bert-base-cased")
    # >>> flat_params = traverse_util.flatten_dict(model.params)
    # >>> mask = {
    # ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
    # ...     for path in flat_params
    # ... }
    # >>> mask = traverse_util.unflatten_dict(mask)
    # >>> model.params = model.to_fp16(model.params, mask)
    # ```
    def to_fp16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        # 调用内部方法 `_cast_floating_to` 将 `params` 中的浮点数类型转换为 `jnp.float16` 类型
        return self._cast_floating_to(params, jnp.float16, mask)
    # 定义一个类方法，用于加载 Flax 模型的权重数据
    def load_flax_weights(cls, resolved_archive_file):
        try:
            # 如果文件名以 ".safetensors" 结尾，使用 safe_load_file 加载状态
            if resolved_archive_file.endswith(".safetensors"):
                state = safe_load_file(resolved_archive_file)
                # 使用特定分隔符将状态字典展开
                state = unflatten_dict(state, sep=".")
            else:
                # 否则，使用二进制方式读取文件并将其反序列化为对象状态
                with open(resolved_archive_file, "rb") as state_f:
                    state = from_bytes(cls, state_f.read())
        # 捕获反序列化过程可能出现的异常
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                # 尝试以文本模式打开文件，检查其内容以确定错误类型
                with open(resolved_archive_file) as f:
                    # 如果文件内容以 "version" 开头，可能是由于缺少 git-lfs 导致的错误
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        # 否则，抛出 ValueError 并将原始异常作为其原因
                        raise ValueError from e
            # 捕获可能的解码或数值错误
            except (UnicodeDecodeError, ValueError):
                # 抛出环境错误，指示无法将文件转换为 Flax 可反序列化对象
                raise EnvironmentError(f"Unable to convert {resolved_archive_file} to Flax deserializable object. ")

        # 返回加载的状态对象
        return state

    @classmethod
    def load_flax_sharded_weights(cls, shard_files):
        """
        This is the same as [`flax.serialization.from_bytes`](https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.

        This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
        loaded in the model.

        Args:
            shard_files (`List[str]`):
                The list of shard files to load.

        Returns:
            `Dict`: A nested dictionary of the model parameters, in the expected format for flax models : `{'model':
            {'params': {'...'}}}`.
        """

        # Load the index
        state_sharded_dict = {}

        for shard_file in shard_files:
            # load using msgpack utils
            try:
                with open(shard_file, "rb") as state_f:
                    state = from_bytes(cls, state_f.read())
            except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                # Handle specific error cases
                with open(shard_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                # Raise an environment error if conversion fails
                raise EnvironmentError(f"Unable to convert {shard_file} to Flax deserializable object. ")

            # Flatten the state dictionary using '/' separator
            state = flatten_dict(state, sep="/")
            # Update the main dictionary with the flattened state
            state_sharded_dict.update(state)
            # Clean up the `state` variable from memory
            del state
            # Perform garbage collection to free up memory
            gc.collect()

        # Unflatten the state_sharded_dict to match the format of model.params
        return unflatten_dict(state_sharded_dict, sep="/")

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternatively, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            # If both conditions are met, return False indicating generation capability is not supported
            return False
        # If not, return True indicating generation capability is supported
        return True
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],  # 接受预训练模型名称或路径作为输入参数
        dtype: jnp.dtype = jnp.float32,  # 指定数据类型，默认为 jnp.float32
        *model_args,  # 其余位置参数
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,  # 预训练配置对象或其路径，可选参数
        cache_dir: Optional[Union[str, os.PathLike]] = None,  # 缓存目录的路径，可选参数
        ignore_mismatched_sizes: bool = False,  # 是否忽略大小不匹配的情况，默认为 False
        force_download: bool = False,  # 是否强制下载，默认为 False
        local_files_only: bool = False,  # 是否仅使用本地文件，默认为 False
        token: Optional[Union[str, bool]] = None,  # token 用于验证，可选参数
        revision: str = "main",  # 版本号，默认为 "main"
        **kwargs,  # 其余关键字参数
    ):
        """
        从预训练模型加载模型参数和配置。

        <Tip warning={true}>
        当前 API 处于实验阶段，未来版本可能会有一些轻微的更改。
        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                预训练模型的名称或路径。
            dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
                指定加载参数时使用的数据类型，默认为 jnp.float32。
            *model_args:
                其余位置参数，传递给具体模型加载函数。
            config (`PretrainedConfig`, `str`, `os.PathLike`, *optional*, defaults to `None`):
                预训练模型的配置对象或其路径，可选参数。
            cache_dir (`str` or `os.PathLike`, *optional*, defaults to `None`):
                缓存目录的路径，可选参数。
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                是否忽略加载参数时大小不匹配的情况，默认为 False。
            force_download (`bool`, *optional*, defaults to `False`):
                是否强制重新下载模型，默认为 False。
            local_files_only (`bool`, *optional*, defaults to `False`):
                是否仅使用本地文件加载模型，默认为 False。
            token (`str` or `bool`, *optional*, defaults to `None`):
                token 用于验证下载的模型，可选参数。
            revision (`str`, *optional*, defaults to `"main"`):
                模型的版本号，默认为 "main"。
            **kwargs:
                其余关键字参数，传递给具体模型加载函数。
        """

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],  # 保存模型的目录路径
        params=None,  # 要保存的模型参数，默认为 None
        push_to_hub=False,  # 是否推送到模型 Hub，默认为 False
        max_shard_size="10GB",  # 最大的分片大小，默认为 "10GB"
        token: Optional[Union[str, bool]] = None,  # token 用于验证，可选参数
        safe_serialization: bool = False,  # 是否进行安全序列化，默认为 False
        **kwargs,  # 其余关键字参数
    ):
        """
        将当前模型保存到指定目录。

        Args:
            save_directory (`str` or `os.PathLike`):
                保存模型的目录路径。
            params:
                要保存的模型参数，默认为 None。
            push_to_hub (`bool`, *optional*, defaults to `False`):
                是否将模型推送到模型 Hub，默认为 False。
            max_shard_size (`str`, *optional*, defaults to `"10GB"`):
                最大的分片大小，默认为 "10GB"。
            token (`str` or `bool`, *optional*, defaults to `None`):
                token 用于验证保存操作，可选参数。
            safe_serialization (`bool`, *optional*, defaults to `False`):
                是否进行安全序列化，默认为 False。
            **kwargs:
                其余关键字参数，传递给具体保存函数。
        """

    @classmethod
    def register_for_auto_class(cls, auto_class="FlaxAutoModel"):
        """
        注册当前模型类到指定的自动加载类。仅用于自定义模型，因为库中的模型已经与自动加载类映射。

        <Tip warning={true}>
        当前 API 处于实验阶段，未来版本可能会有一些轻微的更改。
        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"FlaxAutoModel"`):
                要注册新模型的自动加载类名称或类型。
        """

        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class
# 复制 FlaxPreTrainedModel 类中的 push_to_hub 方法，以确保我们修改的是其副本而非原始方法
FlaxPreTrainedModel.push_to_hub = copy_func(FlaxPreTrainedModel.push_to_hub)

# 如果 push_to_hub 方法已有文档字符串，则使用格式化字符串来更新其文档字符串，将对象类型、对象类名和对象文件类型作为参数插入
if FlaxPreTrainedModel.push_to_hub.__doc__ is not None:
    FlaxPreTrainedModel.push_to_hub.__doc__ = FlaxPreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="FlaxAutoModel", object_files="model checkpoint"
    )


def overwrite_call_docstring(model_class, docstring):
    # 复制 model_class 的 __call__ 方法，以确保仅修改该函数的文档字符串
    model_class.__call__ = copy_func(model_class.__call__)
    # 删除现有的 __call__ 方法的文档字符串
    model_class.__call__.__doc__ = None
    # 设置正确的 __call__ 方法文档字符串，使用指定的 docstring
    model_class.__call__ = add_start_docstrings_to_model_forward(docstring)(model_class.__call__)


def append_call_sample_docstring(
    model_class, checkpoint, output_type, config_class, mask=None, revision=None, real_checkpoint=None
):
    # 复制 model_class 的 __call__ 方法，以确保仅修改该函数的文档字符串
    model_class.__call__ = copy_func(model_class.__call__)
    # 使用 add_code_sample_docstrings 函数为 __call__ 方法添加代码示例的文档字符串，传入相关参数
    model_class.__call__ = add_code_sample_docstrings(
        checkpoint=checkpoint,
        output_type=output_type,
        config_class=config_class,
        model_cls=model_class.__name__,
        revision=revision,
        real_checkpoint=real_checkpoint,
    )(model_class.__call__)


def append_replace_return_docstrings(model_class, output_type, config_class):
    # 复制 model_class 的 __call__ 方法，以确保仅修改该函数的文档字符串
    model_class.__call__ = copy_func(model_class.__call__)
    # 使用 replace_return_docstrings 函数替换 __call__ 方法的返回值相关文档字符串，传入输出类型和配置类参数
    model_class.__call__ = replace_return_docstrings(
        output_type=output_type,
        config_class=config_class,
    )(model_class.__call__)
```