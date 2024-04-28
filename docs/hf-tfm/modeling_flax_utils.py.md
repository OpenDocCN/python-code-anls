# `.\transformers\modeling_flax_utils.py`

```
def flax_shard_checkpoint(params, max_shard_size="10GB"):
    """
    将参数字典分片为多个文件以减小文件大小。

    参数：
        params (FrozenDict): 要分片的参数字典。
        max_shard_size (str): 每个分片的最大大小，以字节为单位。默认为 "10GB"。

    返回：
        List[str]: 分片后的文件路径列表。

    例子：
    ```python
    # 分片参数字典并保存分片文件
    shards = flax_shard_checkpoint(params)
    ```
    """
    # 计算最大分片大小
    max_size = convert_file_size_to_int(max_shard_size)
    # 将参数字典解冻为可变字典
    params = unfreeze(params)
    # 初始化分片列表
    shards = []
    # 初始化当前分片大小和当前分片计数
    current_size = 0
    shard_count = 0
    # 初始化当前分片参数字典
    current_shard = {}
    # 遍历参数字典的键值对
    for key, value in params.items():
        # 将键值对添加到当前分片参数字典中
        current_shard[key] = value
        # 计算当前键值对的大小
        param_size = jnp.prod(jnp.array(value).shape) * dtype_byte_size(value.dtype)
        # 计算当前分片的总大小
        current_size += param_size
        # 如果当前分片大小超过了最大分片大小
        if current_size > max_size:
            # 保存当前分片文件
            shard_path = f"{WEIGHTS_NAME}-shard-{shard_count}.npz"
            with open(shard_path, "wb") as f:
                f.write(to_bytes(current_shard))
            # 将分片文件路径添加到分片列表中
            shards.append(shard_path)
            # 重置当前分片大小、计数和参数字典
            current_size = 0
            shard_count += 1
            current_shard = {}
    # 如果还有剩余的参数未分片
    if current_shard:
        # 保存剩余参数的分片文件
        shard_path = f"{WEIGHTS_NAME}-shard-{shard_count}.npz"
        with open(shard_path, "wb") as f:
            f.write(to_bytes(current_shard))
        # 将分片文件路径添加到分片列表中
        shards.append(shard_path)
    return shards
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size. The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so
    there is no optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For
    example, if the limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as
    [6GB], [6+2GB], [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        params (`Union[Dict, FrozenDict]`): A `PyTree` of model parameters.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    """
    # Convert the maximum shard size to an integer value
    max_shard_size = convert_file_size_to_int(max_shard_size)

    # Initialize a list to store the sharded state dictionaries
    sharded_state_dicts = []
    # Initialize an empty dictionary to hold the current block of weights
    current_block = {}
    # Initialize the size of the current block to 0
    current_block_size = 0
    # Initialize the total size of all weights to 0
    total_size = 0

    # Flatten the model parameters into a dictionary of weights
    weights = flatten_dict(params, sep="/")
    # Iterate through each weight in the flattened dictionary
    for item in weights:
        # Calculate the size of the weight
        weight_size = weights[item].size * dtype_byte_size(weights[item].dtype)

        # If adding this weight would exceed the maximum shard size, start a new shard
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0

        # Add the weight to the current block and update the block size and total size
        current_block[item] = weights[item]
        current_block_size += weight_size
        total_size += weight_size

    # Add the last block to the list of sharded state dictionaries
    sharded_state_dicts.append(current_block)

    # If there is only one shard, return it with the weights name
    if len(sharded_state_dicts) == 1:
        return {FLAX_WEIGHTS_NAME: sharded_state_dicts[0]}, None

    # Otherwise, create an index mapping weight names to shard files
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        # Generate a file name for the shard based on its index
        shard_file = FLAX_WEIGHTS_NAME.replace(".msgpack", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.msgpack")
        shards[shard_file] = shard
        # Map each weight in the shard to the corresponding shard file
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file

    # Create metadata and index dictionaries to store additional information
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    # Return the sharded state dictionaries and the index
    return shards, index
class FlaxPreTrainedModel(PushToHubMixin, FlaxGenerationMixin):
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

    config_class = None
    base_model_prefix = ""
    main_input_name = "input_ids"
    _auto_class = None
    _missing_keys = set()

    def __init__(
        self,
        config: PretrainedConfig,
        module: nn.Module,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        # 检查配置是否为空
        if config is None:
            raise ValueError("config cannot be None")

        # 检查模块是否为空
        if module is None:
            raise ValueError("module cannot be None")

        # 将配置和模块存储为私有属性，以便在派生类上公开为类型化属性
        self._config = config
        self._module = module

        # 初始化 PRNGKey 和其他公共属性
        self.key = PRNGKey(seed)
        self.dtype = dtype
        self.input_shape = input_shape
        # 如果模型可以生成，则生成配置
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None

        # 检查模型是否自动初始化
        self._is_initialized = _do_init

        if _do_init:
            # 随机初始化参数
            random_params = self.init_weights(self.key, input_shape)
            # 计算参数形状树
            params_shape_tree = jax.eval_shape(lambda params: params, random_params)
        else:
            init_fn = partial(self.init_weights, input_shape=input_shape)
            params_shape_tree = jax.eval_shape(init_fn, self.key)

            logger.info(
                "Model weights are not initialized as `_do_init` is set to `False`. "
                f"Make sure to call `{self.__class__.__name__}.init_weights` manually to initialize the weights."
            )

        # 获取参数的形状
        self._params_shape_tree = params_shape_tree

        # 将 required_params 保存为集合
        self._required_params = set(flatten_dict(unfreeze(params_shape_tree)).keys())

        # 初始化参数
        if _do_init:
            self.params = random_params
    # 初始化权重的方法，需要子类实现
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> Dict:
        raise NotImplementedError(f"init method has to be implemented for {self}")

    # 启用梯度检查点的方法，需要子类实现
    def enable_gradient_checkpointing(self):
        raise NotImplementedError(f"gradient checkpointing method has to be implemented for {self}")

    # 从配置中创建模型的类方法
    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.
        """
        return cls(config, **kwargs)

    # 返回框架标识为"flax"
    @property
    def framework(self) -> str:
        """
        :str: Identifies that this is a Flax model.
        """
        return "flax"

    # 返回预训练配置
    @property
    def config(self) -> PretrainedConfig:
        return self._config

    # 返回模块
    @property
    def module(self) -> nn.Module:
        return self._module

    # 返回参数，如果模型未初始化，则引发错误
    @property
    def params(self) -> Union[Dict, FrozenDict]:
        if not self._is_initialized:
            raise ValueError(
                "`params` cannot be accessed from model when the model is created with `_do_init=False`. "
                "You must call `init_weights` manually and store the params outside of the model and "
                "pass it explicitly where needed."
            )
        return self._params

    # 返回所需参数集合
    @property
    def required_params(self) -> Set:
        return self._required_params

    # 返回参数形状树
    @property
    def params_shape_tree(self) -> Dict:
        return self._params_shape_tree

    # 设置参数，如果模型未初始化，则引发错误
    @params.setter
    def params(self, params: Union[Dict, FrozenDict]):
        # don't set params if the model is not initialized
        if not self._is_initialized:
            raise ValueError(
                "`params` cannot be set from model when the model is created with `_do_init=False`. "
                "You store the params outside of the model."
            )

        if isinstance(params, FrozenDict):
            params = unfreeze(params)
        param_keys = set(flatten_dict(params).keys())
        if len(self.required_params - param_keys) > 0:
            raise ValueError(
                "Some parameters are missing. Make sure that `params` include the following "
                f"parameters {self.required_params - param_keys}"
            )
        self._params = params
    def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any = None) -> Any:
        """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """

        # 从给定参数 `PyTree` 中将浮点数值转换为给定的 `dtype` 的辅助方法
        def conditional_cast(param):
            # 如果参数是 jnp.ndarray 类型并且数据类型是浮点型，则将其转换为指定的数据类型
            if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
                param = param.astype(dtype)
            return param

        # 如果没有给定 mask，则对所有参数应用条件转换函数并返回结果
        if mask is None:
            return jax.tree_util.tree_map(conditional_cast, params)

        # 将参数 `params` 扁平化为字典
        flat_params = flatten_dict(params)
        # 将 mask 也扁平化为字典
        flat_mask, _ = jax.tree_util.tree_flatten(mask)

        # 遍历扁平化后的 mask 和参数字典的键值对
        for masked, key in zip(flat_mask, flat_params.keys()):
            # 如果 mask 标记该参数需要转换，则进行条件转换
            if masked:
                param = flat_params[key]
                flat_params[key] = conditional_cast(param)

        # 将扁平化后的参数字典转回原始结构并返回
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
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # By default, the model parameters will be in fp32 precision, to cast these to bfloat16 precision
        >>> model.params = model.to_bf16(model.params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> flat_params = traverse_util.flatten_dict(model.params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> model.params = model.to_bf16(model.params, mask)
        ```"""
        # 调用 _cast_floating_to 方法，将参数转换为 jax.numpy.bfloat16 类型
        return self._cast_floating_to(params, jnp.bfloat16, mask)
    # 将浮点参数 `params` 转换为 `jax.numpy.float32`。此方法可用于显式将模型参数转换为fp32精度。这将返回一个新的 `params` 树，不会直接对 `params` 进行转换。
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
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # By default, the model params will be in fp32, to illustrate the use of this method,
        >>> # we'll first cast to fp16 and back to fp32
        >>> model.params = model.to_f16(model.params)
        >>> # now cast back to fp32
        >>> model.params = model.to_fp32(model.params)
        ```"""
        # 调用 `_cast_floating_to` 方法，将参数 `params` 中的浮点数转换为 `jax.numpy.float32`
        return self._cast_floating_to(params, jnp.float32, mask)
```  
    # 将浮点型的参数转换为 jax.numpy.float16 类型。返回一个新的参数树，并不是在原地进行转换。
    def to_fp16(self, params: Union[Dict, FrozenDict], mask: Any = None):
        r"""
        Cast the floating-point `parmas` to `jax.numpy.float16`. This returns a new `params` tree and does not cast the
        `params` in place.

        This method can be used on GPU to explicitly convert the model parameters to float16 precision to do full
        half-precision training or to save weights in float16 for inference in order to save memory and improve speed.

        Arguments:
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            mask (`Union[Dict, FrozenDict]`):
                A `PyTree` with same structure as the `params` tree. The leaves should be booleans, `True` for params
                you want to cast, and should be `False` for those you want to skip

        Examples:

        ```python
        >>> from transformers import FlaxBertModel

        >>> # load model
        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> # By default, the model params will be in fp32, to cast these to float16
        >>> model.params = model.to_fp16(model.params)
        >>> # If you want don't want to cast certain parameters (for example layer norm bias and scale)
        >>> # then pass the mask as follows
        >>> from flax import traverse_util

        >>> model = FlaxBertModel.from_pretrained("bert-base-cased")
        >>> flat_params = traverse_util.flatten_dict(model.params)
        >>> mask = {
        ...     path: (path[-2] != ("LayerNorm", "bias") and path[-2:] != ("LayerNorm", "scale"))
        ...     for path in flat_params
        ... }
        >>> mask = traverse_util.unflatten_dict(mask)
        >>> model.params = model.to_fp16(model.params, mask)
        ```"""
        # 调用 _cast_floating_to 方法进行参数类型转换，转换为 jnp.float16 类型
        return self._cast_floating_to(params, jnp.float16, mask)

    @classmethod
```  
    # 从文件中加载 FLAX 模型的权重数据
    def load_flax_weights(cls, resolved_archive_file):
        try:
            # 如果文件名以 ".safetensors" 结尾，使用 safe_load_file 函数加载数据
            if resolved_archive_file.endswith(".safetensors"):
                # 加载安全的文件内容并解析成字典形式的状态
                state = safe_load_file(resolved_archive_file)
                # 将字典状态重新整理成以 "." 为分隔符的形式
                state = unflatten_dict(state, sep=".")
            else:
                # 否则，使用二进制模式打开文件，读取其中的数据并解析成对象状态
                with open(resolved_archive_file, "rb") as state_f:
                    state = from_bytes(cls, state_f.read())
        except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
            try:
                # 尝试使用文本模式打开文件，检查文件内容是否以 "version" 开头
                with open(resolved_archive_file) as f:
                    # 如果文件内容以 "version" 开头，则可能是没有安装 git-lfs 的仓库副本
                    if f.read().startswith("version"):
                        # 抛出 OSError 提示用户安装 git-lfs 并拉取数据
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    else:
                        # 否则，抛出 ValueError
                        raise ValueError from e
            except (UnicodeDecodeError, ValueError):
                # 如果无法解码文件内容或转换成 FLAX 可反序列化对象，抛出 EnvironmentError
                raise EnvironmentError(f"Unable to convert {resolved_archive_file} to Flax deserializable object. ")

        # 返回加载的状态数据
        return state

    # 类方法
    @classmethod
    # 定义一个类方法，用于加载分片权重
    def load_flax_sharded_weights(cls, shard_files):
        """
        This is the same as [`flax.serialization.from_bytes`](https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html#from_bytes) but for a sharded checkpoint.

        This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
        loaded in the model.

        Args:
            shard_files (`List[str]`):
                The list of shard files to load.

        Returns:
            `Dict`: A nested dictionary of the model parameters, in the expected format for flax models: `{'model':
            {'params': {'...'}}}`.
        """

        # 创建一个空字典用于存储分片状态
        state_sharded_dict = {}

        # 遍历每个分片文件
        for shard_file in shard_files:
            # 使用 msgpack 工具加载状态
            try:
                # 以二进制方式打开分片文件
                with open(shard_file, "rb") as state_f:
                    # 从字节读取状态
                    state = from_bytes(cls, state_f.read())
            # 捕获可能的异常
            except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
                # 如果文件以版本信息开头，提示用户安装 git-lfs
                with open(shard_file) as f:
                    if f.read().startswith("version"):
                        raise OSError(
                            "You seem to have cloned a repository without having git-lfs installed. Please"
                            " install git-lfs and run `git lfs install` followed by `git lfs pull` in the"
                            " folder you cloned."
                        )
                    # 否则抛出错误
                    else:
                        raise ValueError from e
            # 捕获可能的异常
            except (UnicodeDecodeError, ValueError):
                # 抛出环境错误，提示无法将文件转换为 Flax 可反序列化对象
                raise EnvironmentError(f"Unable to convert {shard_file} to Flax deserializable object. ")

            # 将状态展平为字典，使用 "/" 分隔键
            state = flatten_dict(state, sep="/")
            # 更新分片状态字典
            state_sharded_dict.update(state)
            # 删除状态变量
            del state
            # 手动触发垃圾回收
            gc.collect()

        # 将状态字典展开以匹配模型参数的格式
        return unflatten_dict(state_sharded_dict, sep="/")

    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # 检测 `prepare_inputs_for_generation` 是否被重写，这是生成序列的要求之一。
        # 或者，模型也可以具有自定义的 `generate` 函数。
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True

    @classmethod
```  
    # 从预训练模型名称或路径创建一个新的模型实例
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],  # 预训练模型的名称或路径
        dtype: jnp.dtype = jnp.float32,  # 数据类型，默认为 jnp.float32
        *model_args,  # 模型参数
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,  # 预训练配置，可选
        cache_dir: Optional[Union[str, os.PathLike]] = None,  # 缓存目录，可选
        ignore_mismatched_sizes: bool = False,  # 是否忽略大小不匹配，默认为 False
        force_download: bool = False,  # 是否强制下载，默认为 False
        local_files_only: bool = False,  # 是否只使用本地文件，默认为 False
        token: Optional[Union[str, bool]] = None,  # 认证令牌，可选
        revision: str = "main",  # 版本控制，缺省为 "main"
        **kwargs,  # 其他关键字参数
    # 将模型保存到指定目录
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],  # 要保存的目录路径
        params=None,  # 要保存的模型参数，可选
        push_to_hub=False,  # 是否推送到 Hub，默认为 False
        max_shard_size="10GB",  # 最大分片大小，默认为 "10GB"
        token: Optional[Union[str, bool]] = None,  # 认证令牌，可选
        safe_serialization: bool = False,  # 是否安全序列化，默认为 False
        **kwargs,  # 其他关键字参数
    @classmethod
    def register_for_auto_class(cls, auto_class="FlaxAutoModel"):
        """
        为给定的自动类注册此类。这仅应用于自定义模型，因为库中的模型已经与自动类映射。

        <Tip warning={true}>

        此 API 是实验性的，并且在下一个版本中可能会有一些轻微的变化。

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"FlaxAutoModel"`):
                要注册此新模型的自动类。
        """
        # 如果自动类不是字符串，则获取其类名
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        # 导入自动模块
        import transformers.models.auto as auto_module

        # 如果自动模块中不存在指定的自动类，则引发值错误
        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        # 将自动类赋值给当前类的私有属性 _auto_class
        cls._auto_class = auto_class
```  
# 更新文档字符串，需要复制方法，否则会改变原始文档字符串
FlaxPreTrainedModel.push_to_hub = copy_func(FlaxPreTrainedModel.push_to_hub)
# 如果存在文档字符串
if FlaxPreTrainedModel.push_to_hub.__doc__ is not None:
    # 使用格式化字符串替换文档字符串中的占位符
    FlaxPreTrainedModel.push_to_hub.__doc__ = FlaxPreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="FlaxAutoModel", object_files="model checkpoint"
    )

# 覆盖调用方法的文档字符串
def overwrite_call_docstring(model_class, docstring):
    # 复制 __call__ 方法以确保只更改此函数的文档字符串
    model_class.__call__ = copy_func(model_class.__call__)
    # 删除现有的文档字符串
    model_class.__call__.__doc__ = None
    # 设置正确的文档字符串
    model_class.__call__ = add_start_docstrings_to_model_forward(docstring)(model_class.__call__)

# 添加调用示例的文档字符串
def append_call_sample_docstring(
    model_class, checkpoint, output_type, config_class, mask=None, revision=None, real_checkpoint=None
):
    # 复制 __call__ 方法
    model_class.__call__ = copy_func(model_class.__call__)
    # 使用 add_code_sample_docstrings 函数添加调用示例的文档字符串
    model_class.__call__ = add_code_sample_docstrings(
        checkpoint=checkpoint,
        output_type=output_type,
        config_class=config_class,
        model_cls=model_class.__name__,
        revision=revision,
        real_checkpoint=real_checkpoint,
    )(model_class.__call__)

# 添加替换返回文档字符串
def append_replace_return_docstrings(model_class, output_type, config_class):
    # 复制 __call__ 方法
    model_class.__call__ = copy_func(model_class.__call__)
    # 使用 replace_return_docstrings 函数替换返回的文档字符串
    model_class.__call__ = replace_return_docstrings(
        output_type=output_type,
        config_class=config_class,
    )(model_class.__call__)
```