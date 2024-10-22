# `.\diffusers\callbacks.py`

```py
# 导入类型注解 Any, Dict, List
from typing import Any, Dict, List

# 从配置工具导入基类 ConfigMixin 和注册函数 register_to_config
from .configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入常量 CONFIG_NAME
from .utils import CONFIG_NAME


# 定义一个回调基类，用于管道中的所有官方回调
class PipelineCallback(ConfigMixin):
    """
    Base class for all the official callbacks used in a pipeline. This class provides a structure for implementing
    custom callbacks and ensures that all callbacks have a consistent interface.

    Please implement the following:
        `tensor_inputs`: This should return a list of tensor inputs specific to your callback. You will only be able to
        include
            variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
        `callback_fn`: This method defines the core functionality of your callback.
    """

    # 设置配置名称为 CONFIG_NAME
    config_name = CONFIG_NAME

    # 注册构造函数到配置
    @register_to_config
    def __init__(self, cutoff_step_ratio=1.0, cutoff_step_index=None):
        # 调用父类构造函数
        super().__init__()

        # 检查 cutoff_step_ratio 和 cutoff_step_index 是否同时为 None 或同时存在
        if (cutoff_step_ratio is None and cutoff_step_index is None) or (
            cutoff_step_ratio is not None and cutoff_step_index is not None
        ):
            # 如果同时为 None 或同时存在则抛出异常
            raise ValueError("Either cutoff_step_ratio or cutoff_step_index should be provided, not both or none.")

        # 检查 cutoff_step_ratio 是否为有效的浮点数
        if cutoff_step_ratio is not None and (
            not isinstance(cutoff_step_ratio, float) or not (0.0 <= cutoff_step_ratio <= 1.0)
        ):
            # 如果 cutoff_step_ratio 不在 0.0 到 1.0 之间则抛出异常
            raise ValueError("cutoff_step_ratio must be a float between 0.0 and 1.0.")

    # 定义 tensor_inputs 属性，返回类型为 List[str]
    @property
    def tensor_inputs(self) -> List[str]:
        # 抛出未实现错误，提醒用户必须实现该属性
        raise NotImplementedError(f"You need to set the attribute `tensor_inputs` for {self.__class__}")

    # 定义 callback_fn 方法，接收管道、步骤索引、时间步和回调参数，返回类型为 Dict[str, Any]
    def callback_fn(self, pipeline, step_index, timesteps, callback_kwargs) -> Dict[str, Any]:
        # 抛出未实现错误，提醒用户必须实现该方法
        raise NotImplementedError(f"You need to implement the method `callback_fn` for {self.__class__}")

    # 定义可调用方法，接收管道、步骤索引、时间步和回调参数，返回类型为 Dict[str, Any]
    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        # 调用 callback_fn 方法并返回结果
        return self.callback_fn(pipeline, step_index, timestep, callback_kwargs)


# 定义一个多管道回调类
class MultiPipelineCallbacks:
    """
    This class is designed to handle multiple pipeline callbacks. It accepts a list of PipelineCallback objects and
    provides a unified interface for calling all of them.
    """

    # 初始化方法，接收回调列表
    def __init__(self, callbacks: List[PipelineCallback]):
        # 将回调列表存储为类属性
        self.callbacks = callbacks

    # 定义 tensor_inputs 属性，返回所有回调的输入列表
    @property
    def tensor_inputs(self) -> List[str]:
        # 使用列表推导式从每个回调中获取 tensor_inputs
        return [input for callback in self.callbacks for input in callback.tensor_inputs]

    # 定义可调用方法，接收管道、步骤索引、时间步和回调参数，返回类型为 Dict[str, Any]
    def __call__(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        """
        Calls all the callbacks in order with the given arguments and returns the final callback_kwargs.
        """
        # 遍历所有回调并依次调用
        for callback in self.callbacks:
            # 更新回调参数
            callback_kwargs = callback(pipeline, step_index, timestep, callback_kwargs)

        # 返回最终的回调参数
        return callback_kwargs


# 定义稳定扩散管道的截止回调
class SDCFGCutoffCallback(PipelineCallback):
    """
    Callback function for Stable Diffusion Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    # 回调函数用于在特定步骤禁用 CFG
        `cutoff_step_index`), this callback will disable the CFG.
    
        # 注意：此回调通过将 `_guidance_scale` 属性更改为 0.0 来修改管道，发生在截止步骤之后。
        """
    
        # 定义输入的张量名称列表
        tensor_inputs = ["prompt_embeds"]
    
        # 定义回调函数，接收管道、步骤索引、时间步和其他回调参数
        def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
            # 获取截止步骤比例
            cutoff_step_ratio = self.config.cutoff_step_ratio
            # 获取截止步骤索引
            cutoff_step_index = self.config.cutoff_step_index
    
            # 如果截止步骤索引不为 None，使用该索引，否则根据比例计算截止步骤
            cutoff_step = (
                cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
            )
    
            # 如果当前步骤索引等于截止步骤，进行以下操作
            if step_index == cutoff_step:
                # 从回调参数中获取提示嵌入
                prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
                # 获取最后一个嵌入，表示条件文本标记的嵌入
                prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.
    
                # 将管道的指导比例设置为 0.0
                pipeline._guidance_scale = 0.0
    
                # 更新回调参数中的提示嵌入
                callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
            # 返回更新后的回调参数
            return callback_kwargs
# 定义 SDXLCFGCutoffCallback 类，继承自 PipelineCallback
class SDXLCFGCutoffCallback(PipelineCallback):
    """
    Stable Diffusion XL 管道的回调函数。在指定步骤数后（由 `cutoff_step_ratio` 或 `cutoff_step_index` 设置），
    此回调将禁用 CFG。

    注意：此回调通过将 `_guidance_scale` 属性在截止步骤后更改为 0.0 来改变管道。
    """

    # 定义需要处理的张量输入
    tensor_inputs = ["prompt_embeds", "add_text_embeds", "add_time_ids"]

    # 定义回调函数，接受管道、步骤索引、时间步和回调参数
    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        # 从配置中获取截止步骤比例
        cutoff_step_ratio = self.config.cutoff_step_ratio
        # 从配置中获取截止步骤索引
        cutoff_step_index = self.config.cutoff_step_index

        # 如果截止步骤索引不为 None，则使用该值，否则使用截止步骤比例计算
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )

        # 如果当前步骤等于截止步骤
        if step_index == cutoff_step:
            # 获取条件文本令牌的嵌入
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            # 取最后一个嵌入，表示条件文本令牌的嵌入
            prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.

            # 获取条件池化文本令牌的嵌入
            add_text_embeds = callback_kwargs[self.tensor_inputs[1]]
            # 取最后一个嵌入，表示条件池化文本令牌的嵌入
            add_text_embeds = add_text_embeds[-1:]  # "-1" denotes the embeddings for conditional pooled text tokens

            # 获取条件附加时间向量的 ID
            add_time_ids = callback_kwargs[self.tensor_inputs[2]]
            # 取最后一个 ID，表示条件附加时间向量的 ID
            add_time_ids = add_time_ids[-1:]  # "-1" denotes the embeddings for conditional added time vector

            # 将管道的引导比例设置为 0.0
            pipeline._guidance_scale = 0.0

            # 更新回调参数中的嵌入和 ID
            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
            callback_kwargs[self.tensor_inputs[1]] = add_text_embeds
            callback_kwargs[self.tensor_inputs[2]] = add_time_ids
        # 返回更新后的回调参数
        return callback_kwargs


# 定义 IPAdapterScaleCutoffCallback 类，继承自 PipelineCallback
class IPAdapterScaleCutoffCallback(PipelineCallback):
    """
    适用于任何继承 `IPAdapterMixin` 的管道的回调函数。在指定步骤数后（由 `cutoff_step_ratio` 或
    `cutoff_step_index` 设置），此回调将 IP 适配器的比例设置为 `0.0`。

    注意：此回调通过在截止步骤后将比例设置为 0.0 来改变 IP 适配器注意力处理器。
    """

    # 定义需要处理的张量输入（此类无具体输入）
    tensor_inputs = []

    # 定义回调函数，接受管道、步骤索引、时间步和回调参数
    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        # 从配置中获取截止步骤比例
        cutoff_step_ratio = self.config.cutoff_step_ratio
        # 从配置中获取截止步骤索引
        cutoff_step_index = self.config.cutoff_step_index

        # 如果截止步骤索引不为 None，则使用该值，否则使用截止步骤比例计算
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )

        # 如果当前步骤等于截止步骤
        if step_index == cutoff_step:
            # 将 IP 适配器的比例设置为 0.0
            pipeline.set_ip_adapter_scale(0.0)
        # 返回回调参数
        return callback_kwargs
```