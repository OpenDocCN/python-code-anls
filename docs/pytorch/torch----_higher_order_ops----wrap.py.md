# `.\pytorch\torch\_higher_order_ops\wrap.py`

```
# mypy: allow-untyped-defs
# 引入检查模块
import inspect
# 引入迭代工具模块
import itertools
# 引入日志模块
import logging

# 从 torch._logging 中引入 warning_once 函数
from torch._logging import warning_once

# 从 torch._ops 中引入 HigherOrderOperator 类
from torch._ops import HigherOrderOperator
# 从 torch.utils.checkpoint 中引入 checkpoint 和 CheckpointPolicy 函数
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 生成唯一标识符的计数器
uid = itertools.count(1)


# 用于测试 HigherOrderOperator 机制的类
class Wrap(HigherOrderOperator):
    # 初始化方法
    def __init__(self):
        super().__init__("wrap")

    # 调用方法，接受一个函数和参数，并进行装饰
    def __call__(self, func, *args, **kwargs):
        # Dynamo 在之前已经对 HigherOrderOp 的主体进行了跟踪，因此不需要再次跟踪
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        # 禁用装饰器函数
        @disable
        def wrapper():
            result = func(*args, **kwargs)
            return result

        return wrapper()


# 创建 Wrap 类的实例
wrap = Wrap()


# 用于启用 set_grad_enabled 的 HigherOrderOperator 包装类
class WrapWithSetGradEnabled(HigherOrderOperator):
    # 初始化方法
    def __init__(self):
        super().__init__("wrap_with_set_grad_enabled")

    # 调用方法，接受一个布尔值、一个被包装的函数和参数，并进行装饰
    def __call__(self, enable_grad, wrapped_func, *args, **kwargs):
        # Dynamo 在之前已经对 HigherOrderOp 的主体进行了跟踪，因此不需要再次跟踪
        import torch._dynamo  # noqa: F401
        from torch._dynamo import disable

        # 禁用装饰器函数
        @disable
        def wrapper():
            # 根据 enable_grad 的值设置梯度是否启用，并返回 wrapped_func 的结果
            with torch.set_grad_enabled(enable_grad):
                return wrapped_func(*args, **kwargs)

        return wrapper()


# 创建 WrapWithSetGradEnabled 类的实例
wrap_with_set_grad_enabled = WrapWithSetGradEnabled()


# 用于包装激活检查点的 HigherOrderOperator 类
class WrapActivationCheckpoint(HigherOrderOperator):
    """
    此操作符用于包装 torch.utils.checkpoint。
    这样可以避免 TorchDynamo 查看保存的张量钩子，并直接将控制传递给 AOT Autograd，
    这对于追踪保存的张量钩子是可以接受的。由于 AOT 追踪 torch.utils.checkpoint 代码，
    我们有一个包含重新计算前向节点的反向图。

    然而，我们可能很快会弃用此操作符。困难在于 rng 操作的功能化。
    今天，在 AOT autograd 和 Inductor 中，有两种不同的 rng 操作功能化，
    它们之间很难进行映射。rng 状态也使 Inductor 中的模式匹配复杂化。
    由于实现的便利性，我们目前倾向于在 Inductor 级别进行功能化，
    这意味着在分区器中作为编译器通行证执行重复/重新计算。详细信息请参见 TagActivationCheckpoint。
    """

    # 初始化方法
    def __init__(self):
        super().__init__("wrap_activation_checkpoint")
    def __call__(self, function, *args, **kwargs):
        """
        # 在这个方法中，我们将给定的函数使用 checkpointing 技术进行调用。

        # 设置 use_reentrant 为 False，因为这个操作将被追踪。
        # 我们确保 AOT Autograd 通过非可重入版本的 checkpointing 进行跟踪。
        """
        # 导入需要的模块
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter
        
        # 将 use_reentrant 和 preserve_rng_state 设置为 False
        kwargs["use_reentrant"] = False
        kwargs["preserve_rng_state"] = False
        
        """
        # 使用 Interpreter 允许通过 torch.compile 堆栈保留元数据。
        # 在 fx_traceback 中保留节点元数据。
        """
        with fx_traceback.preserve_node_meta():
            # 调用 checkpoint 函数，并传递给定函数的解释器的运行结果
            return checkpoint(Interpreter(function).run, *args, **kwargs)
wrap_activation_checkpoint = WrapActivationCheckpoint()

class TagActivationCheckpoint(HigherOrderOperator):
    """
    This operator is supposed to be used only with torch.compile stack. This
    accepts a Fx graph module which needs to be checkpointed. This operator adds
    "recomputable" tag to the nodes of the Fx graph that should be recomputed.

    The goal is to:
    1. Avoid using Dynamo to trace through saved tensor hooks.
    2. For selective checkpointing case, let AOTAutograd trace through
       saved tensor hooks but has special logic with TorchDispatchMode to override
       the usual saved_tensor_hooks fn logic in order to tag the nodes.
    3. Rely on the partitioners to actually duplicate the nodes.
    This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops (by setting fixed seed for each random op, see `replace_random_passes`).
    Therefore, the duplication of nodes, by design, respects the rng states in
    the forward and recomputed forward in backward.
    """

    def __init__(self):
        # 调用父类的构造函数，并传入特定的操作符名称
        super().__init__("tag_activation_checkpoint")

    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
        # 获取 checkpoint 函数的签名
        ckpt_signature = inspect.signature(checkpoint)
        checkpoint_keys = set()
        # 遍历签名中的参数，将除了 "function", "args", "kwargs" 之外的参数名加入集合
        for name in ckpt_signature.parameters:
            if name in ("function", "args", "kwargs"):
                continue
            checkpoint_keys.add(name)

        # `preserve_rng_state` 不是常规的关键字参数，需要额外添加到集合中
        checkpoint_keys.add("preserve_rng_state")

        # 根据参数名将 kwargs 分成两部分：checkpoint_kwargs 和 gmod_kwargs
        checkpoint_kwargs = {
            name: kwargs[name] for name in kwargs.keys() if name in checkpoint_keys
        }
        gmod_kwargs = {
            name: kwargs[name] for name in kwargs.keys() if name not in checkpoint_keys
        }
        return checkpoint_kwargs, gmod_kwargs
    def tag_nodes(self, gmod, is_sac):
        # 生成唯一的图标识符
        unique_graph_id = next(uid)
        # 遍历图中的每个节点
        for node in gmod.graph.nodes:
            # 如果节点的操作类型是函数调用、方法调用或模块调用
            if node.op in ("call_function", "call_method", "call_module"):
                # 设置节点的元数据中的"ac_graph_id"字段为生成的唯一图标识符
                node.meta["ac_graph_id"] = unique_graph_id
                # 如果是选择性检查点模式
                if is_sac:
                    # 在选择性检查点中，稍后我们会在_CachingTorchDispatchMode中填充此标记。
                    node.meta["recompute"] = None
                else:
                    # 在普通激活检查点中，所有节点应重新计算。
                    node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
        # 返回带有标记节点的修改后的图模型
        return gmod

    def __call__(self, gmod, *args, **kwargs):
        import torch.fx.traceback as fx_traceback
        from torch.fx import Interpreter

        # 如果图模型的元数据中包含"_checkpoint_context_fn"字段
        if "_checkpoint_context_fn" in gmod.meta:
            # 一次性发出警告，记录日志
            warning_once(
                log,
                """
Detected that context_fn is passed to torch.utils.checkpoint under torch.compile.
Please make sure the checkpointed region does not contain in-place ops (e.g. torch.relu_).
""",
            )
            # 将 use_reentrant 设置为 False，因为这个操作将被追踪。
            # 我们确保 AOT Autograd 通过非可重入版本的 checkpointing 进行追踪。
            kwargs["use_reentrant"] = False
            # 将 preserve_rng_state 设置为 False，因为我们希望阻止 AOTAutograd 通过 `torch.random.fork_rng` 操作进行追踪
            # （该操作在 CUDA 下尚不支持）。
            # 这并不意味着我们不保留 RNG 状态。相反，我们将始终保留 RNG 状态，
            # 无论此标志如何（通过 Inductor 中的 `replace_random_passes` 进行 RNG 功能化，而不是在 AOTAutograd 中进行）。
            kwargs["preserve_rng_state"] = False
            kwargs["context_fn"] = gmod.meta["_checkpoint_context_fn"]
            # 首先在该图中将所有节点标记为 "recompute"，然后在 torch/utils/checkpoint.py 的 _CachingTorchDispatchMode 中取消特定节点的 "recompute" 标记。
            gmod = self.tag_nodes(gmod, is_sac=True)
            # 使用解释器允许通过 torch.compile 堆栈保留元数据。
            with fx_traceback.preserve_node_meta():
                return checkpoint(Interpreter(gmod).run, *args, **kwargs)
        else:
            gmod = self.tag_nodes(gmod, is_sac=False)
            # 使用解释器允许通过 torch.compile 堆栈保留元数据。
            # TODO: 我们希望在这里使用与 `context_fn != None` 情况相同的 `checkpoint(Interpreter(gmod).run, *args, **kwargs)`，
            # 但这取决于 TorchDispatchMode + torch.compile 中对原位操作的支持。
            # （有关原位操作问题的详细信息，请运行 `test_compile_selective_checkpoint_inplace_op` 单元测试）
            with fx_traceback.preserve_node_meta():
                return Interpreter(gmod).run(*args)


tag_activation_checkpoint = TagActivationCheckpoint()
```