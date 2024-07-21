# `.\pytorch\benchmarks\framework_overhead_benchmark\pt_wrapper_module.py`

```
# 导入 PyTorch 库
import torch

# 定义一个包装器类，用于封装 wrapped_type 实例
class WrapperModule:
    """Wraps the instance of wrapped_type.
    For graph_mode traces the instance of wrapped_type.
    Randomaly initializes num_params tensors with single float element.
    Args:
        wrapped_type:
            - Object type to be wrapped.
                Expects the wrapped_type to:
                   - be constructed with pt_fn specified in module_config.
                   - provide forward method that takes module_config.num_params args.
        module_config:
            - Specified pt_fn to construct wrapped_type with, whether graph_mode
              is enabled, and number of parameters wrapped_type's forward method
              takes.
        debug:
            - Whether debug mode is enabled.
        save:
            - In graph mode, whether graph is to be saved.
    """

    def __init__(self, wrapped_type, module_config, debug, save=False):
        # 从 module_config 中获取构造 wrapped_type 所需的 pt_fn 函数
        pt_fn = module_config.pt_fn
        # 使用 pt_fn 函数构造 wrapped_type 对象
        self.module = wrapped_type(pt_fn)
        # 初始化一个空列表，用于存储随机生成的浮点数张量
        self.tensor_inputs = []
        # 记录 wrapped_type 的类名
        self.module_name = wrapped_type.__name__
        
        # 随机初始化 num_params 个张量，每个张量包含一个随机浮点数
        for _ in range(module_config.num_params):
            self.tensor_inputs.append(torch.randn(1))
        
        # 如果启用了 graph_mode，则使用 torch.jit.trace 对模块进行跟踪
        if module_config.graph_mode:
            self.module = torch.jit.trace(self.module, self.tensor_inputs)
            # 如果 save 参数为 True，则保存生成的图形模型
            if save:
                file_name = self.module_name + "_" + pt_fn.__name__ + ".pt"
                torch.jit.save(self.module, file_name)
                print(f"Generated graph is saved in {file_name}")
        
        # 打印模块的基准信息，包括模块名称、使用的函数名称和是否启用了 graph_mode
        print(
            f"Benchmarking module {self.module_name} with fn {pt_fn.__name__}: Graph mode:{module_config.graph_mode}"
        )
        
        # 如果启用了 debug 并且模块是 torch.jit.ScriptModule 类型，则打印模块的图形和代码
        if debug and isinstance(self.module, torch.jit.ScriptModule):
            print(self.module.graph)
            print(self.module.code)

    def forward(self, niters):
        # 在无梯度计算的上下文中运行 forward 方法 niters 次
        with torch.no_grad():
            for _ in range(niters):
                self.module.forward(*self.tensor_inputs)
```