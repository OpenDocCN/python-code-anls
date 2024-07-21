# `.\pytorch\torch\fx\passes\fake_tensor_prop.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型声明模块
from typing import Optional

# 引入相关的torch.fx模块及类
import torch.fx
from torch.fx import Node
from torch.fx.node import map_aggregate
from torch.fx._compatibility import compatibility

# 引入FakeTensor相关的类和模块
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
from torch.fx.experimental.proxy_tensor import snapshot_fake, py_sym_types

# 定义在from torch.fx之后需要公开的符号
__all__ = ['FakeTensorProp']

# FakeTensorProp类继承自torch.fx.Interpreter类，用于执行FX图中的节点，并记录代表节点元数据的虚假张量
@compatibility(is_backward_compatible=False)
class FakeTensorProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """
    
    # 初始化方法，接受一个torch.fx.GraphModule类型的module和一个可选的FakeTensorMode类型的mode参数
    def __init__(self, module: torch.fx.GraphModule, mode: Optional[FakeTensorMode] = None):
        # 调用父类(torch.fx.Interpreter)的初始化方法
        super().__init__(module)
        # 如果mode参数为None，则使用默认的FakeTensorMode对象
        if mode is None:
            mode = FakeTensorMode()
        # 将mode赋值给实例变量self._mode
        self._mode = mode
        # 增加模式的epoch计数
        mode.epoch += 1

    # 执行单个节点的方法，接受一个Node类型的参数n
    def run_node(self, n: Node):
        # 导入相关模块和函数
        from torch.fx.experimental.symbolic_shapes import rebind_unbacked, compute_unbacked_bindings
        
        # 调用父类的run_node方法，执行节点n，并获取结果
        result = super().run_node(n)
        # 根据当前节点n和其执行结果result，重新绑定未支持的（unbacked）变量
        rebind_unbacked(self._mode.shape_env, n, result)

        # 定义一个函数，用于从对象obj中提取值
        def extract_val(obj):
            # 如果obj是FakeTensor类型，则返回其快照（snapshot）
            if isinstance(obj, FakeTensor):
                return snapshot_fake(obj)
            # 如果obj是torch.Tensor类型，则根据当前模式self._mode将其转换为FakeTensor后返回其快照
            elif isinstance(obj, torch.Tensor):
                # TODO: How is it possible that we get a non fake tensor?  We
                # should be running under the mode...
                return snapshot_fake(self._mode.from_tensor(obj, static_shapes=True))
            # 如果obj是py_sym_types类型，则直接返回obj
            elif isinstance(obj, py_sym_types):
                return obj
            # 否则返回None
            else:
                return None

        # 使用map_aggregate函数从result中提取元数据
        meta = map_aggregate(result, extract_val)
        # 如果提取到的元数据不为None，则将其存储到节点n的meta字典中的"val"键下
        if meta is not None:
            n.meta['val'] = meta
            # 如果存在shape_env和symbol_to_path，则计算并存储未支持绑定
            if (shape_env := self._mode.shape_env) and (symbol_to_path := compute_unbacked_bindings(shape_env, result)):
                n.meta["unbacked_bindings"] = symbol_to_path

        # 返回执行结果result
        return result

    # 传播方法，用于执行模型计算
    def propagate(self, *args):
        # 将输入参数args转换为相应的虚假张量后传播计算
        fake_args = [
            self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        return self.propagate_dont_convert_inputs(*fake_args)

    # 不转换输入的传播方法，用于执行模型计算
    def propagate_dont_convert_inputs(self, *args):
        # 使用当前模式self._mode执行不转换输入的计算，并返回结果
        with self._mode:
            return super().run(*args)
```