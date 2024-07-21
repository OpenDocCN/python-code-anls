# `.\pytorch\torch\backends\_nnapi\prepare.py`

```py
# mypy: allow-untyped-defs
# 引入所需的类型注解模块
from typing import List, Optional

# 引入 PyTorch 深度学习框架
import torch
# 从 torch.backends._nnapi.serializer 中导入 _NnapiSerializer
from torch.backends._nnapi.serializer import _NnapiSerializer

# 定义几个 NNAPI 编译偏好常量
ANEURALNETWORKS_PREFER_LOW_POWER = 0
ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1
ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2

# 定义一个继承自 torch.nn.Module 的类 NnapiModule
class NnapiModule(torch.nn.Module):
    """Torch Module that wraps an NNAPI Compilation.

    This module handles preparing the weights, initializing the
    NNAPI TorchBind object, and adjusting the memory formats
    of all inputs and outputs.
    """

    # _nnapi.Compilation 是一个可选的属性，可能为空
    comp: Optional[torch.classes._nnapi.Compilation]  # type: ignore[name-defined]
    # 权重列表，每个元素是一个 torch.Tensor
    weights: List[torch.Tensor]
    # 输出模板列表，每个元素是一个 torch.Tensor
    out_templates: List[torch.Tensor]

    def __init__(
        self,
        shape_compute_module: torch.nn.Module,
        ser_model: torch.Tensor,
        weights: List[torch.Tensor],
        inp_mem_fmts: List[int],
        out_mem_fmts: List[int],
        compilation_preference: int,
        relax_f32_to_f16: bool,
    ):
        super().__init__()
        # 初始化类的各个属性
        self.shape_compute_module = shape_compute_module
        self.ser_model = ser_model
        self.weights = weights
        self.inp_mem_fmts = inp_mem_fmts
        self.out_mem_fmts = out_mem_fmts
        self.out_templates = []
        self.comp = None
        self.compilation_preference = compilation_preference
        self.relax_f32_to_f16 = relax_f32_to_f16

    @torch.jit.export
    def init(self, args: List[torch.Tensor]):
        # 确保编译对象 comp 为空
        assert self.comp is None
        # 调用 shape_compute_module 的 prepare 方法，准备输出模板
        self.out_templates = self.shape_compute_module.prepare(self.ser_model, args)  # type: ignore[operator]
        # 将权重列表中的每个权重张量转换为连续的张量
        self.weights = [w.contiguous() for w in self.weights]
        # 创建一个 torch.classes._nnapi.Compilation 类的实例 comp
        comp = torch.classes._nnapi.Compilation()
        # 调用 comp 的 init2 方法，初始化编译对象
        comp.init2(
            self.ser_model,
            self.weights,
            self.compilation_preference,
            self.relax_f32_to_f16,
        )
        # 将创建的编译对象赋值给类属性 comp
        self.comp = comp
    # 定义一个方法 `forward`，接受一个参数列表 `args`，返回一个 `torch.Tensor` 列表
    def forward(self, args: List[torch.Tensor]) -> List[torch.Tensor]:
        # 如果 `self.comp` 为空，则初始化它
        if self.comp is None:
            self.init(args)
        # 将 `self.comp` 赋值给局部变量 `comp`
        comp = self.comp
        # 确保 `comp` 不为空
        assert comp is not None
        # 根据 `self.out_templates` 中的模板创建一个空的 `torch.Tensor` 列表 `outs`
        outs = [torch.empty_like(out) for out in self.out_templates]

        # 确保 `args` 的长度与 `self.inp_mem_fmts` 的长度相同
        assert len(args) == len(self.inp_mem_fmts)
        fixed_args = []
        # 遍历 `args` 列表的索引范围
        for idx in range(len(args)):
            # 获取当前参数的格式 `fmt`
            fmt = self.inp_mem_fmts[idx]
            # 这些常量与 `serializer.py` 中的 `DimOrder` 中的值匹配
            # TODO: 查看是否可以直接使用那些值。
            # 如果 `fmt` 为 0，则添加连续内存的 `args[idx]` 到 `fixed_args`
            if fmt == 0:
                fixed_args.append(args[idx].contiguous())
            # 如果 `fmt` 为 1，则对 `args[idx]` 进行维度置换，并添加到 `fixed_args`
            elif fmt == 1:
                fixed_args.append(args[idx].permute(0, 2, 3, 1).contiguous())
            else:
                # 如果 `fmt` 不是 0 或 1，则抛出值错误异常
                raise ValueError("Invalid mem_fmt")

        # 使用 `comp` 的 `run` 方法执行 `fixed_args`，将结果写入 `outs`
        comp.run(fixed_args, outs)
        # 确保 `outs` 的长度与 `self.out_mem_fmts` 的长度相同
        assert len(outs) == len(self.out_mem_fmts)
        # 遍历 `self.out_templates` 列表的索引范围
        for idx in range(len(self.out_templates)):
            # 获取当前输出模板的格式 `fmt`
            fmt = self.out_mem_fmts[idx]
            # 这些常量与 `serializer.py` 中的 `DimOrder` 中的值匹配
            # TODO: 查看是否可以直接使用那些值。
            # 如果 `fmt` 在 (0, 2) 中，则不做任何处理
            if fmt in (0, 2):
                pass
            # 如果 `fmt` 为 1，则对 `outs[idx]` 进行维度置换
            elif fmt == 1:
                outs[idx] = outs[idx].permute(0, 3, 1, 2)
            else:
                # 如果 `fmt` 不是 0、1 或 2，则抛出值错误异常
                raise ValueError("Invalid mem_fmt")
        
        # 返回经过处理的 `outs` 列表
        return outs
# 将模型转换为 NNAPI 可用格式的包装函数
def convert_model_to_nnapi(
    model,
    inputs,
    serializer=None,
    return_shapes=None,
    use_int16_for_qint16=False,
    compilation_preference=ANEURALNETWORKS_PREFER_SUSTAINED_SPEED,
    relax_f32_to_f16=False,
):
    # 调用 process_for_nnapi 函数处理模型和输入数据，获取相关信息
    (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    ) = process_for_nnapi(
        model, inputs, serializer, return_shapes, use_int16_for_qint16
    )

    # 创建 NNAPI 模块对象
    nnapi_model = NnapiModule(
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        compilation_preference,
        relax_f32_to_f16,
    )

    # 定义一个继承自 torch.nn.Module 的包装器类，以便处理 NNAPI 的输入输出格式
    class NnapiInterfaceWrapper(torch.nn.Module):
        """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

        def __init__(self, mod):
            super().__init__()
            self.mod = mod

    # 使用 NNAPI 模块实例化包装器类
    wrapper_model_py = NnapiInterfaceWrapper(nnapi_model)
    # 将包装器类转换为 Torch Script 模块
    wrapper_model = torch.jit.script(wrapper_model_py)
    
    # TODO: Maybe make these names match the original.
    # 生成 forward 方法的定义字符串，用于 Torch Script 模块的定义
    arg_list = ", ".join(f"arg_{idx}" for idx in range(len(inputs)))
    if retval_count < 0:
        ret_expr = "retvals[0]"
    else:
        ret_expr = "".join(f"retvals[{idx}], " for idx in range(retval_count))
    
    # 使用 wrapper_model 的 define 方法定义 forward 方法
    wrapper_model.define(
        f"def forward(self, {arg_list}):\n"
        f"    retvals = self.mod([{arg_list}])\n"
        f"    return {ret_expr}\n"
    )
    
    # 返回 Torch Script 包装后的模块
    return wrapper_model


# 处理模型和输入数据以准备转换为 NNAPI 格式
def process_for_nnapi(
    model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False
):
    # 冻结 Torch 模型，以便进行序列化
    model = torch.jit.freeze(model)

    # 如果输入数据是单个 Tensor，则转换为列表形式
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]

    # 初始化 NNAPI 序列化器
    serializer = serializer or _NnapiSerializer(
        config=None, use_int16_for_qint16=use_int16_for_qint16
    )
    
    # 使用序列化器将模型和输入数据序列化为 NNAPI 可接受的格式
    (
        ser_model,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        shape_compute_lines,
        retval_count,
    ) = serializer.serialize_model(model, inputs, return_shapes)
    
    # 将序列化后的模型转换为 Torch Tensor，数据类型为 int32
    ser_model_tensor = torch.tensor(ser_model, dtype=torch.int32)

    # 定义用于计算操作数形状的动态生成模块类
    class ShapeComputeModule(torch.nn.Module):
        """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """

        pass

    # 使用 Torch Script 将 ShapeComputeModule 类转换为模块
    shape_compute_module = torch.jit.script(ShapeComputeModule())

    # 返回序列化后的模型信息和动态生成模块对象
    return (
        shape_compute_module,
        ser_model_tensor,
        used_weights,
        inp_mem_fmts,
        out_mem_fmts,
        retval_count,
    )
    # 将 shape_compute_lines 列表中的每行代码加上缩进，并组成新的列表
    ] + [f"    {line}\n" for line in shape_compute_lines]
    # 将加上缩进的代码列表合并成一个字符串，并传递给 shape_compute_module 定义
    shape_compute_module.define("".join(real_shape_compute_lines))

    # 返回多个值作为元组
    return (
        shape_compute_module,   # 返回 shape_compute_module 对象
        ser_model_tensor,       # 返回 ser_model_tensor 变量
        used_weights,           # 返回 used_weights 变量
        inp_mem_fmts,           # 返回 inp_mem_fmts 变量
        out_mem_fmts,           # 返回 out_mem_fmts 变量
        retval_count,           # 返回 retval_count 变量
    )
```