# `.\pytorch\torch\onnx\_internal\fx\serialization.py`

```py
# mypy: allow-untyped-defs
from __future__ import annotations

import io  # 导入io模块，提供对字节流的支持
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，提供与操作系统交互的功能
from typing import Optional, Tuple, TYPE_CHECKING, Union  # 导入类型提示相关的类和函数

import torch  # 导入PyTorch深度学习库
from torch.onnx import _type_utils as jit_type_utils  # 导入类型工具模块
from torch.onnx._internal import _beartype  # 导入_beartype模块，用于函数参数类型验证

if TYPE_CHECKING:
    import onnx  # 类型检查时导入onnx模块

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@_beartype.beartype
def _create_tensor_proto_with_external_data(
    tensor: torch.Tensor,
    name: str,
    location: str,
    basepath: str,
    dtype_override: Optional["onnx.TypeProto"] = None,  # type: ignore[name-defined]
) -> onnx.TensorProto:  # type: ignore[name-defined]
    """Create a TensorProto with external data from a PyTorch tensor.
    The external data is saved to os.path.join(basepath, location).

    Args:
        tensor: Tensor to be saved.
        name: Name of the tensor (i.e., initializer name in ONNX graph).
        location: Relative location of the external data file
            (e.g., "/tmp/initializers/weight_0" when model is "/tmp/model_name.onnx").
        basepath: Base path of the external data file (e.g., "/tmp/external_data" while model must be in "/tmp").

    Reference for ONNX's external data format:
        How to load?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L187
        How to save?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L43
        How to set ONNX fields?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L88
    """
    import onnx  # 动态导入onnx模块，用于创建TensorProto对象

    scalar_type = (
        jit_type_utils.JitScalarType.from_onnx_type(
            dtype_override.tensor_type.elem_type
        )
        if dtype_override is not None
        else jit_type_utils.JitScalarType.from_dtype(tensor.dtype)
    )

    # 检查点可以以不同于模型期望的dtype存储，因为用户脚本可以显式地将原始类型转换为其他类型，或者PyTorch的类型提升可能会这样做
    if dtype_override is not None and scalar_type.dtype() != tensor.dtype:
        tensor = tensor.to(scalar_type.dtype())

    tensor_proto = onnx.TensorProto()  # 创建一个新的TensorProto对象
    tensor_proto.name = name  # 设置TensorProto的名称
    tensor_proto.data_type = scalar_type.onnx_type()  # 设置TensorProto的数据类型

    tensor_proto.dims.extend(tensor.shape)  # 将Tensor的形状信息添加到TensorProto中
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL  # 设置TensorProto的数据位置为外部文件

    # 设置每个文件保存一个张量的相关设置，因为在同一个文件中没有其他张量，因此偏移量为零
    key_value_pairs = {
        "location": location,
        "offset": 0,
        "length": tensor.untyped_storage().nbytes(),
    }
    for k, v in key_value_pairs.items():
        entry = tensor_proto.external_data.add()  # 向TensorProto的external_data列表中添加条目
        entry.key = k  # 设置条目的键
        entry.value = str(v)  # 将值转换为字符串并设置条目的值

    # 实际写入张量内容的路径。
    # 构建外部数据文件的路径，基于给定的基础路径和位置信息
    external_data_file_path = os.path.join(basepath, location)
    
    # 如果外部数据文件路径已存在，则删除该文件
    if os.path.exists(external_data_file_path):
        os.remove(external_data_file_path)
    
    # 创建外部数据文件所在的文件夹路径，如果该路径不存在的话
    external_data_dir_path = os.path.dirname(external_data_file_path)
    if not os.path.exists(external_data_dir_path):
        # 如果 demo_folder 目录不存在，则创建它
        os.makedirs(external_data_dir_path)
    
    # 创建一个新的文件，并以二进制写入模式打开
    with open(external_data_file_path, "xb") as data_file:
        # 将张量转换为字节流并写入文件
        data_file.write(tensor.numpy(force=True).tobytes())
    
    # 返回生成的张量协议缓冲区对象
    return tensor_proto
def _convert_safetensors_to_torch_format(safetensors_file):
    # 如果调用此函数，safetensors 文件必定存在，
    # 因为 HF 模型已加载并导出为 ONNX 模型，包含了 safetensors
    from safetensors import safe_open  # type: ignore[import-not-found]

    # 创建空字典用于存储张量数据
    tensors = {}
    # 使用 safe_open 函数打开 safetensors 文件，指定框架为 "pt"，设备为 "cpu"
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:  # type: ignore[attr-defined]
        # 遍历文件中的键（张量名称）
        for k in f.keys():
            # 获取张量并转移到 CPU 上存储
            tensors[k] = f.get_tensor(k).cpu()
    return tensors



# TODO: generalize to allow more checkpoints formats (torch or gguf)
@_beartype.beartype
def save_model_with_external_data(
    basepath: str,
    model_location: str,
    initializer_location: str,
    torch_state_dicts: Tuple[Union[dict, str, io.BytesIO], ...],
    onnx_model: onnx.ModelProto,  # type: ignore[name-defined]
    rename_initializer: bool = False,
) -> None:
    """Load PyTorch tensors from files and add to "onnx_model" as external initializers.

    Output files:
        ONNX model file path:
        ONNX initializer folder: os.path.join(basepath, initializer_location)

    After running this function, you can do
        ort_sess = onnxruntime.InferenceSession(os.path.join(basepath, model_location))
    to execute the model.

    Arguments:
        basepath: ONNX 外部数据文件的基本路径（例如，"/path/to/large_model/"）。
        model_location: ONNX 模型文件的相对位置。
            例如，"model.onnx" 表示模型文件将保存在 "<basepath>/model.onnx"。
        initializer_location: ONNX 初始化器文件夹的相对位置。
            例如，"initializers" 表示初始化器将保存在 "<basepath>/initializers/"。
            注意：当初始化器超过 2GB 时，必须与 `model_location` 相同。
        torch_state_dicts: 包含要保存为 ONNX 初始化器的 PyTorch 张量的字典或文件。
            对于非字典参数，将使用 `torch.load` 从类文件对象中加载它们。
        onnx_model: 要保存的带有外部初始化器的 ONNX 模型。
            如果输入名称与从 `torch_state_dicts` 加载的张量匹配，
            则该张量将保存为该输入的外部初始化器。
        rename_initializer: 替换所有 ONNX 初始化器名称中的 "." 为 "_"。
            对于支持 `FXSymbolicTracer` 与伪张量模式的情况，
            不需要官方的 torch.onnx.dynamo_export。这是一种方法，
            `FXSymbolicTracer` 提升 FX 参数（如 self.linear_weight）
            作为输入（`def forward(self, linear_weight)`）并且因此 `.` 不能使用。

    """
    # FIXME: 避免将 onnx 导入到 torch.onnx 中。
    import onnx

    # 使用字典来存储将要删除的初始化器，因为它是**有序的**
    initializers_to_be_deleted = {}
    # 使用字典存储现有初始化器的名称及其索引
    existing_initializers = {
        k.name: idx for idx, k in enumerate(onnx_model.graph.initializer)
    }
    # 获取 ONNX 模型中输入的名称集合
    onnx_input_names = {input.name for input in onnx_model.graph.input}
    # 如果有旧的重复初始化器，按降序删除以避免删除索引失效
    initializers_to_be_deleted = dict(
        sorted(initializers_to_be_deleted.items(), reverse=True)
    )
    # 遍历要删除的初始化器的索引，并从ONNX模型的图中删除对应的初始化器
    for idx in initializers_to_be_deleted.keys():
        del onnx_model.graph.initializer[idx]

    # 将ONNX模型保存到指定路径，model_location应为纯文件名，如 "file_name.onnx"，不应包含文件夹路径
    onnx.save(onnx_model, os.path.join(basepath, model_location))  # type: ignore[attr-defined]
```