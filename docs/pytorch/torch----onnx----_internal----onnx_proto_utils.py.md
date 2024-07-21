# `.\pytorch\torch\onnx\_internal\onnx_proto_utils.py`

```py
# Utilities for manipulating the ONNX and ONNX-script dependencies and ONNX proto.
"""Utilities for manipulating the onnx and onnx-script dependencies and ONNX proto."""

# Import necessary modules
import glob
import io
import os
import shutil
import zipfile
from typing import Any, List, Mapping, Set, Tuple, Union

# Import torch related modules
import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors
from torch.onnx._internal import _beartype, jit_utils, registration


@_beartype.beartype
# Function to export an ONNX model as a self-contained test case
def export_as_test_case(
    model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str
) -> str:
    """Export an ONNX model as a self contained ONNX test case.

    The test case contains the model and the inputs/outputs data. The directory structure
    is as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        model_bytes: The ONNX model in bytes.
        inputs_data: The inputs data, nested data structure of numpy.ndarray.
        outputs_data: The outputs data, nested data structure of numpy.ndarray.

    Returns:
        The path to the test case directory.
    """
    try:
        # Try to import the onnx module
        import onnx
    except ImportError as exc:
        # Raise an ImportError if onnx module is not found
        raise ImportError(
            "Export test case to ONNX format failed: Please install ONNX."
        ) from exc

    # Define the directory for the test case
    test_case_dir = os.path.join(dir, "test_" + name)
    os.makedirs(test_case_dir, exist_ok=True)

    # Export the ONNX model to a file
    _export_file(
        model_bytes,
        os.path.join(test_case_dir, "model.onnx"),
        _exporter_states.ExportTypes.PROTOBUF_FILE,
        {},
    )

    # Define the directory for the test data set
    data_set_dir = os.path.join(test_case_dir, "test_data_set_0")

    # Remove the directory if it already exists
    if os.path.exists(data_set_dir):
        shutil.rmtree(data_set_dir)
    os.makedirs(data_set_dir)

    # Load the ONNX model from the model_bytes
    proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    # Export inputs and outputs to protobuf files
    for i, (input_proto, input) in enumerate(zip(proto.graph.input, inputs_data)):
        export_data(input, input_proto, os.path.join(data_set_dir, f"input_{i}.pb"))
    for i, (output_proto, output) in enumerate(zip(proto.graph.output, outputs_data)):
        export_data(output, output_proto, os.path.join(data_set_dir, f"output_{i}.pb"))

    # Return the path to the test case directory
    return test_case_dir


@_beartype.beartype
# Function to load a self-contained ONNX test case from a directory
def load_test_case(dir: str) -> Tuple[bytes, Any, Any]:
    """Load a self contained ONNX test case from a directory.

    The test case must contain the model and the inputs/outputs data. The directory structure
    should be as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       └── output_0.pb

    Args:
        dir: The directory containing the test case.

    Returns:
        A tuple containing the ONNX model bytes, inputs data, and outputs data.
    """
    # 导入必要的模块和库：onnx用于加载ONNX模型，numpy_helper用于处理numpy数组
    try:
        import onnx
        from onnx import numpy_helper  # type: ignore[attr-defined]
    except ImportError as exc:
        # 如果导入失败，抛出ImportError异常，并提示安装ONNX库
        raise ImportError(
            "Load test case from ONNX format failed: Please install ONNX."
        ) from exc

    # 打开存储在指定目录下的ONNX模型文件，以二进制形式读取模型内容
    with open(os.path.join(dir, "model.onnx"), "rb") as f:
        model_bytes = f.read()

    # 指定测试数据集所在的子目录
    test_data_dir = os.path.join(dir, "test_data_set_0")

    # 初始化空字典，用于存储输入数据，键为输入名称，值为numpy数组
    inputs = {}
    # 查找所有以"input_"开头的.pb文件
    input_files = glob.glob(os.path.join(test_data_dir, "input_*.pb"))
    # 遍历每个输入文件
    for input_file in input_files:
        # 使用onnx.load_tensor函数加载输入文件中的张量数据
        tensor = onnx.load_tensor(input_file)  # type: ignore[attr-defined]
        # 将加载的张量数据转换为numpy数组，并存储到inputs字典中，键为张量名称
        inputs[tensor.name] = numpy_helper.to_array(tensor)

    # 初始化空字典，用于存储输出数据，键为输出名称，值为numpy数组
    outputs = {}
    # 查找所有以"output_"开头的.pb文件
    output_files = glob.glob(os.path.join(test_data_dir, "output_*.pb"))
    # 遍历每个输出文件
    for output_file in output_files:
        # 使用onnx.load_tensor函数加载输出文件中的张量数据
        tensor = onnx.load_tensor(output_file)  # type: ignore[attr-defined]
        # 将加载的张量数据转换为numpy数组，并存储到outputs字典中，键为张量名称
        outputs[tensor.name] = numpy_helper.to_array(tensor)

    # 返回加载的ONNX模型的字节数据，输入数据字典和输出数据字典
    return model_bytes, inputs, outputs
# 使用装饰器 `_beartype.beartype` 对函数进行类型检查和注解
@_beartype.beartype
# 定义函数 `_export_file`，接受以下参数：
# - model_bytes: 模型字节流数据
# - f: 可以是 io.BytesIO 对象或者字符串类型的文件路径，用于写入数据
# - export_type: 导出类型，可以是字符串 "protobuf_file" 或 "zip_archive" 或 "compressed_zip_archive"
# - export_map: 字符串到字节流的映射，用于在 ZIP 导出时存储多个文件
def _export_file(
    model_bytes: bytes,
    f: Union[io.BytesIO, str],
    export_type: str,
    export_map: Mapping[str, bytes],
) -> None:
    """export/write model bytes into directory/protobuf/zip"""
    
    # 如果导出类型是 "protobuf_file"
    if export_type == _exporter_states.ExportTypes.PROTOBUF_FILE:
        # 确保在 protobuf_file 模式下，export_map 是空的
        assert len(export_map) == 0
        # 使用 torch.serialization._open_file_like 打开文件对象，并以二进制写入模型字节流
        with torch.serialization._open_file_like(f, "wb") as opened_file:
            opened_file.write(model_bytes)
    
    # 如果导出类型是 "zip_archive" 或者 "compressed_zip_archive"
    elif export_type in {
        _exporter_states.ExportTypes.ZIP_ARCHIVE,
        _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE,
    }:
        # 确定压缩类型：如果是 "compressed_zip_archive" 使用 ZIP_DEFLATED 压缩，否则使用 ZIP_STORED
        compression = (
            zipfile.ZIP_DEFLATED
            if export_type == _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE
            else zipfile.ZIP_STORED
        )
        # 使用 zipfile.ZipFile 打开文件对象 f，以写入模式，并指定压缩方式
        with zipfile.ZipFile(f, "w", compression=compression) as z:
            # 将模型字节流写入 ZIP 文件中，使用常量 _constants.ONNX_ARCHIVE_MODEL_PROTO_NAME 命名
            z.writestr(_constants.ONNX_ARCHIVE_MODEL_PROTO_NAME, model_bytes)
            # 遍历 export_map 中的每个键值对，将键作为文件名，值作为内容写入 ZIP 文件
            for k, v in export_map.items():
                z.writestr(k, v)
    # 如果导出类型为DIRECTORY
    elif export_type == _exporter_states.ExportTypes.DIRECTORY:
        # 如果给定的f是BytesIO对象或者不是一个目录，则抛出数值错误异常
        if isinstance(f, io.BytesIO) or not os.path.isdir(f):  # type: ignore[arg-type]
            raise ValueError(
                f"f should be directory when export_type is set to DIRECTORY, instead get type(f): {type(f)}"
            )
        # 如果路径f不存在，则创建它
        if not os.path.exists(f):  # type: ignore[arg-type]
            os.makedirs(f)  # type: ignore[arg-type]

        # 构建模型proto文件的完整路径
        model_proto_file = os.path.join(f, _constants.ONNX_ARCHIVE_MODEL_PROTO_NAME)  # type: ignore[arg-type]
        
        # 使用torch的文件操作，以二进制写模式打开模型proto文件
        with torch.serialization._open_file_like(model_proto_file, "wb") as opened_file:
            opened_file.write(model_bytes)

        # 遍历导出映射中的键值对
        for k, v in export_map.items():
            # 构建权重proto文件的完整路径
            weight_proto_file = os.path.join(f, k)  # type: ignore[arg-type]
            
            # 使用torch的文件操作，以二进制写模式打开权重proto文件
            with torch.serialization._open_file_like(
                weight_proto_file, "wb"
            ) as opened_file:
                opened_file.write(v)
    
    # 如果导出类型不是DIRECTORY，则抛出数值错误异常
    else:
        raise ValueError("Unknown export type")
# 使用装饰器 @_beartype.beartype 对函数进行类型检查和验证
@_beartype.beartype
# 将自定义的 ONNX 脚本函数插入到 ModelProto 中
def _add_onnxscript_fn(
    model_bytes: bytes,  # 模型的字节表示
    custom_opsets: Mapping[str, int],  # 自定义操作集合，映射从操作名称到版本号的整数
) -> bytes:  # 函数返回字节表示的模型

    """Insert model-included custom onnx-script function into ModelProto"""

    try:
        import onnx  # 尝试导入 ONNX 模块
    except ImportError as e:
        # 如果导入失败，抛出自定义的 OnnxExporterError 异常
        raise errors.OnnxExporterError("Module onnx is not installed!") from e

    # 将模型字节加载为 ONNX 的 ModelProto 对象
    model_proto = onnx.load_model_from_string(model_bytes)  # type: ignore[attr-defined]

    # 迭代图节点，将包含的自定义函数添加到模型协议中
    onnx_function_list = list()  # type: ignore[var-annotated]
    included_node_func = set()  # type: Set[str]

    # 递归查找 ONNXFunction 操作，可能包含控制流操作
    _find_onnxscript_op(
        model_proto.graph, included_node_func, custom_opsets, onnx_function_list
    )

    # 如果找到了自定义函数，则将其添加到模型协议中的函数列表，并序列化模型协议为字节表示
    if onnx_function_list:
        model_proto.functions.extend(onnx_function_list)
        model_bytes = model_proto.SerializeToString()

    # 返回更新后的模型字节表示
    return model_bytes


# 通过装饰器 @_beartype.beartype 对函数进行类型检查和验证
@_beartype.beartype
# 递归迭代 ModelProto 查找 ONNXFunction 操作，可能包含控制流操作
def _find_onnxscript_op(
    graph_proto,  # 图的协议缩写
    included_node_func: Set[str],  # 包含的节点函数名称集合
    custom_opsets: Mapping[str, int],  # 自定义操作集合，映射从操作名称到版本号的整数
    onnx_function_list: List,  # 存储找到的 ONNXFunction 的列表
):
    """Recursively iterate ModelProto to find ONNXFunction op as it may contain control flow Op."""
    # 遍历图协议中的每个节点
    for node in graph_proto.node:
        # 根据节点的域和操作类型生成节点类型字符串
        node_kind = node.domain + "::" + node.op_type
        # 对控制流节点（如IF/Loop）进行递归查找，因为它们可能包含内部图协议
        for attr in node.attribute:
            # 如果属性中包含子图，则递归查找其中的节点函数
            if attr.g is not None:
                _find_onnxscript_op(
                    attr.g, included_node_func, custom_opsets, onnx_function_list
                )
        
        # 获取注册表中与当前节点类型匹配的函数组
        onnx_function_group = registration.registry.get_function_group(node_kind)
        
        # 排除特定情况：节点域不为空，并且不是 aten、prim、onnx，且在注册表中有对应的函数组
        if (
            node.domain
            and not jit_utils.is_aten(node.domain)
            and not jit_utils.is_prim(node.domain)
            and not jit_utils.is_onnx(node.domain)
            and onnx_function_group is not None
            and node_kind not in included_node_func
        ):
            # 获取指定域的自定义操作集合的版本号，默认为1
            specified_version = custom_opsets.get(node.domain, 1)
            
            # 根据指定版本号获取对应的 ONNX 函数
            onnx_fn = onnx_function_group.get(specified_version)
            
            # 如果找到对应的 ONNX 函数
            if onnx_fn is not None:
                # 如果 ONNX 函数具有 "to_function_proto" 属性，则转换为函数协议
                if hasattr(onnx_fn, "to_function_proto"):
                    onnx_function_proto = onnx_fn.to_function_proto()  # type: ignore[attr-defined]
                    # 将转换后的函数协议添加到列表中
                    onnx_function_list.append(onnx_function_proto)
                    # 将当前节点类型添加到已包含节点函数的集合中
                    included_node_func.add(node_kind)
                continue

            # 如果未找到对应的 ONNX 函数，则抛出不支持的操作错误
            raise errors.UnsupportedOperatorError(
                node_kind,
                specified_version,
                onnx_function_group.get_min_supported()
                if onnx_function_group
                else None,
            )
    
    # 返回最终的 ONNX 函数列表和已包含的节点函数集合
    return onnx_function_list, included_node_func
```