# `.\pytorch\torch\backends\_coreml\preprocess.py`

```py
# mypy: allow-untyped-defs
# 导入 hashlib 模块用于哈希计算，json 模块用于 JSON 数据处理
import hashlib
import json
# 导入 coremltools 库，并忽略类型检查
import coremltools as ct  # type: ignore[import]
# 从 coremltools.converters.mil.input_types 导入 TensorType 类型，忽略类型检查
from coremltools.converters.mil.input_types import TensorType  # type: ignore[import]
# 从 coremltools.converters.mil.mil 导入 types，忽略类型检查
from coremltools.converters.mil.mil import types  # type: ignore[import]
# 从 coremltools.models.neural_network 导入 quantization_utils，忽略类型检查
from coremltools.models.neural_network import quantization_utils  # type: ignore[import]

# 导入 torch 库
import torch

# 定义 CoreML 模型的元数据版本和来源
CT_METADATA_VERSION = "com.github.apple.coremltools.version"
CT_METADATA_SOURCE = "com.github.apple.coremltools.source"


# 定义标量类型类，包括 Float、Double、Int、Long 和 Undefined 类型
class ScalarType:
    Float = 0
    Double = 1
    Int = 2
    Long = 3
    Undefined = 4


# torch 到 coremltools 中支持的 Tensor 类型的映射字典
torch_to_mil_types = {
    ScalarType.Float: types.fp32,
    ScalarType.Double: types.fp64,
    ScalarType.Int: types.int32,
    ScalarType.Long: types.int64,
}


# 定义 CoreML 计算单元的类，包括 CPU、CPUAndGPU 和 ALL 选项
class CoreMLComputeUnit:
    CPU = "cpuOnly"
    CPUAndGPU = "cpuAndGPU"
    ALL = "all"


# 定义 CoreML 量化模式的类，包括 LINEAR、LINEAR_SYMMETRIC 和 NONE 选项
class CoreMLQuantizationMode:
    LINEAR = "linear"
    LINEAR_SYMMETRIC = "linear_symmetric"
    NONE = "none"


# 定义函数 TensorSpec，用于创建 Tensor 的规格，包括形状和数据类型
def TensorSpec(shape, dtype=ScalarType.Float):
    return (shape, dtype)


# 定义函数 CompileSpec，用于创建编译规格，包括输入、输出、计算后端、是否允许低精度、量化模式和输出 MLModel 路径等参数
def CompileSpec(
    inputs,
    outputs,
    backend=CoreMLComputeUnit.CPU,
    allow_low_precision=True,
    quantization_mode=CoreMLQuantizationMode.NONE,
    mlmodel_export_path=None,
):
    return (
        inputs,
        outputs,
        backend,
        allow_low_precision,
        quantization_mode,
        mlmodel_export_path,
    )


# 私有函数 _check_enumerated_shape，用于检查形状是否为枚举类型
def _check_enumerated_shape(shape):
    for s in shape:
        if not isinstance(s, (list, tuple)):
            return False
    return True


# 私有函数 _convert_to_mil_type，将输入形状和数据类型转换为 coremltools 的 TensorType 类型
def _convert_to_mil_type(shape, dtype, name: str):
    mil_shape = shape
    if _check_enumerated_shape(shape):
        mil_shape = ct.EnumeratedShapes(shape)
    ml_type = TensorType(shape=mil_shape, dtype=torch_to_mil_types[dtype])
    ml_type.name = name
    return ml_type


# 函数 preprocess，用于预处理 Torch 脚本模块，生成 MLModel
def preprocess(script_module: torch._C.ScriptObject, compile_spec: Dict[str, Tuple]):
    spec = compile_spec["forward"]
    (
        input_specs,
        output_specs,
        backend,
        allow_low_precision,
        quantization_mode,
        mlmodel_export_path,
    ) = spec
    mil_inputs = []
    inputs = []
    # 遍历输入规格列表，创建输入名称、数据类型和形状的列表，并生成对应的 coremltools 的 TensorType 对象列表
    for index, input in enumerate(input_specs):
        shape, dtype = input
        name = "input_" + str(index)
        inputs.append([name, str(dtype), str(shape)])
        ml_type = _convert_to_mil_type(shape, dtype, name)
        mil_inputs.append(ml_type)
    # 构建 Torch 脚本模型
    model = torch.jit.RecursiveScriptModule._construct(script_module, lambda x: None)
    # 将 Torch 模型转换为 CoreML 模型
    mlmodel = ct.convert(model, inputs=mil_inputs)

    # 如果指定了量化模式，对生成的 MLModel 进行权重量化处理
    if quantization_mode != CoreMLQuantizationMode.NONE:
        quant_model_spec = quantization_utils.quantize_weights(
            mlmodel, nbits=8, quantization_mode=quantization_mode
        )
        mlmodel = ct.models.MLModel(quant_model_spec)

    # 获取最终生成的 MLModel 的规格
    spec = mlmodel.get_spec()
    # 确保描述输出的数量与输出规范的数量相等，忽略类型检查的属性定义
    assert len(spec.description.output) == len(output_specs)  # type: ignore[attr-defined]
    
    # 初始化一个空的输出列表
    outputs = []
    
    # 遍历输出规范的索引和内容
    for index, output in enumerate(output_specs):
        # 解构出形状和数据类型
        shape, dtype = output
        
        # 获取指定索引处的输出名称，忽略类型检查的属性定义
        name = spec.description.output[index].name  # type: ignore[attr-defined]
        
        # 将输出名称、数据类型和形状转换为字符串，添加到输出列表中
        outputs.append([name, str(dtype), str(shape)])
    
    # 创建一个 CoreML 的模型对象，使用给定的规范
    mlmodel = ct.models.model.MLModel(spec)
    
    # 打印 MLModel 对象的信息
    print(mlmodel)

    # 如果指定了导出路径，将 CoreML 的模型保存为 .mlmodel 文件
    if mlmodel_export_path is not None:
        print(f"Saving CoreML .mlmodel file to {mlmodel_export_path}")
        mlmodel.save(mlmodel_export_path)

    # 构建配置信息，包括规范版本、后端、低精度设置
    config = {
        "spec_ver": str(spec.specificationVersion),  # type: ignore[attr-defined]
        "backend": backend,
        "allow_low_precision": str(allow_low_precision),
    }
    
    # 构建元数据信息，包括 coremltools 版本和源 Torch 版本
    metadata = {
        "coremltool_ver": mlmodel.user_defined_metadata[CT_METADATA_VERSION],
        "torch_ver": mlmodel.user_defined_metadata[CT_METADATA_SOURCE],
    }
    
    # 构建 CoreML 编译规范，包括输入、输出、配置和元数据
    coreml_compile_spec = {
        "inputs": inputs,
        "outputs": outputs,
        "config": config,
        "metadata": metadata,
    }
    
    # 将规范序列化为字符串，忽略类型检查的属性定义
    mlmodel = spec.SerializeToString()  # type: ignore[attr-defined]

    # 返回包含模型、哈希和附加信息的字典
    return {
        "model": mlmodel,
        "hash": str(hashlib.sha256(mlmodel).hexdigest()),
        "extra": json.dumps(coreml_compile_spec),
    }
```