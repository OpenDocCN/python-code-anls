# `.\SenseVoiceSmall-src\utils\export_utils.py`

```
# 导入操作系统模块和 PyTorch 库
import os
import torch


# 定义导出模型的函数，支持量化、ONNX 版本和导出类型等参数
def export(
    model, quantize: bool = False, opset_version: int = 14, type="onnx", **kwargs
):
    # 调用模型的 export 方法，获取导出脚本
    model_scripts = model.export(**kwargs)
    # 获取输出目录，默认为初始化参数所在的目录
    export_dir = kwargs.get("output_dir", os.path.dirname(kwargs.get("init_param")))
    # 创建输出目录，如果已经存在则不报错
    os.makedirs(export_dir, exist_ok=True)

    # 确保 model_scripts 是一个列表或元组
    if not isinstance(model_scripts, (list, tuple)):
        model_scripts = (model_scripts,)
    # 遍历每个模型脚本
    for m in model_scripts:
        # 设置模型为评估模式
        m.eval()
        # 根据类型选择导出方式
        if type == "onnx":
            _onnx(
                m,
                quantize=quantize,
                opset_version=opset_version,
                export_dir=export_dir,
                **kwargs,
            )
        # 打印输出目录
        print("output dir: {}".format(export_dir))

    # 返回输出目录
    return export_dir


# 定义内部函数用于导出 ONNX 模型
def _onnx(
    model,
    quantize: bool = False,
    opset_version: int = 14,
    export_dir: str = None,
    **kwargs,
):
    # 获取模型的虚拟输入
    dummy_input = model.export_dummy_inputs()

    # 获取是否详细输出的参数
    verbose = kwargs.get("verbose", False)

    # 获取导出的模型名称
    export_name = model.export_name()
    # 组合导出路径
    model_path = os.path.join(export_dir, export_name)
    # 执行 ONNX 导出
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=verbose,
        opset_version=opset_version,
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
    )

    # 如果需要量化模型
    if quantize:
        # 导入量化相关的模块
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx

        # 生成量化模型的路径
        quant_model_path = model_path.replace(".onnx", "_quant.onnx")
        # 如果量化模型尚不存在，则进行量化
        if not os.path.exists(quant_model_path):
            # 加载原始 ONNX 模型
            onnx_model = onnx.load(model_path)
            # 获取模型中所有节点的名称
            nodes = [n.name for n in onnx_model.graph.node]
            # 定义要排除的节点
            nodes_to_exclude = [
                m for m in nodes if "output" in m or "bias_encoder" in m or "bias_decoder" in m
            ]
            # 执行动态量化
            quantize_dynamic(
                model_input=model_path,
                model_output=quant_model_path,
                op_types_to_quantize=["MatMul"],
                per_channel=True,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
                nodes_to_exclude=nodes_to_exclude,
            )
```