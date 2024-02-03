# `.\segment-anything\scripts\export_onnx_model.py`

```py
# 导入 torch 库
import torch

# 从 segment_anything 模块中导入 sam_model_registry 和 SamOnnxModel
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

# 导入 argparse 和 warnings 库
import argparse
import warnings

# 尝试导入 onnxruntime 库，如果导入失败则设置 onnxruntime_exists 为 False
try:
    import onnxruntime  # type: ignore
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

# 创建命令行参数解析器
parser = argparse.ArgumentParser(
    description="Export the SAM prompt encoder and mask decoder to an ONNX model."
)

# 添加命令行参数：模型检查点路径
parser.add_argument(
    "--checkpoint", type=str, required=True, help="The path to the SAM model checkpoint."
)

# 添加命令行参数：保存 ONNX 模型的文件名
parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

# 添加命令行参数：SAM 模型类型
parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

# 添加命令行参数：是否只返回单个 mask
parser.add_argument(
    "--return-single-mask",
    action="store_true",
    help=(
        "If true, the exported ONNX model will only return the best mask, "
        "instead of returning multiple masks. For high resolution images "
        "this can improve runtime when upscaling masks is expensive."
    ),
)

# 添加命令行参数：ONNX opset 版本
parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

# 添加命令行参数：是否量化输出
parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)

# 添加命令行参数：是否使用 GELU 近似
parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

# 添加命令行参数：是否使用稳定性评分
    # 设置参数为布尔类型，当存在时设为True
    action="store_true",
    # 帮助信息，替换模型预测的掩模质量分数为在低分辨率掩模上使用偏移1.0计算的稳定性分数
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
# 添加命令行参数，用于指定是否返回额外的指标
parser.add_argument(
    "--return-extra-metrics",
    action="store_true",
    help=(
        "The model will return five results: (masks, scores, stability_scores, "
        "areas, low_res_logits) instead of the usual three. This can be "
        "significantly slower for high resolution outputs."
    ),
)

# 运行导出函数，加载模型并生成 ONNX 模型
def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    # 打印信息，加载模型
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    # 创建 SamOnnxModel 对象
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    # 如果启用 GELU 近似，则将模型中的 GELU 层设置为 "tanh" 近似
    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    # 定义动态轴
    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    # 获取模型的相关参数
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    
    # 创建虚拟输入数据
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    # 运行 ONNX 模型
    _ = onnx_model(**dummy_inputs)

    # 定义输出的名称
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    # 使用 warnings 模块捕获警告信息
    with warnings.catch_warnings():
        # 忽略 torch.jit.TracerWarning 类别的警告
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # 忽略 UserWarning 类别的警告
        warnings.filterwarnings("ignore", category=UserWarning)
        # 以二进制写模式打开输出文件
        with open(output, "wb") as f:
            # 打印导出 ONNX 模型的信息
            print(f"Exporting onnx model to {output}...")
            # 导出 ONNX 模型
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    # 如果存在 ONNXRuntime
    if onnxruntime_exists:
        # 将 dummy_inputs 转换为 numpy 格式
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        # 设置 CPU 提供程序为默认
        providers = ["CPUExecutionProvider"]
        # 创建 ONNXRuntime 推理会话
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        # 运行 ONNXRuntime 推理会话
        _ = ort_session.run(None, ort_inputs)
        # 打印成功使用 ONNXRuntime 运行模型的信息
        print("Model has successfully been run with ONNXRuntime.")
# 将 PyTorch 张量转换为 NumPy 数组
def to_numpy(tensor):
    return tensor.cpu().numpy()

# 如果作为脚本运行，则解析命令行参数
if __name__ == "__main__":
    args = parser.parse_args()
    # 运行导出函数，传入各种参数
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    # 如果需要量化输出
    if args.quantize_out is not None:
        # 检查是否存在 onnxruntime 库
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        # 执行动态量化，将量化后的模型写入指定路径
        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
```