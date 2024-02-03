# `.\PaddleOCR\tools\export_model.py`

```py
# 导入所需的库
import os
import sys

# 获取当前文件所在目录的绝对路径
__dir__ = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在目录添加到系统路径中
sys.path.append(__dir__)
# 将当前文件所在目录的上一级目录添加到系统路径中
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

# 导入 argparse 库
import argparse

# 导入 paddle 库
import paddle
# 从 paddle.jit 模块中导入 to_static 函数
from paddle.jit import to_static

# 从 ppocr.modeling.architectures 模块中导入 build_model 函数
from ppocr.modeling.architectures import build_model
# 从 ppocr.postprocess 模块中导入 build_post_process 函数
from ppocr.postprocess import build_post_process
# 从 ppocr.utils.save_load 模块中导入 load_model 函数
from ppocr.utils.save_load import load_model
# 从 ppocr.utils.logging 模块中导入 get_logger 函数
from ppocr.utils.logging import get_logger
# 从 tools.program 模块中导入 load_config, merge_config, ArgsParser 函数

# 定义一个函数，用于导出单个模型
def export_single_model(model,
                        arch_config,
                        save_path,
                        logger,
                        input_shape=None,
                        quanter=None):
    # 如果选择的算法是 SRN
    if arch_config["algorithm"] == "SRN":
        # 获取最大文本长度
        max_text_length = arch_config["Head"]["max_text_length"]
        # 定义其他输入形状
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 64, 256], dtype="float32"), [
                    paddle.static.InputSpec(
                        shape=[None, 256, 1],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, max_text_length, 1], dtype="int64"),
                    paddle.static.InputSpec(
                        shape=[None, 8, max_text_length, max_text_length],
                        dtype="int64"), paddle.static.InputSpec(
                            shape=[None, 8, max_text_length, max_text_length],
                            dtype="int64")
                ]
        ]
        # 将模型转换为静态图模型
        model = to_static(model, input_spec=other_shape)
    # 如果选择的算法是 SAR
    elif arch_config["algorithm"] == "SAR":
        # 定义其他输入形状
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"),
            [paddle.static.InputSpec(
                shape=[None], dtype="float32")]
        ]
        # 将模型转换为静态图模型
        model = to_static(model, input_spec=other_shape)
    # 如果选择的算法是 SVTR_LCNet 或 SVTR_HGNet
    elif arch_config["algorithm"] in ["SVTR_LCNet", "SVTR_HGNet"]:
        # 定义其他输入形状
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, -1], dtype="float32"),
        ]
        # 将模型转换为静态图模型
        model = to_static(model, input_spec=other_shape)
    # 如果选择的算法是 SVTR
    elif arch_config["algorithm"] == "SVTR":
        # 定义其他输入形状
        other_shape = [
            paddle.static.InputSpec(
                shape=[None] + input_shape, dtype="float32"),
        ]
        # 将模型转换为静态图模型
        model = to_static(model, input_spec=other_shape)
    # 如果选择的算法是 PREN
    elif arch_config["algorithm"] == "PREN":
        # 定义其他输入形状
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 256], dtype="float32"),
        ]
        # 将模型转换为静态图模型
        model = to_static(model, input_spec=other_shape)
    # 如果模型类型是 "sr"，设置输入形状为 [None, 3, 16, 64]，数据类型为 float32
    elif arch_config["model_type"] == "sr":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 16, 64], dtype="float32")
        ]
        # 将动态图模型转换为静态图模型，指定输入形状为 other_shape
        model = to_static(model, input_spec=other_shape)
    
    # 如果算法是 "ViTSTR"，设置输入形状为 [None, 1, 224, 224]，数据类型为 float32
    elif arch_config["algorithm"] == "ViTSTR":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 224, 224], dtype="float32"),
        ]
        # 将动态图模型转换为静态图模型，指定输入形状为 other_shape
        model = to_static(model, input_spec=other_shape)
    
    # 如果算法是 "ABINet"，设置输入形状为 [None, 3, 32, 128]，数据类型为 float32
    elif arch_config["algorithm"] == "ABINet":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 32, 128], dtype="float32"),
        ]
        # 将动态图模型转换为静态图模型，指定输入形状为 other_shape
        model = to_static(model, input_spec=other_shape)
    
    # 如果算法是 "NRTR"、"SPIN" 或 "RFL"，设置输入形状为 [None, 1, 32, 100]，数据类型为 float32
    elif arch_config["algorithm"] in ["NRTR", "SPIN", 'RFL']:
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 1, 32, 100], dtype="float32"),
        ]
        # 将动态图模型转换为静态图模型，指定输入形状为 other_shape
        model = to_static(model, input_spec=other_shape)
    
    # 如果算法是 "SATRN"，设置输入形状为 [None, 3, 32, 100]，数据类型为 float32
    elif arch_config["algorithm"] == 'SATRN':
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 32, 100], dtype="float32"),
        ]
        # 将动态图模型转换为静态图模型，指定输入形状为 other_shape
        model = to_static(model, input_spec=other_shape)
    
    # 如果算法是 "VisionLAN"，设置输入形状为 [None, 3, 64, 256]，数据类型为 float32
    elif arch_config["algorithm"] == "VisionLAN":
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 64, 256], dtype="float32"),
        ]
        # 将动态图模型转换为静态图模型，指定输入形状为 other_shape
        model = to_static(model, input_spec=other_shape)
    # 如果算法是RobustScanner，则设置最大文本长度
    elif arch_config["algorithm"] == "RobustScanner":
        # 获取最大文本长度
        max_text_length = arch_config["Head"]["max_text_length"]
        # 定义其他输入形状
        other_shape = [
            paddle.static.InputSpec(
                shape=[None, 3, 48, 160], dtype="float32"), [
                    paddle.static.InputSpec(
                        shape=[None, ], dtype="float32"),
                    paddle.static.InputSpec(
                        shape=[None, max_text_length], dtype="int64")
                ]
        ]
        # 将模型转换为静态图模式，并指定输入形状
        model = to_static(model, input_spec=other_shape)
    # 如果算法是CAN，则设置输入形状
    elif arch_config["algorithm"] == "CAN":
        # 定义其他输入形状
        other_shape = [[
            paddle.static.InputSpec(
                shape=[None, 1, None, None],
                dtype="float32"), paddle.static.InputSpec(
                    shape=[None, 1, None, None], dtype="float32"),
            paddle.static.InputSpec(
                shape=[None, arch_config['Head']['max_text_length']],
                dtype="int64")
        ]]
        # 将模型转换为静态图模式，并指定输入形状
        model = to_static(model, input_spec=other_shape)
    # 如果模型的算法是 LayoutLM、LayoutLMv2 或 LayoutXLM
    elif arch_config["algorithm"] in ["LayoutLM", "LayoutLMv2", "LayoutXLM"]:
        # 定义输入规格列表
        input_spec = [
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # input_ids
            paddle.static.InputSpec(
                shape=[None, 512, 4], dtype="int64"),  # bbox
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(
                shape=[None, 512], dtype="int64"),  # token_type_ids
            paddle.static.InputSpec(
                shape=[None, 3, 224, 224], dtype="int64"),  # image
        ]
        # 如果模型的骨干网络包含'Re'，则添加额外的输入规格
        if 'Re' in arch_config['Backbone']['name']:
            input_spec.extend([
                paddle.static.InputSpec(
                    shape=[None, 512, 3], dtype="int64"),  # entities
                paddle.static.InputSpec(
                    shape=[None, None, 2], dtype="int64"),  # relations
            ])
        # 如果模型的骨干网络不使用视觉骨干网络，则移除图像输入规格
        if model.backbone.use_visual_backbone is False:
            input_spec.pop(4)
        # 将模型转换为静态图模式，指定输入规格
        model = to_static(model, input_spec=[input_spec])
    else:
        # 推断形状为[3, -1, -1]
        infer_shape = [3, -1, -1]
        if arch_config["model_type"] == "rec":
            # 对于rec模型，H必须为32
            infer_shape = [3, 32, -1]
            if "Transform" in arch_config and arch_config["Transform"] is not None and arch_config["Transform"]["name"] == "TPS":
                # 当网络中存在TPS时，不支持可变长度输入，并且输入大小需要与训练期间相同
                logger.info("When there is tps in the network, variable length input is not supported, and the input size needs to be the same as during training")
                infer_shape[-1] = 100
        elif arch_config["model_type"] == "table":
            infer_shape = [3, 488, 488]
            if arch_config["algorithm"] == "TableMaster":
                infer_shape = [3, 480, 480]
            if arch_config["algorithm"] == "SLANet":
                infer_shape = [3, -1, -1]
        # 将模型转换为静态图模型，指定输入规格
        model = to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + infer_shape, dtype="float32")
            ])

    if arch_config["model_type"] != "sr" and arch_config["Backbone"]["name"] == "PPLCNetV3":
        # 对于rep lcnetv3
        for layer in model.sublayers():
            if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
                layer.rep()

    if quanter is None:
        # 保存模型
        paddle.jit.save(model, save_path)
    else:
        # 保存量化后的模型
        quanter.save_quantized_model(model, save_path)
    logger.info("inference model is saved to {}".format(save_path))
    return
# 主函数入口
def main():
    # 解析命令行参数
    FLAGS = ArgsParser().parse_args()
    # 加载配置文件
    config = load_config(FLAGS.config)
    # 合并配置文件
    config = merge_config(config, FLAGS.opt)
    # 获取日志记录器
    logger = get_logger()
    # 构建后处理器

    # 构建后处理器类
    post_process_class = build_post_process(config["PostProcess"],
                                            config["Global"])

    # 构建模型
    # 用于推荐算法
    # 用于超分辨率算法
    if config["Architecture"]["model_type"] == "sr":
        config['Architecture']["Transform"]['infer_mode'] = True
    model = build_model(config["Architecture"])
    # 加载模型
    load_model(config, model, model_type=config['Architecture']["model_type"])
    # 设置模型为评估模式
    model.eval()

    # 保存路径
    save_path = config["Global"]["save_inference_dir"]

    arch_config = config["Architecture"]

    if arch_config["algorithm"] == "SVTR" and arch_config["Head"][
            "name"] != 'MultiHead':
        input_shape = config["Eval"]["dataset"]["transforms"][-2][
            'SVTRRecResizeImg']['image_shape']
    else:
        input_shape = None

    if arch_config["algorithm"] in ["Distillation", ]:  # 蒸馏模型
        archs = list(arch_config["Models"].values())
        for idx, name in enumerate(model.model_name_list):
            sub_model_save_path = os.path.join(save_path, name, "inference")
            export_single_model(model.model_list[idx], archs[idx],
                                sub_model_save_path, logger)
    else:
        save_path = os.path.join(save_path, "inference")
        export_single_model(
            model, arch_config, save_path, logger, input_shape=input_shape)


if __name__ == "__main__":
    # 调用主函数
    main()
```