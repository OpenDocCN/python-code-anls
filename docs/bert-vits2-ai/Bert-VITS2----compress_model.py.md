# `Bert-VITS2\compress_model.py`

```

# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict
# 从 text.symbols 模块中导入 symbols 列表
from text.symbols import symbols
# 导入 torch 模块
import torch
# 从 tools.log 模块中导入 logger 对象
from tools.log import logger
# 导入 utils 模块
import utils
# 从 models 模块中导入 SynthesizerTrn 类
from models import SynthesizerTrn
# 导入 os 模块
import os

# 定义函数 copyStateDict，用于复制状态字典
def copyStateDict(state_dict):
    # 如果状态字典的第一个键以 "module" 开头，则 start_idx 为 1，否则为 0
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    # 创建一个有序字典 new_state_dict
    new_state_dict = OrderedDict()
    # 遍历状态字典的键值对
    for k, v in state_dict.items():
        # 将键按照 "." 分割，取 start_idx 后的部分，用 "," 连接起来作为新的键
        name = ",".join(k.split(".")[start_idx:])
        # 将新的键值对添加到 new_state_dict 中
        new_state_dict[name] = v
    # 返回新的状态字典
    return new_state_dict

# 定义函数 removeOptimizer，用于移除优化器
def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config)

    # 创建 SynthesizerTrn 模型
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    # 创建 AdamW 优化器
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # 加载输入模型的状态字典
    state_dict_g = torch.load(input_model, map_location="cpu")
    # 复制状态字典
    new_dict_g = copyStateDict(state_dict_g)
    keys = []
    # 遍历新状态字典的模型部分
    for k, v in new_dict_g["model"].items():
        # 如果键中包含 "enc_q"，则跳过
        if "enc_q" in k:
            continue  # noqa: E701
        keys.append(k)

    # 根据是否使用半精度，创建新的状态字典
    new_dict_g = (
        {k: new_dict_g["model"][k].half() for k in keys}
        if ishalf
        else {k: new_dict_g["model"][k] for k in keys}
    )

    # 保存新的模型状态字典、迭代次数、优化器状态字典和学习率
    torch.save(
        {
            "model": new_dict_g,
            "iteration": 0,
            "optimizer": optim_g.state_dict(),
            "learning_rate": 0.0001,
        },
        output_model,
    )

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 导入 argparse 模块
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数选项
    parser.add_argument("-c", "--config", type=str, default="configs/config.json")
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument(
        "-hf", "--half", action="store_true", default=False, help="Save as FP16"
    )

    # 解析命令行参数
    args = parser.parse_args()

    output = args.output

    # 如果未指定输出文件名
    if output is None:
        # 导入 os.path 模块
        import os.path
        # 获取输入文件的文件名和扩展名
        filename, ext = os.path.splitext(args.input)
        # 如果使用半精度，则在文件名末尾添加 "_half"，否则为空字符串
        half = "_half" if args.half else ""
        # 构建输出文件名
        output = filename + "_release" + half + ext

    # 调用 removeOptimizer 函数，移除优化器
    removeOptimizer(args.config, args.input, args.half, output)
    # 打印压缩模型成功的信息，输出模型的绝对路径
    logger.info(f"压缩模型成功, 输出模型: {os.path.abspath(output)}")

```