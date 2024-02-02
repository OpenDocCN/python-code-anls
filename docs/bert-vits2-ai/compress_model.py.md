# `Bert-VITS2\compress_model.py`

```py
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

# 定义函数 copyStateDict，接受一个 state_dict 参数
def copyStateDict(state_dict):
    # 如果 state_dict 的第一个键名以 "module" 开头
    if list(state_dict.keys())[0].startswith("module"):
        # 则 start_idx 为 1
        start_idx = 1
    else:
        # 否则 start_idx 为 0
        start_idx = 0
    # 创建一个新的有序字典对象 new_state_dict
    new_state_dict = OrderedDict()
    # 遍历 state_dict 的键值对
    for k, v in state_dict.items():
        # 将键名按 "." 分割，取从 start_idx 开始的部分，用 "," 连接起来作为新的键名
        name = ",".join(k.split(".")[start_idx:])
        # 将新的键值对添加到 new_state_dict 中
        new_state_dict[name] = v
    # 返回新的有序字典对象
    return new_state_dict

# 定义函数 removeOptimizer，接受四个参数：config、input_model、ishalf、output_model
def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config)

    # 创建 SynthesizerTrn 类的实例 net_g
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    # 使用 AdamW 优化器初始化 net_g
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # 从 input_model 加载模型参数到 state_dict_g
    state_dict_g = torch.load(input_model, map_location="cpu")
    # 调用 copyStateDict 函数，将 state_dict_g 处理成新的字典对象 new_dict_g
    new_dict_g = copyStateDict(state_dict_g)
    # 创建一个空列表 keys
    keys = []
    # 遍历 new_dict_g["model"] 的键值对
    for k, v in new_dict_g["model"].items():
        # 如果键名中包含 "enc_q"，则跳过当前循环
        if "enc_q" in k:
            continue  # noqa: E701
        # 将键名添加到 keys 列表中
        keys.append(k)

    # 根据 ishalf 参数，将 new_dict_g["model"] 中的部分参数转换为半精度浮点数
    # 或保持原精度不变，并重新赋值给 new_dict_g
    new_dict_g = (
        {k: new_dict_g["model"][k].half() for k in keys}
        if ishalf
        else {k: new_dict_g["model"][k] for k in keys}
    )

    # 将新的模型参数、迭代次数、优化器状态、学习率保存到 output_model
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

    # 创建 ArgumentParser 对象 parser
    parser = argparse.ArgumentParser()
    # 添加命令行参数 -c/--config，默认值为"configs/config.json"
    parser.add_argument("-c", "--config", type=str, default="configs/config.json")
    # 添加命令行参数 -i/--input
    parser.add_argument("-i", "--input", type=str)
    # 添加命令行参数 -o/--output，默认值为 None
    parser.add_argument("-o", "--output", type=str, default=None)
    # 添加一个名为"-hf"或"--half"的命令行参数，如果存在则设置为True，否则设置为False，帮助信息为"Save as FP16"
    parser.add_argument(
        "-hf", "--half", action="store_true", default=False, help="Save as FP16"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 获取输出文件名
    output = args.output

    # 如果输出文件名为空
    if output is None:
        # 导入os.path模块
        import os.path

        # 获取输入文件的文件名和扩展名
        filename, ext = os.path.splitext(args.input)
        # 如果参数中存在"--half"，则在文件名后添加"_half"，否则为空
        half = "_half" if args.half else ""
        # 设置输出文件名为输入文件名+"_release"+half+扩展名
        output = filename + "_release" + half + ext

    # 调用removeOptimizer函数，传入配置文件、输入文件、是否压缩为半精度、输出文件名
    removeOptimizer(args.config, args.input, args.half, output)
    # 记录日志，输出压缩模型成功，并显示输出模型的绝对路径
    logger.info(f"压缩模型成功, 输出模型: {os.path.abspath(output)}")
```