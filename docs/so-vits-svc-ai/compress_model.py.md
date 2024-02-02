# `so-vits-svc\compress_model.py`

```py
# 从 collections 模块中导入 OrderedDict 类
from collections import OrderedDict

# 导入 torch 模块
import torch

# 导入 utils 模块
import utils
# 从 models 模块中导入 SynthesizerTrn 类
from models import SynthesizerTrn

# 定义函数 copyStateDict，用于复制状态字典
def copyStateDict(state_dict):
    # 如果状态字典的第一个键以 'module' 开头
    if list(state_dict.keys())[0].startswith('module'):
        # 则起始索引为 1
        start_idx = 1
    else:
        # 否则起始索引为 0
        start_idx = 0
    # 创建一个新的有序字典
    new_state_dict = OrderedDict()
    # 遍历状态字典的键值对
    for k, v in state_dict.items():
        # 以点号分割键，并取起始索引后的部分作为新的键名
        name = ','.join(k.split('.')[start_idx:])
        # 将新的键值对添加到新的状态字典中
        new_state_dict[name] = v
    # 返回新的状态字典
    return new_state_dict

# 定义函数 removeOptimizer，用于移除优化器
def removeOptimizer(config: str, input_model: str, ishalf: bool, output_model: str):
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(config)

    # 创建一个 SynthesizerTrn 对象
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model)

    # 使用 AdamW 优化器对 net_g 的参数进行优化
    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hps.train.learning_rate,
                                betas=hps.train.betas,
                                eps=hps.train.eps)

    # 加载输入模型的状态字典
    state_dict_g = torch.load(input_model, map_location="cpu")
    # 复制输入模型的状态字典
    new_dict_g = copyStateDict(state_dict_g)
    # 创建一个空列表 keys
    keys = []
    # 遍历新的状态字典中的模型项
    for k, v in new_dict_g['model'].items():
        # 如果键名中包含 "enc_q"，则跳过当前循环
        if "enc_q" in k: continue  # noqa: E701
        # 否则将键名添加到 keys 列表中
        keys.append(k)
    
    # 如果 ishalf 为真，则将新的状态字典中的模型项转换为半精度
    new_dict_g = {k: new_dict_g['model'][k].half() for k in keys} if ishalf else {k: new_dict_g['model'][k] for k in keys}

    # 保存新的状态字典、迭代次数、优化器状态字典和学习率到输出模型中
    torch.save(
        {
            'model': new_dict_g,
            'iteration': 0,
            'optimizer': optim_g.state_dict(),
            'learning_rate': 0.0001
        }, output_model)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 导入 argparse 模块
    import argparse
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加配置文件参数
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        default='configs/config.json')
    # 添加输入模型参数
    parser.add_argument("-i", "--input", type=str)
    # 添加输出模型参数
    parser.add_argument("-o", "--output", type=str, default=None)
    # 添加是否使用半精度参数
    parser.add_argument('-hf', '--half', action='store_true', default=False, help='Save as FP16')
    # 解析命令行参数
    args = parser.parse_args()

    # 获取输出文件名
    output = args.output

    # 如果输出文件名为空
    if output is None:
        # 导入 os.path 模块
        import os.path
        # 获取输入文件的文件名和扩展名
        filename, ext = os.path.splitext(args.input)
        # 如果参数中包含 half，则在文件名后添加 "_half"
        half = "_half" if args.half else ""
        # 组合输出文件名
        output = filename + "_release" + half + ext

    # 调用 removeOptimizer 函数，传入配置文件、输入文件、是否压缩一半、输出文件名
    removeOptimizer(args.config, args.input, args.half, output)
```