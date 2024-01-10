# `so-vits-svc\train_diff.py`

```
# 导入必要的库
import argparse
import torch
from loguru import logger
from torch.optim import lr_scheduler
from diffusion.data_loaders import get_data_loaders
from diffusion.logger import utils
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel
from diffusion.vocoder import Vocoder

# 解析命令行参数
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

# 如果作为主程序运行
if __name__ == '__main__':
    # 解析命令行参数
    cmd = parse_args()
    
    # 加载配置文件
    args = utils.load_config(cmd.config)
    logger.info(' > config:'+ cmd.config)
    logger.info(' > exp:'+ args.env.expdir)
    
    # 加载声码器
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # 加载模型
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max
                )
    
    logger.info(f' > Now model timesteps is {model.timesteps}, and k_step_max is {model.k_step_max}')
    
    # 加载参数
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * (args.train.gamma ** max(((initial_global_step-2)//args.train.decay_step),0) )
        param_group['weight_decay'] = args.train.weight_decay
    # 创建一个学习率调度器，根据给定的步长和衰减因子来调整优化器的学习率，同时设置初始的全局步数
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma,last_epoch=initial_global_step-2)
    
    # 设置设备，如果选择的设备是 'cuda'，则设置当前 CUDA 设备为指定的 GPU ID，然后将模型移动到指定的设备上
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu_id)
    model.to(args.device)
    
    # 将优化器中的参数状态转移到指定的设备上
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # 获取训练和验证数据加载器
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # 运行训练函数，传入参数和模型相关的信息
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid)
```