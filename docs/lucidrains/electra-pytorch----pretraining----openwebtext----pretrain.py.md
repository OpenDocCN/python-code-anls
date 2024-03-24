# `.\lucidrains\electra-pytorch\pretraining\openwebtext\pretrain.py`

```py
# 导入必要的库
import os
import sys

# 获取当前文件所在目录的绝对路径
dir_path = os.path.dirname(os.path.realpath(__file__))
# 获取当前文件所在目录的父目录的绝对路径
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# 将父目录的路径插入到系统路径中
sys.path.insert(0, parent_dir_path)

# 导入其他必要的库
import random
import logging
from time import time
from dataclasses import dataclass

import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from electra_pytorch import Electra

from openwebtext import arg
from openwebtext.dataset import load_owt, new_tokenizer, wrap_example_builder

logger = logging.getLogger(__name__)

########################################################################################################
## args

# 定义参数类
@dataclass
class Args:
    data_dir: arg.Str = 'data/openwebtext_features'
    data_vocab_file: arg.Str = 'data/vocab.txt'
    data_n_tensors_per_file: arg.Int = 2048
    data_max_seq_length: arg.Int = 128

    gpu: arg.Int = 0
    gpu_enabled: arg.Bool = True
    gpu_deterministic: arg.Bool = False
    gpu_mixed_precision: arg.Bool = False
    distributed_port: arg.Int = 8888
    distributed_enabled: arg.Bool = True
    distributed_world_size: arg.Int = 4

    model_generator: arg.Str = 'pretraining/openwebtext/small_generator.json'
    model_discriminator: arg.Str = 'pretraining/openwebtext/small_discriminator.json'
    model_mask_prob: arg.Float = 0.15

    opt_lr: arg.Float = 5e-4
    opt_batch_size: arg.Int = 128 // (distributed_world_size if distributed_enabled else 1)
    opt_warmup_steps: arg.Int = 10_000
    opt_num_training_steps: arg.Int = 200_000

    step_log: arg.Int = 10
    step_ckpt: arg.Int = 10_000


########################################################################################################
## train

# 定义训练函数
def train(rank, args):

    #######################
    ## distributed

    # 如果启用分布式训练，则初始化进程组
    if args.distributed_enabled:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.distributed_world_size,
            rank=rank)
    # 如果启用 GPU，则选择对应的设备
    if args.gpu_enabled:
        device = torch.device('cuda:{}'.format(rank))
    else:
        device = torch.device('cpu')

    # 判断当前进程是否为主进程
    is_master = True if not args.distributed_enabled else args.distributed_enabled and rank == 0


    #######################
    ## preamble

    # 设置 GPU
    set_gpus(rank)
    # 设置随机种子
    set_seed(rank)
    # 设置 CUDA
    set_cuda(deterministic=args.gpu_deterministic)

    # 创建输出目录
    output_dir = f'{args.output_dir}/{rank}'
    os.makedirs(output_dir, exist_ok=False)

    # 设置日志记录
    setup_logging(filename=f'{output_dir}/output.log', console=is_master)


    #######################
    ## dataset

    # 创建分词器
    tokenizer = new_tokenizer(vocab_file=args.data_vocab_file)
    vocab_size = len(tokenizer.vocab)
    # 加载数据集
    ds_train = wrap_example_builder(dataset=load_owt(owt_dir=args.data_dir, n_tensors_per_file=args.data_n_tensors_per_file), vocab=tokenizer.vocab, max_length=args.data_max_seq_length)

    # 获取特殊标记的 ID
    pad_token_id = tokenizer.vocab['[PAD]']
    mask_token_id = tokenizer.vocab['[MASK]']
    cls_token_id = tokenizer.vocab['[CLS]']
    sep_token_id = tokenizer.vocab['[SEP]']

    # 断言特殊标记的 ID 符合预期
    assert pad_token_id == 0
    assert cls_token_id == 101
    assert sep_token_id == 102
    assert mask_token_id == 103

    # 定义数据加载函数
    def collate_batch(examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([example['input_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        input_mask = torch.nn.utils.rnn.pad_sequence([example['input_mask'] for example in examples], batch_first=True, padding_value=pad_token_id)
        segment_ids = torch.nn.utils.rnn.pad_sequence([example['segment_ids'] for example in examples], batch_first=True, padding_value=pad_token_id)
        return input_ids, input_mask, segment_ids

    # 定义数据集加载器
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    ds_train_loader = iter(cycle(DataLoader(ds_train, batch_size=args.opt_batch_size, collate_fn=collate_batch)))


    #######################
    ## model
    # 如果分布式模式未启用，则返回原始模型；否则返回使用分布式数据并行的模型
    def to_distributed_model(model):
        return model if not args.distributed_enabled else torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    # 将生成器和鉴别器的权重绑定在一起
    def tie_weights(generator, discriminator):
        generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
        generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
        generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings

    # 定义一个适配器类，用于调整模型输出的格式
    class LogitsAdapter(torch.nn.Module):
        def __init__(self, adaptee):
            super().__init__()
            self.adaptee = adaptee

        def forward(self, *args, **kwargs):
            return self.adaptee(*args, **kwargs)[0]

    # 导入所需的库和模型配置
    from transformers import AutoConfig, ElectraForMaskedLM, ElectraForPreTraining

    # 创建生成器和鉴别器模型
    generator = ElectraForMaskedLM(AutoConfig.from_pretrained(args.model_generator))
    discriminator = ElectraForPreTraining(AutoConfig.from_pretrained(args.model_discriminator))

    # 将生成器和鉴别器的权重绑定在一起
    tie_weights(generator, discriminator)

    # 创建分布式模型，并设置相关参数
    model = to_distributed_model(Electra(
        LogitsAdapter(generator),
        LogitsAdapter(discriminator),
        num_tokens = vocab_size,
        mask_token_id = mask_token_id,
        pad_token_id = pad_token_id,
        mask_prob = args.model_mask_prob,
        mask_ignore_token_ids = [tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]'],
        random_token_prob = 0.0).to(device))

    #######################
    ## optimizer

    # 定义一个带有热身阶段的线性学习率调度器
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        def lr_lambda(current_step):
            learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
            learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
            return learning_rate
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    # 获取不需要权重衰减的参数
    def get_params_without_weight_decay_ln(named_params, weight_decay):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
            },
            {
                'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        return optimizer_grouped_parameters

    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(get_params_without_weight_decay_ln(model.named_parameters(), weight_decay=0.1), lr=args.opt_lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.opt_warmup_steps, num_training_steps=args.opt_num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.gpu_mixed_precision)

    #######################
    ## train

    # 记录训练开始时间，步长速度和预计完成时间
    t, steps_s, eta_m = time(), 0., 0
    # 循环执行训练步骤，包括优化器更新、梯度裁剪、学习率调整等
    for step in range(args.opt_num_training_steps+1):
        # 从训练数据加载下一个批次的输入数据
        input_ids, input_mask, segment_ids = next(ds_train_loader)

        # 将输入数据移动到指定设备上
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        # 断言输入数据的序列长度不超过设定的最大长度
        assert input_ids.shape[1] <= args.data_max_seq_length

        # 梯度清零
        optimizer.zero_grad()

        # 使用混合精度训练，计算损失和准确率
        with torch.cuda.amp.autocast(enabled=args.gpu_mixed_precision):
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)

        # 反向传播并调整优化器参数
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # 记录训练指标
        metrics = {
            'step': (step, '{:8d}'),
            'loss': (loss.item(), '{:8.5f}'),
            'loss_mlm': (loss_mlm.item(), '{:8.5f}'),
            'loss_disc': (loss_disc.item(), '{:8.5f}'),
            'acc_gen': (acc_gen.item(), '{:5.3f}'),
            'acc_disc': (acc_disc.item(), '{:5.3f}'),
            'lr': (scheduler.get_last_lr()[0], '{:8.7f}'),
            'steps': (steps_s, '{:4.1f}/s'),
            'eta': (eta_m, '{:4d}m'),
        }

        # 每隔一定步数打印训练指标信息
        if step % args.step_log == 0:
            sep = ' ' * 2
            logger.info(sep.join([f'{k}: {v[1].format(v[0])}' for (k, v) in metrics.items()])

        # 每隔一定步数计算训练速度和预计剩余时间
        if step > 0 and step % 100 == 0:
            t2 = time()
            steps_s = 100. / (t2 - t)
            eta_m = int(((args.opt_num_training_steps - step) / steps_s) // 60)
            t = t2

        # 每隔一定步数打印部分标签和预测结果
        if step % 200 == 0:
            logger.info(np.array2string(disc_labels[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))
            logger.info(np.array2string(disc_pred[0].cpu().numpy(), threshold=sys.maxsize, max_line_width=sys.maxsize))

        # 每隔一定步数保存模型检查点
        if step > 0 and step % args.step_ckpt == 0 and is_master:
            discriminator.electra.save_pretrained(f'{args.output_dir}/ckpt/{step}')
# 设置程序在哪块 GPU 上运行
def set_gpus(gpu):
    torch.cuda.set_device(gpu)

# 设置随机种子
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    如果 CUDA 可用，设置 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 设置 CUDA 是否确定性
def set_cuda(deterministic=True):
    如果 CUDA 可用，设置 CUDA 是否确定性
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

# 获取实验 ID
def get_exp_id(file):
    返回文件名的基本名称（不包含扩展名）
    return os.path.splitext(os.path.basename(file))[0]

# 获取输出目录
def get_output_dir(exp_id):
    导入 datetime 模块
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    创建输出目录路径
    output_dir = os.path.join('output/' + exp_id, t)
    如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    返回输出目录路径
    return output_dir

# 设置日志记录
def setup_logging(filename, console=True):
    设置日志格式
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    获取日志记录器
    logger = logging.getLogger()
    清空日志记录器的处理器
    logger.handlers = []
    创建文件处理器
    file_handler = logging.FileHandler(filename)
    设置文件处理器的格式
    file_handler.setFormatter(log_format)
    添加文件处理器到日志记录器
    logger.addHandler(file_handler)
    如果需要在控制台输出日志
    if console:
        创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        设置控制台处理器的格式
        console_handler.setFormatter(log_format)
        添加控制台处理器到日志记录器
        logger.addHandler(console_handler)
        设置日志记录器的日志级别为 INFO
        logger.setLevel(logging.INFO)
    返回日志记录器
    return logger

# 复制源文件到输出目录
def copy_source(file, output_dir):
    导入 shutil 模块
    复制源文件到输出目录
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

# 主函数
def main():

    # preamble
    获取实验 ID
    exp_id = get_exp_id(__file__)
    获取输出目录
    output_dir = get_output_dir(exp_id)
    如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    创建检查点目录
    os.makedirs(f'{output_dir}/ckpt', exist_ok=False)
    复制源文件到输出目录
    copy_source(__file__, output_dir)

    # args
    解析命令行参数
    args = arg.parse_to(Args)
    设置输出目录和实验 ID
    args.output_dir = output_dir
    args.exp_id = exp_id

    # distributed
    如果启用分布式训练
    if args.distributed_enabled:
        设置主地址和端口
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.distributed_port)
        使用多进程方式启动训练
        torch.multiprocessing.spawn(train, nprocs=args.distributed_world_size, args=(args,))
    否则
    else:
        单机训练
        train(rank=args.gpu, args=args)

# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == '__main__':
    main()
```