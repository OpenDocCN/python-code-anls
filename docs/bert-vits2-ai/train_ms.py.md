# `d:/src/tocomm/Bert-VITS2\train_ms.py`

```
# flake8: noqa: E402
# 导入模块，忽略 E402 错误（flake8 是一个 Python 代码静态检查工具）
import platform  # 导入 platform 模块，用于获取操作系统信息
import os  # 导入 os 模块，用于与操作系统进行交互
import torch  # 导入 torch 模块，用于构建和训练神经网络
from torch.nn import functional as F  # 导入 torch.nn 模块中的 functional 子模块，用于定义神经网络的各种函数
from torch.utils.data import DataLoader  # 导入 torch.utils.data 模块中的 DataLoader 类，用于加载和处理数据
from torch.utils.tensorboard import SummaryWriter  # 导入 torch.utils.tensorboard 模块中的 SummaryWriter 类，用于可视化训练过程
import torch.distributed as dist  # 导入 torch.distributed 模块，用于分布式训练
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入 torch.nn.parallel 模块中的 DistributedDataParallel 类，用于分布式训练
from torch.cuda.amp import autocast, GradScaler  # 导入 torch.cuda.amp 模块中的 autocast 和 GradScaler 类，用于混合精度训练
from tqdm import tqdm  # 导入 tqdm 模块，用于显示训练进度条
import logging  # 导入 logging 模块，用于记录日志信息
from config import config  # 导入 config 模块，用于读取配置文件
import argparse  # 导入 argparse 模块，用于解析命令行参数
import datetime  # 导入 datetime 模块，用于处理日期和时间
import gc  # 导入 gc 模块，用于垃圾回收

# 设置 numba 模块的日志级别为 WARNING
logging.getLogger("numba").setLevel(logging.WARNING)
# 导入自定义模块 commons
import commons
# 导入自定义模块 utils
import utils
# 导入所需的模块和类
from data_utils import (
    TextAudioSpeakerLoader,  # 导入TextAudioSpeakerLoader类，用于加载文本、音频和说话人信息的数据
    TextAudioSpeakerCollate,  # 导入TextAudioSpeakerCollate类，用于将加载的数据进行整理和处理
    DistributedBucketSampler,  # 导入DistributedBucketSampler类，用于分布式训练时对数据进行分桶采样
)
from models import (
    SynthesizerTrn,  # 导入SynthesizerTrn类，用于定义合成器模型
    MultiPeriodDiscriminator,  # 导入MultiPeriodDiscriminator类，用于定义多周期鉴别器模型
    DurationDiscriminator,  # 导入DurationDiscriminator类，用于定义持续时间鉴别器模型
    WavLMDiscriminator,  # 导入WavLMDiscriminator类，用于定义WavLM鉴别器模型
)
from losses import (
    generator_loss,  # 导入generator_loss函数，用于计算生成器的损失
    discriminator_loss,  # 导入discriminator_loss函数，用于计算鉴别器的损失
    feature_loss,  # 导入feature_loss函数，用于计算特征损失
    kl_loss,  # 导入kl_loss函数，用于计算KL散度损失
    WavLMLoss,  # 导入WavLMLoss类，用于计算WavLM损失
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # 导入mel_spectrogram_torch和spec_to_mel_torch函数，用于进行梅尔频谱转换
from text.symbols import symbols  # 导入symbols列表，用于表示文本中的符号
# 设置允许使用 TF32 精度进行矩阵乘法运算
torch.backends.cuda.matmul.allow_tf32 = True
# 设置允许使用 TF32 精度进行 cuDNN 加速
torch.backends.cudnn.allow_tf32 = (
    True  # 如果遇到训练问题，请尝试禁用 TF32
)
# 设置矩阵乘法的浮点精度为 medium
torch.set_float32_matmul_precision("medium")
# 设置 CUDA 的 SDP 内核为 "flash"
torch.backends.cuda.sdp_kernel("flash")
# 启用 CUDA 的 SDP 闪存加速
torch.backends.cuda.enable_flash_sdp(True)
# 启用 CUDA 的 SDP 内存高效模式
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # 如果 torch 版本低于 2.0，则不可用
# 设置全局步数为 0
global_step = 0


def run():
    # 解析环境变量
    envs = config.train_ms_config.env
    for env_name, env_value in envs.items():
        # 如果环境变量名不在当前环境变量的键中
        if env_name not in os.environ.keys():
            # 打印加载配置信息
            print("加载config中的配置{}".format(str(env_value)))
# 设置环境变量
os.environ[env_name] = str(env_value)

# 打印加载的环境变量信息
print(
    "加载环境变量 \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
        os.environ["MASTER_ADDR"],
        os.environ["MASTER_PORT"],
        os.environ["WORLD_SIZE"],
        os.environ["RANK"],
        os.environ["LOCAL_RANK"],
    )
)

# 设置分布式训练的后端，默认为"nccl"，如果是Windows系统，则切换到"gloo"后端
backend = "nccl"
if platform.system() == "Windows":
    backend = "gloo"

# 初始化分布式进程组
dist.init_process_group(
    backend=backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=300),
)

# 获取当前进程的rank
rank = dist.get_rank()
```

这段代码的作用是设置环境变量、打印加载的环境变量信息、设置分布式训练的后端、初始化分布式进程组和获取当前进程的rank。
    local_rank = int(os.environ["LOCAL_RANK"])
    # 获取当前进程的本地排名，用于多GPU训练时的分布式训练
    n_gpus = dist.get_world_size()
    # 获取总的GPU数量，用于多GPU训练时的分布式训练

    # 命令行/config.yml配置解析
    # hps = utils.get_hparams()
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # 非必要不建议使用命令行配置，请使用config.yml文件
    # 添加一个名为config的参数，用于指定配置文件的路径，默认值为config.train_ms_config.config_path
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=config.train_ms_config.config_path,
        help="JSON file for configuration",
    )

    # 添加一个名为model的参数，用于指定数据集文件夹的路径
    # 注意：数据集文件夹不再默认放在/logs文件夹下，如果需要用命令行配置，请声明相对于根目录的路径
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="数据集文件夹路径，请注意，数据不再默认放在/logs文件夹下。如果需要用命令行配置，请声明相对于根目录的路径",
    )
```

这段代码主要是解析命令行参数，其中`local_rank`和`n_gpus`用于多GPU训练时的分布式训练。`parser`是一个参数解析器对象，通过调用`add_argument`方法添加参数，其中`-c`和`--config`用于指定配置文件的路径，默认值为`config.train_ms_config.config_path`；`-m`和`--model`用于指定数据集文件夹的路径。
# 导入必要的模块
import os
import argparse
import torch
import utils

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加一个名为 "model" 的命令行参数，用于指定模型路径，默认值为 config.dataset_path
parser.add_argument(
    "--model",
    type=str,
    default=config.dataset_path,
)
# 解析命令行参数
args = parser.parse_args()

# 根据模型路径和训练配置文件名创建模型目录路径
model_dir = os.path.join(args.model, config.train_ms_config.model)
# 如果模型目录不存在，则创建该目录
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 从配置文件中获取超参数
hps = utils.get_hparams_from_file(args.config)
# 将模型目录路径添加到超参数中
hps.model_dir = model_dir

# 比较命令行参数中的配置文件路径和训练配置文件路径是否相同
if os.path.realpath(args.config) != os.path.realpath(
    config.train_ms_config.config_path
):
    # 如果不相同，则将命令行参数中的配置文件内容写入训练配置文件中
    with open(args.config, "r", encoding="utf-8") as f:
        data = f.read()
    with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
        f.write(data)

# 设置随机种子
torch.manual_seed(hps.train.seed)
# 设置当前使用的 GPU 设备
torch.cuda.set_device(local_rank)
```

注释解释：

1. 导入必要的模块：导入了 os、argparse 和 torch 模块，以及自定义的 utils 模块。
2. 创建命令行参数解析器：创建了一个 argparse.ArgumentParser 对象，用于解析命令行参数。
3. 添加一个名为 "model" 的命令行参数：添加了一个名为 "model" 的命令行参数，用于指定模型路径，默认值为 config.dataset_path。
4. 解析命令行参数：将命令行参数解析为 args 对象。
5. 根据模型路径和训练配置文件名创建模型目录路径：使用 os.path.join() 函数将模型路径和训练配置文件名拼接起来，得到模型目录路径。
6. 如果模型目录不存在，则创建该目录：使用 os.makedirs() 函数创建模型目录。
7. 从配置文件中获取超参数：使用 utils.get_hparams_from_file() 函数从配置文件中读取超参数。
8. 将模型目录路径添加到超参数中：将模型目录路径赋值给超参数 hps 的 model_dir 属性。
9. 比较命令行参数中的配置文件路径和训练配置文件路径是否相同：使用 os.path.realpath() 函数获取命令行参数中配置文件的真实路径，并与训练配置文件的真实路径进行比较。
10. 如果不相同，则将命令行参数中的配置文件内容写入训练配置文件中：使用 with open() 打开命令行参数中的配置文件和训练配置文件，分别读取和写入文件内容。
11. 设置随机种子：使用 torch.manual_seed() 函数设置随机种子，用于保证实验的可重复性。
12. 设置当前使用的 GPU 设备：使用 torch.cuda.set_device() 函数设置当前使用的 GPU 设备。
    global global_step  # 声明全局变量 global_step
    if rank == 0:  # 如果 rank 等于 0
        logger = utils.get_logger(hps.model_dir)  # 获取日志记录器
        logger.info(hps)  # 记录日志信息
        utils.check_git_hash(hps.model_dir)  # 检查 git 的哈希值
        writer = SummaryWriter(log_dir=hps.model_dir)  # 创建一个用于写入训练日志的 SummaryWriter 对象
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))  # 创建一个用于写入评估日志的 SummaryWriter 对象
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)  # 创建训练数据集对象
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],  # 定义 bucket 的边界
        num_replicas=n_gpus,  # GPU 数量
        rank=rank,  # 当前进程的 rank
        shuffle=True,  # 是否打乱数据
    )  # 创建分布式 bucket sampler 对象
    collate_fn = TextAudioSpeakerCollate()  # 创建数据集的 collate 函数
    train_loader = DataLoader(
        train_dataset,
        num_workers=min(config.train_ms_config.num_workers, os.cpu_count() - 1),  # 设置并行加载数据的线程数
        shuffle=False,  # 设置为False，不对数据进行随机打乱
        pin_memory=True,  # 设置为True，将数据加载到固定的内存区域，加速数据传输
        collate_fn=collate_fn,  # 设置数据加载时的自定义函数，用于对样本进行处理和组合
        batch_sampler=train_sampler,  # 设置训练数据的采样器，用于确定每个batch的样本索引
        persistent_workers=True,  # 设置为True，使用持久化的工作进程来加载数据
        prefetch_factor=4,  # 设置预取因子，用于提前加载数据到内存中
    )  # DataLoader的配置可以进行调整。
    if rank == 0:  # 如果rank等于0
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)  # 创建验证数据集对象
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,  # 设置工作进程的数量为0，即不使用多线程加载数据
            shuffle=False,  # 设置为False，不对数据进行随机打乱
            batch_size=1,  # 设置每个batch的样本数量为1
            pin_memory=True,  # 设置为True，将数据加载到固定的内存区域，加速数据传输
            drop_last=False,  # 设置为False，保留最后一个batch中样本数量小于batch_size的样本
            collate_fn=collate_fn,  # 设置数据加载时的自定义函数，用于对样本进行处理和组合
        )
    if (
        "use_noise_scaled_mas" in hps.model.keys()  # 如果"use_noise_scaled_mas"在hps.model的键中
if (
    "use_noise_scaled_mas" in hps.model.keys()  # 检查配置文件中是否存在"use_noise_scaled_mas"键
    and hps.model.use_noise_scaled_mas is True  # 检查"use_noise_scaled_mas"键对应的值是否为True
):
    print("Using noise scaled MAS for VITS2")  # 打印使用噪声缩放的MAS（模型辅助训练）用于VITS2
    mas_noise_scale_initial = 0.01  # 设置噪声缩放的初始值为0.01
    noise_scale_delta = 2e-6  # 设置噪声缩放的增量为2e-6
else:
    print("Using normal MAS for VITS1")  # 打印使用正常的MAS（模型辅助训练）用于VITS1
    mas_noise_scale_initial = 0.0  # 设置噪声缩放的初始值为0.0
    noise_scale_delta = 0.0  # 设置噪声缩放的增量为0.0
if (
    "use_duration_discriminator" in hps.model.keys()  # 检查配置文件中是否存在"use_duration_discriminator"键
    and hps.model.use_duration_discriminator is True  # 检查"use_duration_discriminator"键对应的值是否为True
):
    print("Using duration discriminator for VITS2")  # 打印使用持续时间鉴别器用于VITS2
    net_dur_disc = DurationDiscriminator(
        hps.model.hidden_channels,  # 隐藏通道数
        hps.model.hidden_channels,  # 隐藏通道数
        3,  # 鉴别器的输出通道数
        0.1,  # 鉴别器的dropout率
        gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,  # GIN通道数
        ).cuda(local_rank)
```
这行代码将模型移动到GPU上进行计算。

```
    else:
        net_dur_disc = None
```
如果条件不满足，则将`net_dur_disc`设置为`None`。

```
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        print("Using normal encoder for VITS1")
```
如果`hps.model`中包含键`"use_spk_conditioned_encoder"`且其值为`True`，则检查`hps.data.n_speakers`是否为0。如果为0，则抛出`ValueError`异常，否则打印"Using normal encoder for VITS1"。

```
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
```
创建一个`SynthesizerTrn`对象，并传入参数`len(symbols)`、`hps.data.filter_length // 2 + 1`、`hps.train.segment_size // hps.data.hop_length`、`n_speakers=hps.data.n_speakers`和`mas_noise_scale_initial=mas_noise_scale_initial`。

注：以上代码片段缺少了一些上下文信息，可能无法完全理解其含义和作用。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

```
noise_scale_delta=noise_scale_delta,
**hps.model,
).cuda(local_rank)
```
这段代码是创建一个神经网络模型，并将其放在GPU上进行计算。

```
if getattr(hps.train, "freeze_ZH_bert", False):
    print("Freezing ZH bert encoder !!!")
    for param in net_g.enc_p.bert_proj.parameters():
        param.requires_grad = False
```
这段代码用于冻结中文BERT编码器的参数，即不对其进行梯度更新。

```
if getattr(hps.train, "freeze_EN_bert", False):
    print("Freezing EN bert encoder !!!")
    for param in net_g.enc_p.en_bert_proj.parameters():
        param.requires_grad = False
```
这段代码用于冻结英文BERT编码器的参数，即不对其进行梯度更新。

```
if getattr(hps.train, "freeze_JP_bert", False):
    print("Freezing JP bert encoder !!!")
    for param in net_g.enc_p.ja_bert_proj.parameters():
        param.requires_grad = False
```
这段代码用于冻结日文BERT编码器的参数，即不对其进行梯度更新。

```
net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
```
这段代码是创建一个多周期鉴别器模型，并将其放在GPU上进行计算。
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
    ).cuda(local_rank)
```
创建一个名为`net_wd`的`WavLMDiscriminator`对象，使用`hps.model.slm.hidden`、`hps.model.slm.nlayers`和`hps.model.slm.initial_channel`作为参数进行初始化，并将其移动到GPU上。

```
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
```
创建一个名为`optim_g`的AdamW优化器对象，用于优化`net_g`的参数。只选择那些`requires_grad`属性为True的参数进行优化。使用`hps.train.learning_rate`作为学习率，`hps.train.betas`作为beta参数，`hps.train.eps`作为epsilon参数。

```
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
```
创建一个名为`optim_d`的AdamW优化器对象，用于优化`net_d`的参数。使用`hps.train.learning_rate`作为学习率，`hps.train.betas`作为beta参数，`hps.train.eps`作为epsilon参数。

```
    optim_wd = torch.optim.AdamW(
        net_wd.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
```
创建一个名为`optim_wd`的AdamW优化器对象，用于优化`net_wd`的参数。使用`hps.train.learning_rate`作为学习率，`hps.train.betas`作为beta参数，`hps.train.eps`作为epsilon参数。
    )
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None
    net_g = DDP(net_g, device_ids=[local_rank], bucket_cap_mb=512)
    net_d = DDP(net_d, device_ids=[local_rank], bucket_cap_mb=512)
    net_wd = DDP(net_wd, device_ids=[local_rank], bucket_cap_mb=512)
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            bucket_cap_mb=512,
        )
```

注释如下：

```
    )
    # 如果存在 net_dur_disc，则使用 AdamW 优化器对其参数进行优化
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        # 否则，将优化器设置为 None
        optim_dur_disc = None
    # 使用 DDP 将 net_g 分布式部署到指定的设备上，并设置 bucket 容量为 512MB
    net_g = DDP(net_g, device_ids=[local_rank], bucket_cap_mb=512)
    # 使用 DDP 将 net_d 分布式部署到指定的设备上，并设置 bucket 容量为 512MB
    net_d = DDP(net_d, device_ids=[local_rank], bucket_cap_mb=512)
    # 使用 DDP 将 net_wd 分布式部署到指定的设备上，并设置 bucket 容量为 512MB
    net_wd = DDP(net_wd, device_ids=[local_rank], bucket_cap_mb=512)
    # 如果存在 net_dur_disc，则使用 DDP 将其分布式部署到指定的设备上，并设置 bucket 容量为 512MB
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            bucket_cap_mb=512,
        )
    # 下载底模
    if config.train_ms_config.base["use_base_model"]:
        # 如果配置中指定使用底模型，则下载底模型的检查点文件
        utils.download_checkpoint(
            hps.model_dir,
            config.train_ms_config.base,
            token=config.openi_token,
            mirror=config.mirror,
        )
    # 初始化持续时间判别器的学习率
    dur_resume_lr = hps.train.learning_rate
    # 初始化声学特征判别器的学习率
    wd_resume_lr = hps.train.learning_rate
    # 如果持续时间判别器存在
    if net_dur_disc is not None:
        # 尝试加载最新的持续时间判别器的检查点文件
        _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
            net_dur_disc,
            optim_dur_disc,
            # 如果配置中指定跳过优化器，则跳过优化器的加载
            skip_optimizer=hps.train.skip_optimizer
            if "skip_optimizer" in hps.train
            else True,
        )
```

这段代码主要是根据给定的配置信息进行一系列操作，包括下载底模型的检查点文件、初始化学习率以及加载持续时间判别器的检查点文件。具体注释如下：

- `# 下载底模`：如果配置中指定使用底模型，则下载底模型的检查点文件。
- `# 初始化持续时间判别器的学习率`：将持续时间判别器的学习率初始化为训练配置中指定的学习率。
- `# 初始化声学特征判别器的学习率`：将声学特征判别器的学习率初始化为训练配置中指定的学习率。
- `# 如果持续时间判别器存在`：判断持续时间判别器是否存在。
- `# 尝试加载最新的持续时间判别器的检查点文件`：尝试加载最新的持续时间判别器的检查点文件，并返回加载结果。
- `# 如果配置中指定跳过优化器，则跳过优化器的加载`：根据配置中是否指定跳过优化器的加载来决定是否跳过优化器的加载操作。
# 如果优化器optim_dur_disc的param_groups列表中的第一个元素的"initial_lr"键不存在
if not optim_dur_disc.param_groups[0].get("initial_lr"):
    # 将dur_resume_lr赋值给优化器optim_dur_disc的param_groups列表中的第一个元素的"initial_lr"键
    optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
# 捕获异常
except:
    # 打印"Initialize dur_disc"
    print("Initialize dur_disc")

try:
    # 调用utils.load_checkpoint函数，加载最新的以"G_*.pth"为后缀的检查点文件
    # 将加载的结果分别赋值给_、optim_g、g_resume_lr和epoch_str
    _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
        net_g,
        optim_g,
        # 如果hps.train中存在"skip_optimizer"键，则将其值赋给skip_optimizer；否则赋值为True
        skip_optimizer=hps.train.skip_optimizer
        if "skip_optimizer" in hps.train
        else True,
    )
    # 调用utils.load_checkpoint函数，加载最新的以"D_*.pth"为后缀的检查点文件
    # 将加载的结果分别赋值给_、optim_d、d_resume_lr和epoch_str
    _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
        net_d,
        optim_d,
        # 如果hps.train中存在"skip_optimizer"键，则将其值赋给skip_optimizer；否则赋值为True
        skip_optimizer=hps.train.skip_optimizer
        if "skip_optimizer" in hps.train
            else True,
        )
```
这是一个三元表达式，根据条件判断返回不同的值。如果条件为真，则返回True，否则返回else后面的值。

```
        if not optim_g.param_groups[0].get("initial_lr"):
            optim_g.param_groups[0]["initial_lr"] = g_resume_lr
        if not optim_d.param_groups[0].get("initial_lr"):
            optim_d.param_groups[0]["initial_lr"] = d_resume_lr
```
这两个if语句用于检查优化器optim_g和optim_d的参数组中是否存在"initial_lr"键。如果不存在，则将g_resume_lr和d_resume_lr分别赋值给optim_g和optim_d的参数组中的"initial_lr"键。

```
        epoch_str = max(epoch_str, 1)
```
将epoch_str和1进行比较，取较大的值赋给epoch_str。

```
        global_step = int(
            utils.get_steps(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"))
        )
```
调用utils模块中的函数，获取最新的以"G_"开头的.pth文件的路径，并将路径传递给utils.get_steps函数，获取全局步数，并将其转换为整数类型赋给global_step。

```
        print(
            f"******************检测到模型存在，epoch为 {epoch_str}，gloabl step为 {global_step}*********************"
        )
```
打印检测到模型存在的提示信息，包括当前的epoch和全局步数。

```
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0
```
捕获异常，并打印异常信息。将epoch_str赋值为1，将global_step赋值为0。
try:
    # 调用load_checkpoint函数加载最新的模型检查点文件，返回值包括网络参数、优化器参数、学习率和训练轮数
    _, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
        net_wd,
        optim_wd,
        skip_optimizer=hps.train.skip_optimizer
        if "skip_optimizer" in hps.train
        else True,
    )
    # 如果优化器参数中没有"initial_lr"键，则将wd_resume_lr赋值给它
    if not optim_wd.param_groups[0].get("initial_lr"):
        optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
except Exception as e:
    # 捕获异常并打印错误信息
    print(e)

# 创建指数衰减学习率调度器，用于调整生成器和判别器的学习率
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
    optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
)
```

这段代码主要是用于加载模型检查点文件，并创建学习率调度器。具体注释如下：

1. `try:`：开始异常处理块，尝试执行下面的代码。
2. `_, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(...)`：调用`load_checkpoint`函数加载最新的模型检查点文件，并将返回的网络参数、优化器参数、学习率和训练轮数分别赋值给`_, optim_wd, wd_resume_lr, epoch_str`四个变量。
3. `if not optim_wd.param_groups[0].get("initial_lr"):`：如果优化器参数中没有"initial_lr"键，则执行下面的代码。
4. `optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr`：将`wd_resume_lr`的值赋给优化器参数中的"initial_lr"键。
5. `except Exception as e:`：如果发生异常，则执行下面的代码，并将异常信息赋值给变量`e`。
6. `print(e)`：打印异常信息。
7. `scheduler_g = torch.optim.lr_scheduler.ExponentialLR(...)`：创建指数衰减学习率调度器，用于调整生成器的学习率。其中`optim_g`是生成器的优化器，`gamma`是衰减因子，`last_epoch`是上一轮的训练轮数。
8. `scheduler_d = torch.optim.lr_scheduler.ExponentialLR(...)`：创建指数衰减学习率调度器，用于调整判别器的学习率。其中`optim_d`是判别器的优化器，`gamma`是衰减因子，`last_epoch`是上一轮的训练轮数。
    scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
        optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
```
创建一个学习率调度器对象scheduler_wd，使用指数衰减的方式调整优化器optim_wd的学习率。gamma参数表示衰减因子，last_epoch参数表示上一个epoch的索引。

```
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None
```
如果net_dur_disc不为空，则创建一个学习率调度器对象scheduler_dur_disc，使用指数衰减的方式调整优化器optim_dur_disc的学习率。否则，将scheduler_dur_disc设置为None。

```
    scaler = GradScaler(enabled=hps.train.bf16_run)
```
创建一个GradScaler对象scaler，用于在混合精度训练中缩放梯度。

```
    wl = WavLMLoss(
        hps.model.slm.model,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr,
    ).to(local_rank)
```
创建一个WavLMLoss对象wl，用于计算语音语言模型的损失。将wl对象移动到指定的设备local_rank上。

```
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
```
对于每个epoch，从epoch_str开始迭代到hps.train.epochs + 1。如果rank等于0（表示当前进程为主进程），则执行以下代码块。
# 如果条件成立，执行以下代码块
if condition:
    # 调用train_and_evaluate函数，并传入多个参数
    train_and_evaluate(
        rank,
        local_rank,
        epoch,
        hps,
        [net_g, net_d, net_dur_disc, net_wd, wl],
        [optim_g, optim_d, optim_dur_disc, optim_wd],
        [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
        scaler,
        [train_loader, eval_loader],
        logger,
        [writer, writer_eval],
    )
# 如果条件不成立，执行以下代码块
else:
    # 调用train_and_evaluate函数，并传入多个参数
    train_and_evaluate(
        rank,
        local_rank,
        epoch,
        hps,
        [net_g, net_d, net_dur_disc, net_wd, wl],
[optim_g, optim_d, optim_dur_disc, optim_wd],
```
这行代码定义了一个列表，其中包含了四个优化器对象optim_g、optim_d、optim_dur_disc和optim_wd。

```
[scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
```
这行代码定义了一个列表，其中包含了四个学习率调度器对象scheduler_g、scheduler_d、scheduler_dur_disc和scheduler_wd。

```
scaler,
```
这行代码定义了一个变量scaler，用于进行梯度缩放。

```
[train_loader, None],
```
这行代码定义了一个列表，其中包含了两个元素，一个是train_loader对象，另一个是None。

```
None,
```
这行代码定义了一个None对象。

```
None,
```
这行代码定义了一个None对象。

```
)
```
这行代码表示函数调用的结束。

```
scheduler_g.step()
scheduler_d.step()
scheduler_wd.step()
```
这几行代码分别调用了scheduler_g、scheduler_d和scheduler_wd的step()方法，用于更新对应的学习率。

```
if net_dur_disc is not None:
    scheduler_dur_disc.step()
```
这段代码判断net_dur_disc是否为None，如果不是None，则调用scheduler_dur_disc的step()方法，用于更新持续时间鉴别器的学习率。

```
def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
```
这段代码定义了一个名为train_and_evaluate的函数，该函数接受六个参数：rank、local_rank、epoch、hps、nets和...（省略号表示还有其他参数）。
    optims,  # 优化器列表
    schedulers,  # 调度器列表
    scaler,  # 梯度缩放器
    loaders,  # 数据加载器列表
    logger,  # 日志记录器
    writers,  # 写入器列表
):
    net_g, net_d, net_dur_disc, net_wd, wl = nets  # 网络模型列表
    optim_g, optim_d, optim_dur_disc, optim_wd = optims  # 优化器列表
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers  # 调度器列表
    train_loader, eval_loader = loaders  # 训练数据加载器和评估数据加载器
    if writers is not None:  # 如果写入器列表不为空
        writer, writer_eval = writers  # 获取训练和评估的写入器

    train_loader.batch_sampler.set_epoch(epoch)  # 设置训练数据加载器的 epoch
    global global_step  # 声明全局变量 global_step

    net_g.train()  # 设置生成器网络为训练模式
    net_d.train()  # 设置判别器网络为训练模式
    net_wd.train()  # 设置权重衰减网络为训练模式
    if net_dur_disc is not None:
        net_dur_disc.train()
```
如果`net_dur_disc`不是`None`，则调用`train()`方法，用于训练`net_dur_disc`模型。

```
    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speakers,
        tone,
        language,
        bert,
        ja_bert,
        en_bert,
    ) in enumerate(tqdm(train_loader)):
```
使用`enumerate()`函数遍历`train_loader`，并将每个批次的数据赋值给对应的变量。

```
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
```
如果`net_g.module.use_noise_scaled_mas`为`True`，则计算当前的`mas_noise_scale`值，用于后续的操作。
# 将输入数据和标签转移到GPU上
x, x_lengths = x.cuda(local_rank, non_blocking=True), x_lengths.cuda(local_rank, non_blocking=True)
spec, spec_lengths = spec.cuda(local_rank, non_blocking=True), spec_lengths.cuda(local_rank, non_blocking=True)
y, y_lengths = y.cuda(local_rank, non_blocking=True), y_lengths.cuda(local_rank, non_blocking=True)
speakers = speakers.cuda(local_rank, non_blocking=True)
tone = tone.cuda(local_rank, non_blocking=True)
language = language.cuda(local_rank, non_blocking=True)
bert = bert.cuda(local_rank, non_blocking=True)
ja_bert = ja_bert.cuda(local_rank, non_blocking=True)
en_bert = en_bert.cuda(local_rank, non_blocking=True)

# 使用自动混合精度进行计算
with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
    (
    # 在自动混合精度下执行的代码
    )
```

注释解释了代码的作用，将输入数据和标签转移到GPU上，并使用自动混合精度进行计算。
# 调用 net_g 函数，传入以下参数：
# - x: 输入数据
# - x_lengths: 输入数据的长度
# - spec: 音频特征
# - spec_lengths: 音频特征的长度
# - speakers: 说话人信息
# - tone: 音调信息
# - language: 语言信息
# - bert: BERT 模型
# - ja_bert: 日语 BERT 模型
# - en_bert: 英语 BERT 模型
# 返回以下结果：
# - y_hat: 预测结果
# - l_length: 预测结果的长度
# - attn: 注意力权重
# - ids_slice: 切片的 ID
# - x_mask: 输入数据的掩码
# - z_mask: 音频特征的掩码
# - (z, z_p, m_p, logs_p, m_q, logs_q): 音频特征的编码结果
# - (hidden_x, logw, logw_, logw_sdp): 输入数据的编码结果
# - g: 生成器的输出
y_hat,
l_length,
attn,
ids_slice,
x_mask,
z_mask,
(z, z_p, m_p, logs_p, m_q, logs_q),
(hidden_x, logw, logw_, logw_sdp),
g = net_g(
    x,
    x_lengths,
    spec,
    spec_lengths,
    speakers,
    tone,
    language,
    bert,
    ja_bert,
    en_bert,
)
            )
```
这是一个多行函数调用的结束括号。

```
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
```
调用名为`spec_to_mel_torch`的函数，传入了多个参数`spec, hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax`，并将返回值赋给变量`mel`。

```
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
```
调用名为`commons.slice_segments`的函数，传入了多个参数`mel, ids_slice, hps.train.segment_size // hps.data.hop_length`，并将返回值赋给变量`y_mel`。

```
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
```
调用名为`mel_spectrogram_torch`的函数，传入了多个参数`y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, hps.data.mel_fmin`，并将返回值赋给变量`y_hat_mel`。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
                # 计算判别器的损失
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
            if net_dur_disc is not None:
                # 计算持续时间判别器的损失
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw.detach(),
                    g.detach(),
                )
```
将变量g从计算图中分离，不参与梯度计算。

```
                y_dur_hat_r_sdp, y_dur_hat_g_sdp = net_dur_disc(
                    hidden_x.detach(),
                    x_mask.detach(),
                    logw_.detach(),
                    logw_sdp.detach(),
                    g.detach(),
                )
```
使用分离的变量hidden_x、x_mask、logw_、logw_sdp和g作为输入，调用net_dur_disc函数，得到y_dur_hat_r_sdp和y_dur_hat_g_sdp。

```
                y_dur_hat_r = y_dur_hat_r + y_dur_hat_r_sdp
                y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp
```
将y_dur_hat_r_sdp和y_dur_hat_g_sdp与y_dur_hat_r和y_dur_hat_g相加。

```
                with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
```
启用自动混合精度训练，使用torch.bfloat16作为数据类型。

```
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    (
                        loss_dur_disc,
                        losses_dur_disc_r,
                        losses_dur_disc_g,
                    ) = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    loss_dur_disc_all = loss_dur_disc
```
计算y_dur_hat_r和y_dur_hat_g的损失，得到loss_dur_disc、losses_dur_disc_r和losses_dur_disc_g，并将loss_dur_disc赋值给loss_dur_disc_all。

```
                optim_dur_disc.zero_grad()
```
将优化器optim_dur_disc的梯度置零。
                scaler.scale(loss_dur_disc_all).backward()
                # 使用混合精度进行反向传播，对损失进行缩放
                scaler.unscale_(optim_dur_disc)
                # 取消损失缩放
                # torch.nn.utils.clip_grad_norm_(
                #     parameters=net_dur_disc.parameters(), max_norm=100
                # )
                # 对持续时间鉴别器的梯度进行裁剪，限制梯度的范数不超过100
                grad_norm_dur = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                # 对持续时间鉴别器的优化器进行一步优化
                scaler.step(optim_dur_disc)

        # 清空鉴别器的梯度
        optim_d.zero_grad()
        # 使用混合精度进行反向传播，对总损失进行缩放
        scaler.scale(loss_disc_all).backward()
        # 取消损失缩放
        scaler.unscale_(optim_d)
        # 如果配置中设置了bf16_run为True，则对鉴别器的梯度进行裁剪，限制梯度的范数不超过200
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_d.parameters(), max_norm=200)
        # 对鉴别器的梯度进行裁剪，限制梯度的范数不超过None
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        # 对鉴别器的优化器进行一步优化
        scaler.step(optim_d)

        # 使用自动混合精度进行计算，如果配置中设置了bf16_run为True，则使用torch.bfloat16数据类型
        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            # 计算语言模型的损失
            loss_slm = wl.discriminator(
        y.detach().squeeze(), y_hat.detach().squeeze()
    ).mean()
```
这段代码计算了两个张量 `y` 和 `y_hat` 的均值。`detach()` 方法用于创建一个新的张量，该张量与原始张量共享相同的数据，但不会进行梯度计算。`squeeze()` 方法用于去除张量中维度为1的维度。最后，`mean()` 方法计算了张量的均值。

```
    optim_wd.zero_grad()
    scaler.scale(loss_slm).backward()
    scaler.unscale_(optim_wd)
    grad_norm_wd = commons.clip_grad_value_(net_wd.parameters(), None)
    scaler.step(optim_wd)
```
这段代码用于进行反向传播和参数更新。首先，`zero_grad()` 方法将优化器 `optim_wd` 中的所有参数的梯度置零。然后，`scale()` 方法将损失 `loss_slm` 进行缩放。接着，调用 `backward()` 方法进行反向传播，计算参数的梯度。`unscale_()` 方法将优化器中的梯度进行反缩放。`clip_grad_value_()` 方法用于对参数的梯度进行裁剪，以防止梯度爆炸。最后，调用 `step()` 方法更新参数。

```
    with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        if net_dur_disc is not None:
            _, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw, g)
            _, y_dur_hat_g_sdp = net_dur_disc(hidden_x, x_mask, logw_, logw_sdp, g)
            y_dur_hat_g = y_dur_hat_g + y_dur_hat_g_sdp
        with autocast(enabled=hps.train.bf16_run, dtype=torch.bfloat16):
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
```
这段代码使用了自动混合精度（autocast）技术。首先，使用 `autocast()` 上下文管理器，将计算过程中的张量类型设置为 `torch.bfloat16`，并根据 `hps.train.bf16_run` 的值来决定是否启用自动混合精度。然后，调用 `net_d()` 方法，将输入 `y` 和 `y_hat` 传递给生成器，并将生成的结果赋值给变量 `y_d_hat_r`、`y_d_hat_g`、`fmap_r` 和 `fmap_g`。接下来，如果 `net_dur_disc` 不为 `None`，则调用 `net_dur_disc()` 方法，传递参数 `hidden_x`、`x_mask`、`logw_`、`logw` 和 `g`，并将生成的结果赋值给变量 `y_dur_hat_g` 和 `y_dur_hat_g_sdp`。最后，计算了 `loss_dur` 和 `loss_mel`，分别使用了 `torch.sum()` 和 `F.l1_loss()` 函数。
loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
```
这行代码计算 KL 散度损失，其中 `z_p`、`logs_q`、`m_p`、`logs_p` 和 `z_mask` 是输入参数，`hps.train.c_kl` 是一个常数，用于调整损失的权重。

```
loss_fm = feature_loss(fmap_r, fmap_g)
```
这行代码计算特征匹配损失，其中 `fmap_r` 和 `fmap_g` 是输入参数。

```
loss_gen, losses_gen = generator_loss(y_d_hat_g)
```
这行代码计算生成器损失，其中 `y_d_hat_g` 是输入参数。`loss_gen` 是总体损失，`losses_gen` 是各个子损失的列表。

```
loss_lm = wl(y.detach().squeeze(), y_hat.squeeze()).mean()
```
这行代码计算语言模型损失，其中 `y` 和 `y_hat` 是输入参数。`wl` 是一个语言模型对象，`y.detach().squeeze()` 和 `y_hat.squeeze()` 是对输入进行处理。

```
loss_lm_gen = wl.generator(y_hat.squeeze())
```
这行代码计算生成器的语言模型损失，其中 `y_hat` 是输入参数。

```
loss_gen_all = (
    loss_gen
    + loss_fm
    + loss_mel
    + loss_dur
    + loss_kl
    + loss_lm
    + loss_lm_gen
)
```
这行代码计算所有损失的总和，其中 `loss_gen`、`loss_fm`、`loss_mel`、`loss_dur`、`loss_kl`、`loss_lm` 和 `loss_lm_gen` 是之前计算得到的各个损失。

```
if net_dur_disc is not None:
    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
    loss_gen_all += loss_dur_gen
```
这段代码判断 `net_dur_disc` 是否为 `None`，如果不是，则计算持续时间生成器的损失，并将其加到 `loss_gen_all` 上。其中 `y_dur_hat_g` 是输入参数，`loss_dur_gen` 是持续时间生成器的损失。
        optim_g.zero_grad()
```
将生成器的梯度置零，准备进行反向传播和梯度更新。

```
        scaler.scale(loss_gen_all).backward()
```
根据生成器的损失计算梯度，并进行反向传播。

```
        scaler.unscale_(optim_g)
```
将生成器的梯度进行反缩放，以便进行梯度裁剪。

```
        if getattr(hps.train, "bf16_run", False):
            torch.nn.utils.clip_grad_norm_(parameters=net_g.parameters(), max_norm=500)
```
如果配置中设置了`bf16_run`为True，则对生成器的梯度进行裁剪，以防止梯度爆炸。

```
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
```
对生成器的梯度进行裁剪，以防止梯度爆炸，并返回裁剪后的梯度范数。

```
        scaler.step(optim_g)
```
根据优化器的设置，更新生成器的参数。

```
        scaler.update()
```
更新梯度缩放器的状态。

```
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])
```
如果当前进程的rank为0（即主进程），并且全局步数能够被配置中的log_interval整除，那么记录当前训练的epoch、进度百分比、损失值、全局步数和学习率等信息。
# 创建一个字典scalar_dict，用于存储各个变量的值
scalar_dict = {
    "loss/g/total": loss_gen_all,  # 将loss_gen_all赋值给"loss/g/total"键
    "loss/d/total": loss_disc_all,  # 将loss_disc_all赋值给"loss/d/total"键
    "loss/wd/total": loss_slm,  # 将loss_slm赋值给"loss/wd/total"键
    "learning_rate": lr,  # 将lr赋值给"learning_rate"键
    "grad_norm_d": grad_norm_d,  # 将grad_norm_d赋值给"grad_norm_d"键
    "grad_norm_g": grad_norm_g,  # 将grad_norm_g赋值给"grad_norm_g"键
    "grad_norm_dur": grad_norm_dur,  # 将grad_norm_dur赋值给"grad_norm_dur"键
    "grad_norm_wd": grad_norm_wd,  # 将grad_norm_wd赋值给"grad_norm_wd"键
}

# 更新scalar_dict字典，添加更多的键值对
scalar_dict.update(
    {
        "loss/g/fm": loss_fm,  # 将loss_fm赋值给"loss/g/fm"键
        "loss/g/mel": loss_mel,  # 将loss_mel赋值给"loss/g/mel"键
        "loss/g/dur": loss_dur,  # 将loss_dur赋值给"loss/g/dur"键
        "loss/g/kl": loss_kl,  # 将loss_kl赋值给"loss/g/kl"键
        "loss/g/lm": loss_lm,  # 将loss_lm赋值给"loss/g/lm"键
        "loss/g/lm_gen": loss_lm_gen,  # 将loss_lm_gen赋值给"loss/g/lm_gen"键
    }
)
# 更新scalar_dict字典，将"loss/g/{}".format(i)作为键，losses_gen中的值作为对应的值
scalar_dict.update(
    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
)

# 更新scalar_dict字典，将"loss/d_r/{}".format(i)作为键，losses_disc_r中的值作为对应的值
scalar_dict.update(
    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
)

# 更新scalar_dict字典，将"loss/d_g/{}".format(i)作为键，losses_disc_g中的值作为对应的值
scalar_dict.update(
    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
)

# 如果net_dur_disc不为空
if net_dur_disc is not None:
    # 更新scalar_dict字典，将"loss/dur_disc/total"作为键，loss_dur_disc_all作为对应的值
    scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all})

    # 更新scalar_dict字典，将"loss/dur_disc_g/{}".format(i)作为键，losses_dur_disc_g中的值作为对应的值
    scalar_dict.update(
        {
            "loss/dur_disc_g/{}".format(i): v
            for i, v in enumerate(losses_dur_disc_g)
        }
    )
    scalar_dict.update(
{
    "loss/dur_disc_r/{}".format(i): v
    for i, v in enumerate(losses_dur_disc_r)
}
```
这段代码使用了字典推导式，将`losses_dur_disc_r`列表中的元素作为值，根据索引`i`和字符串模板`"loss/dur_disc_r/{}"`生成键，最终生成一个字典。

```
scalar_dict.update({"loss/g/dur_gen": loss_dur_gen})
```
这行代码使用`update()`方法将一个键值对`{"loss/g/dur_gen": loss_dur_gen}`添加到`scalar_dict`字典中。

```
scalar_dict.update(
    {
        "loss/g/dur_gen_{}".format(i): v
        for i, v in enumerate(losses_dur_gen)
    }
)
```
这段代码使用了字典推导式，将`losses_dur_gen`列表中的元素作为值，根据索引`i`和字符串模板`"loss/g/dur_gen_{}"`生成键，最终生成一个字典，并将该字典中的键值对添加到`scalar_dict`字典中。

```
image_dict = {
    "slice/mel_org": utils.plot_spectrogram_to_numpy(
        y_mel[0].data.cpu().numpy()
    ),
    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
        y_hat_mel[0].data.cpu().numpy()
    )
}
```
这段代码创建了一个字典`image_dict`，其中包含两个键值对。键`"slice/mel_org"`对应的值是通过调用`utils.plot_spectrogram_to_numpy()`函数将`y_mel[0].data.cpu().numpy()`转换为numpy数组得到的结果。键`"slice/mel_gen"`对应的值是通过调用`utils.plot_spectrogram_to_numpy()`函数将`y_hat_mel[0].data.cpu().numpy()`转换为numpy数组得到的结果。
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/attn": utils.plot_alignment_to_numpy(
                        attn[0, 0].data.cpu().numpy()
                    ),
                }
                # 调用utils.summarize函数，将图像和标量数据写入TensorBoard
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            # 如果全局步数能够被hps.train.eval_interval整除
            if global_step % hps.train.eval_interval == 0:
                # 调用evaluate函数，评估模型性能
                evaluate(hps, net_g, eval_loader, writer_eval)
                # 调用utils.save_checkpoint函数，保存模型的checkpoint
                utils.save_checkpoint(
                    net_g,
                    optim_g,
# 保存生成器的模型参数到指定路径
utils.save_checkpoint(
    net_g,  # 生成器网络模型
    optim_g,  # 生成器的优化器
    hps.train.learning_rate,  # 学习率
    epoch,  # 当前训练的轮数
    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),  # 保存路径
)

# 保存判别器的模型参数到指定路径
utils.save_checkpoint(
    net_d,  # 判别器网络模型
    optim_d,  # 判别器的优化器
    hps.train.learning_rate,  # 学习率
    epoch,  # 当前训练的轮数
    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),  # 保存路径
)

# 保存权重衰减器的模型参数到指定路径
utils.save_checkpoint(
    net_wd,  # 权重衰减器网络模型
    optim_wd,  # 权重衰减器的优化器
    hps.train.learning_rate,  # 学习率
    epoch,  # 当前训练的轮数
    os.path.join(hps.model_dir, "WD_{}.pth".format(global_step)),  # 保存路径
)

# 如果存在持续性判别器，则保存其模型参数到指定路径
if net_dur_disc is not None:
    utils.save_checkpoint(
        net_dur_disc,  # 持续性判别器网络模型
        optim_dur_disc,  # 持续性判别器的优化器
        hps.train.learning_rate,  # 学习率
        epoch,  # 当前训练的轮数
        os.path.join(hps.model_dir, "DurDisc_{}.pth".format(global_step)),  # 保存路径
    )
```

这段代码是在训练过程中保存模型的参数。通过调用`utils.save_checkpoint()`函数，将生成器、判别器、权重衰减器和持续性判别器（如果存在）的模型参数、优化器、学习率、当前训练轮数以及保存路径作为参数传入，将它们保存到指定的路径下。每个模型都会保存在不同的文件中，文件名中包含了当前的全局步数（global_step）和模型类型的标识。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
                keep_ckpts = config.train_ms_config.keep_ckpts
                if keep_ckpts > 0:
                    # 清理模型检查点，保留指定数量的检查点
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

    # gc.collect()
    # torch.cuda.empty_cache()
    if rank == 0:
        # 打印当前的训练轮数
        logger.info("====> Epoch: {}".format(epoch))
```

注释已经添加在代码中，解释了每个语句的作用。
# 定义一个函数evaluate，用于评估模型的性能
def evaluate(hps, generator, eval_loader, writer_eval):
    # 将generator设置为评估模式，即不进行梯度计算
    generator.eval()
    # 创建空字典image_dict和audio_dict，用于存储图像和音频数据
    image_dict = {}
    audio_dict = {}
    # 打印"Evaluating ..."，表示正在进行评估
    print("Evaluating ...")
    # 使用torch.no_grad()上下文管理器，确保在评估过程中不进行梯度计算
    with torch.no_grad():
        # 遍历评估数据集中的每个批次
        for batch_idx, (
            x,  # 输入数据x
            x_lengths,  # 输入数据x的长度
            spec,  # 频谱数据
            spec_lengths,  # 频谱数据的长度
            y,  # 目标数据y
            y_lengths,  # 目标数据y的长度
            speakers,  # 说话人信息
            tone,  # 音调信息
            language,  # 语言信息
            bert,  # BERT特征
            ja_bert,  # 日语BERT特征
```

注释解释了每个语句的作用，包括函数定义、模型评估模式设置、创建空字典、打印提示信息、禁用梯度计算和遍历评估数据集等。
# 将en_bert从eval_loader中取出，并使用enumerate函数获取其索引
for i, (
    en_bert,
) in enumerate(eval_loader):
    # 将x, x_lengths, spec, spec_lengths, y, y_lengths, speakers, bert, ja_bert, en_bert, tone, language依次移动到GPU上
    x, x_lengths = x.cuda(), x_lengths.cuda()
    spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
    y, y_lengths = y.cuda(), y_lengths.cuda()
    speakers = speakers.cuda()
    bert = bert.cuda()
    ja_bert = ja_bert.cuda()
    en_bert = en_bert.cuda()
    tone = tone.cuda()
    language = language.cuda()
    # 对于use_sdp为True和False的两种情况进行循环
    for use_sdp in [True, False]:
        # 调用generator模块的infer函数进行推理，传入x, x_lengths, speakers, tone, language, bert, ja_bert, en_bert等参数
        y_hat, attn, mask, *_ = generator.module.infer(
            x,
            x_lengths,
            speakers,
            tone,
            language,
            bert,
            ja_bert,
            en_bert,
            use_sdp=use_sdp
        )
en_bert,  # 定义变量 en_bert

y=spec,  # 将变量 spec 赋值给变量 y

max_len=1000,  # 定义变量 max_len，并赋值为 1000

sdp_ratio=0.0 if not use_sdp else 1.0,  # 如果 use_sdp 为 False，则将变量 sdp_ratio 赋值为 0.0，否则赋值为 1.0

)  # 函数调用的结束括号

y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length  # 计算 mask 在维度 1 和 2 上的和，转换为整型，并与变量 hps.data.hop_length 相乘，将结果赋值给变量 y_hat_lengths

mel = spec_to_mel_torch(  # 调用函数 spec_to_mel_torch，并传入以下参数

spec,  # 变量 spec

hps.data.filter_length,  # 变量 hps.data.filter_length

hps.data.n_mel_channels,  # 变量 hps.data.n_mel_channels

hps.data.sampling_rate,  # 变量 hps.data.sampling_rate

hps.data.mel_fmin,  # 变量 hps.data.mel_fmin

hps.data.mel_fmax,  # 变量 hps.data.mel_fmax

)  # 函数调用的结束括号

y_hat_mel = mel_spectrogram_torch(  # 调用函数 mel_spectrogram_torch，并传入以下参数

y_hat.squeeze(1).float(),  # 将变量 y_hat 在维度 1 上压缩，并转换为浮点型

hps.data.filter_length,  # 变量 hps.data.filter_length

hps.data.n_mel_channels,  # 变量 hps.data.n_mel_channels

hps.data.sampling_rate,  # 变量 hps.data.sampling_rate
# 将模型生成的音频和图像数据添加到字典中
image_dict.update(
    {
        f"gen/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
            y_hat_mel[0].cpu().numpy()
        )
    }
)
audio_dict.update(
    {
        f"gen/audio_{batch_idx}_{use_sdp}": y_hat[
            0, :, : y_hat_lengths[0]
        ]
    }
)
image_dict.update(
```

这段代码将模型生成的音频和图像数据添加到字典中。具体来说：

- `image_dict.update()` 将生成的图像数据添加到 `image_dict` 字典中。
- `f"gen/mel_{batch_idx}"` 是图像数据的键，其中 `batch_idx` 是批次索引。
- `utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())` 是将模型生成的 Mel 频谱图转换为 numpy 数组的函数调用。

- `audio_dict.update()` 将生成的音频数据添加到 `audio_dict` 字典中。
- `f"gen/audio_{batch_idx}_{use_sdp}"` 是音频数据的键，其中 `batch_idx` 是批次索引，`use_sdp` 是一个布尔值。
- `y_hat[0, :, : y_hat_lengths[0]]` 是模型生成的音频数据的切片。

这段代码的作用是将模型生成的音频和图像数据添加到字典中，以便后续使用。
{
    f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
        mel[0].cpu().numpy()
    )
}
```
这段代码将一个键值对添加到字典中，键是以`gt/mel_`开头并以`batch_idx`结尾的字符串，值是通过`utils.plot_spectrogram_to_numpy()`函数将`mel[0].cpu().numpy()`转换而来的数据。

```
audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})
```
这段代码将一个键值对添加到字典中，键是以`gt/audio_`开头并以`batch_idx`结尾的字符串，值是`y[0, :, : y_lengths[0]]`。

```
utils.summarize(
    writer=writer_eval,
    global_step=global_step,
    images=image_dict,
    audios=audio_dict,
    audio_sampling_rate=hps.data.sampling_rate,
)
```
这段代码调用了`utils.summarize()`函数，传入了多个参数，包括`writer`、`global_step`、`images`、`audios`和`audio_sampling_rate`。这个函数用于生成摘要信息，将结果写入`writer_eval`中。

```
generator.train()
```
这段代码调用了`generator`对象的`train()`方法，用于训练生成器模型。

```
if __name__ == "__main__":
    run()
```
这段代码用于判断当前脚本是否作为主程序运行，如果是，则调用`run()`函数。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
# 根据 ZIP 文件名读取其二进制，封装成字节流
bio = BytesIO(open(fname, 'rb').read())
```
这行代码将打开给定的 ZIP 文件，并将其内容读取为二进制数据。然后，使用BytesIO类将二进制数据封装成字节流。

```
# 使用字节流里面内容创建 ZIP 对象
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用字节流里面的内容创建一个ZIP对象。'r'参数表示以只读模式打开ZIP文件。

```
# 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码遍历ZIP对象中包含的所有文件名，并使用zip.read(n)读取每个文件的数据。然后，将文件名和数据组成一个字典。

```
# 关闭 ZIP 对象
zip.close()
```
这行代码关闭ZIP对象，释放资源。

```
# 返回结果字典
return fdict
```
这行代码返回包含文件名到数据的字典作为函数的结果。
```