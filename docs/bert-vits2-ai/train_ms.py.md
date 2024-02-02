# `Bert-VITS2\train_ms.py`

```py
# flake8: noqa: E402
# 导入必要的库
import platform
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from config import config
import argparse
import datetime

# 设置 numba 库的日志级别
logging.getLogger("numba").setLevel(logging.WARNING)
# 导入自定义的模块
import commons
import utils
from data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    DurationDiscriminator,
    WavLMDiscriminator,
)
from losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    WavLMLoss,
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

# 设置 CUDA 相关的参数
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = (
    True  # 如果遇到训练问题，请尝试禁用 TF32
)
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.sdp_kernel("flash")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(
    True
)  # 如果 torch 版本低于 2.0，则不可用
global_step = 0

# 运行函数
def run():
    # 解析环境变量
    envs = config.train_ms_config.env
    for env_name, env_value in envs.items():
        if env_name not in os.environ.keys():
            print("加载config中的配置{}".format(str(env_value)))
            os.environ[env_name] = str(env_value)
    print(
        "加载环境变量 \nMASTER_ADDR: {},\nMASTER_PORT: {},\nWORLD_SIZE: {},\nRANK: {},\nLOCAL_RANK: {}".format(
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
            os.environ["WORLD_SIZE"],
            os.environ["RANK"],
            os.environ["LOCAL_RANK"],
        )
    )

    backend = "nccl"
    # 检查操作系统类型，如果是Windows，则切换到gloo后端
    if platform.system() == "Windows":
        backend = "gloo"  # If Windows,switch to gloo backend.
    # 初始化进程组，使用指定的后端和初始化方法，设置超时时间为300秒
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(seconds=300),
    )  # Use torchrun instead of mp.spawn
    # 获取当前进程的排名
    rank = dist.get_rank()
    # 获取本地GPU的排名
    local_rank = int(os.environ["LOCAL_RANK"])
    # 获取GPU的数量
    n_gpus = dist.get_world_size()

    # 解析命令行参数和配置文件
    parser = argparse.ArgumentParser()
    # 添加配置文件路径的命令行参数，默认为config.train_ms_config.config_path
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=config.train_ms_config.config_path,
        help="JSON file for configuration",
    )
    # 添加模型路径的命令行参数，默认为config.dataset_path
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="数据集文件夹路径，请注意，数据不再默认放在/logs文件夹下。如果需要用命令行配置，请声明相对于根目录的路径",
        default=config.dataset_path,
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 拼接模型路径
    model_dir = os.path.join(args.model, config.train_ms_config.model)
    # 如果模型路径不存在，则创建
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # 从配置文件中获取超参数
    hps = utils.get_hparams_from_file(args.config)
    hps.model_dir = model_dir
    # 比较命令行参数中的配置文件路径和默认配置文件路径是否相同，如果不同则更新默认配置文件
    if os.path.realpath(args.config) != os.path.realpath(
        config.train_ms_config.config_path
    ):
        with open(args.config, "r", encoding="utf-8") as f:
            data = f.read()
        with open(config.train_ms_config.config_path, "w", encoding="utf-8") as f:
            f.write(data)

    # 设置随机种子
    torch.manual_seed(hps.train.seed)
    # 设置当前设备的GPU
    torch.cuda.set_device(local_rank)

    # 初始化全局步数
    global global_step
    # 如果进程排名为0，则获取日志记录器、记录超参数、检查git哈希值、创建SummaryWriter对象
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    # 创建TextAudioSpeakerLoader对象，用于加载训练数据集
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    # 使用分布式桶采样器创建训练数据集的采样器，指定批量大小和桶的边界
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # 创建文本音频说话人数据的整合函数
    collate_fn = TextAudioSpeakerCollate()
    # 创建训练数据集的数据加载器，指定多线程读取数据，不打乱顺序，使用固定内存，整合函数，采样器，持续工作的工作进程数，预取因子
    train_loader = DataLoader(
        train_dataset,
        num_workers=min(config.train_ms_config.num_workers, os.cpu_count() - 1),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )  # DataLoader config could be adjusted.
    # 如果是主进程
    if rank == 0:
        # 创建评估数据集的数据加载器，指定无多线程读取数据，不打乱顺序，批量大小为1，使用固定内存，不丢弃最后一批数据，整合函数
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=0,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    # 如果模型配置中包含使用噪声缩放的 MAS，并且设置为 True
    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas is True
    ):
        # 打印信息
        print("Using noise scaled MAS for VITS2")
        # 初始化 MAS 噪声缩放和噪声缩放增量
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        # 打印信息
        print("Using normal MAS for VITS1")
        # 初始化 MAS 噪声缩放和噪声缩放增量为0
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0
    # 如果模型配置中包含使用持续鉴别器，并且设置为 True
    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator is True
    ):
        # 打印信息
        print("Using duration discriminator for VITS2")
        # 创建持续鉴别器网络
        net_dur_disc = DurationDiscriminator(
            hps.model.hidden_channels,
            hps.model.hidden_channels,
            3,
            0.1,
            gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
        ).cuda(local_rank)
    else:
        # 如果不使用持续鉴别器，设置为 None
        net_dur_disc = None
    # 如果模型配置中包含使用说话人条件编码器，并且设置为 True
    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder is True
    ):
        # 如果数据中的说话者数量为0，则抛出数值错误
        if hps.data.n_speakers == 0:
            raise ValueError(
                "n_speakers must be > 0 when using spk conditioned encoder to train multi-speaker model"
            )
    else:
        # 如果不是使用 spk conditioned encoder 训练多说话者模型，则打印提示信息
        print("Using normal encoder for VITS1")

    # 创建合成器模型对象，并将其部署到 GPU 上
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        **hps.model,
    ).cuda(local_rank)

    # 如果设置了 freeze_ZH_bert 标志，则冻结 ZH bert 编码器的参数
    if getattr(hps.train, "freeze_ZH_bert", False):
        print("Freezing ZH bert encoder !!!")
        for param in net_g.enc_p.bert_proj.parameters():
            param.requires_grad = False

    # 如果设置了 freeze_EN_bert 标志，则冻结 EN bert 编码器的参数
    if getattr(hps.train, "freeze_EN_bert", False):
        print("Freezing EN bert encoder !!!")
        for param in net_g.enc_p.en_bert_proj.parameters():
            param.requires_grad = False

    # 如果设置了 freeze_JP_bert 标志，则冻结 JP bert 编码器的参数
    if getattr(hps.train, "freeze_JP_bert", False):
        print("Freezing JP bert encoder !!!")
        for param in net_g.enc_p.ja_bert_proj.parameters():
            param.requires_grad = False

    # 创建多周期鉴别器模型对象，并将其部署到 GPU 上
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(local_rank)
    # 创建 WavLM 鉴别器模型对象，并将其部署到 GPU 上
    net_wd = WavLMDiscriminator(
        hps.model.slm.hidden, hps.model.slm.nlayers, hps.model.slm.initial_channel
    ).cuda(local_rank)
    # 创建合成器模型的优化器对象
    optim_g = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net_g.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # 创建多周期鉴别器模型的优化器对象
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # 创建 WavLM 鉴别器模型的优化器对象
    optim_wd = torch.optim.AdamW(
        net_wd.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # 如果网络持续时间不为None，则使用AdamW优化器对网络持续时间进行优化
    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        # 否则将优化器设为None
        optim_dur_disc = None
    # 使用分布式数据并行对网络进行初始化
    net_g = DDP(net_g, device_ids=[local_rank], bucket_cap_mb=512)
    net_d = DDP(net_d, device_ids=[local_rank], bucket_cap_mb=512)
    net_wd = DDP(net_wd, device_ids=[local_rank], bucket_cap_mb=512)
    # 如果网络持续时间不为None，则使用分布式数据并行对网络持续时间进行初始化
    if net_dur_disc is not None:
        net_dur_disc = DDP(
            net_dur_disc,
            device_ids=[local_rank],
            bucket_cap_mb=512,
        )

    # 如果配置中使用基础模型，则下载基础模型
    if config.train_ms_config.base["use_base_model"]:
        utils.download_checkpoint(
            hps.model_dir,
            config.train_ms_config.base,
            token=config.openi_token,
            mirror=config.mirror,
        )
    # 将网络持续时间和权重衰减的学习率设为配置中的学习率
    dur_resume_lr = hps.train.learning_rate
    wd_resume_lr = hps.train.learning_rate
    # 如果网络持续时间不为None，则尝试加载网络持续时间的检查点
    if net_dur_disc is not None:
        try:
            _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
                skip_optimizer=hps.train.skip_optimizer
                if "skip_optimizer" in hps.train
                else True,
            )
            # 如果优化器的参数组中没有"initial_lr"，则将其设为网络持续时间的学习率
            if not optim_dur_disc.param_groups[0].get("initial_lr"):
                optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr
        except:
            # 如果加载检查点失败，则打印"Initialize dur_disc"
            print("Initialize dur_disc")
    # 尝试加载生成器模型的最新检查点，包括优化器、学习率和训练轮次信息
    _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
        net_g,
        optim_g,
        skip_optimizer=hps.train.skip_optimizer
        if "skip_optimizer" in hps.train
        else True,
    )
    # 尝试加载判别器模型的最新检查点，包括优化器、学习率和训练轮次信息
    _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
        net_d,
        optim_d,
        skip_optimizer=hps.train.skip_optimizer
        if "skip_optimizer" in hps.train
        else True,
    )
    # 如果生成器优化器的初始学习率不存在，则设置为加载的恢复学习率
    if not optim_g.param_groups[0].get("initial_lr"):
        optim_g.param_groups[0]["initial_lr"] = g_resume_lr
    # 如果判别器优化器的初始学习率不存在，则设置为加载的恢复学习率
    if not optim_d.param_groups[0].get("initial_lr"):
        optim_d.param_groups[0]["initial_lr"] = d_resume_lr

    # 将轮次信息设置为加载的轮次信息或者1（如果加载失败）
    epoch_str = max(epoch_str, 1)
    # 计算全局步数
    global_step = int(
        utils.get_steps(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"))
    )
    # 打印模型存在的信息
    print(
        f"******************检测到模型存在，epoch为 {epoch_str}，gloabl step为 {global_step}*********************"
    )
    # 如果发生异常，打印异常信息，并将轮次信息设置为1，全局步数设置为0
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    # 尝试加载判别器模型的最新检查点，包括优化器、学习率和训练轮次信息
    _, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
        net_wd,
        optim_wd,
        skip_optimizer=hps.train.skip_optimizer
        if "skip_optimizer" in hps.train
        else True,
    )
    # 如果判别器优化器的初始学习率不存在，则设置为加载的恢复学习率
    if not optim_wd.param_groups[0].get("initial_lr"):
        optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr
    # 如果发生异常，打印异常信息
    except Exception as e:
        print(e)

    # 创建生成器的学习率衰减调度器
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    # 创建一个指数衰减的学习率调度器，用于更新判别器的学习率
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    # 创建一个指数衰减的学习率调度器，用于更新生成器的学习率
    scheduler_wd = torch.optim.lr_scheduler.ExponentialLR(
        optim_wd, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    # 如果存在持续器鉴别器，则创建一个指数衰减的学习率调度器，用于更新持续器鉴别器的学习率
    if net_dur_disc is not None:
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        # 如果不存在持续器鉴别器，则将其设为None
        scheduler_dur_disc = None
    # 创建一个梯度缩放器，用于混合精度训练
    scaler = GradScaler(enabled=hps.train.bf16_run)

    # 创建一个WavLMLoss对象，用于计算语音生成的损失
    wl = WavLMLoss(
        hps.model.slm.model,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr,
    ).to(local_rank)

    # 循环遍历每个训练轮次
    for epoch in range(epoch_str, hps.train.epochs + 1):
        # 如果当前进程的rank为0，则进行训练和评估
        if rank == 0:
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
        else:
            # 如果当前进程的rank不为0，则进行训练
            train_and_evaluate(
                rank,
                local_rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc, net_wd, wl],
                [optim_g, optim_d, optim_dur_disc, optim_wd],
                [scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        # 更新生成器的学习率
        scheduler_g.step()
        # 更新判别器的学习率
        scheduler_d.step()
        # 更新持续器鉴别器的学习率
        scheduler_wd.step()
        # 如果存在持续器鉴别器，则更新其学习率
        if net_dur_disc is not None:
            scheduler_dur_disc.step()
# 定义训练和评估函数，接受多个参数
def train_and_evaluate(
    rank,
    local_rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
):
    # 解包nets参数
    net_g, net_d, net_dur_disc, net_wd, wl = nets
    # 解包optims参数
    optim_g, optim_d, optim_dur_disc, optim_wd = optims
    # 解包schedulers参数
    scheduler_g, scheduler_d, scheduler_dur_disc, scheduler_wd = schedulers
    # 解包loaders参数
    train_loader, eval_loader = loaders
    # 如果writers不为空，则解包writers参数
    if writers is not None:
        writer, writer_eval = writers

    # 设置训练数据的epoch
    train_loader.batch_sampler.set_epoch(epoch)
    # 设置全局变量global_step
    global global_step

    # 设置net_g, net_d, net_wd, net_dur_disc为训练模式
    net_g.train()
    net_d.train()
    net_wd.train()
    # 如果net_dur_disc不为空，则设置其为训练模式
    if net_dur_disc is not None:
        net_dur_disc.train()
    # 遍历batch_idx和多个参数
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
    # gc.collect()
    # torch.cuda.empty_cache()
    # 如果rank为0，则记录日志信息
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


# 定义评估函数，接受hps, generator, eval_loader, writer_eval参数
def evaluate(hps, generator, eval_loader, writer_eval):
    # 设置generator为评估模式
    generator.eval()
    # 初始化image_dict和audio_dict
    image_dict = {}
    audio_dict = {}
    # 打印"Evaluating ..."
    print("Evaluating ...")
    # 调用utils.summarize函数，传入writer_eval, global_step, image_dict, audio_dict, hps.data.sampling_rate参数
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    # 设置generator为训练模式
    generator.train()


# 如果当前脚本作为主程序运行，则调用run函数
if __name__ == "__main__":
    run()
```