# `.\lucidrains\DALLE2-pytorch\train_decoder.py`

```py
# 导入所需的模块
from pathlib import Path
from typing import List
from datetime import timedelta

# 导入自定义模块
from dalle2_pytorch.trainer import DecoderTrainer
from dalle2_pytorch.dataloaders import create_image_embedding_dataloader
from dalle2_pytorch.trackers import Tracker
from dalle2_pytorch.train_configs import DecoderConfig, TrainDecoderConfig
from dalle2_pytorch.utils import Timer, print_ribbon
from dalle2_pytorch.dalle2_pytorch import Decoder, resize_image_to
from clip import tokenize

# 导入第三方模块
import torchvision
import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import dataclasses as accelerate_dataclasses
import webdataset as wds
import click

# 定义常量
TRAIN_CALC_LOSS_EVERY_ITERS = 10
VALID_CALC_LOSS_EVERY_ITERS = 10

# 定义辅助函数
def exists(val):
    return val is not None

# 定义主要函数
def create_dataloaders(
    available_shards,
    webdataset_base_url,
    img_embeddings_url=None,
    text_embeddings_url=None,
    shard_width=6,
    num_workers=4,
    batch_size=32,
    n_sample_images=6,
    shuffle_train=True,
    resample_train=False,
    img_preproc = None,
    index_width=4,
    train_prop = 0.75,
    val_prop = 0.15,
    test_prop = 0.10,
    seed = 0,
    **kwargs
):
    """
    随机将可用的数据分片分为训练、验证和测试集，并为每个集合返回一个数据加载器
    """
    # 检查训练、验证和测试集的比例之和是否为1
    assert train_prop + test_prop + val_prop == 1
    # 计算训练集、测试集和验证集的数量
    num_train = round(train_prop*len(available_shards))
    num_test = round(test_prop*len(available_shards))
    num_val = len(available_shards) - num_train - num_test
    # 检查分配是否正确
    assert num_train + num_test + num_val == len(available_shards), f"{num_train} + {num_test} + {num_val} = {num_train + num_test + num_val} != {len(available_shards)}"
    # 使用随机数生成器手动设置种子，将数据集随机分为训练、测试和验证集
    train_split, test_split, val_split = torch.utils.data.random_split(available_shards, [num_train, num_test, num_val], generator=torch.Generator().manual_seed(seed))

    # 根据分片宽度将训练、测试和验证集的 URL 进行格式化
    train_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in train_split]
    test_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in test_split]
    val_urls = [webdataset_base_url.format(str(shard).zfill(shard_width)) for shard in val_split]
    
    # 创建数据加载器的 lambda 函数
    create_dataloader = lambda tar_urls, shuffle=False, resample=False, for_sampling=False: create_image_embedding_dataloader(
        tar_url=tar_urls,
        num_workers=num_workers,
        batch_size=batch_size if not for_sampling else n_sample_images,
        img_embeddings_url=img_embeddings_url,
        text_embeddings_url=text_embeddings_url,
        index_width=index_width,
        shuffle_num = None,
        extra_keys= ["txt"],
        shuffle_shards = shuffle,
        resample_shards = resample, 
        img_preproc=img_preproc,
        handler=wds.handlers.warn_and_continue
    )

    # 创建训练、验证和测试集的数据加载器
    train_dataloader = create_dataloader(train_urls, shuffle=shuffle_train, resample=resample_train)
    train_sampling_dataloader = create_dataloader(train_urls, shuffle=False, for_sampling=True)
    val_dataloader = create_dataloader(val_urls, shuffle=False)
    test_dataloader = create_dataloader(test_urls, shuffle=False)
    test_sampling_dataloader = create_dataloader(test_urls, shuffle=False, for_sampling=True)
    # 返回数据加载器字典
    return {
        "train": train_dataloader,
        "train_sampling": train_sampling_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
        "test_sampling": test_sampling_dataloader
    }

def get_dataset_keys(dataloader):
    """
    # 有时需要获取数据加载器返回的键。由于数据集被嵌入在数据加载器中，我们需要进行一些处理来恢复它。
    """
    # 如果数据加载器实际上是一个WebLoader，则需要提取真正的数据加载器
    if isinstance(dataloader, wds.WebLoader):
        dataloader = dataloader.pipeline[0]
    # 返回数据加载器的数据集键映射
    return dataloader.dataset.key_map
# 从数据加载器中获取示例数据，返回一个包含示例的列表
def get_example_data(dataloader, device, n=5):
    # 初始化空列表
    images = []
    img_embeddings = []
    text_embeddings = []
    captions = []
    # 遍历数据加载器
    for img, emb, txt in dataloader:
        # 获取图像和文本嵌入
        img_emb, text_emb = emb.get('img'), emb.get('text')
        # 如果图像嵌入不为空
        if img_emb is not None:
            # 将图像嵌入转移到指定设备上
            img_emb = img_emb.to(device=device, dtype=torch.float)
            img_embeddings.extend(list(img_emb))
        else:
            # 否则添加与图像形状相同数量的 None
            img_embeddings.extend([None]*img.shape[0])
        # 如果文本嵌入不为空
        if text_emb is not None:
            # 将文本嵌入转移到指定设备上
            text_emb = text_emb.to(device=device, dtype=torch.float)
            text_embeddings.extend(list(text_emb))
        else:
            # 否则添加与图像形状相同数量的 None
            text_embeddings.extend([None]*img.shape[0])
        # 将图像转移到指定设备上
        img = img.to(device=device, dtype=torch.float)
        images.extend(list(img))
        captions.extend(list(txt))
        # 如果示例数量达到指定数量，跳出循环
        if len(images) >= n:
            break
    # 返回示例列表
    return list(zip(images[:n], img_embeddings[:n], text_embeddings[:n], captions[:n]))

# 生成样本并从嵌入中生成图像
def generate_samples(trainer, example_data, clip=None, start_unet=1, end_unet=None, condition_on_text_encodings=False, cond_scale=1.0, device=None, text_prepend="", match_image_size=True):
    # 解压示例数据
    real_images, img_embeddings, text_embeddings, txts = zip(*example_data)
    sample_params = {}
    # 如果图像嵌入为空
    if img_embeddings[0] is None:
        # 从真实图像生成图像嵌入
        imgs_tensor = torch.stack(real_images)
        assert clip is not None, "clip is None, but img_embeddings is None"
        imgs_tensor.to(device=device)
        img_embeddings, img_encoding = clip.embed_image(imgs_tensor)
        sample_params["image_embed"] = img_embeddings
    else:
        # 使用预先计算的图像嵌入
        img_embeddings = torch.stack(img_embeddings)
        sample_params["image_embed"] = img_embeddings
    # 如果基于文本编码条件生成
    if condition_on_text_encodings:
        # 如果文本嵌入为空
        if text_embeddings[0] is None:
            # 从文本生成文本嵌入
            assert clip is not None, "clip is None, but text_embeddings is None"
            tokenized_texts = tokenize(txts, truncate=True).to(device=device)
            text_embed, text_encodings = clip.embed_text(tokenized_texts)
            sample_params["text_encodings"] = text_encodings
        else:
            # 使用预先计算的文本嵌入
            text_embeddings = torch.stack(text_embeddings)
            sample_params["text_encodings"] = text_embeddings
    sample_params["start_at_unet_number"] = start_unet
    sample_params["stop_at_unet_number"] = end_unet
    # 如果只训练上采样器
    if start_unet > 1:
        sample_params["image"] = torch.stack(real_images)
    if device is not None:
        sample_params["_device"] = device
    # 生成样本
    samples = trainer.sample(**sample_params, _cast_deepspeed_precision=False)  # 在采样时不需要转换为 FP16
    generated_images = list(samples)
    captions = [text_prepend + txt for txt in txts]
    # 如果匹配图像大小
    if match_image_size:
        generated_image_size = generated_images[0].shape[-1]
        real_images = [resize_image_to(image, generated_image_size, clamp_range=(0, 1)) for image in real_images]
    # 返回真实图像、生成图像和标题
    return real_images, generated_images, captions

# 生成网格样本
def generate_grid_samples(trainer, examples, clip=None, start_unet=1, end_unet=None, condition_on_text_encodings=False, cond_scale=1.0, device=None, text_prepend=""):
    # 生成样本并使用 torchvision 将其放入并排网格中以便查看
    real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend)
    # 使用torchvision.utils.make_grid函数将每对原始图像和生成图像组合成一个图像网格
    grid_images = [torchvision.utils.make_grid([original_image, generated_image]) for original_image, generated_image in zip(real_images, generated_images)]
    # 返回图像网格列表和对应的文本描述
    return grid_images, captions
def evaluate_trainer(trainer, dataloader, device, start_unet, end_unet, clip=None, condition_on_text_encodings=False, cond_scale=1.0, inference_device=None, n_evaluation_samples=1000, FID=None, IS=None, KID=None, LPIPS=None):
    """
    Computes evaluation metrics for the decoder
    """
    metrics = {}
    # 准备数据
    examples = get_example_data(dataloader, device, n_evaluation_samples)
    if len(examples) == 0:
        print("No data to evaluate. Check that your dataloader has shards.")
        return metrics
    real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, inference_device)
    real_images = torch.stack(real_images).to(device=device, dtype=torch.float)
    generated_images = torch.stack(generated_images).to(device=device, dtype=torch.float)
    # 将像素值从 [0, 1] 转换为 [0, 255]，并将数据类型从 torch.float 转换为 torch.uint8
    int_real_images = real_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)
    int_generated_images = generated_images.mul(255).add(0.5).clamp(0, 255).type(torch.uint8)

    def null_sync(t, *args, **kwargs):
        return [t]

    if exists(FID):
        fid = FrechetInceptionDistance(**FID, dist_sync_fn=null_sync)
        fid.to(device=device)
        fid.update(int_real_images, real=True)
        fid.update(int_generated_images, real=False)
        metrics["FID"] = fid.compute().item()
    if exists(IS):
        inception = InceptionScore(**IS, dist_sync_fn=null_sync)
        inception.to(device=device)
        inception.update(int_real_images)
        is_mean, is_std = inception.compute()
        metrics["IS_mean"] = is_mean.item()
        metrics["IS_std"] = is_std.item()
    if exists(KID):
        kernel_inception = KernelInceptionDistance(**KID, dist_sync_fn=null_sync)
        kernel_inception.to(device=device)
        kernel_inception.update(int_real_images, real=True)
        kernel_inception.update(int_generated_images, real=False)
        kid_mean, kid_std = kernel_inception.compute()
        metrics["KID_mean"] = kid_mean.item()
        metrics["KID_std"] = kid_std.item()
    if exists(LPIPS):
        # 将像素值从 [0, 1] 转换为 [-1, 1]
        renorm_real_images = real_images.mul(2).sub(1).clamp(-1,1)
        renorm_generated_images = generated_images.mul(2).sub(1).clamp(-1,1)
        lpips = LearnedPerceptualImagePatchSimilarity(**LPIPS, dist_sync_fn=null_sync)
        lpips.to(device=device)
        lpips.update(renorm_real_images, renorm_generated_images)
        metrics["LPIPS"] = lpips.compute().item()

    if trainer.accelerator.num_processes > 1:
        # 同步指标
        metrics_order = sorted(metrics.keys())
        metrics_tensor = torch.zeros(1, len(metrics), device=device, dtype=torch.float)
        for i, metric_name in enumerate(metrics_order):
            metrics_tensor[0, i] = metrics[metric_name]
        metrics_tensor = trainer.accelerator.gather(metrics_tensor)
        metrics_tensor = metrics_tensor.mean(dim=0)
        for i, metric_name in enumerate(metrics_order):
            metrics[metric_name] = metrics_tensor[i].item()
    return metrics

def save_trainer(tracker: Tracker, trainer: DecoderTrainer, epoch: int, sample: int, next_task: str, validation_losses: List[float], samples_seen: int, is_latest=True, is_best=False):
    """
    Logs the model with an appropriate method depending on the tracker
    """
    tracker.save(trainer, is_best=is_best, is_latest=is_latest, epoch=epoch, sample=sample, next_task=next_task, validation_losses=validation_losses, samples_seen=samples_seen)
    
def recall_trainer(tracker: Tracker, trainer: DecoderTrainer):
    """
    Loads the model with an appropriate method depending on the tracker
    """
    trainer.accelerator.print(print_ribbon(f"Loading model from {type(tracker.loader).__name__}"))
    state_dict = tracker.recall()
    trainer.load_state_dict(state_dict, only_model=False, strict=True)
    # 返回状态字典中的"epoch"键对应的值，如果不存在则返回默认值0
    # 返回状态字典中的"validation_losses"键对应的值，如果不存在则返回空列表
    # 返回状态字典中的"next_task"键对应的值，如果不存在则返回默认值"train"
    # 返回状态字典中的"sample"键对应的值，如果不存在则返回默认值0
    # 返回状态字典中的"samples_seen"键对应的值，如果不存在则返回默认值0
    return state_dict.get("epoch", 0), state_dict.get("validation_losses", []), state_dict.get("next_task", "train"), state_dict.get("sample", 0), state_dict.get("samples_seen", 0)
# 定义训练函数，用于训练解码器模型
def train(
    dataloaders,  # 数据加载器
    decoder: Decoder,  # 解码器模型
    accelerator: Accelerator,  # 加速器
    tracker: Tracker,  # 追踪器
    inference_device,  # 推断设备
    clip=None,  # 梯度裁剪阈值
    evaluate_config=None,  # 评估配置
    epoch_samples = None,  # 每个周期的样本数
    validation_samples = None,  # 验证样本数
    save_immediately=False,  # 是否立即保存
    epochs = 20,  # 训练周期数
    n_sample_images = 5,  # 样本图像数
    save_every_n_samples = 100000,  # 每隔多少样本保存一次
    unet_training_mask=None,  # UNet训练掩码
    condition_on_text_encodings=False,  # 是否基于文本编码条件
    cond_scale=1.0,  # 条件缩放
    **kwargs  # 其他参数
):
    """
    Trains a decoder on a dataset.
    """
    is_master = accelerator.process_index == 0

    if not exists(unet_training_mask):
        # 如果未提供UNet训练掩码，则默认所有UNet都应训练
        unet_training_mask = [True] * len(decoder.unets)
    assert len(unet_training_mask) == len(decoder.unets), f"The unet training mask should be the same length as the number of unets in the decoder. Got {len(unet_training_mask)} and {trainer.num_unets}"
    trainable_unet_numbers = [i+1 for i, trainable in enumerate(unet_training_mask) if trainable]
    first_trainable_unet = trainable_unet_numbers[0]
    last_trainable_unet = trainable_unet_numbers[-1]
    def move_unets(unet_training_mask):
        for i in range(len(decoder.unets)):
            if not unet_training_mask[i]:
                # 将不可训练的UNet替换为nn.Identity()。此训练脚本不使用未训练的UNet，因此这样做是可以的。
                decoder.unets[i] = nn.Identity().to(inference_device)
    # 移除不可训练的UNet
    move_unets(unet_training_mask)

    trainer = DecoderTrainer(
        decoder=decoder,
        accelerator=accelerator,
        dataloaders=dataloaders,
        **kwargs
    )

    # 根据召回的状态字典设置起始模型和参数
    start_epoch = 0
    validation_losses = []
    next_task = 'train'
    sample = 0
    samples_seen = 0
    val_sample = 0
    step = lambda: int(trainer.num_steps_taken(unet_number=first_trainable_unet))

    if tracker.can_recall:
        start_epoch, validation_losses, next_task, recalled_sample, samples_seen = recall_trainer(tracker, trainer)
        if next_task == 'train':
            sample = recalled_sample
        if next_task == 'val':
            val_sample = recalled_sample
        accelerator.print(f"Loaded model from {type(tracker.loader).__name__} on epoch {start_epoch} having seen {samples_seen} samples with minimum validation loss {min(validation_losses) if len(validation_losses) > 0 else 'N/A'}")
        accelerator.print(f"Starting training from task {next_task} at sample {sample} and validation sample {val_sample}")
    trainer.to(device=inference_device)

    accelerator.print(print_ribbon("Generating Example Data", repeat=40))
    accelerator.print("This can take a while to load the shard lists...")
    if is_master:
        train_example_data = get_example_data(dataloaders["train_sampling"], inference_device, n_sample_images)
        accelerator.print("Generated training examples")
        test_example_data = get_example_data(dataloaders["test_sampling"], inference_device, n_sample_images)
        accelerator.print("Generated testing examples")
    
    send_to_device = lambda arr: [x.to(device=inference_device, dtype=torch.float) for x in arr]

    sample_length_tensor = torch.zeros(1, dtype=torch.int, device=inference_device)
    unet_losses_tensor = torch.zeros(TRAIN_CALC_LOSS_EVERY_ITERS, trainer.num_unets, dtype=torch.float, device=inference_device)
    # 等待所有节点到达此处，以防止它们在不同时间尝试自动恢复当前运行，这没有意义并会导致错误
    accelerator.wait_for_everyone()
    # 使用给定的配置创建跟踪器对象
    tracker: Tracker = tracker_config.create(config, accelerator_config, dummy_mode=dummy)
    # 将配置保存到指定路径下的文件中，文件名为'decoder_config.json'
    tracker.save_config(config_path, config_name='decoder_config.json')
    # 添加保存元数据，键为'state_dict_key'，值为配置模型的转储
    tracker.add_save_metadata(state_dict_key='config', metadata=config.model_dump())
    # 返回跟踪器对象
    return tracker
def initialize_training(config: TrainDecoderConfig, config_path):
    # 确保在不加载时，分布式模型初始化为相同的值
    torch.manual_seed(config.seed)

    # 为可配置的分布式训练设置加速器
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.train.find_unused_parameters, static_graph=config.train.static_graph)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=60*60))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, init_kwargs])

    if accelerator.num_processes > 1:
        # 使用分布式训练，并立即确保所有进程都可以连接
        accelerator.print("Waiting for all processes to connect...")
        accelerator.wait_for_everyone()
        accelerator.print("All processes online and connected")

    # 如果我们处于深度学习 fp16 模式，则必须确保学习的方差关闭
    if accelerator.mixed_precision == "fp16" and accelerator.distributed_type == accelerate_dataclasses.DistributedType.DEEPSPEED and config.decoder.learned_variance:
        raise ValueError("DeepSpeed fp16 mode does not support learned variance")
    
    # 设置数据
    all_shards = list(range(config.data.start_shard, config.data.end_shard + 1))
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    shards_per_process = len(all_shards) // world_size
    assert shards_per_process > 0, "Not enough shards to split evenly"
    my_shards = all_shards[rank * shards_per_process: (rank + 1) * shards_per_process]

    dataloaders = create_dataloaders (
        available_shards=my_shards,
        img_preproc = config.data.img_preproc,
        train_prop = config.data.splits.train,
        val_prop = config.data.splits.val,
        test_prop = config.data.splits.test,
        n_sample_images=config.train.n_sample_images,
        **config.data.model_dump(),
        rank = rank,
        seed = config.seed,
    )

    # 如果模型中有 clip，则需要将其移除以与 deepspeed 兼容
    clip = None
    if config.decoder.clip is not None:
        clip = config.decoder.clip.create()  # 当然我们保留它以在训练期间使用，只是不在解码器中使用会导致问题
        config.decoder.clip = None
    # 创建解码器模型并打印基本信息
    decoder = config.decoder.create()
    get_num_parameters = lambda model, only_training=False: sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_training))

    # 如果我们是主节点，则创建并初始化跟踪器
    tracker = create_tracker(accelerator, config, config_path, dummy = rank!=0)

    has_img_embeddings = config.data.img_embeddings_url is not None
    has_text_embeddings = config.data.text_embeddings_url is not None
    conditioning_on_text = any([unet.cond_on_text_encodings for unet in config.decoder.unets])

    has_clip_model = clip is not None
    data_source_string = ""

    if has_img_embeddings:
        data_source_string += "precomputed image embeddings"
    elif has_clip_model:
        data_source_string += "clip image embeddings generation"
    else:
        raise ValueError("No image embeddings source specified")
    if conditioning_on_text:
        if has_text_embeddings:
            data_source_string += " and precomputed text embeddings"
        elif has_clip_model:
            data_source_string += " and clip text encoding generation"
        else:
            raise ValueError("No text embeddings source specified")

    accelerator.print(print_ribbon("Loaded Config", repeat=40))
    accelerator.print(f"Running training with {accelerator.num_processes} processes and {accelerator.distributed_type} distributed training")
    accelerator.print(f"Training using {data_source_string}. {'conditioned on text' if conditioning_on_text else 'not conditioned on text'}")
    # 打印解码器的参数数量，包括总数和仅训练时的数量
    accelerator.print(f"Number of parameters: {get_num_parameters(decoder)} total; {get_num_parameters(decoder, only_training=True)} training")
    # 遍历解码器中的每个 UNet 模型
    for i, unet in enumerate(decoder.unets):
        # 打印每个 UNet 模型的参数数量，包括总数和仅训练时的数量
        accelerator.print(f"Unet {i} has {get_num_parameters(unet)} total; {get_num_parameters(unet, only_training=True)} training")

    # 调用训练函数，传入数据加载器、解码器、加速器等参数
    train(dataloaders, decoder, accelerator,
        clip=clip,
        tracker=tracker,
        inference_device=accelerator.device,
        evaluate_config=config.evaluate,
        condition_on_text_encodings=conditioning_on_text,
        **config.train.model_dump(),
    )
# 创建一个简单的 click 命令行接口，用于加载配置并启动训练
@click.command()
@click.option("--config_file", default="./train_decoder_config.json", help="Path to config file")
def main(config_file):
    # 将配置文件路径转换为 Path 对象
    config_file_path = Path(config_file)
    # 从 JSON 文件路径加载训练配置
    config = TrainDecoderConfig.from_json_path(str(config_file_path))
    # 初始化训练，传入配置和配置文件路径
    initialize_training(config, config_path=config_file_path)

if __name__ == "__main__":
    # 如果作为脚本直接运行，则调用 main 函数
    main()
```