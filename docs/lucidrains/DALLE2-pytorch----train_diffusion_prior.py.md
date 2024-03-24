# `.\lucidrains\DALLE2-pytorch\train_diffusion_prior.py`

```
import click  # 导入 click 库，用于创建命令行界面
import torch  # 导入 PyTorch 库

from torch import nn  # 从 PyTorch 中导入 nn 模块
from typing import List  # 导入 List 类型提示
from accelerate import Accelerator  # 从 accelerate 库中导入 Accelerator 类
from accelerate.utils import set_seed  # 从 accelerate 库中导入 set_seed 函数
from torch.utils.data import DataLoader  # 从 PyTorch 中导入 DataLoader 类
from embedding_reader import EmbeddingReader  # 导入自定义的 embedding_reader 模块
from accelerate.utils import dataclasses as accelerate_dataclasses  # 从 accelerate 库中导入 dataclasses 模块
from dalle2_pytorch.utils import Timer  # 从 dalle2_pytorch.utils 中导入 Timer 类
from dalle2_pytorch.trackers import Tracker  # 从 dalle2_pytorch.trackers 中导入 Tracker 类
from dalle2_pytorch import DiffusionPriorTrainer  # 导入自定义的 DiffusionPriorTrainer 类
from dalle2_pytorch.dataloaders import get_reader, make_splits  # 从 dalle2_pytorch.dataloaders 中导入 get_reader 和 make_splits 函数
from dalle2_pytorch.train_configs import (  # 从 dalle2_pytorch.train_configs 中导入 TrainDiffusionPriorConfig 相关配置
    DiffusionPriorConfig,
    DiffusionPriorTrainConfig,
    TrainDiffusionPriorConfig,
)


# helpers


cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # 创建一个计算余弦相似度的对象


def exists(val):
    return val is not None  # 判断值是否为 None


def all_between(values: list, lower_bound, upper_bound):
    for value in values:
        if value < lower_bound or value > upper_bound:
            return False

    return True


def make_model(
    prior_config: DiffusionPriorConfig,
    train_config: DiffusionPriorTrainConfig,
    device: str = None,
    accelerator: Accelerator = None,
):
    # 根据配置创建模型
    diffusion_prior = prior_config.create()

    # 实例化训练器
    trainer = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=train_config.lr,
        wd=train_config.wd,
        max_grad_norm=train_config.max_grad_norm,
        amp=train_config.amp,
        use_ema=train_config.use_ema,
        device=device,
        accelerator=accelerator,
        warmup_steps=train_config.warmup_steps,
    )

    return trainer


def create_tracker(
    accelerator: Accelerator,
    config: TrainDiffusionPriorConfig,
    config_path: str,
    dummy: bool = False,
) -> Tracker:
    tracker_config = config.tracker

    accelerator_config = {
        "Distributed": accelerator.distributed_type
        != accelerate_dataclasses.DistributedType.NO,
        "DistributedType": accelerator.distributed_type,
        "NumProcesses": accelerator.num_processes,
        "MixedPrecision": accelerator.mixed_precision,
    }

    tracker: Tracker = tracker_config.create(
        config, accelerator_config, dummy_mode=dummy
    )

    tracker.save_config(config_path, config_name="prior_config.json")

    return tracker


def pad_gather_reduce(trainer: DiffusionPriorTrainer, x, method="mean"):
    """
    pad a value or tensor across all processes and gather

    params:
        - trainer: a trainer that carries an accelerator object
        - x: a number or torch tensor to reduce
        - method: "mean", "sum", "max", "min"

    return:
        - the average tensor after maskin out 0's
        - None if the gather resulted in an empty tensor
    """

    assert method in [
        "mean",
        "sum",
        "max",
        "min",
    ], "This function has limited capabilities [sum, mean, max, min]"
    assert type(x) is not None, "Cannot reduce a None type object"

    # 等待所有进���到达此处后再进��聚合

    if type(x) is not torch.Tensor:
        x = torch.tensor([x])

    # 确保张量在正确的设备上
    x = x.to(trainer.device)

    # 跨进程填充
    padded_x = trainer.accelerator.pad_across_processes(x, dim=0)

    # 聚合所有进程
    gathered_x = trainer.accelerator.gather(padded_x)

    # 掩码掉零值
    masked_x = gathered_x[gathered_x != 0]

    # 如果张量为空，则警告并返回 None
    if len(masked_x) == 0:
        click.secho(
            f"The call to this method resulted in an empty tensor after masking out zeros. The gathered tensor was this: {gathered_x} and the original value passed was: {x}.",
            fg="red",
        )
        return None

    if method == "mean":
        return torch.mean(masked_x)
    elif method == "sum":
        return torch.sum(masked_x)
    elif method == "max":
        return torch.max(masked_x)
    elif method == "min":
        return torch.min(masked_x)


def save_trainer(
    tracker: Tracker,
    # 定义一个名为trainer的变量，类型为DiffusionPriorTrainer
    trainer: DiffusionPriorTrainer,
    # 定义一个名为is_latest的变量，类型为bool，表示是否为最新的
    is_latest: bool,
    # 定义一个名为is_best的变量，类型为bool，表示是否为最佳的
    is_best: bool,
    # 定义一个名为epoch的变量，类型为int，表示当前的训练轮数
    epoch: int,
    # 定义一个名为samples_seen的变量，类型为int，表示已经处理的样本数量
    samples_seen: int,
    # 定义一个名为best_validation_loss的变量，类型为float，表示最佳验证损失值
    best_validation_loss: float,
# 记录模型的状态，根据追踪器选择适当的方法
def log_model(tracker: Tracker, trainer: DiffusionPriorTrainer, is_best: bool, is_latest: bool, epoch: int, samples_seen: int, best_validation_loss: float):
    # 等待所有进程完成
    trainer.accelerator.wait_for_everyone()

    # 如果是主进程
    if trainer.accelerator.is_main_process:
        # 打印保存模型的信息，包括最佳和最新状态
        click.secho(
            f"RANK:{trainer.accelerator.process_index} | Saving Model | Best={is_best} | Latest={is_latest}",
            fg="magenta",
        )

    # 保存模型
    tracker.save(
        trainer=trainer,
        is_best=is_best,
        is_latest=is_latest,
        epoch=int(epoch),
        samples_seen=int(samples_seen),
        best_validation_loss=best_validation_loss,
    )


# 恢复训练器状态
def recall_trainer(tracker: Tracker, trainer: DiffusionPriorTrainer):
    # 如果是主进程
    if trainer.accelerator.is_main_process:
        # 打印加载模型的信息
        click.secho(f"Loading model from {type(tracker.loader).__name__}", fg="yellow")

    # 从追踪器中恢复模型状态
    state_dict = tracker.recall()

    # 加载模型状态到训练器
    trainer.load(state_dict, strict=True)

    return (
        int(state_dict.get("epoch", 0)),
        state_dict.get("best_validation_loss", 0),
        int(state_dict.get("samples_seen", 0)),
    )


# 评估函数

# 报告验证集上的损失
def report_validation_loss(trainer: DiffusionPriorTrainer, dataloader: DataLoader, text_conditioned: bool, use_ema: bool, tracker: Tracker, split: str, tracker_folder: str, loss_type: str):
    # 如果是主进程
    if trainer.accelerator.is_main_process:
        # 打印评估性能的信息
        click.secho(
            f"Measuring performance on {use_ema}-{split} split",
            fg="green",
            blink=True,
        )

    # 初始化总损失
    total_loss = torch.zeros(1, dtype=torch.float, device=trainer.device)

    # 遍历数据加载器中的数据
    for image_embeddings, text_data in dataloader:
        image_embeddings = image_embeddings.to(trainer.device)
        text_data = text_data.to(trainer.device)

        input_args = dict(image_embed=image_embeddings)

        if text_conditioned:
            input_args = dict(**input_args, text=text_data)
        else:
            input_args = dict(**input_args, text_embed=text_data)

        if use_ema:
            loss = trainer.ema_diffusion_prior(**input_args)
        else:
            loss = trainer(**input_args)

        total_loss += loss

    # 计算所有进程的平均损失
    avg_loss = pad_gather_reduce(trainer, total_loss, method="mean")
    stats = {f"{tracker_folder}/{loss_type}-loss": avg_loss}

    # 打印和记录结果到主进程
    tracker.log(stats, step=trainer.step.item() + 1)

    return avg_loss


# 报告余弦相似度
def report_cosine_sims(trainer: DiffusionPriorTrainer, dataloader: DataLoader, text_conditioned: bool, tracker: Tracker, split: str, timesteps: int, tracker_folder: str):
    # 设置为评估模式
    trainer.eval()
    # 如果是主进程
    if trainer.accelerator.is_main_process:
        # 打印余弦相似度的信息
        click.secho(
            f"Measuring Cosine-Similarity on {split} split with {timesteps} timesteps",
            fg="green",
            blink=True,
        )
    # 遍历数据加载器，获取测试图像嵌入和文本数据
    for test_image_embeddings, text_data in dataloader:
        # 将测试图像嵌入和文本数据移动到训练器所在的设备上
        test_image_embeddings = test_image_embeddings.to(trainer.device)
        text_data = text_data.to(trainer.device)

        # 如果是文本条件下，从标记化文本中生成嵌入
        if text_conditioned:
            text_embedding, text_encodings = trainer.embed_text(text_data)
            text_cond = dict(text_embed=text_embedding, text_encodings=text_encodings)
        else:
            text_embedding = text_data
            text_cond = dict(text_embed=text_embedding)

        # 复制文本嵌入以进行混洗
        text_embed_shuffled = text_embedding.clone()

        # 滚动文本以模拟“不相关”的标题
        rolled_idx = torch.roll(torch.arange(text_embedding.shape[0]), 1)
        text_embed_shuffled = text_embed_shuffled[rolled_idx]
        text_embed_shuffled = text_embed_shuffled / text_embed_shuffled.norm(
            dim=1, keepdim=True
        )

        if text_conditioned:
            text_encodings_shuffled = text_encodings[rolled_idx]
        else:
            text_encodings_shuffled = None

        text_cond_shuffled = dict(
            text_embed=text_embed_shuffled, text_encodings=text_encodings_shuffled
        )

        # 准备文本嵌入
        text_embed = text_embedding / text_embedding.norm(dim=1, keepdim=True)

        # 准备图像嵌入
        test_image_embeddings = test_image_embeddings / test_image_embeddings.norm(
            dim=1, keepdim=True
        )

        # 在未混洗的文本嵌入上进行预测
        predicted_image_embeddings = trainer.p_sample_loop(
            test_image_embeddings.shape,
            text_cond,
            timesteps=timesteps,
        )

        predicted_image_embeddings = (
            predicted_image_embeddings
            / predicted_image_embeddings.norm(dim=1, keepdim=True)
        )

        # 在混洗的嵌入上进行预测
        predicted_unrelated_embeddings = trainer.p_sample_loop(
            test_image_embeddings.shape,
            text_cond_shuffled,
            timesteps=timesteps,
        )

        predicted_unrelated_embeddings = (
            predicted_unrelated_embeddings
            / predicted_unrelated_embeddings.norm(dim=1, keepdim=True)
        )

        # 计算相似度
        orig_sim = pad_gather_reduce(
            trainer, cos(text_embed, test_image_embeddings), method="mean"
        )
        pred_sim = pad_gather_reduce(
            trainer, cos(text_embed, predicted_image_embeddings), method="mean"
        )
        unrel_sim = pad_gather_reduce(
            trainer, cos(text_embed, predicted_unrelated_embeddings), method="mean"
        )
        pred_img_sim = pad_gather_reduce(
            trainer,
            cos(test_image_embeddings, predicted_image_embeddings),
            method="mean",
        )

        # 统计结果
        stats = {
            f"{tracker_folder}/baseline similarity [steps={timesteps}]": orig_sim,
            f"{tracker_folder}/similarity with text [steps={timesteps}]": pred_sim,
            f"{tracker_folder}/similarity with original image [steps={timesteps}]": pred_img_sim,
            f"{tracker_folder}/similarity with unrelated caption [steps={timesteps}]": unrel_sim,
            f"{tracker_folder}/difference from baseline similarity [steps={timesteps}]": pred_sim
            - orig_sim,
        }

        # 记录统计结果
        tracker.log(stats, step=trainer.step.item() + 1)
# 定义评估模型的函数，用于在模型上运行评估并跟踪指标，返回损失（如果请求）
def eval_model(
    trainer: DiffusionPriorTrainer,  # 训练器对象
    dataloader: DataLoader,  # 数据加载器对象
    text_conditioned: bool,  # 是否基于文本条件
    split: str,  # 数据集划分
    tracker: Tracker,  # 追踪器对象
    use_ema: bool,  # 是否使用指数移动平均
    report_cosine: bool,  # 是否报告余弦相似度
    report_loss: bool,  # 是否报告损失
    timesteps: List[int],  # 时间步列表
    loss_type: str = None,  # 损失类型，默认为None
):
    """
    Run evaluation on a model and track metrics

    returns: loss if requested
    """
    # 将模型设置为评估模式
    trainer.eval()

    # 根据是否使用指数移动平均设置使用的模式
    use_ema = "ema" if use_ema else "online"
    # 设置追踪器文件夹路径
    tracker_folder = f"metrics/{use_ema}-{split}"

    # 检查传入的时间步是否有效
    min_timesteps = trainer.accelerator.unwrap_model(
        trainer.diffusion_prior
    ).sample_timesteps
    max_timesteps = trainer.accelerator.unwrap_model(
        trainer.diffusion_prior
    ).noise_scheduler.num_timesteps

    assert all_between(
        timesteps, lower_bound=min_timesteps, upper_bound=max_timesteps
    ), f"all timesteps values must be between {min_timesteps} and {max_timesteps}: got {timesteps}"

    # 如果需要报告余弦相似度，则在不同的eta和时间步上测量余弦相似度指标
    if report_cosine:
        for timestep in timesteps:
            report_cosine_sims(
                trainer,
                dataloader=dataloader,
                text_conditioned=text_conditioned,
                tracker=tracker,
                split=split,
                timesteps=timestep,
                tracker_folder=tracker_folder,
            )

    # 如果需要报告损失，则在数据的另一个划分上测量损失
    if report_loss:
        # 报告验证集上的损失
        loss = report_validation_loss(
            trainer=trainer,
            dataloader=dataloader,
            text_conditioned=text_conditioned,
            use_ema=use_ema,
            tracker=tracker,
            split=split,
            tracker_folder=tracker_folder,
            loss_type=loss_type,
        )

        return loss


# 训练脚本

# 定义训练函数
def train(
    trainer: DiffusionPriorTrainer,  # 训练器对象
    tracker: Tracker,  # 追踪器对象
    train_loader: DataLoader,  # 训练数据加载器对象
    eval_loader: DataLoader,  # 评估数据加载器对象
    test_loader: DataLoader,  # 测试数据加载器对象
    config: DiffusionPriorTrainConfig,  # 训练配置对象
):
    # 初始化计时器
    save_timer = Timer()  # 保存计时器
    samples_timer = Timer()  # 样本速率计时器
    validation_profiler = Timer()  # 验证时间计时器
    validation_countdown = Timer()  # 验证倒计时计时器

    # 跟踪最佳验证损失
    best_validation_loss = config.train.best_validation_loss
    samples_seen = config.train.num_samples_seen

    # 开始训练
    start_epoch = config.train.current_epoch

    # 在测试数据上进行评估
    if trainer.accelerator.is_main_process:
        click.secho(f"Starting Test", fg="red")

    # 在开始验证之前最后保存一次最新模型
    save_trainer(
        tracker=tracker,
        trainer=trainer,
        is_best=False,
        is_latest=True,
        samples_seen=samples_seen,
        epoch=epoch,
        best_validation_loss=best_validation_loss,
    )

    # 在测试数据上评估模型
    test_loss = eval_model(
        trainer=trainer,
        dataloader=test_loader,
        text_conditioned=config.prior.condition_on_text_encodings,
        split="test",
        tracker=tracker,
        use_ema=True,
        report_cosine=False,
        report_loss=True,
        timesteps=config.train.eval_timesteps,
        loss_type=config.prior.loss_type,
    )

    # 如果测试损失小于最佳验证损失，则更新最佳验证损失并保存模型
    if test_loss < best_validation_loss:
        best_validation_loss = test_loss

        # 保存最佳模型
        save_trainer(
            trainer=trainer,
            tracker=tracker,
            is_best=True,
            is_latest=False,
            samples_seen=samples_seen,
            epoch=epoch,
            best_validation_loss=test_loss,
        )


# 初始化训练
def initialize_training(config_file, accelerator):
    """
    Parse the configuration file, and prepare everything necessary for training
    """
    # 加载配置文件
    if accelerator.is_main_process:
        click.secho(f"Loading configuration from {config_file}", fg="green")

    # 从JSON路径加载训练配置
    config = TrainDiffusionPriorConfig.from_json_path(config_file)
    # 设置随机种子

    set_seed(config.train.random_seed)

    # 获取设备

    device = accelerator.device

    # 创建训练器（如果可能且已配置，将自动分发）

    trainer: DiffusionPriorTrainer = make_model(
        config.prior, config.train, device, accelerator
    ).to(device)

    # 创建一个追踪器

    tracker = create_tracker(
        accelerator, config, config_file, dummy=accelerator.process_index != 0
    )

    # 从检查点重新加载

    if tracker.can_recall:
        current_epoch, best_validation_loss, samples_seen = recall_trainer(
            tracker=tracker, trainer=trainer
        )

        # 显示最佳值
        if trainer.accelerator.is_main_process:
            click.secho(f"Current Epoch: {current_epoch} | Best Val Loss: {best_validation_loss} | Samples Seen: {samples_seen}", fg="yellow")

        # 更新配置以反映已召回的值
        config.train.num_samples_seen = samples_seen
        config.train.current_epoch = current_epoch
        config.train.best_validation_loss = best_validation_loss

    # 获取并准备数据

    if trainer.accelerator.is_main_process:
        click.secho("Grabbing data...", fg="blue", blink=True)

    trainer.accelerator.wait_for_everyone()
    img_reader = get_reader(
        text_conditioned=trainer.text_conditioned,
        img_url=config.data.image_url,
        meta_url=config.data.meta_url,
    )

    # 计算在 epoch 中的起始点

    trainer.accelerator.wait_for_everyone()

    train_loader, eval_loader, test_loader = make_splits(
        text_conditioned=trainer.text_conditioned,
        batch_size=config.data.batch_size,
        num_data_points=config.data.num_data_points,
        train_split=config.data.splits.train,
        eval_split=config.data.splits.val,
        image_reader=img_reader,
        rank=accelerator.state.process_index,
        world_size=accelerator.state.num_processes,
        start=0,
    )

    # 更新起始点以完成在恢复运行时的 epoch

    if tracker.can_recall:
        samples_seen = config.train.num_samples_seen
        length = (
            config.data.num_data_points
            if samples_seen <= img_reader.count
            else img_reader.count
        )
        scaled_samples = length * config.train.current_epoch
        start_point = (
            scaled_samples - samples_seen if scaled_samples > samples_seen else samples_seen
        )

        if trainer.accelerator.is_main_process:
            click.secho(f"Resuming at sample: {start_point}", fg="yellow")

        train_loader.dataset.set_start(start_point)

    # 开始训练

    if trainer.accelerator.is_main_process:
        click.secho(
            f"Beginning Prior Training : Distributed={accelerator.state.distributed_type != accelerate_dataclasses.DistributedType.NO}",
            fg="yellow",
        )

    train(
        trainer=trainer,
        tracker=tracker,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_loader=test_loader,
        config=config,
    )
# 创建一个命令行接口
@click.command()
# 添加一个命令行选项，指定配置文件，默认为"configs/train_prior_config.example.json"
@click.option("--config_file", default="configs/train_prior_config.example.json")
def main(config_file):
    # 初始化加速器对象
    accelerator = Accelerator()

    # 设置训练环境
    initialize_training(config_file, accelerator)


# 如果当前脚本被直接执行，则调用main函数
if __name__ == "__main__":
    main()
```