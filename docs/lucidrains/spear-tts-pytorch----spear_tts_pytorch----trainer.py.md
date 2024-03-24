# `.\lucidrains\spear-tts-pytorch\spear_tts_pytorch\trainer.py`

```
# 导入必要的库
import re
from pathlib import Path
from shutil import rmtree

# 导入 beartype 库中的函数和类型
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Union, Optional, Tuple

# 导入 PyTorch 库
import torch
from torch import nn, LongTensor, IntTensor
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, random_split

# 导入 audiolm_pytorch 库中的模型和函数
from audiolm_pytorch import FairseqVQWav2Vec, HubertWithKmeans
from audiolm_pytorch.data import get_dataloader
from audiolm_pytorch.optimizer import get_optimizer

# 导入 spear_tts_pytorch 库中的模型和数据集
from spear_tts_pytorch.spear_tts_pytorch import SpeechSpeechPretrainWrapper, TextToSemantic, SemanticToTextWrapper, TextToSemanticWrapper
from spear_tts_pytorch.data import GeneratedAudioTextDataset

# 导入 accelerate 库中的加速器和分布式类型
from accelerate import Accelerator, DistributedType

# 定义类型别名
IndicesTensor = Union[LongTensor, IntTensor]

# 确保只有一个 Trainer 实例化
ONE_TRAINER_INSTANTIATED = False

def check_one_trainer():
    global ONE_TRAINER_INSTANTIATED
    assert not ONE_TRAINER_INSTANTIATED, 'only one Trainer can be instantiated at a time for training'
    ONE_TRAINER_INSTANTIATED = True

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 空操作函数
def noop(*args, **kwargs):
    pass

# 无限循环生成数据集
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 将输入转换为元组
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# 询问用户是或否
def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

# 累积日志信息
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# 从检查点文件名中获取训练步数
def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/speech.speech.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path)

    if len(results) == 0:
        return 0

    return int(results[-1])

# 定义 SpeechSpeechPretrainer 类
class SpeechSpeechPretrainer(nn.Module):
    @beartype
    def __init__(
        self,
        model: TextToSemantic,
        wav2vec: Optional[Union[FairseqVQWav2Vec, HubertWithKmeans]],
        *,
        num_train_steps,
        num_warmup_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        deletion_prob: float = 0.6,
        reconstruct_seq: bool = False,
        mask_id = None,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        log_every = 10,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None
        ):
        # 调用父类的构造函数
        super().__init__()
        # 检查是否只有一个训练器
        check_one_trainer()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            **accelerate_kwargs
        )

        # 设置模型和wav2vec
        self.model = model
        self.wav2vec = wav2vec

        # 初始化训练包装器
        self.train_wrapper = SpeechSpeechPretrainWrapper(
            model = model,
            wav2vec = wav2vec,
            deletion_prob = deletion_prob,
            reconstruct_seq = reconstruct_seq,
            mask_id = mask_id
        )

        # 注册缓冲区
        self.register_buffer('steps', torch.Tensor([0]))

        # 设置训练步数、热身步数、批量大小、梯度累积频率
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 优化器
        self.lr = lr
        self.initial_lr = initial_lr
        self.optim = get_optimizer(model.parameters(), lr = lr, wd = wd)
        self.scheduler = CosineAnnealingLR(self.optim, T_max = num_train_steps)

        # 最大梯度范数
        self.max_grad_norm = max_grad_norm

        # 创建数据集
        self.ds = dataset

        # 划分验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # 断言确保数据集和验证集的样本数足够
        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # 数据加载器
        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)
        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # 使用加速器准备训练所需的对象
        (
            self.train_wrapper,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        )

        # 数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        # 设置日志、保存模型和保存结果的频率
        self.log_every = log_every
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # 设置结果文件夹路径
        self.results_folder = Path(results_folder)

        # 如果是主进程且需要清除之前的结果，则清除结果文件夹
        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        # 创建结果文件夹
        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        # 初始化超参数跟踪器
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": num_warmup_steps, "learning_rate": lr, "initial_learning_rate": lr}
        self.accelerator.init_trackers("speechspeech", config=hps)

    # 保存模型
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        torch.save(pkg, path)
    # 加载模型参数和优化器状态
    def load(self, path):
        # 获取未封装的模型
        model = self.accelerator.unwrap_model(self.model)
        # 加载模型
        pkg = model.load(path)

        # 加载优化器状态
        self.optim.load_state_dict(pkg['optim'])
        # 加载调度器状态
        self.scheduler.load_state_dict(pkg['scheduler'])

        # 从下一个步骤开始，避免覆盖最后一个检查点
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 生成结果
    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    # 获取设备
    @property
    def device(self):
        return self.accelerator.device

    # 判断是否分布式训练
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 判断是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 判断是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 热身训练
    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr
    
    # 训练步骤
    def train_step(self):
        steps = int(self.steps.item())

        self.model.train()
        
        # 根据调度器调整学习率
        
        if steps < self.num_warmup_steps:
            # 应用热身训练
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            # 热身训练后，开始应用余弦退火学习率调度器
            self.scheduler.step()

        # 日志

        logs = {}

        # 更新 VAE（生成器）

        for _ in range(self.grad_accum_every):
            x, = next(self.dl_iter)

            loss, _ = self.train_wrapper(x)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # 日志

        if not (steps % self.log_every):
            self.print(f"{steps}: loss: {logs['loss']:0.3f}")

        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 定期采样结果

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            x, = next(self.valid_dl_iter)

            with torch.inference_mode():
                self.train_wrapper.eval()
                valid_loss, _ = self.train_wrapper(x)

            self.print(f'{steps}: valid loss {valid_loss:0.3f}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 定期保存模型

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'speech.speech.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    # 训练模型
    def train(self, log_fn = noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
# 定义一个用于将语义转换为文本的训练器类
class SemanticToTextTrainer(nn.Module):
    # 初始化方法，接受多个参数
    @beartype
    def __init__(
        self,
        model: TextToSemantic,  # 模型参数，用于将文本转换为语义
        *,
        num_train_steps,  # 训练步数
        num_warmup_steps,  # 热身步数
        batch_size,  # 批量大小
        dataset: Optional[Dataset] = None,  # 数据集，默认为None
        lr = 3e-4,  # 学习率，默认为3e-4
        initial_lr = 1e-5,  # 初始学习率，默认为1e-5
        grad_accum_every = 1,  # 梯度累积频率，默认为1
        wd = 0.,  # 权重衰减，默认为0
        max_grad_norm = 0.5,  # 最大梯度范数，默认为0.5
        valid_frac = 0.05,  # 验证集比例，默认为0.05
        random_split_seed = 42,  # 随机拆分种子，默认为42
        log_every = 10,  # 每隔多少步记录日志，默认为10
        save_results_every = 100,  # 每隔多少步保存结果，默认为100
        save_model_every = 1000,  # 每隔多少步保存模型，默认为1000
        results_folder = './results',  # 结果保存文件夹，默认为'./results'
        accelerate_kwargs: dict = dict(),  # 加速参数，默认为空字典
        split_batches = False,  # 是否拆分批次，默认为False
        drop_last = False,  # 是否丢弃最后一批数据，默认为False
        force_clear_prev_results = None  # 强制清除之前的结果，默认为None
        ):
        # 调用父类的构造函数
        super().__init__()
        # 检查是否只有一个训练器
        check_one_trainer()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            **accelerate_kwargs
        )

        # 设置模型
        self.model = model

        # 创建训练包装器
        self.train_wrapper = SemanticToTextWrapper(model = model)

        # 注册缓冲区
        self.register_buffer('steps', torch.Tensor([0]))

        # 设置训练步数、预热步数、批量大小、梯度累积频率
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 在进行反向翻译时，冻结编码器和语音嵌入
        model.unfreeze_all()
        model.freeze_speech_emb()
        model.freeze_encoder()

        # 优化器
        # get_optimizer应该过滤掉冻结的参数（requires_grad设置为False的参数）
        self.optim = get_optimizer(
            model.parameters(),
            lr = lr,
            wd = wd,
            filter_by_requires_grad = True
        )

        self.lr = lr
        self.initial_lr = initial_lr
        self.scheduler = CosineAnnealingLR(self.optim, T_max = num_train_steps)

        # 最大梯度范数
        self.max_grad_norm = max_grad_norm

        # 创建数据集
        self.ds = dataset

        # 划分验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # 数据加载器
        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # 使用加速器准备
        (
            self.train_wrapper,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.scheduler,
            self.dl,
            self.valid_dl
        )

        # 数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.log_every = log_every
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        # 如果是主进程并且强制清除之前的结果或者（force_clear_prev_results不存在且结果文件夹中有文件且用户确认清除）
        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        # 创建结果文件夹
        self.results_folder.mkdir(parents = True, exist_ok = True)
        
        # 初始化超参数跟踪器
        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": num_warmup_steps, "learning_rate": lr, "initial_learning_rate": lr}
        self.accelerator.init_trackers("semantictext", config=hps)

    # 保存模型
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        torch.save(pkg, path)
    # 加载模型参数和优化器状态
    def load(self, path, restore_optimizer = True):
        # 获取未封装的模型对象
        model = self.accelerator.unwrap_model(self.model)
        # 加载模型参数
        pkg = model.load(path)

        # 如果需要恢复优化器状态
        if restore_optimizer:
            # 加载优化器状态
            self.optim.load_state_dict(pkg['optim'])
            # 加载学习率调度器状态
            self.scheduler.load_state_dict(pkg['scheduler'])

            # 从下一个步骤开始，避免覆盖最后一个检查点
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 生成结果
    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    # 获取设备
    @property
    def device(self):
        return self.accelerator.device

    # 判断是否分布式训练
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 判断是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 判断是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 热身训练
    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr
    
    # 训练步骤
    def train_step(self):
        steps = int(self.steps.item())

        # 设置模型为训练模式
        self.model.train()
        
        # 根据调度器调整学习率

        if steps < self.num_warmup_steps:
            # 应用热身训练
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            # 热身训练后，开始应用余弦退火学习率调度器
            self.scheduler.step()

        # 日志

        logs = {}

        # 更新 VAE（生成器）

        for _ in range(self.grad_accum_every):
            semantic_token_ids, grapheme_token_ids = next(self.dl_iter)

            loss, _ = self.train_wrapper(semantic_token_ids = semantic_token_ids, grapheme_token_ids = grapheme_token_ids)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 如果存在最大梯度范数，则进行梯度裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # 记录日志

        if not (steps % self.log_every):
            self.print(f"{steps}: loss: {logs['loss']:0.3f}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 定期采样结果

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            semantic_token_ids, grapheme_token_ids = next(self.valid_dl_iter)

            with torch.inference_mode():
                self.train_wrapper.eval()
                valid_loss, _ = self.train_wrapper(semantic_token_ids = semantic_token_ids, grapheme_token_ids = grapheme_token_ids)

            self.print(f'{steps}: valid loss {valid_loss:0.3f}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 定期保存模型

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'semantic.text.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    # 训练模型
    def train(self, log_fn = noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
# 定义一个用于训练文本到语义模型的类
class TextToSemanticTrainer(nn.Module):
    # 初始化函数，接受模型、训练步数、预热步数等参数
    @beartype
    def __init__(
        self,
        model: TextToSemantic,
        *,
        num_train_steps,
        num_warmup_steps,
        batch_size,
        dataset: Optional[Dataset] = None,
        generated_audio_text_dataset_folder = None,
        dataset_delimiter_id = -1,
        lr = 3e-4,
        initial_lr = 1e-5,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        log_every = 10,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None,
        freeze_encoder_layers_below = 2,
        should_train_early_exit_layer_if_available = True
    # 保存模型参数到指定路径
    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.model),
            optim = self.optim.state_dict(),
            scheduler = self.scheduler.state_dict()
        )
        torch.save(pkg, path)

    # 从指定路径加载模型参数，可选择是否还原优化器状态
    def load(self, path, restore_optimizer = True):
        model = self.accelerator.unwrap_model(self.model)
        pkg = model.load(path)

        if restore_optimizer:
            self.optim.load_state_dict(pkg['optim'])
            self.scheduler.load_state_dict(pkg['scheduler'])

            # + 1 to start from the next step and avoid overwriting the last checkpoint
            self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 生成结果
    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    # 返回设备信息
    @property
    def device(self):
        return self.accelerator.device

    # 判断是否为分布式训练
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 判断是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 判断是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 根据当前步数计算学习率
    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr
    # 定义训练步骤函数
    def train_step(self):
        # 获取当前步数
        steps = int(self.steps.item())

        # 设置模型为训练模式
        self.model.train()
        
        # 根据训练步数调整学习率
        
        if steps < self.num_warmup_steps:
            # 如果步数小于预热步数，应用预热
            lr = self.warmup(steps)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
        else:
            # 预热期后，开始应用余弦退火学习率调度器
            self.scheduler.step()

        # 日志

        logs = {}

        # 更新 VAE（生成器）

        for _ in range(self.grad_accum_every):
            semantic_token_ids, grapheme_token_ids = next(self.dl_iter)

            # 计算损失并进行训练
            loss, _ = self.train_wrapper(semantic_token_ids=semantic_token_ids, grapheme_token_ids=grapheme_token_ids, return_early_exit_loss=self.train_early_exit)

            self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 如果存在最大梯度范数，对梯度进行裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 更新优化器
        self.optim.step()
        self.optim.zero_grad()

        # 记录日志

        if not (steps % self.log_every):
            self.print(f"{steps}: loss: {logs['loss']:0.3f}")
        
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # 定期采样结果

        self.accelerator.wait_for_everyone()

        if self.is_main and not (steps % self.save_results_every):
            semantic_token_ids, grapheme_token_ids = next(self.valid_dl_iter)

            with torch.inference_mode():
                self.train_wrapper.eval()
                valid_loss, _ = self.train_wrapper(semantic_token_ids=semantic_token_ids, grapheme_token_ids=grapheme_token_ids, return_early_exit_loss=self.train_early_exit)

            self.print(f'{steps}: valid loss {valid_loss:0.3f}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # 定期保存模型

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'text.semantic.{steps}.pt')
            self.save(model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        # 更新步数并返回日志
        self.steps += 1
        return logs

    # 训练函数
    def train(self, log_fn=noop):
        # 在未达到训练步数前循环执行训练步骤
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')
```