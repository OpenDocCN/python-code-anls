# `.\lucidrains\phenaki-pytorch\phenaki_pytorch\phenaki_trainer.py`

```
# 导入数学库
import math
# 导入复制库
import copy
# 导入路径库
from pathlib import Path
# 导入随机库
from random import random, choices
# 导入偏函数库
from functools import partial
# 导入命名元组库
from collections import namedtuple
# 导入 CPU 核心数库
from multiprocessing import cpu_count

# 导入 beartype 库
from beartype import beartype
# 导入 beartype.door 库
from beartype.door import is_bearable
# 导入 beartype.vale 库
from beartype.vale import Is
# 导入类型提示库
from typing import Optional, List, Iterable, Tuple
# 导入类型扩展库
from typing_extensions import Annotated

# 导入 PyTorch 库
import torch
# 从 PyTorch 中导入神经网络库和张量乘法库
from torch import nn, einsum
# 从 PyTorch 中导入函数库
import torch.nn.functional as F
# 从 PyTorch 中导入数据集库
from torch.utils.data import Dataset
# 从 PyTorch 中导入优化器库
from torch.optim import Adam

# 从 torchvision 中导入变换库
from torchvision import transforms as T
# 从 torchvision 中导入图像处理库
from torchvision.utils import make_grid, save_image

# 从 einops 中导入重排库和减少库
from einops import rearrange, reduce
# 从 einops.layers.torch 中导入重排层
from einops.layers.torch import Rearrange

# 从 PIL 中导入图像库
from PIL import Image
# 从 tqdm.auto 中导入进度条库
from tqdm.auto import tqdm

# 从 phenaki_pytorch.optimizer 中导入获取优化器函数
from phenaki_pytorch.optimizer import get_optimizer
# 从 accelerate 中导入加速器库
from accelerate import Accelerator

# 从 phenaki_pytorch.phenaki_pytorch 中导入 Phenaki 类
from phenaki_pytorch.phenaki_pytorch import Phenaki

# 从 phenaki_pytorch.data 中导入图像数据集、视频数据集、视频张量转 GIF、数据加载器
from phenaki_pytorch.data import ImageDataset, VideoDataset, video_tensor_to_gif, DataLoader

# 常量

# 数据集字段类型配置
DATASET_FIELD_TYPE_CONFIG = dict(
    videos = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {4, 5}]
    ],
    texts = List[str],
    video_codebook_ids = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.long]
    ],
    video_frame_mask = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.bool]
    ],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

# 辅助函数

# 检查变量是否存在
def exists(x):
    return x is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 无限循环生成数据
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 检查整数是否有平方根
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# 将数字分组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 将元素转移到指定设备
def elements_to_device_if_tensor(arr, device):
    output = []
    for el in arr:
        if isinstance(el, torch.Tensor):
            el = el.to(device)
        output.append(el)
    return output

# 分割可迭代对象
def split_iterable(it, split_size):
    accum = []
    for ind in range(math.ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

# 分割数据
def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

# 查找第一个符合条件的元素
def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

# 分割参数和关键字参数
def split_args_and_kwargs(*args, batch_size = None, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)

    if not exists(batch_size):
        first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
        assert exists(first_tensor)
        batch_size = len(first_tensor)

    split_size = default(split_size, batch_size)
    num_chunks = math.ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))
    # 遍历元组中的每个元素，元素包含一个 chunk_size 和对应的参数列表
    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        # 将参数列表拆分为位置参数和关键字参数值
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        # 将关键字参数的键和值组成字典
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        # 计算当前 chunk 的大小占总 batch 大小的比例
        chunk_size_frac = chunk_size / batch_size
        # 生成当前 chunk 的比例和参数元组
        yield chunk_size_frac, (chunked_args, chunked_kwargs)
# 简单的文本转换函数，将特定字符替换为指定字符，去除空格和特殊字符，并截取指定长度
def simple_slugify(text, max_length = 255):
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_')[:max_length]

# 检查元组中是否存在重复元素
def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))

# 根据配置确定数据的类型
def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

# 训练器类
@beartype
class PhenakiTrainer(object):
    def __init__(
        self,
        phenaki: Phenaki,
        *,
        folder = None,
        train_on_images = False,
        batch_size = 16,
        grad_accum_every = 1,
        num_frames = 17,
        sample_num_frames = None,
        train_lr = 1e-4,
        train_num_steps = 100000,
        max_grad_norm = None,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        wd = 0,
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        sample_texts_file_path = None,  # path to a text file with video captions, delimited by newline
        sample_texts: Optional[List[str]] = None,
        dataset: Optional[Dataset] = None,
        dataset_fields: Optional[Tuple[str, ...]] = None
    ):
        # 调用父类的构造函数
        super().__init__()
        # 导入 phenaki 模块中的 maskgit 和 cvivit
        maskgit = phenaki.maskgit
        cvivit = phenaki.cvivit

        # 确保 cvivit 在 phenaki 中存在
        assert exists(cvivit), 'cvivit must be present on phenaki'

        # 定义加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        # 设置加速器的本地自动混合精度
        self.accelerator.native_amp = amp

        # 设置模型为 phenaki
        self.model = phenaki

        # 确保样本数量具有整数平方根
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        # 设置是否无条件生成
        self.unconditional = maskgit.unconditional

        # 训练相关变量
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm
        self.train_num_steps = train_num_steps
        self.image_size = cvivit.image_size

        # 采样相关变量
        self.num_samples = num_samples
        self.sample_texts = None

        # 如果存在采样文本文件路径，则读取文本内容
        if exists(sample_texts_file_path):
            sample_texts_file_path = Path(sample_texts_file_path)
            assert sample_texts_file_path.exists()
            captions = sample_texts_file_path.read_text().split('\n')
            self.sample_texts = list(filter(len, captions))

        # 如果存在采样文本，则设置为采样文本
        elif exists(self.sample_texts):
            self.sample_texts = sample_texts

        # 如果是无条件生成或存在采样文本，则继续，否则报错
        assert maskgit.unconditional or exists(self.sample_texts), 'if maskgit is to be trained text conditioned, `sample_texts` List[str] or `sample_texts_file_path` must be given'

        # 设置保存和采样频率
        self.save_and_sample_every = save_and_sample_every

        # 数据集和数据加载器
        dataset_klass = ImageDataset if train_on_images else VideoDataset
        self.sample_num_frames = default(sample_num_frames, num_frames)
        self.train_on_images = train_on_images

        # 如果存在数据集，则使用该数据集，否则根据训练类型选择数据集
        if dataset:
            self.ds = dataset
        elif train_on_images:
            assert exists(folder)
            self.ds = ImageDataset(folder, self.image_size)
        else:
            assert exists(folder)
            self.ds = VideoDataset(folder, self.image_size, num_frames = num_frames)

        # 创建数据加载器
        dl = DataLoader(self.ds, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # 如果存在数据集字段，则检查字段是否合法
        if exists(dataset_fields):
            assert not has_duplicates(dataset_fields), 'dataset fields must not have duplicate field names'
            valid_dataset_fields = set(DATASET_FIELD_TYPE_CONFIG.keys())
            assert len(set(dataset_fields) - valid_dataset_fields) == 0, f'dataset fields must be one of {valid_dataset_fields}'

        self.dataset_fields = dataset_fields

        # 优化器
        self.opt = get_optimizer(maskgit.parameters(), lr = train_lr, wd = wd, betas = adam_betas)

        # 步数计数器
        self.step = 0

        # 准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # 设置结果文件���
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents = True, exist_ok = True)

    # 将数据元组转换为关键字参数
    def data_tuple_to_kwargs(self, data):
        if not exists(self.dataset_fields):
            self.dataset_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
            assert not has_duplicates(self.dataset_fields), 'dataset fields must not have duplicate field names'

        return dict(zip(self.dataset_fields, data))

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 设备属性
    @property
    def device(self):
        return self.accelerator.device

    # 是否分布式属性
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 是否主进程属性
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 是否本地主进程属性
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process
    # 保存模型的当前状态
    def save(self, milestone):
        # 如果不是本地主进程，则直接返回
        if not self.accelerator.is_local_main_process:
            return

        # 构建保存的数据字典
        data = {
            'step': self.step,  # 保存当前步数
            'model': self.accelerator.get_state_dict(self.model),  # 保存模型的状态字典
            'opt': self.opt.state_dict(),  # 保存优化器的状态字典
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None  # 保存混合精度训练器的状态字典
        }

        # 将数据保存到文件中
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # 加载指定里程碑的模型状态
    def load(self, milestone):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 从文件中加载数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        # 获取模型并加载状态
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        # 加载步数和优化器状态
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        # 如果混合精度训练器存在且数据中也存在，则加载混合精度训练器状态
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练步骤函数
    def train_step(
        self,
        only_train_generator=False,  # 是否只训练生成器
        only_train_critic=False  # 是否只训练评论家
    # 定义 train 方法，用于训练模型
    def train(
        self,
        only_train_generator = False,
        only_train_critic = False
        ):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = self.device

        # 初始化总损失
        total_loss = 0.

        # 循环执行梯度累积
        for _ in range(self.grad_accum_every):
            # 从数据加载器中获取数据
            data = next(self.dl)
            # 将数据转移到指定设备
            data = elements_to_device_if_tensor(data, device)
            # 将数据转换为关键字参数
            data_kwargs = self.data_tuple_to_kwargs(data)

            # 检查是否训练图像，数据维度是否正确
            assert not (self.train_on_images and data_kwargs['videos'].ndim != 4), 'you have it set to train on images, but the dataset is not returning tensors of 4 dimensions (batch, channels, height, width)'

            # 使用混合精度进行训练
            with self.accelerator.autocast():
                # 模型前向传播计算损失
                loss = self.model(**{
                    **data_kwargs,
                    'only_train_generator': only_train_generator,
                    'only_train_critic': only_train_critic
                })

                # 将损失除以梯度累积次数
                loss = loss / self.grad_accum_every
                # 累加总损失
                total_loss += loss.item()

            # 反向传播
            self.accelerator.backward(loss)

        # 如果存在最大梯度范数，则进行梯度裁剪
        if exists(self.max_grad_norm):
            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 等待所有进程完成
        accelerator.wait_for_everyone()

        # 更新优化器参数
        self.opt.step()
        self.opt.zero_grad()

        # 等待所有进程完成
        accelerator.wait_for_everyone()

        # 如果是主进程且满足保存和采样间隔条件
        if self.is_main and self.step % self.save_and_sample_every == 0:
            # 模型转为评估模式
            self.model.eval()
            milestone = self.step // self.save_and_sample_every

            # 是否传入文本
            sample_kwargs = dict()

            if not self.unconditional:
                texts = choices(self.sample_texts, k = self.num_samples)
            else:
                texts = (None,) * self.num_samples

            sample_kwargs = {'texts': texts}

            # 选择采样方法
            if self.train_on_images:
                sample_method = self.model.sample_images
            else:
                sample_method = partial(self.model.sample, num_frames = self.sample_num_frames)

            # 分组评估，适当拆分参数
            with torch.no_grad():
                groups = num_to_groups(self.num_samples, self.batch_size)
                args_kwargs_iter = split_args_and_kwargs(batch_size = self.num_samples, split_size = self.batch_size, **sample_kwargs)

                all_sampled = []
                for group_batch_size, (_, (_, kwargs)) in zip(groups, args_kwargs_iter):
                    _kwargs = kwargs if not self.unconditional else dict()
                    sampled = sample_method(num_frames = self.sample_num_frames, batch_size = group_batch_size, **_kwargs)
                    all_sampled.append(sampled)

            # 保存视频和图像
            if not self.train_on_images:
                sampled_videos = torch.cat(all_sampled, dim = 0)
                milestone_folder = self.results_folder / f'videos.{milestone}'
                milestone_folder.mkdir(parents = True, exist_ok = True)

                for ind, (video_tensor, video_caption) in enumerate(zip(sampled_videos.unbind(dim = 0), texts)):
                    slugged_video_caption = simple_slugify(video_caption) if exists(video_caption) else str(ind)
                    video_tensor_to_gif(video_tensor, str(milestone_folder / f'{slugged_video_caption}.gif'))
            else:
                nrows = int(math.sqrt(self.num_samples))

                sampled_images = sampled_videos.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(sampled_images, nrow = nrows, normalize = True, value_range = (0, 1))

                save_image(grid, str(self.results_folder / f'{milestone}.png'))

            # 保存检查点
            self.save(milestone)

        # 更新步数
        self.step += 1
        return total_loss
    ):  
        # 使用 tqdm 创建一个进度条，设置初始值为 self.step，总步数为 self.train_num_steps，如果不是主进程则禁用
        with tqdm(
            initial = self.step,
            total = self.train_num_steps,
            disable = not self.is_main
        ) as pbar:
            # 当 self.step 小于 self.train_num_steps 时循环
            while self.step < self.train_num_steps:
                # 调用 train_step 方法进行训练，传入参数 only_train_generator 和 only_train_critic
                loss = self.train_step(
                    only_train_generator = only_train_generator,
                    only_train_critic = only_train_critic
                )
                # 设置进度条的描述为当前 loss 值，保留四位小数
                pbar.set_description(f'loss: {loss:.4f}')
                # 更新进度条
                pbar.update(1)
        # 训练完成后打印信息
        self.print('training complete')
```