# `.\lucidrains\muse-maskgit-pytorch\muse_maskgit_pytorch\trainers.py`

```py
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 random 模块中导入 choice 函数
from random import choice
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 shutil 模块中导入 rmtree 函数
from shutil import rmtree
# 从 functools 模块中导入 partial 函数

# 从 beartype 模块中导入 beartype 装饰器
from beartype import beartype

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.optim 模块中导入 Adam 类
from torch.optim import Adam
# 从 torch.utils.data 模块中导入 Dataset, DataLoader, random_split 类
from torch.utils.data import Dataset, DataLoader, random_split

# 从 torchvision.transforms 模块中导入 T 别名
import torchvision.transforms as T
# 从 torchvision.datasets 模块中导入 ImageFolder 类
from torchvision.datasets import ImageFolder
# 从 torchvision.utils 模块中导入 make_grid, save_image 函数

# 从 muse_maskgit_pytorch.vqgan_vae 模块中导入 VQGanVAE 类

# 从 einops 模块中导入 rearrange 函数

# 从 accelerate 模块中导入 Accelerator, DistributedType, DistributedDataParallelKwargs 类

# 从 ema_pytorch 模块中导入 EMA 类

# 从 PIL 模块中导入 Image, ImageFile 类
from PIL import Image, ImageFile
# 设置 ImageFile.LOAD_TRUNCATED_IMAGES 为 True

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 什么也不做
def noop(*args, **kwargs):
    pass

# 查找满足条件的元素的索引
def find_index(arr, cond):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return None

# 查找并弹出满足条件的元素
def find_and_pop(arr, cond, default = None):
    ind = find_index(arr, cond)

    if exists(ind):
        return arr.pop(ind)

    if callable(default):
        return default()

    return default

# 无限循环生成数据
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

# 累积更新日志
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# 将输入转换为元组
def pair(val):
    return val if isinstance(val, tuple) else (val, val)

# 将图像转换为指定格式
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 与图像相关的辅助函数和数据集

# 图像数据集类
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# 主训练器类

# 使用 beartype 装饰器定义 VQGanVAETrainer 类
@beartype
class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae: VQGanVAE,
        *,
        folder,
        num_train_steps,
        batch_size,
        image_size,
        lr = 3e-4,
        grad_accum_every = 1,
        max_grad_norm = None,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        use_ema = True,
        ema_beta = 0.995,
        ema_update_after_step = 0,
        ema_update_every = 1,
        apply_grad_penalty_every = 4,
        accelerate_kwargs: dict = dict()
        ):
        # 调用父类的构造函数
        super().__init__()

        # 实例化加速器
        kwargs_handlers = accelerate_kwargs.get('kwargs_handlers', [])

        # 查找并弹出 DistributedDataParallelKwargs 对象
        ddp_kwargs = find_and_pop(
            kwargs_handlers,
            lambda x: isinstance(x, DistributedDataParallelKwargs),
            partial(DistributedDataParallelKwargs, find_unused_parameters = True)
        )

        # 设置参数 find_unused_parameters 为 True
        ddp_kwargs.find_unused_parameters = True
        kwargs_handlers.append(ddp_kwargs)
        accelerate_kwargs.update(kwargs_handlers = kwargs_handlers)

        # 实例化加速器对象
        self.accelerator = Accelerator(**accelerate_kwargs)

        # 设置 VAE 模型
        self.vae = vae

        # 设置训练参数
        self.register_buffer('steps', torch.Tensor([0]))
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 获取所有参数和判别器参数
        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters
        self.vae_parameters = vae_parameters

        # 设置优化器
        self.optim = Adam(vae_parameters, lr = lr)
        self.discr_optim = Adam(discr_parameters, lr = lr)
        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # 创建数据集
        self.ds = ImageDataset(folder, image_size)

        # 划分验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # 创建数据加载器
        self.dl = DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        )

        # 使用加速器准备模型和数据加载器
        (
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        ) = self.accelerator.prepare(
            self.vae,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl
        )

        # 设置是否使用指数移动平均
        self.use_ema = use_ema

        # 如果使用指数移动平均，创建 EMA 对象并使用加速器准备
        if use_ema:
            self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)
            self.ema_vae = self.accelerator.prepare(self.ema_vae)

        # 创建数据加载器迭代器
        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        # 设置保存模型和结果的频率
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # 设置应用梯度惩罚的频率
        self.apply_grad_penalty_every = apply_grad_penalty_every

        # 设置结果文件夹路径
        self.results_folder = Path(results_folder)

        # 如果结果文件夹不为空，询问是否清除之前的实验检查点和结果
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        # 创建结果文件夹
        self.results_folder.mkdir(parents = True, exist_ok = True)

    # 保存模型
    def save(self, path):
        # 如果不是本地主进程，则返回
        if not self.accelerator.is_local_main_process:
            return

        # 保存模型参数和优化器状态字典
        pkg = dict(
            model = self.accelerator.get_state_dict(self.vae),
            optim = self.optim.state_dict(),
            discr_optim = self.discr_optim.state_dict()
        )
        torch.save(pkg, path)
    # 加载模型参数和优化器状态
    def load(self, path):
        # 将路径转换为Path对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()
        # 加载模型参数
        pkg = torch.load(path)

        # 获取未封装的VAE模型
        vae = self.accelerator.unwrap_model(self.vae)
        # 加载模型参数
        vae.load_state_dict(pkg['model'])

        # 加载优化器状态
        self.optim.load_state_dict(pkg['optim'])
        self.discr_optim.load_state_dict(pkg['discr_optim'])

    # 打印消息
    def print(self, msg):
        self.accelerator.print(msg)

    # 返回设备
    @property
    def device(self):
        return self.accelerator.device

    # 返回是否分布式
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 返回是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 返回是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process
    # 定义训练步骤函数
    def train_step(self):
        # 获取设备信息
        device = self.device

        # 获取当前步数
        steps = int(self.steps.item())
        # 判断是否需要应用梯度惩罚
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        # 设置 VAE 模型为训练模式
        self.vae.train()
        # 获取鉴别器模型
        discr = self.vae.module.discr if self.is_distributed else self.vae.discr
        # 如果使用指数移动平均模型，获取指数移动平均 VAE 模型
        if self.use_ema:
            ema_vae = self.ema_vae.module if self.is_distributed else self.ema_vae

        # 初始化日志字典
        logs = {}

        # 更新 VAE（生成器）

        # 根据梯度累积次数进行更新
        for _ in range(self.grad_accum_every):
            # 获取下一个数据批次
            img = next(self.dl_iter)
            img = img.to(device)

            # 使用自动混合精度计算损失
            with self.accelerator.autocast():
                # 计算 VAE 模型的损失
                loss = self.vae(
                    img,
                    add_gradient_penalty = apply_grad_penalty,
                    return_loss = True
                )

            # 反向传播
            self.accelerator.backward(loss / self.grad_accum_every)

            # 累积损失日志
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 如果存在最大梯度范数，对梯度进行裁剪
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.vae.parameters(), self.max_grad_norm)

        # 更新优化器
        self.optim.step()
        self.optim.zero_grad()

        # 更新鉴别器

        if exists(discr):
            self.discr_optim.zero_grad()

            for _ in range(self.grad_accum_every):
                img = next(self.dl_iter)
                img = img.to(device)

                loss = self.vae(img, return_discr_loss = True)

                self.accelerator.backward(loss / self.grad_accum_every)

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

            # 记录日志
            self.print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        # 更新指数移动平均生成器

        if self.use_ema:
            ema_vae.update()

        # 定期采样结果

        if not (steps % self.save_results_every):
            vaes_to_evaluate = ((self.vae, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((ema_vae.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            for model, filename in vaes_to_evaluate:
                model.eval()

                valid_data = next(self.valid_dl_iter)
                valid_data = valid_data.to(device)

                recons = model(valid_data, return_recons = True)

                # 保存图像网格

                imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                logs['reconstructions'] = grid

                save_image(grid, str(self.results_folder / f'{filename}.png'))

            self.print(f'{steps}: saving to {str(self.results_folder)}')

        # 定期保存模型
        self.accelerator.wait_for_everyone()
        if self.is_main and not (steps % self.save_model_every):
            state_dict = self.accelerator.unwrap_model(self.vae).state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            self.accelerator.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.accelerator.unwrap_model(self.ema_vae).state_dict()
                model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
                self.accelerator.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        # 更新步数并返回日志
        self.steps += 1
        return logs
    # 定义一个训练方法，接受一个日志函数作为参数，默认为一个空操作函数
    def train(self, log_fn = noop):
        # 获取 VAE 模型参数中的设备信息
        device = next(self.vae.parameters()).device

        # 当训练步数小于总训练步数时，执行训练步骤并记录日志
        while self.steps < self.num_train_steps:
            # 执行一次训练步骤，返回日志信息
            logs = self.train_step()
            # 调用日志函数记录日志信息
            log_fn(logs)

        # 打印训练完成信息
        self.print('training complete')
```