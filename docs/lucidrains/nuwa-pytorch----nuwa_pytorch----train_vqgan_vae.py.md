# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\train_vqgan_vae.py`

```py
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 copy 模块中导入 copy 函数
import copy
# 从 random 模块中导入 choice 函数
from random import choice
# 从 pathlib 模块中导入 Path 类
from pathlib import Path
# 从 shutil 模块中导入 rmtree 函数

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 导入 numpy 模块
import numpy as np

# 从 PIL 模块中导入 Image 类
from PIL import Image
# 从 torchvision.datasets 模块中导入 ImageFolder 类
from torchvision.datasets import ImageFolder
# 从 torchvision.transforms 模块中导入 T 别名
import torchvision.transforms as T
# 从 torch.utils.data 模块中导入 Dataset, DataLoader, random_split 类
from torch.utils.data import Dataset, DataLoader, random_split
# 从 torchvision.utils 模块中导入 make_grid, save_image 函数

# 从 einops 模块中导入 rearrange 函数
from einops import rearrange
# 从 nuwa_pytorch.vqgan_vae 模块中导入 VQGanVAE 类
from nuwa_pytorch.vqgan_vae import VQGanVAE
# 从 nuwa_pytorch.optimizer 模块中导入 get_optimizer 函数

# helpers

# 定义 exists 函数，判断值是否存在
def exists(val):
    return val is not None

# 定义 noop 函数，空函数
def noop(*args, **kwargs):
    pass

# 定义 cycle 函数，循环生成数据
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 定义 cast_tuple 函数，将输入转换为元组
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# 定义 yes_or_no 函数，询问用户是否为是或否
def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

# 定义 accum_log 函数，累积日志
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# classes

# 定义 MemmappedImageDataset 类，继承自 Dataset 类
class MemmappedImageDataset(Dataset):
    def __init__(
        self,
        *,
        path,
        shape,
        random_rotate = True
    ):
        super().__init__()
        path = Path(path)
        assert path.exists(), f'path {path} must exist'
        self.memmap = np.memmap(str(path), mode = 'r', dtype = np.uint8, shape = shape)
        self.random_rotate = random_rotate

        image_size = shape[-1]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return self.memmap.shape[0]

    def __getitem__(self, index):
        arr = self.memmap[index]

        if arr.shape[0] == 1:
            arr = rearrange(arr, '1 ... -> ...')

        img = Image.fromarray(arr)
        img = self.transform(img)

        if self.random_rotate:
            img = T.functional.rotate(img, choice([0, 90, 180, 270]))
        return img

# 定义 ImageDataset 类，继承自 Dataset 类
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

# exponential moving average wrapper

# 定义 EMA 类，继承自 nn.Module 类
class EMA(nn.Module):
    def __init__(
        self,
        model,
        beta = 0.99,
        ema_update_after_step = 1000,
        ema_update_every = 10,
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = copy.deepcopy(model)

        self.ema_update_after_step = ema_update_after_step # only start EMA after this step number, starting at 0
        self.ema_update_every = ema_update_every

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0.]))

    def update(self):
        self.step += 1

        if self.step <= self.ema_update_after_step or (self.step % self.ema_update_every) != 0:
            return

        if not self.initted:
            self.ema_model.state_dict(self.online_model.state_dict())
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)
    # 更新移动平均模型的参数
    def update_moving_average(self, ma_model, current_model):
        # 定义计算指数移动平均的函数
        def calculate_ema(beta, old, new):
            # 如果旧值不存在，则直接返回新值
            if not exists(old):
                return new
            # 计算指数移动平均值
            return old * beta + (1 - beta) * new

        # 遍历当前模型和移动平均模型的参数，更新移动平均值
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = calculate_ema(self.beta, old_weight, up_weight)

        # 遍历当前模型和移动平均模型的缓冲区，更新移动平均值
        for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
            new_buffer_value = calculate_ema(self.beta, ma_buffer, current_buffer)
            ma_buffer.copy_(new_buffer_value)

    # 调用函数，返回移动平均模型的结果
    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
# 主要的训练器类

class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        num_train_steps,
        lr,
        batch_size,
        grad_accum_every,
        wd = 0.,
        images_memmap_path = None,
        images_memmap_shape = None,
        folder = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        ema_beta = 0.995,
        ema_update_after_step = 2000,
        ema_update_every = 10,
        apply_grad_penalty_every = 4,
    ):
        super().__init__()
        assert isinstance(vae, VQGanVAE), 'vae must be instance of VQGanVAE'
        image_size = vae.image_size

        self.vae = vae
        self.ema_vae = EMA(vae, ema_update_after_step = ema_update_after_step, ema_update_every = ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
        self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)

        # 创建数据集

        assert exists(folder) ^ exists(images_memmap_path), 'either folder or memmap path to images must be supplied'

        if exists(images_memmap_path):
            assert exists(images_memmap_shape), 'shape of memmapped images must be supplied'

        if exists(folder):
            self.ds = ImageDataset(folder, image_size = image_size)
        elif exists(images_memmap_path):
            self.ds = MemmappedImageDataset(path = images_memmap_path, shape = images_memmap_shape)

        # 划分验证集

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # 数据加载器

        self.dl = cycle(DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        ))

        self.valid_dl = cycle(DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        ))

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)
    # 定义训练步骤函数
    def train_step(self):
        # 获取模型参数所在设备
        device = next(self.vae.parameters()).device
        # 获取当前步数
        steps = int(self.steps.item())
        # 是否应用梯度惩罚
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        # 设置 VAE 模型为训练模式
        self.vae.train()

        # 初始化日志字典
        logs = {}

        # 更新 VAE（生成器）

        # 多次执行梯度累积
        for _ in range(self.grad_accum_every):
            # 获取下一个数据批次
            img = next(self.dl)
            img = img.to(device)

            # 计算损失
            loss = self.vae(
                img,
                return_loss = True,
                apply_grad_penalty = apply_grad_penalty
            )

            # 累积损失到日志中
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

            # 反向传播
            (loss / self.grad_accum_every).backward()

        # 更新优化器
        self.optim.step()
        self.optim.zero_grad()

        # 更新鉴别器

        if exists(self.vae.discr):
            self.discr_optim.zero_grad()
            discr_loss = 0

            for _ in range(self.grad_accum_every):
                img = next(self.dl)
                img = img.to(device)

                loss = self.vae(img, return_discr_loss = True)
                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

                (loss / self.grad_accum_every).backward()

            self.discr_optim.step()

            # 打印日志
            print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        # 更新指数移动平均生成器
        self.ema_vae.update()

        # 定期采样结果

        if not (steps % self.save_results_every):
            for model, filename in ((self.ema_vae.ema_model, f'{steps}.ema'), (self.vae, str(steps))):
                model.eval()

                imgs = next(self.dl)
                imgs = imgs.to(device)

                recons = model(imgs)
                nrows = int(sqrt(self.batch_size))

                imgs_and_recons = torch.stack((imgs, recons), dim = 0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

                logs['reconstructions'] = grid

                save_image(grid, str(self.results_folder / f'{filename}.png'))

            print(f'{steps}: saving to {str(self.results_folder)}')

        # 定期保存模型

        if not (steps % self.save_model_every):
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)

            ema_state_dict = self.ema_vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
            torch.save(ema_state_dict, model_path)

            print(f'{steps}: saving model to {str(self.results_folder)}')

        # 更新步数
        self.steps += 1
        return logs

    # 训练函数
    def train(self, log_fn = noop):
        # 获取模型参数所在设备
        device = next(self.vae.parameters()).device

        # 在训练步数未达到总训练步数前循环执行训练步骤
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        # 训练完成
        print('training complete')
```