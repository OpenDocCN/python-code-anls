# `.\lucidrains\parti-pytorch\parti_pytorch\vit_vqgan_trainer.py`

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
from shutil import rmtree
# 从 PIL 模块中导入 Image 类
from PIL import Image

# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn 模块
from torch import nn
# 从 torch.cuda.amp 模块中导入 autocast, GradScaler 类
from torch.cuda.amp import autocast, GradScaler
# 从 torch.utils.data 模块中导入 Dataset, DataLoader, random_split 类
from torch.utils.data import Dataset, DataLoader, random_split

# 导入 torchvision.transforms 模块，并重命名为 T
import torchvision.transforms as T
# 从 torchvision.datasets 模块中导入 ImageFolder 类
from torchvision.datasets import ImageFolder
# 从 torchvision.utils 模块中导入 make_grid, save_image 函数
from torchvision.utils import make_grid, save_image

# 从 einops 模块中导入 rearrange 函数
from einops import rearrange

# 从 parti_pytorch.vit_vqgan 模块中导入 VitVQGanVAE 类
from parti_pytorch.vit_vqgan import VitVQGanVAE
# 从 parti_pytorch.optimizer 模块中导入 get_optimizer 函数
from parti_pytorch.optimizer import get_optimizer

# 从 ema_pytorch 模块中导入 EMA 类

from ema_pytorch import EMA

# 辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 空操作函数
def noop(*args, **kwargs):
    pass

# 生成数据循环的函数
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 将输入转换为元组的函数
def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

# 询问用户是否为是或否的函数
def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

# 累积日志的函数
def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# 类

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

class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        num_train_steps,
        batch_size,
        folder,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        ema_beta = 0.995,
        ema_update_after_step = 500,
        ema_update_every = 10,
        apply_grad_penalty_every = 4,
        amp = False
        ):
        # 调用父类的构造函数
        super().__init__()
        # 断言确保 vae 是 VitVQGanVAE 的实例
        assert isinstance(vae, VitVQGanVAE), 'vae must be instance of VitVQGanVAE'
        # 获取 VAE 的图像大小
        image_size = vae.image_size

        # 设置 VAE 和 EMA VAE
        self.vae = vae
        self.ema_vae = EMA(vae, update_after_step = ema_update_after_step, update_every = ema_update_every)

        # 注册缓冲区 'steps'，用于记录步数
        self.register_buffer('steps', torch.Tensor([0]))

        # 设置训练步数、批量大小、梯度累积频率
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # 获取所有参数、判别器参数、VAE 参数
        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        # 获取优化器
        self.optim = get_optimizer(vae_parameters, lr = lr, wd = wd)
        self.discr_optim = get_optimizer(discr_parameters, lr = lr, wd = wd)

        # 设置混合精度训练相关参数
        self.amp = amp
        self.scaler = GradScaler(enabled = amp)
        self.discr_scaler = GradScaler(enabled = amp)

        # 创建数据集
        self.ds = ImageDataset(folder, image_size = image_size)

        # 划分验证集
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # 创建数据加载器
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

        # 设置保存模型和结果的频率
        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        # 设置应用梯度惩罚的频率
        self.apply_grad_penalty_every = apply_grad_penalty_every

        # 设置结果文件夹
        self.results_folder = Path(results_folder)

        # 如果结果文件夹中有文件且用户确认清除，则清除文件夹
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        # 创建结果文件夹
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

        # 根据梯度累积次数进行更新
        for _ in range(self.grad_accum_every):
            # 获取下一个数据批次
            img = next(self.dl)
            img = img.to(device)

            # 开启自动混合精度
            with autocast(enabled = self.amp):
                # 计算损失
                loss = self.vae(
                    img,
                    return_loss = True,
                    apply_grad_penalty = apply_grad_penalty
                )

                # 反向传播并缩放损失
                self.scaler.scale(loss / self.grad_accum_every).backward()

            # 累积日志
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        # 梯度更新
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad()

        # 更新鉴别器

        if exists(self.vae.discr):
            self.discr_optim.zero_grad()

            discr_loss = 0
            for _ in range(self.grad_accum_every):
                img = next(self.dl)
                img = img.to(device)

                with autocast(enabled = self.amp):
                    loss = self.vae(img, return_discr_loss = True)

                    self.discr_scaler.scale(loss / self.grad_accum_every).backward()

                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

            self.discr_scaler.step(self.discr_optim)
            self.discr_scaler.update()

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

        # 更新步数并返回日志
        self.steps += 1
        return logs

    # 训练函数
    def train(self, log_fn = noop):
        # 获取模型参数所在设备
        device = next(self.vae.parameters()).device

        # 在训练步数未达到总训练步数前循环训练步骤
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        # 训练完成
        print('training complete')
```