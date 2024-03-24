# `.\lucidrains\deep-daze\deep_daze\deep_daze.py`

```
# 导入所需的库
import os
import subprocess
import sys
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from siren_pytorch import SirenNet, SirenWrapper
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch_optimizer import DiffGrad, AdamP
import numpy as np

from PIL import Image
from imageio import imread, mimsave
import torchvision.transforms as T

from tqdm import trange, tqdm

from .clip import load, tokenize

# 定义一些辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 对图像进行插值
def interpolate(image, size):
    return F.interpolate(image, (size, size), mode='bilinear', align_corners=False)

# 随机裁剪图像
def rand_cutout(image, size, center_bias=False, center_focus=2):
    # 生成随机裁剪的位置
    # 如果center_bias为True，则在图像中心附近采样
    # 否则在整个图像范围内随机采样
    # 返回裁剪后的图像
    ...

# 创建用于处理CLIP图像的转换函数
def create_clip_img_transform(image_width):
    # 定义CLIP图像的均值和标准差
    # 创建图像转换序列
    transform = T.Compose([
                    T.Resize(image_width),
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std)
            ])
    return transform

# 打开文件夹
def open_folder(path):
    # 如果路径是文件，则获取其所在目录
    # 如果路径不存在或不是文件夹，则返回
    # 根据操作系统选择打开文件夹的命令
    # 尝试打开文件夹，忽略可能的错误
    ...

# 将Siren模型输出归一化到0-1范围
def norm_siren_output(img):
    return ((img + 1) * 0.5).clamp(0.0, 1.0)

# 创建文本或图像的文件名
def create_text_path(context_length, text=None, img=None, encoding=None, separator=None):
    # 如果提供了文本，则根据指定分隔符截取文件名
    # 如果提供了图像，则根据文件名生成文件名
    # 否则使用默认文件名
    return input_name

# 定义DeepDaze类
class DeepDaze(nn.Module):
    ...
    # 初始化函数，设置模型参数和超参数
    def __init__(
            self,
            clip_perceptor,  # CLIP 模型
            clip_norm,  # 归一化图像
            input_res,  # 输入分辨率
            total_batches,  # 总批次数
            batch_size,  # 批次大小
            num_layers=8,  # 神经网络层数，默认为8
            image_width=512,  # 图像宽度，默认为512
            loss_coef=100,  # 损失系数，默认为100
            theta_initial=None,  # 初始 theta 值，默认为 None
            theta_hidden=None,  # 隐藏层 theta 值，默认为 None
            lower_bound_cutout=0.1,  # 切割下界，应小于0.8
            upper_bound_cutout=1.0,  # 切割上界
            saturate_bound=False,  # 是否饱和边界
            gauss_sampling=False,  # 是否高斯采样
            gauss_mean=0.6,  # 高斯均值
            gauss_std=0.2,  # 高斯标准差
            do_cutout=True,  # 是否进行切割
            center_bias=False,  # 是否中心偏置
            center_focus=2,  # 中心焦点
            hidden_size=256,  # 隐藏层大小
            averaging_weight=0.3,  # 平均权重
    ):
        super().__init__()
        # 加载 CLIP 模型
        self.perceptor = clip_perceptor
        self.input_resolution = input_res
        self.normalize_image = clip_norm
        
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

        # 设置初始 theta 值
        w0 = default(theta_hidden, 30.)
        w0_initial = default(theta_initial, 30.)

        # 创建 Siren 网络
        siren = SirenNet(
            dim_in=2,
            dim_hidden=hidden_size,
            num_layers=num_layers,
            dim_out=3,
            use_bias=True,
            w0=w0,
            w0_initial=w0_initial
        )

        # 创建 SirenWrapper 模型
        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_width
        )

        self.saturate_bound = saturate_bound
        self.saturate_limit = 0.75  # 超过此值的切割会导致不稳定
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout
        self.gauss_sampling = gauss_sampling
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std
        self.do_cutout = do_cutout
        self.center_bias = center_bias
        self.center_focus = center_focus
        self.averaging_weight = averaging_weight
        
    # 根据给定的下界、上界、宽度和高斯均值采样切割大小
    def sample_sizes(self, lower, upper, width, gauss_mean):
        if self.gauss_sampling:
            # 使用高斯分布采样
            gauss_samples = torch.zeros(self.batch_size).normal_(mean=gauss_mean, std=self.gauss_std)
            outside_bounds_mask = (gauss_samples > upper) | (gauss_samples < upper)
            gauss_samples[outside_bounds_mask] = torch.zeros((len(gauss_samples[outside_bounds_mask]),)).uniform_(lower, upper)
            sizes = (gauss_samples * width).int()
        else:
            lower *= width
            upper *= width
            sizes = torch.randint(int(lower), int(upper), (self.batch_size,))
        return sizes
    # 定义一个前向传播函数，接受文本嵌入和是否返回损失值以及是否进行干预运行的参数
    def forward(self, text_embed, return_loss=True, dry_run=False):
        # 使用模型进行前向传播
        out = self.model()
        # 对输出进行规范化处理
        out = norm_siren_output(out)

        # 如果不需要返回损失值，则直接返回输出
        if not return_loss:
            return out
                
        # 确定上下采样边界
        width = out.shape[-1]
        lower_bound = self.lower_bound_cutout
        # 如果饱和边界为真，则根据进度比例调整下限边界
        if self.saturate_bound:
            progress_fraction = self.num_batches_processed / self.total_batches
            lower_bound += (self.saturate_limit - self.lower_bound_cutout) * progress_fraction

        # 在下限和上限边界之间采样切割大小
        sizes = self.sample_sizes(lower_bound, self.upper_bound_cutout, width, self.gauss_mean)

        # 创建归一化的随机切割
        if self.do_cutout:   
            image_pieces = [rand_cutout(out, size, center_bias=self.center_bias, center_focus=self.center_focus) for size in sizes]
            image_pieces = [interpolate(piece, self.input_resolution) for piece in image_pieces]
        else:
            image_pieces = [interpolate(out.clone(), self.input_resolution) for _ in sizes]

        # 对图像进行规范化
        image_pieces = torch.cat([self.normalize_image(piece) for piece in image_pieces])
        
        # 计算图像嵌入
        with autocast(enabled=False):
            image_embed = self.perceptor.encode_image(image_pieces)
            
        # 计算损失值
        # 对切割特征的平均值进行损失计算
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)
        averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
        # 对所有切割进行损失计算
        general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        # 合并损失值
        loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

        # 计算批次数
        if not dry_run:
            self.num_batches_processed += self.batch_size
        
        return out, loss
# 定义 Imagine 类，继承自 nn.Module
class Imagine(nn.Module):
    # 初始化函数
    def __init__(
            self,
            *,
            text=None,
            img=None,
            clip_encoding=None,
            lr=1e-5,
            batch_size=4,
            gradient_accumulate_every=4,
            save_every=100,
            image_width=512,
            num_layers=16,
            epochs=20,
            iterations=1050,
            save_progress=True,
            seed=None,
            open_folder=True,
            save_date_time=False,
            start_image_path=None,
            start_image_train_iters=10,
            start_image_lr=3e-4,
            theta_initial=None,
            theta_hidden=None,
            model_name="ViT-B/32",
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            averaging_weight=0.3,
            create_story=False,
            story_start_words=5,
            story_words_per_epoch=5,
            story_separator=None,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            center_bias=False,
            center_focus=2,
            optimizer="AdamP",
            jit=True,
            hidden_size=256,
            save_gif=False,
            save_video=False,
    # 创建 clip_encoding 函数
    def create_clip_encoding(self, text=None, img=None, encoding=None):
        # 设置 text 和 img 属性
        self.text = text
        self.img = img
        # 如果 encoding 不为空，则转移到设备上
        if encoding is not None:
            encoding = encoding.to(self.device)
        # 如果需要创建 story，则更新编码
        elif self.create_story:
            encoding = self.update_story_encoding(epoch=0, iteration=1)
        # 如果 text 和 img 都不为空，则计算平均编码
        elif text is not None and img is not None:
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        # 如果只有 text，则计算文本编码
        elif text is not None:
            encoding = self.create_text_encoding(text)
        # 如果只有 img，则计算图像编码
        elif img is not None:
            encoding = self.create_img_encoding(img)
        return encoding

    # 创建文本编码函数
    def create_text_encoding(self, text):
        # 对文本进行标记化，并转移到设备上
        tokenized_text = tokenize(text).to(self.device)
        # 使用感知器编码���本
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    # 创建图像编码函数
    def create_img_encoding(self, img):
        # 如果 img 是字符串，则打开图像
        if isinstance(img, str):
            img = Image.open(img)
        # 对图像进行规范化处理，并转移到设备上
        normed_img = self.clip_transform(img).unsqueeze(0).to(self.device)
        # 使用感知器编码图像
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    # 设置 clip_encoding 函数
    def set_clip_encoding(self, text=None, img=None, encoding=None):
        # 创建 clip_encoding
        encoding = self.create_clip_encoding(text=text, img=img, encoding=encoding)
        # 将编码转移到设备上
        self.clip_encoding = encoding.to(self.device)
    
    # 返回第一个分隔符的索引
    def index_of_first_separator(self) -> int:
        for c, word in enumerate(self.all_words):
            if self.separator in str(word):
                return c + 1
    def update_story_encoding(self, epoch, iteration):
        # 如果存在分隔符，则将所有单词拼接成字符串，去除分隔符
        if self.separator is not None:
            self.words = " ".join(self.all_words[:self.index_of_first_separator()])
            # 从 epoch-text 中移除分隔符
            self.words = self.words.replace(self.separator,'')
            self.all_words = self.all_words[self.index_of_first_separator():]
        else:
            if self.words is None:
                self.words = " ".join(self.all_words[:self.num_start_words])
                self.all_words = self.all_words[self.num_start_words:]
            else:
                # 添加 words_per_epoch 个新单词
                count = 0
                while count < self.words_per_epoch and len(self.all_words) > 0:
                    new_word = self.all_words[0]
                    self.words = " ".join(self.words.split(" ") + [new_word])
                    self.all_words = self.all_words[1:]
                    count += 1
                # 移除单词直到符合上下文长度
                while len(self.words) > self.perceptor.context_length:
                    # 移除第一个单词
                    self.words = " ".join(self.words.split(" ")[1:])
        # 获取新的编码
        print("Now thinking of: ", '"', self.words, '"')
        sequence_number = self.get_img_sequence_number(epoch, iteration)
        # 将新单词保存到磁盘
        with open("story_transitions.txt", "a") as f:
            f.write(f"{epoch}, {sequence_number}, {self.words}\n")
        
        encoding = self.create_text_encoding(self.words)
        return encoding

    def image_output_path(self, sequence_number=None):
        """
        返回下划线分隔的路径。
        如果设置了 `self.save_date_time`，则在前面加上当前时间戳。
        如果设置了 `save_every`，则在后面加上左填充为 6 个零的序列号。
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        if self.save_date_time:
            current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
            output_path = f"{current_time}_{output_path}"
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast(enabled=True):
                out, loss = self.model(self.clip_encoding)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward()    
        out = out.cpu().float().clamp(0., 1.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if (iteration % self.save_every == 0) and self.save_progress:
            self.save_image(epoch, iteration, img=out)

        return out, total_loss
    
    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None):
        sequence_number = self.get_img_sequence_number(epoch, iteration)

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)
        
        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(self.filename, quality=95, subsampling=0)
        pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)

        tqdm.write(f'image updated at "./{str(self.filename)}"')
    # 生成 GIF 动画
    def generate_gif(self):
        # 初始化空列表用于存储图片
        images = []
        # 遍历当前目录下的文件
        for file_name in sorted(os.listdir('./')):
            # 如果文件名以指定前缀开头且不是指定文件名，则将其读取为图片并添加到列表中
            if file_name.startswith(self.textpath) and file_name != f'{self.textpath}.jpg':
                images.append(imread(os.path.join('./', file_name)))

        # 如果需要保存视频，则将图片列表保存为 MP4 格式
        if self.save_video:
            mimsave(f'{self.textpath}.mp4', images)
            print(f'Generated image generation animation at ./{self.textpath}.mp4')
        # 如果需要保存 GIF，则将图片列表保存为 GIF 格式
        if self.save_gif:
            mimsave(f'{self.textpath}.gif', images)
            print(f'Generated image generation animation at ./{self.textpath}.gif')

    # 向前推进生成过程
    def forward(self):
        # 如果初始图片存在，则进行初始化操作
        if exists(self.start_image):
            tqdm.write('Preparing with initial image...')
            # 使用 DiffGrad 优化器对模型参数进行优化
            optim = DiffGrad(self.model.model.parameters(), lr=self.start_image_lr)
            # 创建进度条
            pbar = trange(self.start_image_train_iters, desc='iteration')
            try:
                # 迭代训练初始图片
                for _ in pbar:
                    loss = self.model.model(self.start_image)
                    loss.backward()
                    pbar.set_description(f'loss: {loss.item():.2f}')

                    optim.step()
                    optim.zero_grad()
            except KeyboardInterrupt:
                print('interrupted by keyboard, gracefully exiting')
                return exit()

            # 释放资源
            del self.start_image
            del optim

        # 输出提示信息
        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        # 禁用梯度计算，进行一次预热步骤以解决 CLIP 和 CUDA 的潜在问题
        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True)

        # 如果需要打开文件夹，则打开当前目录
        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        try:
            # 迭代训练过程
            for epoch in trange(self.epochs, desc='epochs'):
                pbar = trange(self.iterations, desc='iteration')
                for i in pbar:
                    _, loss = self.train_step(epoch, i)
                    pbar.set_description(f'loss: {loss.item():.2f}')

                # 如果正在创建故事，则更新 clip_encoding
                if self.create_story:
                    self.clip_encoding = self.update_story_encoding(epoch, i)
        except KeyboardInterrupt:
            print('interrupted by keyboard, gracefully exiting')
            return

        # 在结束时保存图片
        self.save_image(epoch, i)

        # 如果需要保存 GIF 或视频，并且保存进度，则生成 GIF 动画
        if (self.save_gif or self.save_video) and self.save_progress:
            self.generate_gif()
```