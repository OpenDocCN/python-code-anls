# `.\lucidrains\nuwa-pytorch\nuwa_pytorch\train_nuwa.py`

```py
# 从 random 模块中导入 randrange 函数
from random import randrange
# 从 pathlib 模块中导入 Path 类
from pathlib import Path

# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.utils.data 模块中导入 Dataset 和 DataLoader 类
from torch.utils.data import Dataset, DataLoader
# 从 torch.nn.utils.rnn 模块中导入 pad_sequence 函数
from torch.nn.utils.rnn import pad_sequence
# 从 einops 库中导入 rearrange 函数
from einops import rearrange

# 从 tqdm 模块中导入 tqdm 函数
from tqdm import tqdm
# 导入 numpy 库
import numpy as np
# 从 shutil 模块中导入 rmtree 函数
from shutil import rmtree

# 导入 nuwa_pytorch 库中的 tokenizer 模块和 optimizer 模块
from nuwa_pytorch.tokenizer import tokenizer
from nuwa_pytorch.optimizer import get_optimizer
# 导入 nuwa_pytorch 库中的 image_utils 模块
from nuwa_pytorch.image_utils import gif_to_tensor
# 从 nuwa_pytorch 模块中导入 NUWA 类

# 从 torchvision.transforms 模块中导入 T 别名
import torchvision.transforms as T
# 从 torchvision.utils 模块中导入 make_grid 和 save_image 函数

# 辅助函数

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 空操作函数
def noop(*args, **kwargs):
    pass

# 生成循环迭代器的函数
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

# 数据加载器辅助函数

# 数据填充函数
def pad_collate_fn(batch):
    texts, videos = zip(*batch)
    return pad_sequence(texts, batch_first = True), torch.stack(videos)

# 数据处理流水线函数

# 将视频张量数据集转换为索引的函数
def convert_video_tensor_dataset_to_indices(
    *,
    vae,
    raw_video_dataset,
    num_frames,
    path,
):
    vae_device = next(vae.parameters()).device
    num_videos = len(raw_video_dataset)
    assert num_videos > 0, 'there must be at least 1 video'

    fmap_size = vae.image_size // (vae.num_layers ** 2)
    shape = (num_videos, num_frames * fmap_size * fmap_size)

    video_indices_memmap = np.memmap(path, mode = 'w+', dtype = np.int64, shape = shape)

    for ind in tqdm(range(num_videos)):
        _, video = raw_video_dataset[ind]
        video = rearrange(video, '... -> 1 ...')
        video = video.to(vae_device)
        indices = vae.get_video_indices(video)
        indices = rearrange(indices, '1 f h w -> (f h w)')
        video_indices_memmap[ind] = indices.cpu().numpy()

    print(f'completed conversion of {num_videos} videos to indices at {path}')

# 数据集类

# Mnist 数据集类
class MnistDataset(Dataset):
    def __init__(
        self,
        num_videos,
        videos_memmap_path,
        text_memmap_path,
        num_digits = 2,
        num_frames = 10,
        image_size = 64,
        channels = 1,
        random_rotate = False
    ):
        super().__init__()
        self.num_videos = num_videos
        self.videos_memmap = np.memmap(videos_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_frames, channels, image_size, image_size))
        self.text_memmap = np.memmap(text_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_digits))
        self.random_rotate = random_rotate

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        video = torch.from_numpy(self.videos_memmap[idx].copy()).float()
        label = torch.from_numpy(self.text_memmap[idx].copy())

        video /= 255
        video = video.to(torch.float32)

        text = tokenizer.encode(' '.join(map(str, label.tolist())))
        text = torch.Tensor(text).long()

        if self.random_rotate:
            video = T.functional.rotate(video, choice([0, 90, 180, 270]))

        return text, video

# 视频索引数据集类
class VideoIndicesDataset(Dataset):
    def __init__(
        self,
        *,
        videos_memmap_path,
        text_memmap_path,
        vae,
        num_videos,
        num_frames,
        num_digits = 2,
    ):
        self.num_videos = num_videos
        fmap_size = vae.image_size // (vae.num_layers ** 2)
        self.videos_memmap = np.memmap(videos_memmap_path, mode = 'r', dtype = np.int64, shape = (num_videos, num_frames * (fmap_size ** 2)))
        self.text_memmap = np.memmap(text_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_digits))

    def __len__(self):
        return self.num_videos
    # 定义一个特殊方法，用于获取数据集中指定索引位置的数据
    def __getitem__(self, idx):
        # 从内存映射中读取视频数据，并转换为PyTorch张量
        video = torch.from_numpy(self.videos_memmap[idx].copy())
        # 从内存映射中读取文本数据，并转换为PyTorch张量
        text = torch.from_numpy(self.text_memmap[idx].copy())

        # 将文本数据转换为字符串，使用空格连接后编码为token，再转换为PyTorch张量
        text = tokenizer.encode(' '.join(map(str, text.tolist())))
        text = torch.Tensor(text).long()

        # 将视频数据转换为长整型张量
        video = video.long()
        # 返回处理后的文本和视频数据
        return text, video
# 从视频文件夹中创建用于训练的数据集类
class GifVideoDataset(Dataset):
    def __init__(
        self,
        *,
        folder,  # 视频文件夹路径
        channels = 1  # 通道数，默认为1
    ):
        # 将文件夹路径转换为 Path 对象
        folder = Path(folder)
        # 获取所有 GIF 文件和对应的文本文件
        gifs = folder.glob('**/*.gif')
        txts = folder.glob('**/*.txt')

        # 获取 GIF 文件和文本文件的路径前缀
        gif_path_stems = set(map(lambda t: str(t.with_suffix('')), gifs))
        txt_path_stems = set(map(lambda t: str(t.with_suffix('')), txts))
        # 获取共同的路径前缀作为数据集的路径
        self.path_stems = list(gif_path_stems.intersection(txt_path_stems))

        self.channels = channels  # 设置通道数
        print(f'{len(self.path_stems)} video / text pairs found')  # 打印找到的视频/文本对数量

    def __len__(self):
        return len(self.path_stems)  # 返回数据集的长度

    def __getitem__(self, idx):
        path_stem = self.path_stems[idx]  # 获取指定索引的路径前缀

        txt_path = Path(f'{path_stem}.txt')  # 构建文本文件路径
        txt_str = txt_path.read_text()  # 读取文本文件内容
        text_tensor = torch.Tensor(tokenizer.encode(txt_str)).long()  # 将文本内容编码为张量

        video_tensor = gif_to_tensor(f'{path_stem}.gif', channels = self.channels)  # 将 GIF 文件转换为张量
        return text_tensor, video_tensor  # 返回文本张量和视频张量的元组

# 训练类
class NUWATrainer(nn.Module):
    def __init__(
        self,
        *,
        nuwa,  # NUWA 模型实例
        dataset,  # 数据集实例
        num_train_steps,  # 训练步数
        lr = 3e-4,  # 学习率，默认为 3e-4
        wd = 0.01,  # 权重衰减，默认为 0.01
        batch_size = 4,  # 批量大小，默认为 4
        grad_accum_every = 8,  # 梯度累积间隔，默认为 8
        max_grad_norm = 0.5,  # 最大梯度范数，默认为 0.5
        save_model_every = 2500,  # 每隔多少步保存模型，默认为 2500
        save_results_every = 1000,  # 每隔多少步保存结果，默认为 1000
        results_folder = './results-nuwa',  # 结果文件夹路径，默认为 './results-nuwa'
        num_sampled_frames = float('inf')  # 抽样帧数，默认为无穷大
    ):
        super().__init__()
        assert isinstance(nuwa, NUWA), 'nuwa must be an instance of NUWA'  # 断言 nuwa 必须是 NUWA 类的实例
        self.nuwa = nuwa  # 设置 NUWA 模型实例

        self.steps = 0  # 训练步数初始化为 0
        self.num_train_steps = num_train_steps  # 设置训练步数
        self.batch_size = batch_size  # 设置批量大小
        self.grad_accum_every = grad_accum_every  # 设置梯度累积间隔
        self.max_grad_norm = max_grad_norm  # 设置最大梯度范数

        self.optim = get_optimizer(nuwa.parameters(), lr = lr, wd = wd)  # 获取优化器

        # 数据集
        self.ds = dataset  # 设置数据集

        # 数据加载器
        self.dl = cycle(DataLoader(
            self.ds,
            batch_size = batch_size,
            collate_fn = pad_collate_fn,
            shuffle = True
        ))  # 创建循环数据加载器

        self.save_model_every = save_model_every  # 设置保存模型间隔
        self.save_results_every = save_results_every  # 设置保存结果间隔
        self.num_sampled_frames = num_sampled_frames  # 设置抽样帧数

        self.results_folder = Path(results_folder)  # 设置结果文件夹路径

        # 如果结果文件夹中有文件且确认清除之前的实验检查点和结果
        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))  # 清除之前的实验检查点和结果

        self.results_folder.mkdir(parents = True, exist_ok = True)  # 创建结果文件夹
    # 定义训练步骤函数
    def train_step(self):
        # 获取模型参数所在设备
        device = next(self.nuwa.parameters()).device
        # 设置模型为训练模式
        self.nuwa.train()

        # 初始化日志字典
        logs = {}

        # 循环执行梯度累积次数
        for _ in range(self.grad_accum_every):
            # 从数据加载器中获取文本和视频数据
            text, video = next(self.dl)
            # 将文本和视频数据移动到指定设备
            text, video = map(lambda t: t.to(device), (text, video))

            # 计算模型损失
            loss = self.nuwa(
                text = text,
                video = video,
                return_loss = True
            )
            # 累积损失到日志中
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

            # 反向传播梯度
            (loss / self.grad_accum_every).backward()

        # 打印当前步骤的损失值
        print(f'{self.steps} loss: {logs["loss"]}')

        # 对模型参数进行梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.nuwa.parameters(), self.max_grad_norm)
        # 更新优化器参数
        self.optim.step()
        # 清空梯度
        self.optim.zero_grad()

        # 每隔一定步骤保存生成结果
        if not (self.steps % self.save_results_every):
            # 设置模型为评估模式
            self.nuwa.eval()
            print(f'{self.steps} sampling')

            # 随机选择一个数据样本
            rand_idx = randrange(0, len(self.ds))

            text, video = self.ds[rand_idx]
            text, video = next(self.dl)
            text = text.to(device)

            # 生成视频序列
            video = self.nuwa.generate(text = text[:1], num_frames = min(video.shape[1], self.num_sampled_frames))
            one_video = video[0].cpu().clamp(0., 1.)

            # 解码文本数据
            text_str = tokenizer.decode(text[0])

            # 保存生成的文本和视频结果
            logs['sampled_text'] = text_str
            logs['sampled_video'] = one_video.numpy()

            # 重新排列视频帧以保存为图像
            image = rearrange(one_video, 'f c h w -> c (f h) w')
            save_image(image, str(self.results_folder / f'{self.steps}.png'))

            print(f'{self.steps}: saving to {str(self.results_folder)}')

        # 每隔一定步骤保存模型
        if not (self.steps % self.save_model_every):
            # 获取模型状态字典
            state_dict = self.nuwa.state_dict()
            model_path = str(self.results_folder / f'nuwa.{self.steps}.pt')
            # ���存模型参数
            torch.save(state_dict, model_path)

            print(f'{self.steps}: saving model to {str(self.results_folder)}')

        # 更新步骤数
        self.steps += 1
        return logs

    # 定义训练函数
    def train(self, log_fn = noop):
        # 循环执行训练步骤直到达到指定训练步数
        while self.steps < self.num_train_steps:
            # 执行训练步骤并记录日志
            logs = self.train_step()
            log_fn(logs)

        # 打印训练完成信息
        print('training complete')
```