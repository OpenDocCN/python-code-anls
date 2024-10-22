# `.\cogview3-finetune\sat\sgm\modules\autoencoding\lpips\util.py`

```py
# 导入所需的库
import hashlib  # 导入 hashlib 库用于计算文件的 MD5 哈希
import os  # 导入 os 库用于文件和目录操作

import requests  # 导入 requests 库用于发送 HTTP 请求
import torch  # 导入 PyTorch 库用于深度学习
import torch.nn as nn  # 导入 nn 模块用于构建神经网络
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

# 定义模型 URL 映射字典
URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

# 定义模型检查点文件名映射字典
CKPT_MAP = {"vgg_lpips": "vgg.pth"}

# 定义模型 MD5 哈希值映射字典
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


# 定义下载函数
def download(url, local_path, chunk_size=1024):
    # 创建本地路径的父目录，如果不存在则创建
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    # 发送 GET 请求以流式下载文件
    with requests.get(url, stream=True) as r:
        # 获取响应头中的内容长度
        total_size = int(r.headers.get("content-length", 0))
        # 使用 tqdm 显示下载进度条
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            # 以二进制写入模式打开本地文件
            with open(local_path, "wb") as f:
                # 分块读取响应内容
                for data in r.iter_content(chunk_size=chunk_size):
                    # 如果读取到数据，则写入文件
                    if data:
                        f.write(data)  # 写入数据到文件
                        pbar.update(chunk_size)  # 更新进度条


# 定义 MD5 哈希函数
def md5_hash(path):
    # 以二进制读取模式打开指定路径的文件
    with open(path, "rb") as f:
        content = f.read()  # 读取文件内容
    # 返回文件内容的 MD5 哈希值
    return hashlib.md5(content).hexdigest()


# 定义获取检查点路径的函数
def get_ckpt_path(name, root, check=False):
    # 确保给定的模型名称在 URL 映射中
    assert name in URL_MAP
    # 组合根目录和检查点文件名，形成完整路径
    path = os.path.join(root, CKPT_MAP[name])
    # 检查文件是否存在或是否需要重新下载
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        # 打印下载信息
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)  # 下载文件
        md5 = md5_hash(path)  # 计算下载文件的 MD5 哈希值
        # 确保下载的文件 MD5 值与预期匹配
        assert md5 == MD5_MAP[name], md5
    return path  # 返回检查点路径


# 定义一个自定义的神经网络模块
class ActNorm(nn.Module):
    # 构造函数，初始化参数
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine  # 确保启用仿射变换
        super().__init__()  # 调用父类构造函数
        self.logdet = logdet  # 保存 logdet 标志
        # 定义可学习的均值参数
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # 定义可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init  # 保存是否允许反向初始化标志

        # 注册一个缓冲区，用于记录初始化状态
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    # 定义初始化函数
    def initialize(self, input):
        with torch.no_grad():  # 在不计算梯度的上下文中
            # 将输入张量重排列并展平
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            # 计算展平后的均值
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            # 计算展平后的标准差
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            # 将均值复制到 loc 参数
            self.loc.data.copy_(-mean)
            # 将标准差的倒数复制到 scale 参数
            self.scale.data.copy_(1 / (std + 1e-6))
    # 定义前向传播函数，接受输入和反向标志
        def forward(self, input, reverse=False):
            # 如果反向标志为真，调用反向函数处理输入
            if reverse:
                return self.reverse(input)
            # 检查输入的形状是否为二维
            if len(input.shape) == 2:
                # 将二维输入扩展为四维，增加两个新的维度
                input = input[:, :, None, None]
                squeeze = True  # 标记为需要压缩
            else:
                squeeze = False  # 不需要压缩
    
            # 解包输入的高度和宽度
            _, _, height, width = input.shape
    
            # 如果处于训练状态且尚未初始化
            if self.training and self.initialized.item() == 0:
                # 初始化参数
                self.initialize(input)
                # 标记为已初始化
                self.initialized.fill_(1)
    
            # 根据比例因子和位移量调整输入
            h = self.scale * (input + self.loc)
    
            # 如果需要压缩，移除最后两个维度
            if squeeze:
                h = h.squeeze(-1).squeeze(-1)
    
            # 如果需要计算对数行列式
            if self.logdet:
                # 计算比例因子的绝对值的对数
                log_abs = torch.log(torch.abs(self.scale))
                # 计算对数行列式的值
                logdet = height * width * torch.sum(log_abs)
                # 生成与批量大小相同的对数行列式张量
                logdet = logdet * torch.ones(input.shape[0]).to(input)
                # 返回调整后的输出和对数行列式
                return h, logdet
    
            # 返回调整后的输出
            return h
    
    # 定义反向传播函数，接受输出
        def reverse(self, output):
            # 如果处于训练状态且尚未初始化
            if self.training and self.initialized.item() == 0:
                # 如果不允许在反向方向初始化，则抛出错误
                if not self.allow_reverse_init:
                    raise RuntimeError(
                        "Initializing ActNorm in reverse direction is "
                        "disabled by default. Use allow_reverse_init=True to enable."
                    )
                else:
                    # 初始化参数
                    self.initialize(output)
                    # 标记为已初始化
                    self.initialized.fill_(1)
    
            # 检查输出的形状是否为二维
            if len(output.shape) == 2:
                # 将二维输出扩展为四维，增加两个新的维度
                output = output[:, :, None, None]
                squeeze = True  # 标记为需要压缩
            else:
                squeeze = False  # 不需要压缩
    
            # 根据比例因子和位移量调整输出
            h = output / self.scale - self.loc
    
            # 如果需要压缩，移除最后两个维度
            if squeeze:
                h = h.squeeze(-1).squeeze(-1)
            # 返回调整后的输出
            return h
```