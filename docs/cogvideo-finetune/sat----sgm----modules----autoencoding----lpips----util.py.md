# `.\cogvideo-finetune\sat\sgm\modules\autoencoding\lpips\util.py`

```py
# 导入所需的库
import hashlib  # 用于计算 MD5 哈希
import os  # 用于操作文件和目录

import requests  # 用于发送 HTTP 请求
import torch  # 用于深度学习框架
import torch.nn as nn  # 用于构建神经网络模块
from tqdm import tqdm  # 用于显示进度条

# 定义 URL 映射字典，包含模型名称及其下载链接
URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

# 定义检查点映射字典，包含模型名称及其本地文件名
CKPT_MAP = {"vgg_lpips": "vgg.pth"}

# 定义 MD5 哈希映射字典，包含模型名称及其对应的哈希值
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}

# 下载指定 URL 的文件到本地路径
def download(url, local_path, chunk_size=1024):
    # 创建存储文件的目录（如果不存在的话）
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    # 发送 GET 请求以流式方式下载文件
    with requests.get(url, stream=True) as r:
        # 获取响应头中的内容长度
        total_size = int(r.headers.get("content-length", 0))
        # 使用 tqdm 显示下载进度
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            # 以二进制写模式打开本地文件
            with open(local_path, "wb") as f:
                # 逐块读取内容并写入文件
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:  # 如果读取到数据
                        f.write(data)  # 将数据写入文件
                        pbar.update(chunk_size)  # 更新进度条

# 计算指定文件的 MD5 哈希值
def md5_hash(path):
    # 以二进制读取模式打开文件
    with open(path, "rb") as f:
        content = f.read()  # 读取文件内容
    # 返回内容的 MD5 哈希值
    return hashlib.md5(content).hexdigest()

# 获取检查点路径，如果需要则下载模型
def get_ckpt_path(name, root, check=False):
    # 确保模型名称在 URL 映射中
    assert name in URL_MAP
    # 组合根目录和检查点文件名生成完整路径
    path = os.path.join(root, CKPT_MAP[name])
    # 如果文件不存在或需要检查 MD5 值
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        # 打印下载信息
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        # 下载文件
        download(URL_MAP[name], path)
        # 计算下载后的文件 MD5 哈希
        md5 = md5_hash(path)
        # 确保 MD5 哈希匹配
        assert md5 == MD5_MAP[name], md5
    # 返回检查点文件的路径
    return path

# 定义一个标准化的神经网络模块
class ActNorm(nn.Module):
    # 初始化模块，设置参数和属性
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine  # 确保启用仿射变换
        super().__init__()  # 调用父类初始化
        self.logdet = logdet  # 是否计算对数行列式
        # 定义位置参数
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        # 定义缩放参数
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init  # 是否允许反向初始化

        # 注册一个用于标记初始化状态的缓冲区
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    # 初始化函数，接受输入并计算位置和缩放参数
    def initialize(self, input):
        with torch.no_grad():  # 禁用梯度计算
            # 重排输入并扁平化
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            # 计算每个特征的均值
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            # 计算每个特征的标准差
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            # 更新位置参数为负均值
            self.loc.data.copy_(-mean)
            # 更新缩放参数为标准差的倒数
            self.scale.data.copy_(1 / (std + 1e-6))
    # 定义前向传播函数，接受输入和是否反向的参数
    def forward(self, input, reverse=False):
        # 如果需要反向传播，调用反向函数处理输入
        if reverse:
            return self.reverse(input)
        # 如果输入是二维数组，扩展维度以适应后续处理
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True  # 标记为需要挤压的状态
        else:
            squeeze = False  # 标记为不需要挤压的状态
    
        # 获取输入的高度和宽度
        _, _, height, width = input.shape
    
        # 如果处于训练模式且未初始化，进行初始化
        if self.training and self.initialized.item() == 0:
            self.initialize(input)  # 初始化
            self.initialized.fill_(1)  # 标记为已初始化
    
        # 计算h，考虑缩放和偏移
        h = self.scale * (input + self.loc)
    
        # 如果需要挤压，去掉多余的维度
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
    
        # 如果需要计算对数行列式，执行相应的计算
        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))  # 计算缩放的对数绝对值
            logdet = height * width * torch.sum(log_abs)  # 计算对数行列式
            logdet = logdet * torch.ones(input.shape[0]).to(input)  # 创建与输入批次相同大小的张量
            return h, logdet  # 返回h和对数行列式
    
        return h  # 返回计算结果h
    
    # 定义反向传播函数，接受输出作为输入
    def reverse(self, output):
        # 如果处于训练模式且未初始化，进行初始化
        if self.training and self.initialized.item() == 0:
            # 如果不允许反向初始化，抛出错误
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)  # 初始化
                self.initialized.fill_(1)  # 标记为已初始化
    
        # 如果输出是二维数组，扩展维度以适应后续处理
        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True  # 标记为需要挤压的状态
        else:
            squeeze = False  # 标记为不需要挤压的状态
    
        # 根据缩放和偏移计算h
        h = output / self.scale - self.loc
    
        # 如果需要挤压，去掉多余的维度
        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h  # 返回计算结果h
```