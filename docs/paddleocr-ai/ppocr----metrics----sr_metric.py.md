# `.\PaddleOCR\ppocr\metrics\sr_metric.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""
# 导入所需的库
from math import exp

# 导入 PaddlePaddle 深度学习框架
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import string

# 定义 SSIM 类，用于计算结构相似性指标
class SSIM(nn.Layer):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        # 创建窗口函数
        self.window = self.create_window(window_size, self.channel)

    # 定义高斯函数
    def gaussian(self, window_size, sigma):
        # 计算高斯权重
        gauss = paddle.to_tensor([
            exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    # 创建窗口
    def create_window(self, window_size, channel):
        # 生成一维高斯窗口
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # 生成二维高斯窗口
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        # 扩展为多通道窗口
        window = _2D_window.expand([channel, 1, window_size, window_size])
        return window
    # 计算结构相似性指数（SSIM）的私有方法
    def _ssim(self, img1, img2, window, window_size, channel,
              size_average=True):
        # 计算图像的均值
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        # 计算均值的平方
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 计算方差
        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=window_size // 2,
            groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=window_size // 2,
            groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=window_size // 2,
            groups=channel) - mu1_mu2

        # 定义常数C1和C2
        C1 = 0.01**2
        C2 = 0.03**2

        # 计算SSIM图
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        # 如果需要对结果进行平均，则返回平均值
        if size_average:
            return ssim_map.mean()
        # 否则返回指定维度上的平均值
        else:
            return ssim_map.mean([1, 2, 3])

    # 计算两个图像的SSIM
    def ssim(self, img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.shape
        # 创建窗口
        window = self.create_window(window_size, channel)

        return self._ssim(img1, img2, window, window_size, channel,
                          size_average)

    # 前向传播方法
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.shape

        # 如果通道数和窗口数据类型匹配，则使用已有窗口
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        # 否则重新创建窗口
        else:
            window = self.create_window(self.window_size, channel)

            self.window = window
            self.channel = channel

        return self._ssim(img1, img2, window, self.window_size, channel,
                          self.size_average)
class SRMetric(object):
    # 定义一个SRMetric类
    def __init__(self, main_indicator='all', **kwargs):
        # 初始化函数，设置主要指标为'all'，并接收额外的关键字参数
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.psnr_result = []
        self.ssim_result = []
        self.calculate_ssim = SSIM()
        # 初始化一些变量和对象
        self.reset()

    def reset(self):
        # 重置函数，重置一些计数和结果
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.psnr_result = []
        self.ssim_result = []

    def calculate_psnr(self, img1, img2):
        # 计算图像的峰值信噪比
        # img1和img2的取值范围为[0, 1]
        mse = ((img1 * 255 - img2 * 255)**2).mean()
        if mse == 0:
            return float('inf')
        return 20 * paddle.log10(255.0 / paddle.sqrt(mse))

    def _normalize_text(self, text):
        # 规范化文本，只保留数字和字母，并转换为小写
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, pred_label, *args, **kwargs):
        # 调用函数，计算评价指标
        metric = {}
        images_sr = pred_label["sr_img"]
        images_hr = pred_label["hr_img"]
        psnr = self.calculate_psnr(images_sr, images_hr)
        ssim = self.calculate_ssim(images_sr, images_hr)
        self.psnr_result.append(psnr)
        self.ssim_result.append(ssim)

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        # 获取评价指标
        self.psnr_avg = sum(self.psnr_result) / len(self.psnr_result)
        self.psnr_avg = round(self.psnr_avg.item(), 6)
        self.ssim_avg = sum(self.ssim_result) / len(self.ssim_result)
        self.ssim_avg = round(self.ssim_avg.item(), 6)

        self.all_avg = self.psnr_avg + self.ssim_avg

        self.reset()
        # 重置计数和结果，并返回指标结果
        return {
            'psnr_avg': self.psnr_avg,
            "ssim_avg": self.ssim_avg,
            "all": self.all_avg
        }
```