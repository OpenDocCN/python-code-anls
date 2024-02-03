# `.\PaddleOCR\StyleText\utils\math_functions.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
import paddle

# 计算输入图像的均值和协方差
def compute_mean_covariance(img):
    # 获取图像的批处理大小、通道数、高度和宽度
    batch_size = img.shape[0]
    channel_num = img.shape[1]
    height = img.shape[2]
    width = img.shape[3]
    num_pixels = height * width

    # 计算图像的均值
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # 对图像进行中心化处理
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.reshape([batch_size, channel_num, num_pixels])
    
    # 转置图像以便计算协方差
    img_hat_transpose = img_hat.transpose([0, 2, 1])
    
    # 计算图像的协方差矩阵
    covariance = paddle.bmm(img_hat, img_hat_transpose)
    covariance = covariance / num_pixels

    return mu, covariance

# 计算 Dice 系数，用于评估分割模型的性能
def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    eps = 1e-5
    intersection = paddle.sum(y_true_cls * y_pred_cls * training_mask)
    union = paddle.sum(y_true_cls * training_mask) + paddle.sum(
        y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    return loss
```