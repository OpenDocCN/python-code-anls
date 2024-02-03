# `.\PaddleOCR\ppocr\utils\iou.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
此代码参考自:
https://github.com/whai362/PSENet/blob/python3/models/loss/iou.py
"""

# 导入 paddle 模块
import paddle

# 定义一个极小值 EPS
EPS = 1e-6

# 计算单个样本的 Intersection over Union
def iou_single(a, b, mask, n_class):
    # 获取有效的像素点
    valid = mask == 1
    a = a.masked_select(valid)
    b = b.masked_select(valid)
    miou = []
    for i in range(n_class):
        # 如果 a 的形状为 [0] 且 a 的形状与 b 的形状相同
        if a.shape == [0] and a.shape == b.shape:
            inter = paddle.to_tensor(0.0)
            union = paddle.to_tensor(0.0)
        else:
            # 计算交集和并集
            inter = ((a == i).logical_and(b == i)).astype('float32')
            union = ((a == i).logical_or(b == i)).astype('float32')
        miou.append(paddle.sum(inter) / (paddle.sum(union) + EPS))
    miou = sum(miou) / len(miou)
    return miou

# 计算整个 batch 的 Intersection over Union
def iou(a, b, mask, n_class=2, reduce=True):
    batch_size = a.shape[0]

    a = a.reshape([batch_size, -1])
    b = b.reshape([batch_size, -1])
    mask = mask.reshape([batch_size, -1])

    iou = paddle.zeros((batch_size, ), dtype='float32')
    for i in range(batch_size):
        iou[i] = iou_single(a[i], b[i], mask[i], n_class)

    if reduce:
        iou = paddle.mean(iou)
    return iou
```