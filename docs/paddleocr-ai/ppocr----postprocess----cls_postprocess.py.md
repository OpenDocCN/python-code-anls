# `.\PaddleOCR\ppocr\postprocess\cls_postprocess.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
import paddle

# 定义一个类 ClsPostProcess，用于在文本标签和文本索引之间进行转换
class ClsPostProcess(object):

    def __init__(self, label_list=None, key=None, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list
        self.key = key

    # 定义 __call__ 方法，用于处理预测结果和标签
    def __call__(self, preds, label=None, *args, **kwargs):
        # 如果指定了 key，则从预测结果中获取对应的值
        if self.key is not None:
            preds = preds[self.key]

        label_list = self.label_list
        # 如果标签列表为空，则默认为索引和值相同的字典
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        # 如果预测结果是 paddle.Tensor 类型，则转换为 numpy 数组
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        # 获取预测结果中概率最大的索引
        pred_idxs = preds.argmax(axis=1)
        # 解码输出，将索引转换为对应的标签和概率
        decode_out = [(label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)]
        # 如果没有标签，则返回解码输出
        if label is None:
            return decode_out
        # 将标签转换为对应的标签和概率，返回解码输出和标签
        label = [(label_list[idx], 1.0) for idx in label]
        return decode_out, label
```