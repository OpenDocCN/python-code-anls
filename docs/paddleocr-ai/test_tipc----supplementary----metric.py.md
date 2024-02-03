# `.\PaddleOCR\test_tipc\supplementary\metric.py`

```
import paddle
import paddle.nn.functional as F
from collections import OrderedDict

# 创建模型准确度的度量，例如 top1 和 top5
def create_metric(out,
                  label,
                  architecture=None,
                  topk=5,
                  classes_num=1000,
                  use_distillation=False,
                  mode="train"):
    """
    Create measures of model accuracy, such as top1 and top5

    Args:
        out(variable): model output variable
        feeds(dict): dict of model input variables(included label)
        topk(int): usually top5
        classes_num(int): num of classes
        use_distillation(bool): whether to use distillation training
        mode(str): mode, train/valid

    Returns:
        fetchs(dict): dict of measures
    """
    # 如果架构名称为 "GoogLeNet"
    # assert 语句用于确保条件为真，否则会引发 AssertionError
    # GoogLeNet 应该有 3 个输出
    # 如果不是 GoogLeNet，则只需要学生标签来获取度量
    # 如果使用蒸馏训练，则使用第二个输出
    # softmax_out 是输出 out 的 softmax 结果
    softmax_out = F.softmax(out)

    fetchs = OrderedDict()
    # 设置 top1 到 fetchs
    top1 = paddle.metric.accuracy(softmax_out, label=label, k=1)
    # 设置 topk 到 fetchs
    k = min(topk, classes_num)
    topk = paddle.metric.accuracy(softmax_out, label=label, k=k)

    # 多卡评估
    # 如果模式不是 "train" 并且世界大小大于 1
    # 使用 paddle.distributed.all_reduce 对 top1 和 topk 进行求和并除以世界大小
    if mode != "train" and paddle.distributed.get_world_size() > 1:
        top1 = paddle.distributed.all_reduce(
            top1, op=paddle.distributed.ReduceOp.
            SUM) / paddle.distributed.get_world_size()
        topk = paddle.distributed.all_reduce(
            topk, op=paddle.distributed.ReduceOp.
            SUM) / paddle.distributed.get_world_size()

    fetchs['top1'] = top1
    topk_name = 'top{}'.format(k)
    fetchs[topk_name] = topk

    return fetchs
```