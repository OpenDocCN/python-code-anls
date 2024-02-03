# `.\PaddleOCR\test_tipc\supplementary\loss.py`

```
# 导入 paddle 库
import paddle
# 导入 paddle 中的神经网络函数模块
import paddle.nn.functional as F

# 定义 Loss 类
class Loss(object):
    """
    Loss
    """

    # 初始化函数，设置类别数和平滑因子
    def __init__(self, class_dim=1000, epsilon=None):
        # 断言类别数大于1
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        # 如果平滑因子不为空且在合理范围内，则启用标签平滑
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

    # 标签平滑函数
    def _labelsmoothing(self, target):
        # 如果目标形状的最后一个维度不等于类别数，则进行 one-hot 编码
        if target.shape[-1] != self._class_dim:
            one_hot_target = F.one_hot(target, self._class_dim)
        else:
            one_hot_target = target
        # 对 one-hot 编码的目标进行标签平滑处理
        soft_target = F.label_smooth(one_hot_target, epsilon=self._epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, self._class_dim])
        return soft_target

    # 交叉熵损失函数
    def _crossentropy(self, input, target, use_pure_fp16=False):
        # 如果启用标签平滑，则对目标进行处理
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            input = -F.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = F.cross_entropy(input=input, label=target)
        # 如果使用纯 fp16，则计算总损失
        if use_pure_fp16:
            avg_cost = paddle.sum(cost)
        else:
            avg_cost = paddle.mean(cost)
        return avg_cost

    # 调用函数，计算损失
    def __call__(self, input, target):
        return self._crossentropy(input, target)

# 构建损失函数
def build_loss(config, epsilon=None):
    class_dim = config['class_dim']
    loss_func = Loss(class_dim=class_dim, epsilon=epsilon)
    return loss_func

# LossDistill 类继承自 Loss 类
class LossDistill(Loss):
    # 初始化函数，接受模型名称列表、类别维度和 epsilon 参数
    def __init__(self, model_name_list, class_dim=1000, epsilon=None):
        # 断言类别维度大于1
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        # 设置类别维度
        self._class_dim = class_dim
        # 如果 epsilon 存在且在 [0, 1] 范围内，则启用标签平滑
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            # 否则不启用标签平滑
            self._epsilon = None
            self._label_smoothing = False

        # 设置模型名称列表
        self.model_name_list = model_name_list
        # 断言模型名称列表长度大于1
        assert len(self.model_name_list) > 1, "error"

    # 调用函数，计算损失
    def __call__(self, input, target):
        # 初始化损失字典
        losses = {}
        # 遍历模型名称列表
        for k in self.model_name_list:
            # 获取输入数据
            inp = input[k]
            # 计算交叉熵损失
            losses[k] = self._crossentropy(inp, target)
        # 返回损失字典
        return losses
# 定义 KLJSLoss 类，用于计算 KL 散度或 JS 散度损失
class KLJSLoss(object):
    # 初始化函数，设置损失计算模式为 kl 或 js
    def __init__(self, mode='kl'):
        # 检查模式是否为 kl 或 js，否则抛出异常
        assert mode in ['kl', 'js', 'KL', 'JS'
                        ], "mode can only be one of ['kl', 'js', 'KL', 'JS']"
        self.mode = mode

    # 定义调用函数，计算两个概率分布的 KL 散度或 JS 散度损失
    def __call__(self, p1, p2, reduction="mean"):
        # 对输入的概率分布进行 softmax 归一化
        p1 = F.softmax(p1, axis=-1)
        p2 = F.softmax(p2, axis=-1)

        # 计算 KL 散度或 JS 散度损失
        loss = paddle.multiply(p2, paddle.log((p2 + 1e-5) / (p1 + 1e-5) + 1e-5))

        # 如果模式为 js，则计算 JS 散度损失
        if self.mode.lower() == "js":
            loss += paddle.multiply(
                p1, paddle.log((p1 + 1e-5) / (p2 + 1e-5) + 1e-5))
            loss *= 0.5
        # 根据 reduction 参数计算损失的平均值、总和或直接返回
        if reduction == "mean":
            loss = paddle.mean(loss)
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = paddle.sum(loss)
        return loss


# 定义 DMLLoss 类，用于计算模型之间的 KL 散度或 JS 散度损失
class DMLLoss(object):
    # 初始化函数，设置模型名称对和损失计算模式
    def __init__(self, model_name_pairs, mode='js'):
        # 检查模型名称对格式是否正确，初始化 KLJSLoss 对象
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.kljs_loss = KLJSLoss(mode=mode)

    # 检查模型名称对格式的私有函数
    def _check_model_name_pairs(self, model_name_pairs):
        # 如果模型名称对不是列表格式，则返回空列表
        if not isinstance(model_name_pairs, list):
            return []
        # 如果模型名称对格式正确，则直接返回
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    # 定义调用函数，计算模型之间的 KL 散度或 JS 散度损失
    def __call__(self, predicts, target=None):
        # 初始化损失字典
        loss_dict = dict()
        # 遍历模型名称对，计算对应模型之间的损失
        for pairs in self.model_name_pairs:
            p1 = predicts[pairs[0]]
            p2 = predicts[pairs[1]]

            loss_dict[pairs[0] + "_" + pairs[1]] = self.kljs_loss(p1, p2)

        return loss_dict


# 注释掉的函数，暂时不使用
# def build_distill_loss(config, epsilon=None):
#     class_dim = config['class_dim']
#     loss = LossDistill(model_name_list=['student', 'student1'], )
#     return loss_func
```