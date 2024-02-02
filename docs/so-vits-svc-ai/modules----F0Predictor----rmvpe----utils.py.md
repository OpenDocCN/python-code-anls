# `so-vits-svc\modules\F0Predictor\rmvpe\utils.py`

```py
# 导入 sys 模块
import sys
# 从 functools 模块中导入 reduce 函数
from functools import reduce

# 导入 librosa、numpy 和 torch 模块
import librosa
import numpy as np
import torch
# 从 torch.nn.modules.module 模块中导入 _addindent 函数
from torch.nn.modules.module import _addindent
# 从当前目录下的 constants 模块中导入所有内容
from .constants import *  # noqa: F403

# 定义一个循环生成器函数
def cycle(iterable):
    while True:
        for item in iterable:
            yield item

# 定义一个函数用于打印模型的摘要信息
def summary(model, file=sys.stdout):
    # 定义一个内部函数用于生成模型的字符串表示和参数数量
    def repr(model):
        # 将额外的字符串表示看作子模块，每行一个
        extra_lines = []
        extra_repr = model.extra_repr()
        # 如果额外的字符串表示不为空，则按换行符分割成列表
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        # 遍历模型的子模块
        for key, module in model._modules.items():
            # 递归调用 repr 函数生成子模块的字符串表示和参数数量
            mod_str, num_params = repr(module)
            # 添加缩进
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        # 计算模型参数数量
        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # 如果只有一行额外信息且没有子模块，则使用简单的一行信息
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        # 根据输出文件类型打印模型参数数量信息
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    # 调用 repr 函数生成模型的字符串表示和参数数量
    string, count = repr(model)
    # 如果输出文件不为空，则将字符串打印到文件中
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count

# 定义一个函数用于计算加权平均音分近似值
def to_local_average_cents(salience, center=None, thred=0.05):
    """
    find the weighted average cents near the argmax bin
    """
    # 如果 to_local_average_cents 没有属性 'cents_mapping'，则执行以下操作
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        # 定义 bin number-to-cents 映射关系
        to_local_average_cents.cents_mapping = (
                20 * torch.arange(N_CLASS) + CONST).to(salience.device)  # noqa: F405

    # 如果 salience 的维度为 1
    if salience.ndim == 1:
        # 如果 center 为空，则将其设为 salience 中最大值的索引
        if center is None:
            center = int(torch.argmax(salience))
        # 设置起始位置为 center-4，结束位置为 center+5
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        # 截取 salience 的部分数据
        salience = salience[start:end]
        # 计算加权和
        product_sum = torch.sum(
            salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = torch.sum(salience)
        # 如果 salience 的最大值大于 thred，则返回加权平均值，否则返回 0
        return product_sum / weight_sum if torch.max(salience) > thred else 0
    # 如果 salience 的维度为 2
    if salience.ndim == 2:
        # 对 salience 的每一行进行计算，返回结果组成的张量
        return torch.Tensor([to_local_average_cents(salience[i, :], None, thred) for i in
                         range(salience.shape[0])]).to(salience.device)

    # 如果 salience 的维度既不是 1 也不是 2，则抛出异常
    raise Exception("label should be either 1d or 2d ndarray")
# 将音高显著性转换为维特比频率中心
def to_viterbi_cents(salience, thred=0.05):
    # 创建维特比转移矩阵
    if not hasattr(to_viterbi_cents, 'transition'):
        # 创建一个 N_CLASS x N_CLASS 的网格矩阵
        xx, yy = torch.meshgrid(range(N_CLASS), range(N_CLASS))  # noqa: F405
        # 计算转移矩阵的值，根据频率差异计算权重
        transition = torch.maximum(30 - abs(xx - yy), 0)
        # 对转移矩阵进行归一化，使每行和为1
        transition = transition / transition.sum(axis=1, keepdims=True)
        # 将转移矩阵保存在函数属性中，以便下次使用
        to_viterbi_cents.transition = transition

    # 将显著性转换为概率
    prob = salience.T
    prob = prob / prob.sum(axis=0)    

    # 执行维特比解码
    path = librosa.sequence.viterbi(prob.detach().cpu().numpy(), to_viterbi_cents.transition).astype(np.int64)

    # 返回转换后的结果
    return torch.Tensor([to_local_average_cents(salience[i, :], path[i], thred) for i in
                     range(len(path))]).to(salience.device)
```