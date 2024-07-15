# `.\Chat-Haruhi-Suzumiya\yuki_builder\audio_feature_ext\utils\utility.py`

```py
# 导入用于解析命令行参数的库
import distutils.util

# 导入numpy库，并为其命名空间起别名为np
import numpy as np
# 从tqdm库中导入tqdm函数
from tqdm import tqdm


# 打印命令行参数的配置信息
def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    # 遍历并按字母顺序打印参数和对应的值
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


# 添加命令行参数到参数解析器中
def add_arguments(argname, type, default, help, argparser, **kwargs):
    # 如果参数类型是布尔型，则使用distutils.util.strtobool进行类型转换
    type = distutils.util.strtobool if type == bool else type
    # 向参数解析器添加参数，包括参数名、默认值、类型、帮助信息等
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


# 根据预测分数和真实标签计算最佳准确率和最优阈值
def cal_accuracy_threshold(y_score, y_true):
    # 将预测分数和真实标签转换为numpy数组
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_accuracy = 0
    best_threshold = 0
    # 迭代从0到99，计算不同阈值下的准确率
    for i in tqdm(range(0, 100)):
        threshold = i * 0.01
        y_test = (y_score >= threshold)
        # 计算当前阈值下的准确率
        acc = np.mean((y_test == y_true).astype(int))
        # 更新最佳准确率和最优阈值
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold


# 根据预测分数和真实标签计算指定阈值下的准确率
def cal_accuracy(y_score, y_true, threshold=0.5):
    # 将预测分数和真实标签转换为numpy数组
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    # 根据指定的阈值计算预测结果
    y_test = (y_score >= threshold)
    # 计算准确率
    accuracy = np.mean((y_test == y_true).astype(int))
    return accuracy


# 计算两个向量的余弦相似度
def cosin_metric(x1, x2):
    # 计算向量x1和x2的点积，再除以它们的范数乘积，得到余弦相似度
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
```