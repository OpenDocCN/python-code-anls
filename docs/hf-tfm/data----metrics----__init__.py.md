# `.\transformers\data\metrics\__init__.py`

```
# 导入警告模块
import warnings
# 导入检查Scikit-learn是否可用的函数和后端需求检查函数
from ...utils import is_sklearn_available, requires_backends

# 如果Scikit-learn可用，则导入相关函数
if is_sklearn_available():
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, matthews_corrcoef

# 警告消息，提示即将移除该度量方法，建议使用🤗 Evaluate库
DEPRECATION_WARNING = (
    "This metric will be removed from the library soon, metrics should be handled with the 🤗 Evaluate "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py"
)

# 简单准确度度量函数
def simple_accuracy(preds, labels):
    # 发出即将弃用警告
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    # 检查并要求Scikit-learn可用
    requires_backends(simple_accuracy, "sklearn")
    # 计算简单准确度
    return (preds == labels).mean()

# 准确度和F1度量函数
def acc_and_f1(preds, labels):
    # 发出即将弃用警告
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    # 检查并要求Scikit-learn可用
    requires_backends(acc_and_f1, "sklearn")
    # 计算准确度
    acc = simple_accuracy(preds, labels)
    # 计算F1分数
    f1 = f1_score(y_true=labels, y_pred=preds)
    # 返回准确度、F1分数和二者平均值
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

# 皮尔逊相关系数和斯皮尔曼等级相关系数度量函数
def pearson_and_spearman(preds, labels):
    # 发出即将弃用警告
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    # 检查并要求Scikit-learn可用
    requires_backends(pearson_and_spearman, "sklearn")
    # 计算皮尔逊相关系数
    pearson_corr = pearsonr(preds, labels)[0]
    # 计算斯皮尔曼等级相关系数
    spearman_corr = spearmanr(preds, labels)[0]
    # 返回皮尔逊相关系数、斯皮尔曼等级相关系数和二者平均值
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

# GLUE任务的度量函数
def glue_compute_metrics(task_name, preds, labels):
    # 发出即将弃用警告
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    # 检查并要求Scikit-learn可用
    requires_backends(glue_compute_metrics, "sklearn")
    # 检查预测值和标签值长度是否一致
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    # 根据任务名称执行不同的度量
    if task_name == "cola":
        # 对于CoLA任务，返回马修斯相关系数
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        # 对于SST-2任务，返回简单准确度
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        # 对于MRPC任务，返回准确度和F1分数
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        # 对于STS-B任务，返回皮尔逊相关系数和斯皮尔曼等级相关系数
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        # 对于QQP任务，返回准确度和F1分数
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        # 对于MNLI任务，返回MNLI准确度
        return {"mnli/acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        # 对于MNLI-MM任务，返回MNLI-MM准确度
        return {"mnli-mm/acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        # 对于QNLI任务，返回准确度
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        # 对于RTE任务，返回准确度
        return {"acc": simple_accuracy(preds, labels)}
```  
    # 如果任务名称为 "wnli"，则返回一个包含准确率的字典
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    # 如果任务名称为 "hans"，则返回一个包含准确率的字典
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    # 如果任务名称既不是 "wnli" 也不是 "hans"，则引发一个 KeyError 异常
    else:
        raise KeyError(task_name)
# 定义一个函数，用于计算 xnli 任务的指标
def xnli_compute_metrics(task_name, preds, labels):
    # 发出警告，提示该函数即将被弃用
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    # 检查是否安装了所需的后端库 "sklearn"
    requires_backends(xnli_compute_metrics, "sklearn")
    # 检查预测值和标签值的长度是否一致
    if len(preds) != len(labels):
        # 如果长度不一致，则抛出数值错误异常
        raise ValueError(f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}")
    # 如果任务名为 "xnli"，则返回准确率指标
    if task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        # 如果任务名不是 "xnli"，则抛出键错误异常
        raise KeyError(task_name)
```