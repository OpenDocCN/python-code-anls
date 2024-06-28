# `.\data\metrics\__init__.py`

```
# 导入警告模块，用于发出关于未来版本不推荐使用的警告信息
import warnings

# 从工具包中导入检查函数和后端依赖的函数
from ...utils import is_sklearn_available, requires_backends

# 如果检测到 sklearn 可用，则导入相关的指标函数
if is_sklearn_available():
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, matthews_corrcoef

# 警告信息，指出当前指标将在未来版本中移除，推荐使用 Evaluate 库处理指标
DEPRECATION_WARNING = (
    "This metric will be removed from the library soon, metrics should be handled with the 🤗 Evaluate "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py"
)

# 计算简单准确率的函数定义，发出未来版本警告，并检查 sklearn 后端依赖
def simple_accuracy(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_backends(simple_accuracy, "sklearn")
    return (preds == labels).mean()

# 计算准确率和 F1 分数的函数定义，发出未来版本警告，并检查 sklearn 后端依赖
def acc_and_f1(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_backends(acc_and_f1, "sklearn")
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

# 计算 Pearson 相关系数和 Spearman 秩相关系数的函数定义，发出未来版本警告，并检查 sklearn 后端依赖
def pearson_and_spearman(preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_backends(pearson_and_spearman, "sklearn")
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

# 计算 GLUE 任务中指标的函数定义，发出未来版本警告，并检查 sklearn 后端依赖
def glue_compute_metrics(task_name, preds, labels):
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    requires_backends(glue_compute_metrics, "sklearn")
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"mnli/acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"mnli-mm/acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    # 如果任务名为 "wnli"，则返回一个包含准确率的字典，使用 simple_accuracy 函数计算
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    # 如果任务名为 "hans"，则返回一个包含准确率的字典，使用 simple_accuracy 函数计算
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    # 如果任务名既不是 "wnli" 也不是 "hans"，则抛出 KeyError 异常
    else:
        raise KeyError(task_name)
# 计算 xnli 任务的评估指标
def xnli_compute_metrics(task_name, preds, labels):
    # 发出警告，指示此函数即将弃用
    warnings.warn(DEPRECATION_WARNING, FutureWarning)
    # 确保需要的后端库被加载，这里是 sklearn
    requires_backends(xnli_compute_metrics, "sklearn")
    # 检查预测值和标签的长度是否一致，如果不一致则引发数值错误
    if len(preds) != len(labels):
        raise ValueError(f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}")
    # 如果任务名为 "xnli"，返回精度（accuracy）指标的字典
    if task_name == "xnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        # 否则引发任务名错误
        raise KeyError(task_name)
```