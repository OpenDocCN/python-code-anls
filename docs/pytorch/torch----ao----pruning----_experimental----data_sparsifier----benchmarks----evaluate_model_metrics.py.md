# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\benchmarks\evaluate_model_metrics.py`

```
# mypy: allow-untyped-defs
# 引入类型注解所需的模块
from typing import Dict, List
# 引入PyTorch库
import torch
# 从dlrm_s_pytorch中导入unpack_batch函数，忽略类型检查
from dlrm_s_pytorch import unpack_batch  # type: ignore[import]
# 导入NumPy库，忽略类型检查
import numpy as np  # type: ignore[import]
# 导入Sklearn库，忽略类型检查
import sklearn  # type: ignore[import]
# 从dlrm_utils中导入make_test_data_loader, dlrm_wrap, fetch_model函数，忽略类型检查
from dlrm_utils import make_test_data_loader, dlrm_wrap, fetch_model  # type: ignore[import]
# 导入Pandas库，忽略类型检查
import pandas as pd  # type: ignore[import]
# 导入argparse模块，用于命令行解析
import argparse

# 定义函数：对测试数据集进行推断和评估
def inference_and_evaluation(dlrm, test_dataloader, device):
    """Perform inference and evaluation on the test dataset.
    The function returns the dictionary that contains evaluation metrics such as accuracy, f1, auc,
    precision, recall.
    Note: This function is a rewritten version of ```inference()``` present in dlrm_s_pytorch.py

    Args:
        dlrm (nn.Module)
            dlrm model object
        test_data_loader (torch dataloader):
            dataloader for the test dataset
        device (torch.device)
            device on which the inference happens
    """
    # 计算测试数据集的批次数
    nbatches = len(test_dataloader)
    # 初始化分数和目标列表
    scores = []
    targets = []

    # 遍历测试数据集的每个批次
    for i, testBatch in enumerate(test_dataloader):
        # 如果用户设置了nbatches并且超过了设定值，则提前退出循环
        if nbatches > 0 and i >= nbatches:
            break

        # 解包测试批次数据
        X_test, lS_o_test, lS_i_test, T_test, _, _ = unpack_batch(
            testBatch
        )
        # 执行前向传播
        X_test, lS_o_test, lS_i_test = dlrm_wrap(X_test, lS_o_test, lS_i_test, device, ndevices=1)

        # 获取模型的输出
        Z_test = dlrm(X_test, lS_o_test, lS_i_test)
        # 将输出转换为NumPy数组
        S_test = Z_test.detach().cpu().numpy()  # numpy array
        T_test = T_test.detach().cpu().numpy()  # numpy array
        # 将得分和目标添加到列表中
        scores.append(S_test)
        targets.append(T_test)

    # 合并所有批次的得分和目标
    scores = np.concatenate(scores, axis=0)
    targets = np.concatenate(targets, axis=0)

    # 定义评估指标的字典
    metrics = {
        "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "ap": sklearn.metrics.average_precision_score,
        "roc_auc": sklearn.metrics.roc_auc_score,
        "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
            y_true=y_true, y_pred=np.round(y_score)
        ),
        "log_loss": lambda y_true, y_score: sklearn.metrics.log_loss(
            y_true=y_true, y_pred=y_score
        )
    }

    # 初始化存储所有评估指标结果的字典
    all_metrics = {}
    # 遍历每个评估指标和相应的函数，并计算评估结果
    for metric_name, metric_function in metrics.items():
        all_metrics[metric_name] = round(metric_function(targets, scores), 3)

    # 返回所有评估指标的结果字典
    return all_metrics


def evaluate_metrics(test_dataloader, sparse_model_metadata):
    """Evaluates the metrics the sparsified metrics for the dlrm model on various sparsity levels,
    block shapes and norms. This function evaluates the model on the test dataset and dumps
    """
    Load sparse model metadata from a CSV file and evaluate each model's performance.
    Save evaluation metrics in a CSV file [model_performance.csv]
    """
    # 从稀疏模型元数据的 CSV 文件中读取元数据
    metadata = pd.read_csv(sparse_model_metadata)
    
    # 根据 GPU 是否可用选择设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 初始化一个空字典，用于存储各种评估指标的列表
    metrics_dict: Dict[str, List] = {
        "norm": [],
        "sparse_block_shape": [],
        "sparsity_level": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "accuracy": [],
        "log_loss": []
    }
    
    # 遍历元数据的每一行
    for _, row in metadata.iterrows():
        # 从当前行中获取 norm, sparse_block_shape 和 sparsity_level 的值
        norm, sbs, sl = row['norm'], row['sparse_block_shape'], row['sparsity_level']
        # 获取模型文件的路径
        model_path = row['path']
        # 根据路径和设备加载模型
        model = fetch_model(model_path, device)
    
        # 对模型进行推断和评估，获取评估指标
        model_metrics = inference_and_evaluation(model, test_dataloader, device)
        # 生成当前模型的唯一键，形如 "norm_value_sparse_block_shape_value_sparsity_level_value"
        key = f"{norm}_{sbs}_{sl}"
        # 打印当前模型的键和其评估指标
        print(key, "=", model_metrics)
    
        # 将当前模型的 norm, sparse_block_shape 和 sparsity_level 添加到 metrics_dict 对应的列表中
        metrics_dict['norm'].append(norm)
        metrics_dict['sparse_block_shape'].append(sbs)
        metrics_dict['sparsity_level'].append(sl)
    
        # 将模型的各项评估指标添加到 metrics_dict 中对应的列表中
        for key, value in model_metrics.items():
            if key in metrics_dict:
                metrics_dict[key].append(value)
    
    # 根据 metrics_dict 创建 DataFrame
    sparse_model_metrics = pd.DataFrame(metrics_dict)
    # 打印稀疏模型的评估指标 DataFrame
    print(sparse_model_metrics)
    
    # 将稀疏模型的评估指标保存到 CSV 文件中
    filename = 'sparse_model_metrics.csv'
    sparse_model_metrics.to_csv(filename, index=False)
    # 打印保存文件的消息
    print(f"Model metrics file saved to {filename}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    parser.add_argument('--raw-data-file', '--raw_data_file', type=str)
    # 添加命令行参数，用于指定原始数据文件的路径

    parser.add_argument('--processed-data-file', '--processed_data_file', type=str)
    # 添加命令行参数，用于指定处理后数据文件的路径

    parser.add_argument('--sparse-model-metadata', '--sparse_model_metadata', type=str)
    # 添加命令行参数，用于指定稀疏模型的元数据文件路径

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在args对象中

    # Fetch test data loader
    # 获取测试数据加载器，使用命令行参数中指定的原始数据文件和处理后数据文件
    test_dataloader = make_test_data_loader(args.raw_data_file, args.processed_data_file)

    # Evaluate metrics
    # 评估模型性能指标，使用测试数据加载器和命令行参数中指定的稀疏模型元数据文件
    evaluate_metrics(test_dataloader, args.sparse_model_metadata)
```