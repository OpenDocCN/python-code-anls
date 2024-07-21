# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\benchmarks\evaluate_forward_time.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库，包括类型提示被忽略的导入
from typing import Dict, List
import torch
from dlrm_s_pytorch import unpack_batch  # type: ignore[import]
import numpy as np  # type: ignore[import]
import time
from dlrm_utils import make_test_data_loader, fetch_model, dlrm_wrap  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import argparse

def run_forward(model, **batch):
    """The purpose of this function is to time the forward run of the model.
    The model forward happens a 100 times and each pass is timed. The average
    of this 100 runs is returned as avg_time.
    """
    # 初始化一个空列表来存储每次前向传播的时间
    time_list = []
    # 从输入的batch中获取数据
    X, lS_o, lS_i = batch['X'], batch['lS_o'], batch['lS_i']
    # 执行100次前向传播并测量时间
    for _ in range(100):
        start = time.time()
        # 使用torch.no_grad()上下文管理器避免计算梯度
        with torch.no_grad():
            model(X, lS_o, lS_i)
        end = time.time()
        time_taken = end - start
        time_list.append(time_taken)
    # 计算除第一次外的平均时间作为最终结果
    avg_time = np.mean(time_list[1:])
    return avg_time

def make_sample_test_batch(raw_data_path, processed_data_path, device):
    """Create the test_data_loader and sample a batch from it. This batch will be used
    to measure the forward pass of the model throughout this experiment.
    """
    # 创建测试数据加载器
    test_data_loader = make_test_data_loader(raw_data_path, processed_data_path)
    # 从数据加载器中获取一个批次的数据
    test_iter = iter(test_data_loader)
    test_batch = next(test_iter)
    # 解包批次数据
    X_test, lS_o_test, lS_i_test, _, _, _ = unpack_batch(test_batch)
    # 将数据移动到指定的设备上，并封装成字典形式返回
    X, lS_o, lS_i = dlrm_wrap(X_test, lS_o_test, lS_i_test, device)
    batch = {
        'X': X,
        'lS_o': lS_o,
        'lS_i': lS_i
    }
    return batch

def measure_forward_pass(sparse_model_metadata, device, sparse_dlrm, **batch):
    """Measures and tracks the forward pass of the model for all the sparsity levels, block shapes and norms
    available in sparse_model_metadata file.
    If sparse_dlrm=True, then the SparseDLRM model is loaded, otherwise the standard one is.
    """
    # 初始化一个字典来存储测量结果
    time_taken_dict: Dict[str, List] = {
        "norm": [],
        "sparse_block_shape": [],
        "sparsity_level": [],
        "time_taken": [],
    }
    # 从CSV文件中读取模型的元数据
    metadata = pd.read_csv(sparse_model_metadata)
    # 遍历元数据的每一行
    for _, row in metadata.iterrows():
        # 获取当前行的规范化、稀疏块形状和稀疏度水平
        norm, sbs, sl = row['norm'], row['sparse_block_shape'], row['sparsity_level']
        # 获取当前行的模型路径并加载模型
        model_path = row['path']
        model = fetch_model(model_path, device, sparse_dlrm=sparse_dlrm)
        # 测量当前模型在给定批次上的前向传播时间
        time_taken = run_forward(model, **batch)
        out_str = f"{norm}_{sbs}_{sl}={time_taken}"
        # 打印格式化的输出字符串
        print(out_str)
        # 将测量结果添加到时间字典中对应的列表中
        time_taken_dict["norm"].append(norm)
        time_taken_dict["sparse_block_shape"].append(sbs)
        time_taken_dict["sparsity_level"].append(sl)
        time_taken_dict["time_taken"].append(time_taken)
    # 创建包含测量结果的DataFrame
    time_df = pd.DataFrame(time_taken_dict)
    # 根据sparse_dlrm参数添加dlrm_type列
    if sparse_dlrm:
        time_df['dlrm_type'] = 'with_torch_sparse'
    else:
        time_df['dlrm_type'] = 'without_torch_sparse'
    # 返回包含测量结果的DataFrame
    return time_df

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加解析命令行参数的选项，指定原始数据文件名
    parser.add_argument('--raw-data-file', '--raw_data_file', type=str)
    # 添加解析命令行参数的选项，指定处理后的数据文件名
    parser.add_argument('--processed-data-file', '--processed_data_file', type=str)
    # 添加解析命令行参数的选项，指定稀疏模型元数据文件名
    parser.add_argument('--sparse-model-metadata', '--sparse_model_metadata', type=str)
    
    # 解析命令行参数，将结果存储在 args 中
    args = parser.parse_args()
    
    # 根据 CUDA 是否可用选择设备，若可用则选择 CUDA 设备，否则选择 CPU 设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 打印当前选择的设备（CUDA 或 CPU）
    print(device)
    
    # 调用函数创建一个样本测试批次，使用指定的原始数据文件、处理后的数据文件和设备
    batch = make_sample_test_batch(args.raw_data_file, args.processed_data_file, device)
    
    # 打印信息：稀疏 DLRM 模型的前向传播时间
    print("Forward Time for Sparse DLRM")
    # 测量稀疏 DLRM 模型的前向传播时间，并将结果存储在 sparse_dlrm_time_df 中
    sparse_dlrm_time_df = measure_forward_pass(args.sparse_model_metadata, device, sparse_dlrm=True, **batch)
    # 打印稀疏 DLRM 模型前向传播时间的数据框
    print(sparse_dlrm_time_df)
    
    # 打印信息：普通 DLRM 模型的前向传播时间
    print("Forward Time for Normal DLRM")
    # 测量普通 DLRM 模型的前向传播时间，并将结果存储在 norm_dlrm_time_df 中
    norm_dlrm_time_df = measure_forward_pass(args.sparse_model_metadata, device, sparse_dlrm=False, **batch)
    # 打印普通 DLRM 模型前向传播时间的数据框
    print(norm_dlrm_time_df)
    
    # 将稀疏 DLRM 和普通 DLRM 模型前向传播时间的数据框合并为一个数据框
    forward_time_all = pd.concat([sparse_dlrm_time_df, norm_dlrm_time_df])
    # 将合并后的数据框保存为 CSV 文件，文件名为 'dlrm_forward_time_info.csv'，并且不保存索引
    forward_time_all.to_csv('dlrm_forward_time_info.csv', index=False)
```