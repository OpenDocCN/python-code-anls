# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\benchmarks\evaluate_disk_savings.py`

```py
# mypy: allow-untyped-defs
# 引入类型检查相关库，允许未指定类型的函数定义
from typing import Dict, List
# 引入PyTorch库
import torch
# 引入时间模块
import time
# 引入PyTorch稀疏化模块
from torch.ao.pruning._experimental.data_sparsifier import DataNormSparsifier
# 引入操作系统相关功能
import os
# 引入dlrm_utils工具函数
from dlrm_utils import get_dlrm_model, get_valid_name  # type: ignore[import]
# 引入复制功能模块
import copy
# 引入ZIP文件处理模块
import zipfile
# 从zipfile中引入ZipFile类
from zipfile import ZipFile
# 引入Pandas数据处理库
import pandas as pd  # type: ignore[import]
# 引入命令行参数解析模块
import argparse


def create_attach_sparsifier(model, **sparse_config):
    """Create a DataNormSparsifier and the attach it to the model embedding layers

    Args:
        model (nn.Module)
            layer of the model that needs to be attached to the sparsifier
        sparse_config (Dict)
            Config to the DataNormSparsifier. Should contain the following keys:
                - sparse_block_shape
                - norm
                - sparsity_level
    """
    # 创建DataNormSparsifier对象
    data_norm_sparsifier = DataNormSparsifier(**sparse_config)
    # 遍历模型的命名参数
    for name, parameter in model.named_parameters():
        # 如果参数名包含'emb_l'，表示是嵌入层参数
        if 'emb_l' in name:
            # 获取有效的参数名称
            valid_name = get_valid_name(name)
            # 将数据添加到稀疏化对象中
            data_norm_sparsifier.add_data(name=valid_name, data=parameter)
    return data_norm_sparsifier


def save_model_states(state_dict, sparsified_model_dump_path, save_file_name, sparse_block_shape, norm, zip=True):
    """Dumps the state_dict() of the model.

    Args:
        state_dict (Dict)
            The state_dict() as dumped by dlrm_s_pytorch.py. Only the model state will be extracted
            from this dictionary. This corresponds to the 'state_dict' key in the state_dict dictionary.
            >>> model_state = state_dict['state_dict']
        save_file_name (str)
            The filename (not path) when saving the model state dictionary
        sparse_block_shape (Tuple)
            The block shape corresponding to the data norm sparsifier. **Used for creating save directory**
        norm (str)
            type of norm (L1, L2) for the datanorm sparsifier. **Used for creating save directory**
        zip (bool)
            if True, the file is zip-compressed.
    """
    # 模型状态保存路径的文件夹名
    folder_name = os.path.join(sparsified_model_dump_path, str(norm))
    
    # 保存仅模型状态
    folder_str = f"config_{sparse_block_shape}"
    model_state = state_dict['state_dict']
    model_state_path = os.path.join(folder_name, folder_str, save_file_name)
    
    # 确保保存路径存在
    os.makedirs(os.path.dirname(model_state_path), exist_ok=True)
    # 保存模型状态到指定路径
    torch.save(model_state, model_state_path)
    
    # 如果需要压缩为ZIP文件
    if zip:
        # 指定压缩文件路径，并创建ZIP文件
        zip_path = model_state_path.replace('.ckpt', '.zip')
        with ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip:
            zip.write(model_state_path, save_file_name)
        # 移除未压缩的文件
        os.remove(model_state_path)
        # 更新模型状态路径为压缩文件路径
        model_state_path = zip_path
    
    # 获取模型状态路径的绝对路径
    model_state_path = os.path.abspath(model_state_path)
    # 获取文件大小（单位MB）
    file_size = os.path.getsize(model_state_path)
    file_size = file_size >> 20  # size in mb
    return model_state_path, file_size


def sparsify_model(path_to_model, sparsified_model_dump_path):
    """Sparsifies the embedding layers of the dlrm model for different sparsity levels, norms and block shapes
    using the DataNormSparsifier.
    The function tracks the step time of the sparsifier and the size of the compressed checkpoint and collates
    it into a csv.

    Note::
        This function dumps a csv sparse_model_metadata.csv in the current directory.

    Args:
        path_to_model (str)
            path to the trained criteo model ckpt file
        sparsity_levels (List of float)
            list of sparsity levels to be sparsified on
        norms (List of str)
            list of norms to be sparsified on
        sparse_block_shapes (List of tuples)
            List of sparse block shapes to be sparsified on
    """
    # Define sparsity levels from 0.0 to 1.0 with increments of 0.1, and additional specific levels
    sparsity_levels = [sl / 10 for sl in range(0, 10)]
    sparsity_levels += [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

    # Define norms to apply: L1 and L2
    norms = ["L1", "L2"]
    
    # Define sparse block shapes: (1, 1) and (1, 4)
    sparse_block_shapes = [(1, 1), (1, 4)]

    # Determine device type based on GPU availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Print configurations for sparsity levels, sparse block shapes, and norms
    print("Running for sparsity levels - ", sparsity_levels)
    print("Running for sparse block shapes - ", sparse_block_shapes)
    print("Running for norms - ", norms)

    # Retrieve the original DLRM model
    orig_model = get_dlrm_model()

    # Load the state dictionary from the specified model checkpoint path
    saved_state = torch.load(path_to_model, map_location=device)
    orig_model.load_state_dict(saved_state['state_dict'])

    # Move the original model to the appropriate device (CPU or GPU)
    orig_model = orig_model.to(device)

    # Initialize an empty dictionary to store step times
    step_time_dict = {}

    # Initialize a dictionary to store statistical information with predefined keys
    stat_dict: Dict[str, List] = {'norm': [], 'sparse_block_shape': [], 'sparsity_level': [],
                                  'step_time_sec': [], 'zip_file_size': [], 'path': []}

    # Iterate over each norm and sparse block shape combination
    for norm in norms:
        for sbs in sparse_block_shapes:
            # Skip combination of L2 norm with (1, 1) block shape
            if norm == "L2" and sbs == (1, 1):
                continue
            
            # Iterate over each sparsity level
            for sl in sparsity_levels:
                # Create a deep copy of the original model
                model = copy.deepcopy(orig_model)
                
                # Create a sparsifier object with specified parameters
                sparsifier = create_attach_sparsifier(model, sparse_block_shape=sbs, norm=norm, sparsity_level=sl)

                # Measure the time taken for one step of sparsification
                t1 = time.time()
                sparsifier.step()
                t2 = time.time()

                # Calculate and store the step time in seconds
                step_time = t2 - t1
                norm_sl = f"{norm}_{sbs}_{sl}"
                print(f"Step Time for {norm_sl}=: {step_time} s")

                # Record the step time in the step_time_dict
                step_time_dict[norm_sl] = step_time

                # Finalize sparsification (apply mask)
                sparsifier.squash_mask()

                # Update the state dictionary with sparsified model states
                saved_state['state_dict'] = model.state_dict()
                
                # Generate file name and save the sparsified model states
                file_name = f'criteo_model_norm={norm}_sl={sl}.ckpt'
                state_path, file_size = save_model_states(saved_state, sparsified_model_dump_path, file_name, sbs, norm=norm)

                # Append statistics to the stat_dict for later CSV generation
                stat_dict['norm'].append(norm)
                stat_dict['sparse_block_shape'].append(sbs)
                stat_dict['sparsity_level'].append(sl)
                stat_dict['step_time_sec'].append(step_time)
                stat_dict['zip_file_size'].append(file_size)
                stat_dict['path'].append(state_path)

    # Convert stat_dict to a Pandas DataFrame
    df = pd.DataFrame(stat_dict)

    # Define the filename for the CSV output
    filename = 'sparse_model_metadata.csv'
    # 将DataFrame保存为CSV文件，不包含索引列
    df.to_csv(filename, index=False)
    
    # 打印保存成功的消息，包括文件名
    print(f"Saved sparsified metadata file in {filename}")
# 如果这个脚本作为主程序运行，执行以下操作
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数：模型路径，类型为字符串
    parser.add_argument('--model-path', '--model_path', type=str)
    # 添加命令行参数：稀疏化后模型的保存路径，类型为字符串
    parser.add_argument('--sparsified-model-dump-path', '--sparsified_model_dump_path', type=str)
    # 解析命令行参数，并将它们存储在args变量中
    args = parser.parse_args()

    # 调用sparsify_model函数，传入模型路径和稀疏化后模型保存路径作为参数
    sparsify_model(args.model_path, args.sparsified_model_dump_path)
```