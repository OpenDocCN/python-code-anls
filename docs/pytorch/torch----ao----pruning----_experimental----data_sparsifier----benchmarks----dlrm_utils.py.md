# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\benchmarks\dlrm_utils.py`

```
# mypy: allow-untyped-defs
# 导入PyTorch和相关模块，忽略类型检查
import torch
from dlrm_s_pytorch import DLRM_Net  # type: ignore[import]
import numpy as np  # type: ignore[import]
from dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo_offset  # type: ignore[import]
import zipfile
import os

# 定义 SparseDLRM 类，继承自 DLRM_Net
class SparseDLRM(DLRM_Net):
    """The SparseDLRM model is a wrapper around the DLRM_Net model that tries
    to use torch.sparse tensors for the features obtained after the ```interact_features()```
    call. The idea is to do a simple torch.mm() with the weight matrix of the first linear
    layer of the top layer.
    """
    def __init__(self, **args):
        super().__init__(**args)

    def forward(self, dense_x, lS_o, lS_i):
        # 应用 MLP 到稠密特征 dense_x
        x = self.apply_mlp(dense_x, self.bot_l)
        # 应用 embedding bag 到稀疏特征 lS_o, lS_i
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # 交互特征 x 和 ly
        z = self.interact_features(x, ly)

        # 将 z 转换为稀疏张量格式
        z = z.to_sparse_coo()
        # 执行 z 和顶层第一个线性层权重矩阵的矩阵乘法，并加上偏置
        z = torch.mm(z, self.top_l[0].weight.T).add(self.top_l[0].bias)
        # 对于顶层的其他层，逐层应用
        for layer in self.top_l[1:]:
            z = layer(z)

        return z


# 定义函数 get_valid_name，用于替换文件名中的 '.' 为 '_'
def get_valid_name(name):
    """Replaces '.' with '_' as names with '.' are invalid in data sparsifier
    """
    return name.replace('.', '_')


# 定义函数 get_dlrm_model，用于获取 DLRM 模型
def get_dlrm_model(sparse_dlrm=False):
    """Obtain dlrm model. The configs specified are based on the script in
    bench/dlrm_s_criteo_kaggle.sh. The same config is used to train the model
    for benchmarking on data sparsifier.
    """
    # DLRM 模型的配置信息
    dlrm_model_config = {
        'm_spa': 16,
        'ln_emb': np.array([1460, 583, 10131227, 2202608, 305, 24,
                            12517, 633, 3, 93145, 5683, 8351593,
                            3194, 27, 14992, 5461306, 10, 5652,
                            2173, 4, 7046547, 18, 15, 286181,
                            105, 142572], dtype=np.int32),
        'ln_bot': np.array([13, 512, 256, 64, 16]),
        'ln_top': np.array([367, 512, 256, 1]),
        'arch_interaction_op': 'dot',
        'arch_interaction_itself': False,
        'sigmoid_bot': -1,
        'sigmoid_top': 2,
        'sync_dense_params': True,
        'loss_threshold': 0.0,
        'ndevices': 1,
        'qr_flag': False,
        'qr_operation': 'mult',
        'qr_collisions': 4,
        'qr_threshold': 200,
        'md_flag': False,
        'md_threshold': 200,
        'weighted_pooling': None,
        'loss_function': 'bce'
    }
    # 根据 sparse_dlrm 参数决定使用 SparseDLRM 还是 DLRM_Net
    if sparse_dlrm:
        dlrm_model = SparseDLRM(**dlrm_model_config)
    else:
        dlrm_model = DLRM_Net(**dlrm_model_config)
    return dlrm_model


# 定义函数 dlrm_wrap，简化版的 dlrm_wrap() 函数，用于将输入张量移动到指定设备，但不进行前向传播
def dlrm_wrap(X, lS_o, lS_i, device, ndevices=1):
    """Rewritten simpler version of ```dlrm_wrap()``` found in dlrm_s_pytorch.py.
    This function simply moves the input tensors into the device and without the forward pass
    """
    # 如果设备数为1，将输入张量（或列表中的每个张量）移动到指定设备
    if ndevices == 1:
        # 如果输入是列表，将列表中的每个张量移动到指定设备；否则移动单个输入张量到指定设备
        lS_i = (
            [S_i.to(device) for S_i in lS_i]
            if isinstance(lS_i, list)
            else lS_i.to(device)
        )
        # 如果输出是列表，将列表中的每个张量移动到指定设备；否则移动单个输出张量到指定设备
        lS_o = (
            [S_o.to(device) for S_o in lS_o]
            if isinstance(lS_o, list)
            else lS_o.to(device)
        )
    # 返回移动后的输入张量、输出张量列表和输入张量列表
    return X.to(device), lS_o, lS_i
# 创建用于测试数据集的数据加载器函数
def make_test_data_loader(raw_data_file_path, processed_data_file):
    """Function to create dataset and dataloaders for the test dataset.
    Rewritten simpler version of ```make_criteo_and_loaders()``` from the dlrm_data_pytorch.py
    that makes the test dataset and dataloaders only for the ***kaggle criteo dataset***
    """
    # 创建一个 Kaggle Criteo 数据集对象，用于测试数据集
    test_data = CriteoDataset(
        "kaggle",                # 数据集类型为 Kaggle
        -1,                      # 特定设置，具体含义需要查看 CriteoDataset 类的文档
        0.0,                     # 特定设置，具体含义需要查看 CriteoDataset 类的文档
        "total",                 # 使用全部数据集
        "test",                  # 使用测试数据集部分
        raw_data_file_path,      # 原始数据文件路径
        processed_data_file,     # 处理后的数据文件路径
        False,                   # 不进行额外的数据预处理
        False,                   # 不进行额外的数据处理
    )
    
    # 创建用于加载测试数据集的 DataLoader 对象
    test_loader = torch.utils.data.DataLoader(
        test_data,                        # 使用上面创建的测试数据集对象
        batch_size=16384,                 # 批大小为 16384
        shuffle=False,                    # 不进行数据集洗牌
        num_workers=7,                    # 使用 7 个工作线程加载数据
        collate_fn=collate_wrapper_criteo_offset,  # 使用特定的数据集整理函数
        pin_memory=False,                 # 不使用 pin memory 加速
        drop_last=False,                  # 允许最后一个不完整的批次存在
    )
    
    return test_loader  # 返回创建的测试数据集 DataLoader 对象


# 获取模型函数
def fetch_model(model_path, device, sparse_dlrm=False):
    """This function unzips the zipped model checkpoint (if zipped) and returns a
    model object

    Args:
        model_path (str)
            path pointing to the zipped/raw model checkpoint file that was dumped in evaluate disk savings
        device (torch.device)
            device to which model needs to be loaded to
    """
    # 如果模型路径是一个 zip 文件
    if zipfile.is_zipfile(model_path):
        # 解压缩模型文件到与其同一目录下
        with zipfile.ZipFile(model_path, 'r', zipfile.ZIP_DEFLATED) as zip_ref:
            zip_ref.extractall(os.path.dirname(model_path))
            unzip_path = model_path.replace('.zip', '.ckpt')  # 获取解压后的模型文件路径
    else:
        unzip_path = model_path  # 否则，直接使用模型路径

    # 获取 DLRM 模型对象，可选择稀疏模型
    model = get_dlrm_model(sparse_dlrm=sparse_dlrm)
    # 加载模型状态字典到指定设备上
    model.load_state_dict(torch.load(unzip_path, map_location=device))
    model = model.to(device)  # 将模型移动到指定设备上
    model.eval()  # 设置模型为评估模式

    # 如果模型路径是一个 zip 文件，清理解压后的文件
    if zipfile.is_zipfile(model_path):
        os.remove(unzip_path)  # 删除解压后的模型文件

    return model  # 返回加载并准备好的模型对象
```