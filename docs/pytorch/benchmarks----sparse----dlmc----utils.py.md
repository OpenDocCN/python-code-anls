# `.\pytorch\benchmarks\sparse\dlmc\utils.py`

```
import math  # 导入数学库
from pathlib import Path  # 导入路径操作模块

from scipy import sparse  # 导入稀疏矩阵处理库

import torch  # 导入PyTorch深度学习库


def to_coo_scipy(x):
    # 将稀疏张量转换为SciPy稀疏矩阵的COO格式
    indices_1 = x._indices().numpy()
    values_1 = x._values().numpy()
    return sparse.coo_matrix((values_1, (indices_1[0], indices_1[1])), shape=x.shape)


def sparse_grad_output(a, b):
    # 计算稀疏张量a和b的乘积
    c = torch.sparse.mm(a, b)
    if c.is_sparse:
        # 如果结果张量c为稀疏张量，创建一个与c同样形状的随机稀疏张量c2，并返回c2对c的稀疏掩码结果
        c2 = torch.rand_like(c.to_dense())
        return c2.sparse_mask(c.coalesce())
    else:
        # 如果结果张量c不是稀疏张量，创建一个与c同样形状的随机张量，并返回
        return torch.rand_like(c)


def read_matrix_params(path):
    # 从文件中读取矩阵的行数、列数和非零元素个数
    with open(path) as file:
        line = file.readline()
        nrows, ncols, nnz = (int(el) for el in line.split(", "))
        return (nrows, ncols), nnz


def csr_to_coo(indices, indptr, shape):
    # 将CSR格式的矩阵索引和行指针转换为COO格式的张量索引
    n_rows, n_cols = shape
    cols = indices
    rows = [0] * len(cols)
    for i in range(n_rows):
        for j in range(indptr[i], indptr[i + 1]):
            rows[j] = i
    return torch.tensor([rows, cols], dtype=torch.long)


def load_sparse_matrix(path, device):
    # 从文件中加载稀疏矩阵数据，并返回PyTorch稀疏COO张量
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(", "))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())

    index_pointers = list(index_pointers)
    indices = list(indices)
    data = torch.randn(nnz, dtype=torch.double)
    shape = (nrows, ncols)
    return torch.sparse_coo_tensor(
        csr_to_coo(indices, index_pointers, shape), data, shape, device=device
    )


def gen_vector(path, device):
    # 从文件中生成指定大小的随机向量
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(", "))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())
        return torch.randn(nrows, dtype=torch.double, device=device)


def gen_matrix(path, device):
    # 从文件中生成指定大小的随机矩阵
    with open(path) as file:
        nrows, ncols, nnz = (int(el) for el in file.readline().split(", "))
        index_pointers = (int(el) for el in file.readline().split())
        indices = (int(el) for el in file.readline().split())
        return torch.randn(nrows, ncols, dtype=torch.double, device=device)


def load_spmv_dataset(dataset_path, hidden_size, sparsity, device, n_limit=math.inf):
    """load_spmv_dataset loads a DLMC dataset for a sparse matrix-vector multiplication (SPMV) performance test.
    Args:
        dataset_path:
            path of the dataset from DLMC collection.
        hidden_size
            This value allows tensors of varying sizes.
        sparsity:
            This value allows tensors of varying sparsities.
        device:
            Whether to place the Tensor on a GPU or CPU.
        n_limit:
            This value allows a dataset with some limit size.
    """
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    files = path.glob("**/*.smtx")
    print(dataset_path, hidden_size, sparsity)
    index = 0
    x_files, y_files = [], []
    # 对于每个文件对象 f 在 files 中循环处理
    for f in files:
        # 如果索引 index 大于等于 n_limit，则跳出循环
        if index >= n_limit:
            break
        # 打印一个点符号，不换行
        print(".", end="")
        # 读取文件 f 的矩阵参数，返回大小和非零元素个数
        size, nnz = read_matrix_params(f.as_posix())
        # 如果矩阵的列数 size[1] 等于 hidden_size，则将文件路径添加到 x_files 列表中
        if size[1] == hidden_size:
            x_files.append(f.as_posix())
        # 如果矩阵的行数 size[0] 等于 hidden_size，则将文件路径添加到 y_files 列表中
        if size[0] == hidden_size:
            y_files.append(f.as_posix())
        # 索引加一
        index += 1
    # 打印换行符，结束本行点符号的输出
    print()

    # 对 x_files 和 y_files 中的路径进行并行处理
    for fx, fy in zip(x_files, y_files):
        # 加载稀疏矩阵文件 fx，并将其放在指定的设备上
        x = load_sparse_matrix(fx, device)
        # 生成文件 fy 的向量数据，并放在指定的设备上
        y = gen_vector(fy, device)
        # 使用生成器 yield 返回 x 和 y 的元组
        yield (x, y)
# 加载 DLMC 数据集用于稀疏矩阵-矩阵乘法（SPMM）性能测试
def load_spmm_dataset(
    dataset_path, hidden_size, sparsity, spmm_type, device, n_limit=math.inf
):
    """load_spmm_dataset loads a DLMC dataset for a sparse matrix-matrix multiplication (SPMM) performance test.
    Args:
        dataset_path:
            DLMC 数据集的路径。
        hidden_size:
            允许不同大小的张量。
        sparsity:
            允许不同稀疏度的张量。
        spmm_type:
            允许 `sparse@sparse` 或 `sparse@dense` 操作的张量。
        device:
            指定张量是在 GPU 还是 CPU 上。
        n_limit:
            允许指定数据集的最大大小限制。
    """
    # 构建当前稀疏度对应的文件夹路径
    current_folder_path = f"{dataset_path}/{sparsity}"
    path = Path(current_folder_path)
    # 获取所有.smtx文件的路径
    files = path.glob("**/*.smtx")
    # 输出用于检查的调试信息
    print(dataset_path, hidden_size, sparsity)
    index = 0
    x_files, y_files = [], []
    # 遍历文件列表
    for f in files:
        if index >= n_limit:
            break
        # 输出加载进度的点号
        print(".", end="")
        # 读取矩阵的大小和非零元素数量
        size, nnz = read_matrix_params(f.as_posix())
        # 根据矩阵的大小选择存储文件路径
        if size[1] == hidden_size:
            x_files.append(f.as_posix())
        if size[0] == hidden_size:
            y_files.append(f.as_posix())
        index += 1
    # 输出换行，结束加载进度信息
    print()

    # 逐一加载 x_files 和 y_files 中的文件，生成 (x, y) 对
    for fx, fy in zip(x_files, y_files):
        x = load_sparse_matrix(fx, device)
        # 根据 spmm_type 决定如何加载 y
        y = (
            gen_matrix(fy, device)
            if spmm_type == "sparse@dense"
            else load_sparse_matrix(fy, device)
        )
        yield (x, y)


def load_dlmc_dataset(
    dataset_path,
    operation,
    hidden_size,
    sparsity,
    device,
    requires_grad,
    n_limit=math.inf,
):
    """load_dlmc_dataset loads a DLMC dataset for a matmul performance test.
    Args:
        dataset_path:
            DLMC 数据集的路径。
        operation:
            允许 `sparse@sparse` | `sparse@dense` | `sparse@vector` 操作的张量。
        hidden_size:
            允许不同大小的张量。
        sparsity:
            允许不同稀疏度的张量。
        device:
            指定张量是在 GPU 还是 CPU 上。
        requires_grad:
            加载数据集以进行反向传播测试。
        n_limit:
            允许指定数据集的最大大小限制。
    """
    # 根据操作类型选择不同的数据集加载方式
    if operation == "sparse@sparse" or operation == "sparse@dense":
        collection = load_spmm_dataset(
            dataset_path, hidden_size, sparsity, operation, device, n_limit
        )
    elif operation == "sparse@vector":
        collection = load_spmv_dataset(
            dataset_path, hidden_size, sparsity, device, n_limit
        )
    # 用于存储 SciPy 变量和反向传播变量的空字典
    scipy_vars = {}
    backward_vars = {}
    # 遍历集合中的每对元素 x, y
    for x, y in collection:
        # 如果设备为 "cpu"，则创建 SciPy 变量字典
        if device == "cpu":
            # 如果 x 是稀疏张量，则将其转换为 COO 格式的 SciPy 稀疏矩阵，否则转换为 NumPy 数组
            scipy_vars = {
                "sx": to_coo_scipy(x) if x.is_sparse else x.numpy(),
                # 如果 y 是稀疏张量，则将其转换为 COO 格式的 SciPy 稀疏矩阵，否则转换为 NumPy 数组
                "sy": to_coo_scipy(y) if y.is_sparse else y.numpy(),
            }
        
        # 如果不需要梯度计算
        if not requires_grad:
            # 如果 x 是稀疏张量，则将其转换为密集张量，否则保持不变
            dx = x.to_dense() if x.is_sparse else x
            # 如果 y 是稀疏张量，则将其转换为密集张量，否则保持不变
            dy = y.to_dense() if y.is_sparse else y
        else:
            # 计算稀疏梯度输出
            c = sparse_grad_output(x, y)
            # 创建反向传播变量字典
            backward_vars = {
                "sparse_grad_output": c,
                # 如果 c 是稀疏张量，则将其转换为密集张量，否则保持不变
                "grad_output": c.to_dense() if c.is_sparse else c,
            }
            # 启用 x 和 y 的梯度计算
            x.requires_grad_(True)
            y.requires_grad_(True)
            # 将 x 转换为密集张量并分离其计算图，如果 x 是稀疏张量，则保持不变
            dx = x.to_dense().detach() if x.is_sparse else x.clone().detach()
            # 将 y 转换为密集张量并分离其计算图，如果 y 是稀疏张量，则保持不变
            dy = y.to_dense().detach() if y.is_sparse else y.clone().detach()
            # 启用 dx 和 dy 的梯度计算
            dx.requires_grad_(True)
            dy.requires_grad_(True)
        
        # 生成器返回包含 x, y, dx, dy 以及其他变量的字典
        yield {"x": x, "y": y, "dx": dx, "dy": dy, **scipy_vars, **backward_vars}
```