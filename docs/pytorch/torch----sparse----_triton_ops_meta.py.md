# `.\pytorch\torch\sparse\_triton_ops_meta.py`

```
# mypy: allow-untyped-defs
"""Provides optimal triton kernel parameters.

Aim
---

The usage of optimal triton kernel parameters may increase the
performance of operations several times. For example, for large tensor
shapes, the usage of a bsr tensor as mat1 argument in addmm-based
operations typically outperforms the corresponding operation with
strided-only inputs when the blocked representation of a tensor
provides a better alignement with memory access than what the strided
representation would provide.

Pre-computed kernel parameters
------------------------------

This script finds and stores the optimal triton kernel parameters for
a specific set of shape configurations. For instance, the set of shape
configurations of the bsr_dense_addmm kernel is defined as

  input, out: M x N strided tensor
  mat1: M x K bsr tensor with blocksize (BM, BK) and given sparsity
  mat2: M x N strided tensor
  dtype = float16, bfloat16, float32
  sparsity = 0.5
  M = 256, 512, ..., 16384
  K = M
  N = 256, 512, ..., 131072
  BM = 16, 32, ..., 128
  BK = BM
  alpha = 1
  beta = 0, 1
  GPUs: NVIDIA A100-SXM4-80GB

Approximations
--------------

It is practically infeasible to pre-compute optimal kernel parameter
for all possible shape configurations as well as for all existing
GPUs. Therefore, we'll assume that the pre-computed optimal parameters
are good enough approximations when
1) the used GPU is any of NVIDIA A100 Tensor Core GPUs,
2) the actual sparsity of mat1 is different from sparsity value 0.5.

If a particular shape configuration does not fall in the set of
pre-computed kernel parameters, or it does not match with the listed
approximations above, or the used GPU device is not a NVIDIA A100 GPU,
then a reference set of triton kernel parameters will be used when
executing operations. The reference kernel parameters are defined in
torch/sparse/_triton_ops.py, see bsr_dense_addmm_meta function, for
instance.

Computing optimal kernel parameters
-----------------------------------

If the approximations listed above are unacceptable, e.g. when one
seeks a maximal performance possible, the optimal kernel parameters
for a particular GPU can be computed by simply running this script in
the pytorch developement tree::

  cd /path/to/pytorch
  python setup.py develop
  python torch/sparse/_triton_ops_meta.py

This will compute the optimal kernel parameters for the GPU device
available in the host system for all shape configurations listed in
"Pre-computed kernel parameters" above. The results will be stored in
the database of kernel parameters. Currently, this database is defined
as this module (see "BEGIN GENERATED DATA" comment below) that will be
modified when the script is run. Create a pytorch PR with the
corresponding modifications in this file to make the computed optimal
kernel parameters available for other users as pre-computed kernel
parameters.

Moreover, one can compute the optimal kernel parameters for a specific
"""

# The script continues beyond this point
"""
__all__ = ["get_meta", "tune_bsr_dense_addmm"]  # 定义模块对外公开的接口列表

import inspect  # 导入用于获取对象信息的模块
import itertools  # 导入用于生成迭代器的模块
import re  # 导入用于正则表达式操作的模块
import warnings  # 导入警告处理模块
from typing import Any, Dict  # 导入用于类型提示的工具

import torch  # 导入PyTorch库
from torch.hub import tqdm  # 从torch.hub中导入进度条模块
from torch.testing import make_tensor  # 导入用于生成测试张量的函数


def get_meta(op, key, device_name=None, version=(0, torch.float16, 0.5), exact=False):
    """Return triton kernel meta parameters of the specified op and its inputs key.

    Parameters
    ----------
    op (str): The name of an operation that implementation uses meta parameters.
    key (tuple): A tuple of op input parameters, e.g. shapes, etc.
    device_name (optional, str): The name of a device for which op
      parameters are provided.
    version (optional, hashable): Specifies the version of parameters.
    exact (optional, bool): When True, the returned data (if
      available) corresponds exactly to the specified device_name and
      version information. Otherwise, if the corresponding data is not
      available but there exists a data set that is computed for a
      similar GPU device, then this data set will be returned.

    Returns
    -------
    result (dict): The requested mapping of parameter names and
      values, or None when no data is available. If the input `key`
      contains `"*"`, the result will be a dictionary of keys and
      mappings that match with the given `key`.
    """
    if device_name is None:
        device_name = torch.cuda.get_device_name()  # 如果设备名称未指定，使用当前CUDA设备的名称

    op_data = _operation_device_version_data.get((op, device_name, version))  # 获取操作、设备和版本对应的元数据
    # 如果没有提供操作数据并且不要求精确匹配
    if op_data is None and not exact:
        # 如果缺少操作数据，可能是因为使用了与已计算出最佳元参数的模型稍有不同的 GPU 模型。
        # 在接下来的代码中，我们假设有一组 GPU 模型，它们都具有类似的最佳元参数集。
        
        # 如果设备名称匹配到 "NVIDIA A100" 开头的模式
        if re.match(r"NVIDIA A100[^\d]", device_name) is not None:
            # 将设备名称更改为具体型号 "NVIDIA A100-SXM4-80GB"
            device_name = "NVIDIA A100-SXM4-80GB"
        else:
            # 如果设备名称不匹配任何已知模式，则直接返回
            return
        
        # 根据操作、设备名称和版本号获取操作设备版本数据
        op_data = _operation_device_version_data.get((op, device_name, version))
    
    # 如果操作数据仍然为空，则直接返回
    if op_data is None:
        return

    # 用于存储匹配的数据的空字典
    matching_data = {}

    # 如果键中包含通配符 "*"
    if "*" in key:
        # 遍历操作数据中的每个键
        for op_key in op_data:
            # 如果通配符 "*" 所在位置不是通配符，并且对应位置的字符不相等，则跳过当前键
            if [None for k1, k2 in zip(op_key, key) if k2 != "*" and k1 != k2]:
                continue
            # 将匹配的操作键及其对应的数据存入匹配数据字典中
            matching_data[op_key] = op_data[op_key]
    else:
        # 否则，直接获取键对应的值
        values = op_data.get(key)
        if values is not None:
            matching_data[key] = values
    
    # 用于存储匹配的元数据的空字典
    matching_meta = {}

    # 遍历匹配的数据字典中的每个操作键及其对应的值
    for op_key, values in matching_data.items():
        # 根据不同的操作类型，创建相应的元数据字典
        if op == "scatter_mm":
            names = (
                "GROUP_SIZE",
                "SPLIT_N",
                "TILE_M",
                "TILE_N",
                "num_stages",
                "num_warps",
            )
            meta = dict(zip(names, values))
        elif op == "bsr_dense_addmm":
            meta = dict(
                zip(("GROUP_SIZE_ROW", "SPLIT_N", "num_stages", "num_warps"), values)
            )
        else:
            # 如果操作类型不在已知的处理范围内，则抛出未实现错误
            raise NotImplementedError(f"names for {op=}")
        
        # 如果键中不包含通配符 "*", 直接返回当前的元数据
        if "*" not in key:
            return meta
        
        # 将当前操作键及其对应的元数据存入匹配的元数据字典中
        matching_meta[op_key] = meta

    # 如果键中包含通配符 "*", 直接返回所有匹配的元数据字典
    if "*" in key:
        return matching_meta
# 更新操作参数数据库的函数，根据操作名、设备名和版本号更新对应的键值对
def update(op, device_name, version, key, value):
    # 如果 (op, device_name, version) 存在于 _operation_device_version_data 中
    if (op, device_name, version) in _operation_device_version_data:
        # 如果指定键的当前值与新值相同，则直接返回，不进行更新
        if _operation_device_version_data[op, device_name, version].get(key) == value:
            return
        # 否则更新指定键的值为新值
        _operation_device_version_data[op, device_name, version][key] = value
    else:
        # 否则将 (op, device_name, version) 作为新键，并赋值为包含指定键值对的字典
        _operation_device_version_data[op, device_name, version] = {key: value}


# 存储当前运行时数据库状态到模块文件的函数
def dump():
    # 获取当前函数 dump 所在的文件路径
    current_file = inspect.getfile(dump)
    # 打开当前文件以读取内容
    f = open(current_file)
    current_content = f.read()
    f.close()
    # 定义生成数据开始和结束标记
    begin_data_str = "# BEGIN GENERATED DATA\n"
    end_data_index = current_content.find("    # END GENERATED DATA\n")
    begin_data_index = current_content.find(begin_data_str)
    # 如果找不到开始或结束标记，则发出警告并返回
    if begin_data_index == -1 or end_data_index == -1:
        warnings.warn(
            f"{current_file} cannot be updated:"
            " BEGIN/END GENERATED DATA comment blocks appear to be corrupted"
        )
        return

    # 定义用于排序键的函数
    def sort_key(key):
        op, device_name, version = key
        # 将版本号中的 torch.dtype 转换为字符串以进行排序
        version = tuple(
            (str(item) if isinstance(item, torch.dtype) else item) for item in version
        )
        return (op, device_name, version)

    # 分割当前内容，以便在开始和结束标记之间插入新生成的数据
    part1 = current_content[: begin_data_index + len(begin_data_str)]
    part2 = current_content[end_data_index:]
    data_part = []
    # 遍历 _operation_device_version_data 中的键，并按照 sort_key 函数排序
    for op_key in sorted(_operation_device_version_data, key=sort_key):
        # 构建每个操作键对应的数据部分，并使用 repr 转换为字符串表示形式
        data_part.append("    " + repr(op_key).replace("'", '"') + ": {")
        op_data = _operation_device_version_data[op_key]
        # 遍历每个操作键的数据部分，并构建以键和值对表示的字符串
        for key in sorted(op_data):
            data_part.append(f"        {key}: {op_data[key]},")
        data_part.append("    },")
    # 构建新的文件内容，包括开始部分、新生成的数据部分和结束部分
    new_content = part1 + "\n".join(data_part) + "\n" + part2
    # 如果当前内容与新内容不同，则将新内容写回到文件中
    if current_content != new_content:
        f = open(current_file, "w")
        f.write(new_content)
        f.close()


# 最小化目标函数的参数字典的函数
def minimize(
    target_func,
    initial_parameters,
    reference_parameters,
    step_func,
    max_step=2,
    verbose=False,
    all_values=None,
):
    """Find a dict of parameters that minimizes the target function using
    the initial dict of parameters and a step function that progresses
    a specified parameter in a dict of parameters.

    Parameters
    ----------
    target_func (callable): a functional with the signature
      ``target_func(parameters: dict) -> float``
    initial_parameters (dict): a set of parameters used as an initial
      value to the minimization process.
    reference_parameters (dict): a set of parameters used as an
      reference value with respect to which the speed up is computed.
    """
    step_func (callable): a functional with the signature
      ``step_func(parameter_name:str, parameter_value:int, direction:int, parameters:dict) -> int``
      that increments or decrements (when ``direction`` is positive or
      negative, respectively) the parameter with given name and value.
      When return value is equal to ``parameter_value``, it means that
      no step along the given direction can be made.

    Returns
    -------
    parameters (dict): a set of parameters that minimizes the target
      function.
    speedup_incr (float): a speedup change given in percentage.
    timing (float): the value of the target function at the parameters.
    sensitivity_message (str): a message containing sensitivity.
      information of parameters around the target function minimizer.
    """

    # 将参数字典转换为可哈希的元组键
    def to_key(parameters):
        return tuple(parameters[k] for k in sorted(parameters))

    # 将哈希的元组键还原为参数字典
    def from_key(key, parameters):
        return dict(zip(sorted(parameters), key))

    # 如果未提供 all_values，则初始化为空字典
    if all_values is None:
        all_values = dict()

    # 创建方向列表，从 -max_step 到 max_step
    directions = list(range(-max_step, max_step + 1))
    # 对初始参数按名称排序
    names = sorted(initial_parameters)
    # 生成所有可能的方向组合
    all_directions = []
    for d_tuple in itertools.product(*((directions,) * len(names))):
        dist = sum(map(abs, d_tuple))
        # 只保留总步数在 1 到 max_step 之间的方向组合
        if dist > 0 and dist <= max_step:
            all_directions.append((dist, d_tuple))
    # 按总步数排序所有方向组合
    all_directions.sort()

    try:
        # 计算参考参数的目标函数值
        reference_target = target_func(reference_parameters)
    except Exception as msg:
        # 如果计算失败，并且不是资源耗尽错误，输出错误信息
        if verbose and "out of resource" not in str(msg):
            print(f"{reference_parameters=} lead to failure: {msg}.")
        reference_target = None

    # 如果参考目标不为 None，则将其存储在 all_values 中
    if reference_target is not None:
        all_values[to_key(reference_parameters)] = reference_target

    # 使用初始参数调用目标函数，计算初始目标函数值
    parameters = initial_parameters
    try:
        initial_target = target_func(parameters)
    except Exception as msg:
        # 如果初始计算失败，并且没有参考目标，则输出错误信息并返回空字典和负数值
        if reference_target is None:
            if verbose:
                print(
                    f"{initial_parameters=} lead to failure: {msg}. Optimization failed!"
                )
            return {}, -1, -1, f"{msg}"
        # 如果 verbose 开启且不是资源耗尽错误，输出错误信息
        if verbose and "out of resource" not in str(msg):
            print(
                f"{initial_parameters=} lead to failure: {msg}. Using reference parameters instead of initial parameters."
            )
        # 使用参考参数替换初始参数，并重新计算目标函数值
        parameters = reference_parameters
        initial_target = reference_target

    # 如果没有参考目标，则使用初始目标作为参考目标
    if reference_target is None:
        if verbose:
            print("Using initial parameters instead of reference parameters.")
        reference_target = initial_target

    # 计算初始参数的哈希键
    initial_key = to_key(parameters)
    # 将初始目标函数值作为最小目标值，并存储在 all_values 中
    minimal_target = all_values[initial_key] = initial_target

    # 初始化进度条
    pbar = tqdm(
        total=len(all_directions),
        desc="Tuning...",
        disable=not verbose,
        ncols=75,
    )
# 创建一个稀疏的块状张量 A，以便用于稀疏矩阵-矩阵乘法 (scatter_mm) 的优化
def create_blocked_tensor(B, M, N, blocksize, sparsity, dtype, device):
    # 断言稀疏度 sparsity 处于合理范围内（0到1之间）
    assert (
        sparsity <= 1.0 and sparsity >= 0.0
    ), "sparsity should be a value between 0 and 1"
    # 断言 M 能够被 blocksize[0] 整除
    assert M % blocksize[0] == 0
    # 断言 N 能够被 blocksize[1] 整除
    assert N % blocksize[1] == 0
    # 根据 blocksize 计算 A 的形状
    shape = (B, M // blocksize[0], N // blocksize[1])[int(B == 0) :]
    # 使用 torch.bernoulli 生成一个二元随机张量 A，稀疏度为 1 - sparsity
    A = torch.bernoulli(torch.full(shape, 1 - sparsity, dtype=dtype, device=device))
    # 计算预期的非零元素数量
    expected_nnz = int((1 - sparsity) * M * N / (blocksize[0] * blocksize[1]))
    # 找到 A 中非零元素的索引
    nonzero_indices = A.flatten().nonzero()
    # 实际的非零元素数量
    actual_nnz = nonzero_indices.shape[0]
    # 如果实际非零元素数量大于预期数量，则随机选择一部分非零元素置为零
    if actual_nnz > expected_nnz:
        selected_nonzeros = torch.randperm(actual_nnz)[: actual_nnz - expected_nnz]
        A.flatten()[nonzero_indices[selected_nonzeros]] = 0
    # 如果实际非零元素数量小于预期数量，则在零元素中随机选择一部分置为非零
    elif actual_nnz < expected_nnz:
        zero_indices = (A == 0).flatten().nonzero()
        selected_zeros = torch.randperm(zero_indices.shape[0])[
            : expected_nnz - actual_nnz
        ]
        A.flatten()[zero_indices[selected_zeros]] = 1
    # 根据 blocksize 扩展 A 的维度
    A = torch.repeat_interleave(A, blocksize[0], dim=-2)
    A = torch.repeat_interleave(A, blocksize[1], dim=-1)
    # 返回创建的稀疏块状张量 A
    return A


# 优化稀疏矩阵-矩阵乘法 (scatter_mm) 的函数
def optimize_scatter_mm(
    m, k, n, bm, bk, dtype=torch.float16, device="cuda", sparsity=0.5, force=False
):
    # 导入 Triton 库
    import triton
    # 导入 Triton 的稀疏矩阵-矩阵乘法相关函数
    from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data

    # 定义计算优化的关键参数
    key = (m, k, n, bm, bk)
    # 定义版本信息
    version = (0, dtype, sparsity)
    # 获取当前设备的名称
    device_name = torch.cuda.get_device_name()

    # 参考的元数据信息
    reference_meta = dict(
        GROUP_SIZE=1,
        TILE_M=16,
        TILE_N=16,
        SPLIT_N=n // 16,
        num_stages=1,
        num_warps=1,
    )

    # 获取 scatter_mm 的初始元数据
    initial_meta = get_meta(
        "scatter_mm", key, device_name=device_name, version=version, exact=True
    )
    # 如果初始元数据为 None，尝试获取稠密矩阵-矩阵乘法的元数据
    if initial_meta is None:
        initial_meta = get_meta(
            "bsr_dense_addmm",
            key,
            device_name=device_name,
            version=(0, dtype, 0.5),
            exact=True,
        )
        # 如果仍然为 None，则使用参考的元数据
        if initial_meta is None:
            initial_meta = reference_meta
    # 如果不强制更新并且初始元数据已存在，则直接返回
    elif not force:
        return

    # 设置随机数种子
    torch.manual_seed(0)
    # 创建一个稀疏块状张量 bsr，用于 scatter_mm
    bsr = create_blocked_tensor(
        0, m, k, (bm, bk), sparsity, dtype, device
    ).to_sparse_bsr((bm, bk))
    # 创建一个密集张量 dense，用于 scatter_mm
    dense = make_tensor(k, n, dtype=dtype, device=device)

    # 定义性能测试函数 bench，用于测试 scatter_mm 的性能
    def bench(meta, bsr=bsr, dense=dense):
        # 获取稀疏矩阵的索引数据
        indices_data = bsr_scatter_mm_indices_data(
            bsr, dense, indices_format="bsr_strided_mm_compressed", **meta
        )

        # 定义测试函数
        def test_func():
            return bsr_scatter_mm(bsr, dense, indices_data=indices_data)

        # 进行基准测试，返回最小的时间
        ms_min = triton.testing.do_bench(
            test_func, warmup=500, rep=100, fast_flush=False
        )

        return ms_min
    def step_meta_parameter(name, value, direction, meta, m=m, n=n, k=k, bm=bm, bk=bk):
        # 定义一个函数，根据指定的参数名、当前值、方向和其他元数据来计算下一个合法的参数值
        # 如果参数是对数值的操作，则使用指定的方向进行乘法或除法操作
        # 如果参数的范围超出预设的最小值或最大值，则将其调整到合适的范围内

        # 判断当前参数是否是对数操作的参数
        is_log = name in {"SPLIT_N", "TILE_M", "TILE_N", "num_warps"}
        
        # 根据参数名获取其允许的最小值
        min_value = dict(
            SPLIT_N=1, TILE_M=16, TILE_N=16, num_warps=1, num_stages=1, GROUP_SIZE=1
        )[name]
        
        # 根据参数名获取其允许的最大值
        max_value = dict(
            SPLIT_N=n // meta["TILE_N"], TILE_M=bm, TILE_N=n // meta["SPLIT_N"]
        ).get(name)
        
        # 根据参数名获取其对应的步进值
        value_step = dict(
            SPLIT_N=2, TILE_M=2, TILE_N=2, num_warps=2, num_stages=1, GROUP_SIZE=1
        )[name]
        
        # 根据参数的类型（对数或线性）计算下一个可能的参数值
        if is_log:
            next_value = (
                value * value_step**direction
                if direction > 0
                else value // (value_step ** abs(direction))
            )
        else:
            next_value = value + value_step * direction
        
        # 确保下一个参数值不小于预设的最小值
        if min_value is not None:
            next_value = max(next_value, min_value)
        
        # 确保下一个参数值不大于预设的最大值
        if max_value is not None:
            next_value = min(next_value, max_value)
        
        # 对于特定的参数组合，跳过会破坏 PyTorch CUDA 状态的情况
        if (dtype, name, next_value, m, n, k, bm, bk) in {
            (torch.float32, "num_warps", 32, 256, 256, 256, 16, 16),
            (torch.float32, "num_warps", 16, 256, 256, 256, 32, 32),
            (torch.float32, "num_warps", 16, 256, 256, 256, 64, 64),
            (torch.float32, "num_warps", 16, 256, 256, 256, 128, 128),
            (torch.float32, "num_warps", 16, 512, 512, 256, 128, 128),
        } and re.match(r"NVIDIA A100[^\d]", device_name) is not None:
            return value
        
        # 返回计算出的下一个合法参数值
        return next_value

    # 使用 minimize 函数对 bench 函数进行参数优化
    meta, speedup, timing, sensitivity_message = minimize(
        bench, initial_meta, reference_meta, step_meta_parameter
    )
    
    # 如果初始元数据不等于参考元数据，且最终的元数据与初始元数据相等且不强制更新，则直接返回
    if initial_meta is not reference_meta and initial_meta == meta and not force:
        return
    
    # 打印优化后的元数据、加速比和时间
    print(f"{meta=} {speedup=:.1f} % {timing=:.3f} ms")
    
    # 如果加速比小于0，则直接返回
    if speedup < 0:
        return
    
    # 获取当前设备的名称
    device_name = torch.cuda.get_device_name()
    
    # 更新 scatter_mm 函数，记录设备名称、版本和元数据的排序列表
    update(
        "scatter_mm", device_name, version, key, tuple(meta[k] for k in sorted(meta))
    )
    def tune_bsr_dense_addmm(
        input,
        bsr,
        dense,
        *,
        beta=1,
        alpha=1,
        out=None,
        store=False,
        verbose=False,
        force=False,
    ):
        """Tune bsr_dense_addmm kernel parameters against the given inputs.

        When store is True, the tuning results will be stored in the
        database of kernel parameters.
        """
        import triton  # 导入 Triton 库

        from torch.sparse._triton_ops import bsr_dense_addmm  # 导入 Triton 上的 bsr_dense_addmm 函数

        N = dense.shape[-1]  # 获取 dense 张量的最后一个维度大小作为 N
        values = bsr.values()  # 获取 bsr 稀疏张量的值
        crow_indices = bsr.crow_indices()  # 获取 bsr 稀疏张量的行索引
        batch_ndim = crow_indices.dim() - 1  # 计算批处理维度数目
        M, K = bsr.shape[batch_ndim : batch_ndim + 2]  # 获取 bsr 稀疏张量的形状 M 和 K
        BM, BK = values.shape[batch_ndim + 1 : batch_ndim + 3]  # 获取 bsr 稀疏张量值的形状 BM 和 BK

        # Reference parameters is a set of parameters that leads to a
        # successful kernel call and the corresponding timing is used as a
        # reference for computing speedups. Avoid changing the reference
        # parameters when possible.
        reference_meta = dict(
            GROUP_SIZE_ROW=1, num_stages=1, num_warps=4, SPLIT_N=max(N // BM, 1)
        )

        # Compute the key of parameters:
        sparsity = round(1 - bsr._nnz() * BM * BK / (M * K), 2)  # 计算稀疏度
        dtype = bsr.dtype  # 获取 bsr 稀疏张量的数据类型
        version = (0, dtype, sparsity)  # 定义版本信息
        key = (M, K, N, BM, BK, beta == 0, beta == 1, alpha == 1)  # 构建参数的关键字

        # For tuning, for an initial state, use parameters from the
        # database if available, otherwise, use the reference parameters.
        initial_meta = get_meta("bsr_dense_addmm", key, version=version, exact=True)  # 从数据库获取初始元数据
        if initial_meta is None:
            may_skip_update = False
            initial_meta = get_meta(
                "bsr_dense_addmm", key, version=(0, dtype, 0.5), exact=True
            )  # 尝试从数据库获取备选的初始元数据
            if initial_meta is None:
                initial_meta = reference_meta  # 如果数据库中无可用元数据，则使用参考元数据
        elif not force:
            return initial_meta  # 如果不强制更新且初始元数据已存在，则直接返回初始元数据
        else:
            may_skip_update = True

        # The target function that is minimized in the tuning process:
        def bench(meta, input=input, bsr=bsr, dense=dense, alpha=alpha, out=out):
            def test_func():
                return bsr_dense_addmm(
                    input, bsr, dense, beta=beta, alpha=alpha, meta=meta, out=out
                )  # 调用 Triton 中的 bsr_dense_addmm 函数

            return triton.testing.do_bench(test_func, warmup=500, rep=100, fast_flush=False)  # 使用 Triton 进行性能测试的函数调用

        # The step function that increments a specified meta parameter:
    # 定义一个函数，用于根据给定的参数调整下一个步骤的元数据参数
    def step_meta_parameter(name, value, direction, meta, M=M, N=N, K=K, BM=BM, BK=BK):
        # 如果参数名在{"SPLIT_N", "num_warps"}中，则设定为对数变换
        is_log = name in {"SPLIT_N", "num_warps"}
        # 根据参数名设定最小值
        min_value = dict(SPLIT_N=1, num_warps=1, num_stages=1, GROUP_SIZE_ROW=1)[name]
        # 根据参数名设定最大值（仅对SPLIT_N参数有效）
        max_value = dict(SPLIT_N=max(N // BM, 1)).get(name)
        # 根据参数名设定步长
        value_step = dict(SPLIT_N=2, num_warps=2, num_stages=1, GROUP_SIZE_ROW=1)[name]
        
        # 根据方向调整下一个值，如果是对数变换则使用指数计算，否则直接加减步长
        if is_log:
            next_value = (
                value * value_step**direction
                if direction > 0
                else value // (value_step ** abs(direction))
            )
        else:
            next_value = value + value_step * direction
        
        # 如果设置了最小值，则保证下一个值不低于最小值
        if min_value is not None:
            next_value = max(next_value, min_value)
        # 如果设置了最大值，则保证下一个值不超过最大值（仅对SPLIT_N参数有效）
        if max_value is not None:
            next_value = min(next_value, max_value)
        
        # 特殊情况：如果参数是SPLIT_N，并且N除以下一个值有余数，则返回当前值而不更新
        if name == "SPLIT_N" and N % next_value != 0:
            return value
        
        # 返回计算得到的下一个值
        return next_value
    
    # 调用minimize函数优化参数：
    # bench为性能基准函数，initial_meta为初始元数据，reference_meta为参考元数据，
    # step_meta_parameter为步进参数调整函数，max_step为最大步长，verbose为是否详细输出信息
    meta, speedup, timing, sensitivity_message = minimize(
        bench,
        initial_meta,
        reference_meta,
        step_meta_parameter,
        max_step=2,
        verbose=verbose,
    )
    
    # 如果verbose为True，打印灵敏度信息、加速比和时间
    if verbose:
        print(f"-> {sensitivity_message}, {speedup=:.1f} %, {timing=:.3f} ms")
    
    # 如果需要存储并且不跳过更新，更新指定参数到存储中
    if store and not (
        may_skip_update and meta == initial_meta and initial_meta is not reference_meta
    ):
        device_name = torch.cuda.get_device_name()
        update(
            "bsr_dense_addmm",
            device_name,
            version,
            key,
            tuple(meta[k] for k in sorted(meta)),
        )
    
    # 返回优化后的元数据
    return meta
def optimize_bsr_dense_addmm(
    m,
    k,
    n,
    bm,
    bk,
    beta=1,
    alpha=1,
    dtype=torch.float16,
    device="cuda",
    sparsity=0.5,
    force=False,
    verbose=False,
):
    # 设置随机种子为0
    torch.manual_seed(0)
    # 创建稀疏的块状张量 BSR，其大小为 (m, k)，使用给定的稀疏度和数据类型，在指定设备上生成
    bsr = create_blocked_tensor(
        0, m, k, (bm, bk), sparsity, dtype, device
    ).to_sparse_bsr((bm, bk))
    # 创建大小为 (k, n) 的稠密张量 dense，使用给定的数据类型和设备
    dense = make_tensor(k, n, dtype=dtype, device=device)
    # 创建大小为 (m, n) 的输入张量 input，使用给定的数据类型和设备
    input = make_tensor(m, n, dtype=dtype, device=device)
    # 调用函数来优化 BSR-Dense 加法和乘法操作
    tune_bsr_dense_addmm(
        input,
        bsr,
        dense,
        beta=beta,
        alpha=alpha,
        store=True,
        force=force,
        verbose=verbose,
    )


def main(op="scatter_mm", force=False, dtype=torch.float16, verbose=True):
    import itertools

    # 不同的尺寸列表
    sizes_lst = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # 三倍尺寸列表，限制尺寸最大为2048
    sizes3_lst = [3 * sz for sz in [64, 128] + sizes_lst if sz <= 2048]
    # 形状列表，包含元组 (sz, sz)
    shapes_lst = [(sz, sz) for sz in sizes_lst[:-3] + sizes3_lst]
    # 块大小列表，包含元组 (16, 16), (32, 32), (64, 64), (128, 128)
    blocksize_lst = [(16, 16), (32, 32), (64, 64), (128, 128)]
    # 稀疏度列表，只包含0.5
    sparsity_lst = [0.5, 0.7, 0.3][:1]
    # 对于每个稀疏度
    for sparsity in sparsity_lst:
        # 打印操作类型、数据类型和稀疏度
        print(f"{op, dtype, sparsity=}")
        try:
            # 对于形状列表、尺寸列表和块大小列表的每个组合
            for (M, K), N, (BM, BK) in itertools.product(
                shapes_lst, sizes_lst, blocksize_lst
            ):
                # 如果不满足块大小约束，跳过此次循环
                if not (BM <= M and BK <= K and M % BM == 0 and K % BK == 0):
                    continue
                # 如果操作是 scatter_mm
                if op == "scatter_mm":
                    # 优化 scatter_mm 操作
                    optimize_scatter_mm(
                        M, K, N, BM, BK, force=force, sparsity=sparsity, dtype=dtype
                    )
                # 如果操作是 bsr_dense_addmm
                elif op == "bsr_dense_addmm":
                    # 打印 M、K、N、块大小 (BM, BK)
                    print(f"{M, K, N, (BM, BK)=}")
                    # 对于不同的 alpha 和 beta 组合，优化 bsr_dense_addmm 操作
                    for alpha, beta in [(1, 1), (1, 0)]:
                        optimize_bsr_dense_addmm(
                            M,
                            K,
                            N,
                            BM,
                            BK,
                            beta=beta,
                            alpha=alpha,
                            force=force,
                            sparsity=sparsity,
                            dtype=dtype,
                            verbose=verbose,
                        )
                # 如果操作未实现
                else:
                    raise NotImplementedError(op)
        # 捕获键盘中断异常
        except KeyboardInterrupt:
            break
        # 捕获其它异常，调用 dump 函数
        except Exception as msg:
            dump()
            raise
    # 调用 dump 函数
    dump()

# 定义一个字典 _operation_device_version_data，用于存储操作、设备和版本数据
_operation_device_version_data: Dict[Any, Dict] = {
    # 警告：下面的数据在 BEGIN/END DATA 注释之间是生成的。
    # 可以手动更新或通过上面定义的 dump 函数更新。
    #
    # 图例 [op: key -> data]:
    #   scatter_mm : M, K, N, Ms, Ks -> GROUP_SIZE, SPLIT_N, TILE_M, TILE_N, num_stages, num_warps
    #   bsr_dense_addmm : M, K, N, Ms, Ks, beta==0, beta==1, alpha==1  -> GROUP_SIZE_ROW, SPLIT_N, num_stages, num_warps
    #
    # BEGIN GENERATED DATA
    },
    # 定义一个大的字典，包含不同配置下的性能数据
    ("bsr_dense_addmm", "NVIDIA A100-SXM4-80GB", (0, torch.bfloat16, 0.56)): {
        # 第一个配置 (192, 192, 256, 64, 64, False, True, True) 对应的性能指标
        (192, 192, 256, 64, 64, False, True, True): (3, 4, 3, 4),
        # 第二个配置 (192, 192, 256, 64, 64, True, False, True) 对应的性能指标
        (192, 192, 256, 64, 64, True, False, True): (1, 4, 4, 4),
        # 第三个配置 (192, 192, 512, 64, 64, False, True, True) 对应的性能指标
        (192, 192, 512, 64, 64, False, True, True): (2, 8, 3, 4),
        # 第四个配置 (192, 192, 512, 64, 64, True, False, True) 对应的性能指标
        (192, 192, 512, 64, 64, True, False, True): (2, 8, 3, 4),
        # ...
        # 这里包含了多个不同配置的性能指标，每个配置作为键，对应的性能数据作为值
    },
    # 定义一个大的字典，包含多个元组作为键和元组作为值
    (
        # 第一个元组 ("bsr_dense_addmm", "NVIDIA A100-SXM4-80GB", (0, torch.float16, 0.56))
        "bsr_dense_addmm",  # 键的第一个元素，字符串 "bsr_dense_addmm"
        "NVIDIA A100-SXM4-80GB",  # 键的第二个元素，字符串 "NVIDIA A100-SXM4-80GB"
        (0, torch.float16, 0.56)  # 键的第三个元素，元组 (0, torch.float16, 0.56)
    ): {
        # 键为第一个元组时，对应的值为一个字典
        # 每个键是一个元组，每个值也是一个元组
        (
            # 第一个元组 (192, 192, 256, 64, 64, False, True, True)
            192,  # 第一个元组的第一个元素，整数 192
            192,  # 第一个元组的第二个元素，整数 192
            256,  # 第一个元组的第三个元素，整数 256
            64,   # 第一个元组的第四个元素，整数 64
            64,   # 第一个元组的第五个元素，整数 64
            False,  # 第一个元组的第六个元素，布尔值 False
            True,   # 第一个元组的第七个元素，布尔值 True
            True    # 第一个元组的第八个元素，布尔值 True
        ): (
            1,  # 第一个值元组的第一个元素，整数 1
            4,  # 第一个值元组的第二个元素，整数 4
            3,  # 第一个值元组的第三个元素，整数 3
            4   # 第一个值元组的第四个元素，整数 4
        ),
        # 以下类似地，每个键元组对应一个值元组
        (
            192, 192, 256, 64, 64, True, False, True
        ): (
            1, 4, 3, 4
        ),
        ...
        # 其余键值对同上
    }
    # 定义一个嵌套字典，包含不同的元组作为键，对应的元组作为值
    ("bsr_dense_addmm", "NVIDIA A100-SXM4-80GB", (0, torch.float32, 0.56)): {
        # 第一个元组 (192, 192, 256, 64, 64, False, True, True) 的对应值是 (1, 4, 3, 8)
        (192, 192, 256, 64, 64, False, True, True): (1, 4, 3, 8),
        # 第二个元组 (192, 192, 256, 64, 64, True, False, True) 的对应值是 (1, 4, 3, 8)
        (192, 192, 256, 64, 64, True, False, True): (1, 4, 3, 8),
        # 第三个元组 (192, 192, 512, 64, 64, False, True, True) 的对应值是 (2, 8, 3, 8)
        (192, 192, 512, 64, 64, False, True, True): (2, 8, 3, 8),
        # 第四个元组 (192, 192, 512, 64, 64, True, False, True) 的对应值是 (5, 8, 3, 8)
        (192, 192, 512, 64, 64, True, False, True): (5, 8, 3, 8),
        # 第五个元组 (192, 192, 1024, 64, 64, False, True, True) 的对应值是 (2, 16, 4, 8)
        (192, 192, 1024, 64, 64, False, True, True): (2, 16, 4, 8),
        # 第六个元组 (192, 192, 1024, 64, 64, True, False, True) 的对应值是 (1, 16, 3, 8)
        (192, 192, 1024, 64, 64, True, False, True): (1, 16, 3, 8),
        # 第七个元组 (192, 192, 2048, 64, 64, False, True, True) 的对应值是 (3, 32, 3, 8)
        (192, 192, 2048, 64, 64, False, True, True): (3, 32, 3, 8),
        # 第八个元组 (192, 192, 2048, 64, 64, True, False, True) 的对应值是 (5, 32, 5, 8)
        (192, 192, 2048, 64, 64, True, False, True): (5, 32, 5, 8),
        # 第九个元组 (192, 192, 4096, 64, 64, False, True, True) 的对应值是 (3, 64, 2, 8)
        (192, 192, 4096, 64, 64, False, True, True): (3, 64, 2, 8),
        # 第十个元组 (192, 192, 4096, 64, 64, True, False, True) 的对应值是 (1, 64, 3, 8)
        (192, 192, 4096, 64, 64, True, False, True): (1, 64, 3, 8),
        # 第十一个元组 (192, 192, 8192, 64, 64, False, True, True) 的对应值是 (3, 128, 3, 8)
        (192, 192, 8192, 64, 64, False, True, True): (3, 128, 3, 8),
        # 第十二个元组 (192, 192, 8192, 64, 64, True, False, True) 的对应值是 (6, 128, 3, 4)
        (192, 192, 8192, 64, 64, True, False, True): (6, 128, 3, 4),
        # 第十三个元组 (192, 192, 16384, 64, 64, False, True, True) 的对应值是 (1, 256, 1, 8)
        (192, 192, 16384, 64, 64, False, True, True): (1, 256, 1, 8),
        # 第十四个元组 (192, 192, 16384, 64, 64, True, False, True) 的对应值是 (1, 256, 3, 4)
        (192, 192, 16384, 64, 64, True, False, True): (1, 256, 3, 4),
        # 第十五个元组 (192, 192, 32768, 64, 64, False, True, True) 的对应值是 (1, 512, 1, 8)
        (192, 192, 32768, 64, 64, False, True, True): (1, 512, 1, 8),
        # 第十六个元组 (192, 192, 32768, 64, 64, True, False, True) 的对应值是 (1, 512, 3, 4)
        (192, 192, 32768, 64, 64, True, False, True): (1, 512, 3, 4),
        # 第十七个元组 (192, 192, 65536, 64, 64, False, True, True) 的对应值是 (1, 1024, 1, 8)
        (192, 192, 65536, 64, 64, False, True, True): (1, 1024, 1, 8),
        # 第十八个元组 (192, 192, 65536, 64, 64, True, False, True) 的对应值是 (1, 1024, 3, 4)
        (192, 192, 65536, 64, 64, True, False, True): (1, 1024, 3, 4),
        # 第十九个元组 (192, 192, 131072, 64, 64, False, True, True) 的对应值是 (1, 2048, 1, 8)
        (192, 192, 131072, 64, 64, False, True, True): (1, 2048, 1, 8),
        # 第二十个元组 (192, 192, 131072, 64, 64, True, False, True) 的对应值是 (3, 2048, 1, 4)
        (192, 192, 131072, 64, 64, True, False, True): (3, 2048, 1, 4),
        # 第二十一个元组 (384, 384, 256, 128, 128, False, True, True) 的对应值是 (1, 2, 1, 32)
        (384, 384, 256, 128, 128, False, True, True): (1, 2, 1, 32),
        # 第二十二个元组 (384, 384, 256, 128, 128, True, False, True) 的对应值是 (1, 2, 1, 32)
        (384, 384, 256, 128, 128, True, False, True): (1, 2, 1, 32),
        # 第二十三个元组 (384, 384, 512, 128, 128, False, True, True) 的对应值是 (1, 4, 1, 32)
        (384, 384, 512, 128, 128, False, True, True): (1, 4, 1, 32),
        # 第二十四个元组 (384, 384, 512, 128, 128, True, False, True) 的对应值是 (2, 4, 1, 32)
        (384, 384, 512, 128, 128, True, False, True): (2, 4, 1, 32),
        # 第二十五个元组 (384, 384, 1024
}

if __name__ == "__main__":
    # 循环遍历三种数据类型：torch.float16, torch.bfloat16, torch.float32
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        # 循环遍历操作列表，此处仅包含一个操作 "bsr_dense_addmm"
        for op in ["bsr_dense_addmm"]:
            # 调用主函数 main，传入操作名称 op、force 参数为 False、数据类型为 dtype
            main(op=op, force=False, dtype=dtype)
```