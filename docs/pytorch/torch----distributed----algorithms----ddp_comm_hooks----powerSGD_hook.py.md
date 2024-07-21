# `.\pytorch\torch\distributed\algorithms\ddp_comm_hooks\powerSGD_hook.py`

```py
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入数学模块
import math
# 引入默认字典模块
from collections import defaultdict
# 引入类型提示模块中的字典类型
from typing import Dict

# 引入 PyTorch 模块
import torch
# 引入分布式训练模块
import torch.distributed as dist
# 引入分布式训练模块的 C10d 部分
from torch.distributed import distributed_c10d

# 引入默认挂钩模块
from . import default_hooks as default

# 将以下类和函数添加到模块的公开接口中
__all__ = ["PowerSGDState", "powerSGD_hook", "batched_powerSGD_hook"]

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def _orthogonalize(matrices, epsilon=0):
    """
    Decide between Gram-Schmidt or QR factorization to orthogonalize a batch of matrices.

    QR factorization doesn't work with half-precision, but it is usually faster with a rank > 2.
    """
    # 断言输入的矩阵张量为三维，并且最后一个维度小于等于第二个维度
    assert len(matrices.shape) == 3 and matrices.shape[2] <= matrices.shape[1]

    # 获取矩阵批次的数量
    num_matrices = matrices.shape[0]
    # 获取每个矩阵的秩（列数）
    rank = matrices.shape[2]
    # 获取矩阵的数据类型
    dtype = matrices.dtype
    # 根据秩和数据类型选择使用 Gram-Schmidt 还是 QR 分解来正交化矩阵
    if rank <= 2 or dtype in [torch.float16, torch.bfloat16]:
        # 如果秩小于等于2或者数据类型为半精度，则使用 Gram-Schmidt 方法
        _orthogonalize_gram_schmidt(matrices, epsilon=epsilon)
    else:
        # 否则使用 QR 分解
        torch.linalg.qr(
            matrices,
            out=(
                matrices,
                torch.empty(
                    num_matrices, rank, rank, device=matrices.device, dtype=dtype
                ),
            ),
        )


def _orthogonalize_gram_schmidt(matrices, epsilon=0):
    """
    Apply Gram-Schmidt procedure to orthogonalize a batch of matrices.

    If epsilon is 0, this is equivalent to `torch.qr(matrices, out=(matrices, _))`,
    """
    # 获取矩阵中每个矩阵的列数
    num_cols = matrices.shape[2]
    # 遍历每一列进行 Gram-Schmidt 过程
    for i in range(num_cols):
        # 获取第 i 列
        col = matrices[:, :, i : i + 1]
        # 如果 epsilon 为 0，避免除以零错误，通过增加一个小的 epsilon
        if epsilon == 0:
            # 归一化第 i 列，避免梯度消失引起的除以零错误
            try:
                col /= torch.norm(col, dim=1, keepdim=True)
            except ZeroDivisionError:
                # 若出现除以零错误，记录错误信息并将 NaN 值恢复为 0
                logger.error(
                    "The matrices to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 "
                    "as `orthogonalization_epsilon` in PowerSGD state."
                )
                col.fill_(0.0)  # 将 NaN 值恢复为 0
        else:
            # 使用给定的 epsilon 归一化第 i 列
            col /= torch.norm(col, dim=1, keepdim=True) + epsilon
        # 将第 i 列在剩余列上进行投影并移除
        if i + 1 < num_cols:
            rest = matrices[:, :, i + 1 :]
            rest -= torch.sum(col * rest, dim=1, keepdim=True) * col


def _should_compress(
    num_rows, num_cols, matrix_approximation_rank, min_compression_rate
):
    """
    Recommend if tensor given is worth compressing.

    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,
    """
    # 推荐是否值得压缩给定的张量
    return num_rows * matrix_approximation_rank >= min_compression_rate * num_cols
    # 计算未压缩状态下的元素数量，即行数乘以列数
    uncompressed_size = num_rows * num_cols
    # 计算压缩后的元素数量，即（行数加上列数）乘以矩阵近似秩
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    # 返回一个元组，包含压缩建议、未压缩元素数量和压缩后元素数量
    return (
        # 压缩建议为真，如果压缩率小于未压缩大小与压缩大小之比
        compressed_size * min_compression_rate < uncompressed_size,
        # 未压缩元素数量
        uncompressed_size,
        # 压缩后元素数量
        compressed_size,
    )
def _report_compression_stats(bucket, state):
    """Report compression stats at frequency of ``compression_stats_logging_frequency`` specified in PowerSGD state."""
    # 检查当前 bucket 是否为最后一个，并且当前迭代次数大于等于下一个统计报告的迭代次数
    if bucket.is_last() and state.iter >= state.next_stats_report:
        # 获取压缩统计信息
        stats = state.compression_stats()
        # 记录压缩统计信息到日志中
        logger.info(
            "Compression stats: iter %s, total before compression %s, total after compression %s, "
            "rate %s",
            state.iter,
            stats[1],
            stats[2],
            stats[0],
        )
        # 更新下一个统计报告的迭代次数
        state.next_stats_report = state.iter + state.compression_stats_logging_frequency


class PowerSGDState:
    r"""
    Store both the algorithm's hyperparameters and internal state for all gradients during training.

    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.

    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.

        1.1. If ``matrix_approximation_rank`` is too low, the full model quality will need more training steps to reach or will never reach and yield loss in accuracy.

        1.2. The increase of ``matrix_approximation_rank`` can substantially increase the computation costs of the compression, and the accuracy may not be further improved beyond a certain ``matrix_approximation_rank`` threshold.

    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an exponential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached. Typically only a small value 1-4 is used. For some NLP tasks (as shown in Appendix D of the original paper), this value has been increased to 32.

    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``. This hybrid scheme of **vanilla allreduce + PowerSGD** can effectively improve the accuracy, even a relatively small ``matrix_approximation_rank`` is used. This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, which can result in an irrecoverable impact on the accuracy.

    To tune ``start_powerSGD_iter``, we suggest to start with 10% of total training steps, and increase it until a satisfactory accuracy is reached. If there is a warm-up stage in the training, ``start_powerSGD_iter`` typically should be no less than the number of warm-up steps.
    """
    # __slots__ 定义了该类的特殊属性，限制了该类实例的属性，这些属性需要在类定义时指定
    __slots__ = [
        "process_group",  # MPI 进程组对象
        # 用户经常需要调整的超参数
        "matrix_approximation_rank",  # 矩阵逼近的秩
        "start_powerSGD_iter",  # 开始使用 PowerSGD 的迭代步数
        # 用户很少需要调整的超参数
        "min_compression_rate",  # 最小压缩率，决定是否进行张量压缩
        "orthogonalization_epsilon",  # 正交化步骤中的小值，用于避免除以零错误
        # 推荐开启以提高性能和准确性的二进制超参数
        "use_error_feedback",  # 是否使用误差反馈
        "warm_start",  # 是否启用热启动
        "batch_tensors_with_same_shape",  # 是否批量处理相同形状的张量以提高并行性
        # 内部状态
        "rng",  # 随机数生成器对象
        "error_dict",  # 错误字典，用于误差反馈
        "p_memory_dict",  # 内存字典，用于存储历史 P 值
        "q_memory_dict",  # 内存字典，用于存储历史 Q 值
        "iter",  # 当前迭代步数
        # 记录压缩统计信息
        "total_numel_before_compression",  # 压缩前总元素数量
        "total_numel_after_compression",  # 压缩后总元素数量
        "compression_stats_logging_frequency",  # 记录压缩统计信息的频率（迭代次数）
        "next_stats_report",  # 下一个统计报告的迭代步数
    ]
    def __init__(
        self,
        process_group,
        matrix_approximation_rank=1,
        start_powerSGD_iter=1_000,
        min_compression_rate=2,
        use_error_feedback=True,
        warm_start=True,
        orthogonalization_epsilon=0,
        random_seed=0,
        compression_stats_logging_frequency=10_000,
        batch_tensors_with_same_shape: bool = False,
    ):
        """
        Initialize the PowerSGDState object with specified parameters.

        Args:
        - process_group: The process group for distributed training.
        - matrix_approximation_rank: Rank for matrix approximation.
        - start_powerSGD_iter: Iteration number to start PowerSGD.
        - min_compression_rate: Minimum compression rate.
        - use_error_feedback: Flag indicating whether error feedback is used.
        - warm_start: Flag indicating whether to use warm start.
        - orthogonalization_epsilon: Epsilon value for orthogonalization.
        - random_seed: Random seed for reproducibility.
        - compression_stats_logging_frequency: Frequency for logging compression statistics.
        - batch_tensors_with_same_shape: Flag indicating whether to batch tensors with the same shape.
        """

    def __getstate__(self):
        """
        Return a dictionary representing the object state for pickling.

        Notes:
        - process_group is excluded from the returned state as it's not serializable.
        """
        logger.warning(
            "NOTE: Process group is not serializable and excluded from a saved state."
        )
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot != "process_group"
        }

    def __setstate__(self, state):
        """
        Set the state of the object from the provided dictionary.

        Notes:
        - Sets process_group to a default group for distributed training.
        """
        self.process_group = distributed_c10d._get_default_group()
        logger.warning(
            "NOTE: Process group will be set to a default group (i.e. the world size).\
                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded."
        )
        for slot, value in state.items():
            setattr(self, slot, value)

    def maybe_increase_iter(self, bucket):
        """
        Track iterations and trigger log message at the start of local SGD.

        Args:
        - bucket: Bucket object representing the current iteration's state.

        Notes:
        - Only increments `iter` when bucket 0 is processed.
        """
        if bucket.is_last():
            self.iter += 1

        if self.iter == self.start_powerSGD_iter:
            logger.info("Start to apply PowerSGD after %s iterations.", self.iter)

    def compression_stats(self):
        """
        Return latest compression statistics as a tuple.

        Returns:
        - Tuple of (compress_rate, numel_before_compression, numel_after_compression).

        Notes:
        - compress_rate: Effective compression rate.
        - numel_before_compression: Total number of elements before compression.
        - numel_after_compression: Total number of elements after compression.
        """
        compress_rate = (
            self.total_numel_before_compression / self.total_numel_after_compression
            if self.total_numel_after_compression > 0
            else 0
        )
        return (
            compress_rate,
            self.total_numel_before_compression,
            self.total_numel_after_compression,
        )
# 实现 PowerSGD 算法的通信钩子函数
def powerSGD_hook(
    state: PowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    r"""
    Implement PowerSGD algorithm.

    This DDP communication hook implements PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a list of per-parameter tensors, and divides all the tensors into two groups:

        1.1 The tensors that should be compressed before allreduce, because the compression can give enough saving in bandwidth.

        1.2 Rest of the tensors will be directly allreduced without compression, including all the vector tensors (for biases).

    2. Handles uncompressed tensors:

        2.1. Allocate contiguous memory for those uncompressed tensors, and allreduces all the uncompressed tensors as a batch, without compression;

        2.2. Copies the individual uncompressed tensors from the contiguous memory back to the input tensor.

    3. Handles the tensors that should be compressed by PowerSGD compression:

        3.1. For each tensor M, creates two low-rank tensors P and Q for decomposing M,
        such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

        3.2. Computes each P in Ps, which is equal to MQ;

        3.3. Allreduces Ps as a batch;

        3.4. Orthogonalizes each P in Ps;

        3.5. Computes each Q in Qs, which is approximately equal to M^TP;

        3.6. Allreduces Qs as a batch;

        3.7. Computes each M among all the compressed tensors, which is approximately equal to PQ^T.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank``, ``start_powerSGD_iter``
            and ``min_compression_rate``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.
    """
    # 获取处理组
    process_group = state.process_group
    # 如果处理组存在，则使用该组；否则使用全局默认组
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # 获取处理组的大小（节点数量）
    world_size = group_to_use.size()

    # 输入张量是一个展平的一维张量
    input_tensor = bucket.buffer()

    # 在前 `start_powerSGD_iter` 次迭代中运行原始的全局归约操作
    if state.iter < state.start_powerSGD_iter:
        # 可能增加迭代计数器
        state.maybe_increase_iter(bucket)
        # 返回默认的全局归约操作的 future 对象
        return default._allreduce_fut(group_to_use, input_tensor)

    # 超过 `start_powerSGD_iter` 次迭代后应用 PowerSGD 算法
    device = input_tensor.device
    dtype = input_tensor.dtype

    # 将前一个状态的误差合并到梯度中
    bucket_index = bucket.index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        # 如果错误反馈启用且存在对应 bucket 的误差信息，则添加到输入张量中
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            # 否则创建一个长度为 total_length 的零张量，表示本地误差
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                total_length,
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )

        # 创建输入张量的副本，用于稍后计算由压缩引起的本地误差
        input_tensor_cp = torch.clone(input_tensor).detach()

    # 将输入张量展开为每个参数张量，以进行逐层压缩
    tensors = bucket.gradients()

    # 第一步：将所有张量分为两组，
    # 一组在全局归约之前进行压缩，另一组直接进行全局归约而不压缩。
    tensors_to_compress, uncompressed_tensors = [], []
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in tensors:
        # 将张量视为矩阵，形状为 (n, m)
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        # 选择较小的值作为矩阵近似秩，以进行压缩
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        # 判断是否应该对矩阵进行压缩
        compress_test = _should_compress(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        # 更新压缩前后的元素总数统计信息
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            # 如果需要压缩，则将该矩阵添加到待压缩的张量组中，并更新相关统计信息
            tensors_to_compress.append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compress_test[2]
        else:
            # 否则将张量添加到未压缩的张量组中，并更新相关统计信息
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += compress_test[1]

    # 报告压缩统计信息
    _report_compression_stats(bucket, state)
    """
    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle the tensors that should be compressed.
    # Allocate contiguous memory for Ps and Qs to allreduce efficiently.
    # If warm-start is enabled, reuse Ps and Qs from the previous iteration if possible.
    # The memory spaces of Ps and Qs need to be allocated in the first iteration when PowerSGD is applied.
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        # Only log this if warm-start to avoid spamming.
        if state.warm_start:
            logger.info(
                "Allocating contiguous memory of length %s for Ps, and of length %s for Qs, respectively.",
                total_Ps_size,
                total_Qs_size,
            )
        # Allocate memory for Ps and Qs based on their respective sizes
        state.p_memory_dict[bucket_index] = torch.empty(
            total_Ps_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            total_Qs_size, device=device, dtype=dtype
        )

    # Batch tensors to compress by shape.
    shape_to_tensors = defaultdict(list)
    for tensor in tensors_to_compress:
        # Group tensors by their shapes
        shape_to_tensors[tensor.shape].append(tensor)

    # This function decides whether to batch tensors with the same shape or not according to the argument,
    # so the following process could share the same code.
    def maybe_batched_tensors_to_compress():
        for tensors in shape_to_tensors.values():
            if state.batch_tensors_with_same_shape:
                batch_size = len(tensors)
                if batch_size == 1:
                    # Use the original tensor to avoid copy if batch size is 1
                    yield tensors[0].unsqueeze(0)
                else:
                    # Stack tensors into a single batch tensor
                    yield torch.stack(tensors)
            else:
                # Yield each tensor individually as a batch of size 1
                for tensor in tensors:
                    yield tensor.unsqueeze(0)

    # Create Ps and Qs that point to the allocated memory.
    tensors_to_compress = []
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    # 遍历可能需要批处理压缩的张量
    for tensor in maybe_batched_tensors_to_compress():
        # 获取当前张量的批大小、行数和列数
        batch_size, n, m = tensor.shape
        # 确定用于矩阵近似的秩，取三者中的最小值
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        # 将当前张量添加到待压缩张量列表中
        tensors_to_compress.append(tensor)
        # 从内存字典中取出对应的P，并按照一定规则进行形状变换
        ps.append(
            state.p_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * n * matrix_approximation_rank
            ].view(batch_size, n, matrix_approximation_rank)
        )
        # 从内存字典中取出对应的Q，并按照一定规则进行形状变换
        qs.append(
            state.q_memory_dict[bucket_index][
                q_idx : q_idx + batch_size * m * matrix_approximation_rank
            ].view(batch_size, m, matrix_approximation_rank)
        )
        # 更新P索引，以便下一个张量的处理
        p_idx += batch_size * n * matrix_approximation_rank
        # 更新Q索引，以便下一个张量的处理
        q_idx += batch_size * m * matrix_approximation_rank

    # 如果启用了热启动，则尽可能重用上一迭代的Q，并跳过填充随机值的步骤
    if not need_randomize_qs:
        # 对每个Q张量进行正交化处理
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        # 使用torch.random.fork_rng()在本地生成随机种子，以避免全局种子的变化影响训练中的其他随机采样
        with torch.random.fork_rng(devices=[]):
            # 设置随机种子确保每一步的初始随机值都相同，这对所有DDP副本保持一致性至关重要
            torch.manual_seed(state.rng.randint(1_000_000_000))
            # 对每个Q张量生成随机值，然后进行正交化处理
            for q in qs:
                q.copy_(
                    torch.randn(
                        *q.shape,
                        device="cpu",
                        dtype=dtype,
                    )
                )
                _orthogonalize(q, state.orthogonalization_epsilon)

    # 计算压缩后的P张量
    for tensor, q, p in zip(tensors_to_compress, qs, ps):
        torch.bmm(tensor, q, out=p)

    # 对未压缩的张量执行全局归约操作，以减少通信成本
    # 该操作仅应用于未压缩的张量，因此在上述压缩张量的计算之前就应该启动
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()
    # 定义函数unpack_uncompressed_tensors_and_allreduce_ps，接受一个Future对象作为参数
    def unpack_uncompressed_tensors_and_allreduce_ps(fut):
        # 计算未压缩张量的内存，将其平均分配给各个进程
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        # 遍历未压缩张量列表
        for tensor in uncompressed_tensors:
            # 将分配的内存复制到当前张量中
            tensor.copy_(
                uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        # 由于这些P将在后续进行正交化处理，因此无需除以world_size
        return (
            # 对p_memory_dict[bucket_index]执行全局allreduce操作，使用指定的通信组，异步操作
            dist.all_reduce(
                state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]  # 等待异步操作完成并返回结果
        )

    # 定义函数compute_qs，接受一个Future对象作为参数
    def compute_qs(fut):
        # 将Future对象的值设置为p_memory_dict[bucket_index]
        state.p_memory_dict[bucket_index] = fut.value()
        # 对每个P进行正交化处理，使用给定的epsilon值
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)

        # 计算Qs
        for tensor, p, q in zip(tensors_to_compress, ps, qs):
            # 执行张量的批次矩阵乘积，并将结果存储在q中
            torch.bmm(tensor.transpose(1, 2), p, out=q)

        # TODO: 上述过程每次迭代执行两次matmul+allreduce步骤 --
        # 一次左乘和一次右乘。对于热启动，可以一次执行一个步骤，并在它们之间交替。

        # 对Qs执行全局allreduce操作
        return (
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]  # 等待异步操作完成并返回结果
        )

    # 定义函数decompress，接受一个Future对象作为参数
    def decompress(fut):
        # 将Future对象的值设置为q_memory_dict[bucket_index]，并除以world_size
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)

        # 对每对P、Q和待压缩张量执行批次矩阵乘积的逆过程
        for p, q, tensor in zip(ps, qs, tensors_to_compress):
            torch.bmm(p, q.transpose(1, 2), out=tensor)

        # 如果批量张量具有相同形状，将批量张量复制回原始缓冲区
        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                if tensor.shape[0] == 1:
                    # 跳过批次大小为1的张量，因为它本身就是原始张量
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])

        # 如果CUDA可用，同步所有CUDA设备上的操作
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        # 如果使用错误反馈，记录本地误差
        if state.use_error_feedback:
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        # 如果不是热启动，清空p_memory_dict和q_memory_dict
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()

        # 可能增加迭代次数
        state.maybe_increase_iter(bucket)

        # 返回输入张量
        return input_tensor

    # 返回一个Future对象链，按顺序执行三个函数：unpack_uncompressed_tensors_and_allreduce_ps，compute_qs，decompress
    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_ps
        )
        .then(compute_qs)
        .then(decompress)
    )
def batched_powerSGD_hook(
    state: PowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    r"""
    Implement simplified PowerSGD algorithm.

    This DDP communication hook implements a simplified PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    This variant does not compress the gradients layer by layer,
    but instead compresses the flattened input tensor that batches all the gradients.
    Therefore, it is **faster** than :meth:`powerSGD_hook`,
    but usually results in a **much lower accuracy**, unless ``matrix_approximation_rank`` is 1.

    .. warning ::
        Increasing ``matrix_approximation_rank`` here may not necessarily increase the accuracy,
        because batching per-parameter tensors without column/row alignment can destroy low-rank structure.
        Therefore, the user should always consider :meth:`powerSGD_hook` first,
        and only consider this variant when a satisfactory accuracy can be achieved when ``matrix_approximation_rank`` is 1.

    Once gradient tensors are aggregated across all workers, this hook applies
    compression as follows:

    1. Views the input flattened 1D gradient tensor as a square-shaped tensor M with 0 paddings;

    2. Creates two low-rank tensors P and Q for decomposing M, such that M = PQ^T, where Q is initialized from a standard normal distribution and orthogonalized;

    3. Computes P, which is equal to MQ;

    4. Allreduces P;

    5. Orthogonalizes P;

    6. Computes Q, which is approximately equal to M^TP;

    7. Allreduces Q;

    8. Computes M, which is approximately equal to PQ^T.

    9. Truncates the input tensor to the original length.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.
    This not only gives the user more control over the tradeoff between speedup and accuracy,
    but also helps abstract away some complexity of the internal optimization of DDP for future communication hook developers.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
            To tune the compression configs, mainly need to tune ``matrix_approximation_rank`` and ``start_powerSGD_iter``.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)
        >>> ddp_model.register_comm_hook(state, batched_powerSGD_hook)
    """  # noqa: B950
    process_group = state.process_group
    # 根据传入的 process_group 参数决定使用哪个通信组，若为空则使用默认的 WORLD 组
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    # 获取通信组的大小，即进程的总数
    world_size = group_to_use.size()

    # 输入张量是一个扁平化的一维张量
    input_tensor = bucket.buffer()

    # 在开始的 `start_powerSGD_iter` 次迭代中运行普通的 allreduce 操作
    if state.iter < state.start_powerSGD_iter:
        # 增加迭代次数计数并返回普通 allreduce 的异步操作
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # 当超过 `start_powerSGD_iter` 次迭代后应用 PowerSGD
    device = input_tensor.device
    total_length = input_tensor.shape[0]
    # 更新压缩前的总元素数量统计
    state.total_numel_before_compression += total_length

    # 将输入张量视为一个二维方形张量，并在必要时填充 0
    square_side_length = math.ceil(math.sqrt(total_length))
    # 更新压缩后的总元素数量统计
    state.total_numel_after_compression += (
        square_side_length * state.matrix_approximation_rank * 2
    )
    padded_total_length = square_side_length**2
    # 调整输入张量的大小为填充后的总长度，并用 0 填充扩展的部分
    input_tensor.resize_(padded_total_length)
    input_tensor[total_length:padded_total_length].fill_(0)

    # 报告压缩统计信息
    _report_compression_stats(bucket, state)

    # 将前一个状态的误差合并到梯度中
    bucket_index = bucket.index()
    input_tensor_cp = None
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            # 将存储的误差张量加到输入张量中
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            # 如果误差字典中没有对应的项，则创建一个表示本地误差的零张量
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                padded_total_length,
            )
            state.error_dict[bucket_index] = torch.zeros(
                padded_total_length, device=device, dtype=input_tensor.dtype
            )

        # 创建输入张量的副本，用于后续解压缩后计算本地误差
        input_tensor_cp = torch.clone(input_tensor).detach()

    # 将输入张量重新视为方形矩阵
    matrix = input_tensor.view(square_side_length, square_side_length)

    # 如果可能的话，重用上一次迭代中的 P 和 Q
    # 当应用 PowerSGD 时，需要在第一次迭代中分配 P 和 Q 的内存空间
    # 如果禁用了热启动或者在状态字典中找不到给定桶索引的内存项，则执行以下操作
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        
        # 如果禁用了热启动，则每一步都会初始化低秩张量
        # 只有在启用热启动时才记录这个消息，避免过多的日志信息
        if state.warm_start:
            logger.info(
                "Initializing low-rank tensors P and Q, each of which has a shape of %s x %s.",
                square_side_length,
                state.matrix_approximation_rank,
            )

        # 定义一个函数，用于创建低秩张量
        def create_low_rank_tensor(fill_random_values, rng):
            """Return a low-rank 2D tensor of square_side_length * matrix_approximation_rank."""
            if fill_random_values:
                with torch.random.fork_rng(devices=[]):
                    # 使用torch.random.fork_rng()来保证在全局不改变随机种子的情况下进行随机采样
                    # 这个种子确保了初始随机值在所有的分布式数据并行 replicas 中都是相同的
                    # 每一步的种子应该不同
                    # 由于在所有 CUDA 设备上分叉 RNG 状态非常慢，
                    # 只在 CPU 上分叉，然后将生成的张量移动到 CUDA 设备上
                    torch.manual_seed(rng.randint(1_000_000_000))
                    return torch.randn(
                        square_side_length,
                        state.matrix_approximation_rank,
                        device="cpu",
                        dtype=input_tensor.dtype,
                    ).to(device)
            else:
                # 创建一个空的张量
                return torch.empty(
                    square_side_length,
                    state.matrix_approximation_rank,
                    device=device,
                    dtype=input_tensor.dtype,
                )

        # 将创建的低秩张量存储到状态字典的对应桶索引位置
        state.p_memory_dict[bucket_index] = create_low_rank_tensor(
            fill_random_values=False, rng=state.rng
        )
        state.q_memory_dict[bucket_index] = create_low_rank_tensor(
            fill_random_values=True, rng=state.rng
        )

    # 对 Q 内存项进行正交化处理
    _orthogonalize(state.q_memory_dict[bucket_index])

    # 计算 matrix 和 Q 内存项的矩阵乘法，结果存储到 P 内存项
    torch.matmul(
        matrix, state.q_memory_dict[bucket_index], out=state.p_memory_dict[bucket_index]
    )

    # 对 P 内存项进行全局的 allreduce 操作，使用指定的通信组，并设置为异步操作
    allreduce_p_fut = dist.all_reduce(
        state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
    ).get_future()
    # 定义一个函数，用于计算 q 值。
    def compute_q(fut):
        # 将 future 的值放入 p_memory_dict 中的指定索引位置
        state.p_memory_dict[bucket_index] = fut.value()[0]
        # 对 p_memory_dict 中的数据进行正交化处理
        _orthogonalize(state.p_memory_dict[bucket_index])

        # 执行矩阵乘法操作，将 matrix 与 p_memory_dict[bucket_index] 的转置相乘，结果存入 q_memory_dict[bucket_index] 中
        torch.matmul(
            matrix.t(),
            state.p_memory_dict[bucket_index],
            out=state.q_memory_dict[bucket_index],
        )

        # TODO: 上述过程每次迭代执行两次 matmul 和 allreduce 步骤 --
        # 一次左乘法和一次右乘法。
        # 对于热启动，可以一次执行其中一步，并在它们之间交替。

        # 返回异步操作的 future，该操作会对 q_memory_dict[bucket_index] 执行全局归约
        return (
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()  # 获取 future 对象
            .wait()[0]      # 等待操作完成并获取结果的第一个元素
        )

    # 定义一个函数，用于解压缩操作。
    def decompress(fut):
        # 将 future 的值除以 world_size，并存入 q_memory_dict[bucket_index] 中
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        
        # 执行矩阵乘法操作，将 p_memory_dict[bucket_index] 与 q_memory_dict[bucket_index] 的转置相乘，结果存入 matrix 中
        torch.matmul(
            state.p_memory_dict[bucket_index],
            state.q_memory_dict[bucket_index].t(),
            out=matrix,
        )

        # 如果启用错误反馈，保存本地误差
        if state.use_error_feedback:
            # 记录本地误差
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        
        # 移除这个看似不必要的同步，可能导致失败。
        # 参见：https://github.com/pytorch/pytorch/pull/54838
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        
        # 如果不是热启动，则清空 p_memory_dict 和 q_memory_dict
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        
        # 调整输入张量的大小为 total_length，并返回
        ret = input_tensor.resize_(total_length)

        # 可能增加迭代计数
        state.maybe_increase_iter(bucket)

        return ret

    # 返回一个异步操作链，依次执行 allreduce_p_fut -> compute_q -> decompress
    return allreduce_p_fut.then(compute_q).then(decompress)
```