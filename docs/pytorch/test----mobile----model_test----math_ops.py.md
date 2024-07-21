# `.\pytorch\test\mobile\model_test\math_ops.py`

```py
# 引入 math 和 torch 库，用于数学操作和张量计算
import math
import torch

# 定义一个 PyTorch 模块，用于执行点对点操作
class PointwiseOpsModule(torch.nn.Module):
    def forward(self):
        return self.pointwise_ops()

# 定义一个 PyTorch 模块，用于执行归约操作
class ReductionOpsModule(torch.nn.Module):
    def forward(self):
        return self.reduction_ops()

    # 执行各种张量的归约操作，返回结果数量
    def reduction_ops(self):
        # 创建随机张量 a 和 b
        a = torch.randn(4)
        b = torch.randn(4)
        c = torch.tensor(0.5)
        # 返回以下各种张量的归约操作结果的数量
        return len(
            torch.argmax(a),             # 获取张量 a 中最大值的索引
            torch.argmin(a),             # 获取张量 a 中最小值的索引
            torch.amax(a),               # 返回张量 a 的最大值
            torch.amin(a),               # 返回张量 a 的最小值
            torch.aminmax(a),            # 返回张量 a 的最小值和最大值
            torch.all(a),                # 检查张量 a 中所有元素是否为 True
            torch.any(a),                # 检查张量 a 中是否有任意元素为 True
            torch.max(a),                # 返回张量 a 的最大值
            a.max(a),                    # 返回张量 a 的最大值
            torch.max(a, 0),             # 沿着指定维度返回张量 a 的最大值和对应索引
            torch.min(a),                # 返回张量 a 的最小值
            a.min(a),                    # 返回张量 a 的最小值
            torch.min(a, 0),             # 沿着指定维度返回张量 a 的最小值和对应索引
            torch.dist(a, b),            # 返回张量 a 和 b 之间的欧氏距离
            torch.logsumexp(a, 0),       # 计算张量 a 指数的和的对数
            torch.mean(a),               # 返回张量 a 的平均值
            torch.mean(a, 0),            # 沿着指定维度返回张量 a 的平均值
            torch.nanmean(a),            # 返回张量 a 的非 NaN 值的平均值
            torch.median(a),             # 返回张量 a 的中位数
            torch.nanmedian(a),          # 返回张量 a 的非 NaN 值的中位数
            torch.mode(a),               # 返回张量 a 的众数
            torch.norm(a),               # 返回张量 a 的范数
            a.norm(2),                   # 返回张量 a 的 L2 范数
            torch.norm(a, dim=0),        # 沿着指定维度返回张量 a 的范数
            torch.norm(c, torch.tensor(2)),  # 返回张量 c 的指定范数
            torch.nansum(a),             # 返回张量 a 的非 NaN 值的和
            torch.prod(a),               # 返回张量 a 的所有元素的乘积
            torch.quantile(a, torch.tensor([0.25, 0.5, 0.75])),  # 返回张量 a 的分位数
            torch.quantile(a, 0.5),      # 返回张量 a 的指定分位数
            torch.nanquantile(a, torch.tensor([0.25, 0.5, 0.75])),  # 返回张量 a 的非 NaN 值的分位数
            torch.std(a),                # 返回张量 a 的标准差
            torch.std_mean(a),           # 返回张量 a 的标准差和平均值
            torch.sum(a),                # 返回张量 a 的所有元素的和
            torch.unique(a),             # 返回张量 a 的唯一值
            torch.unique_consecutive(a), # 返回张量 a 中连续的唯一值
            torch.var(a),                # 返回张量 a 的方差
            torch.var_mean(a),           # 返回张量 a 的方差和平均值
            torch.count_nonzero(a),      # 返回张量 a 中非零元素的数量
        )

# 定义一个 PyTorch 模块，用于执行比较操作
class ComparisonOpsModule(torch.nn.Module):
    def forward(self):
        a = torch.tensor(0)
        b = torch.tensor(1)
        # 返回以下各种张量的比较操作结果的数量
        return len(
            torch.allclose(a, b),         # 检查张量 a 和 b 是否在允许误差范围内相等
            torch.argsort(a),             # 返回张量 a 的排序索引
            torch.eq(a, b),               # 检查张量 a 和 b 是否逐元素相等
            torch.eq(a, 1),               # 检查张量 a 是否逐元素等于标量 1
            torch.equal(a, b),            # 检查张量 a 和 b 是否形状和元素都相等
            torch.ge(a, b),               # 检查张量 a 是否逐元素大于或等于张量 b
            torch.ge(a, 1),               # 检查张量 a 是否逐元素大于或等于标量 1
            torch.greater_equal(a, b),    # 检查张量 a 是否逐元素大于或等于张量 b
            torch.greater_equal(a, 1),    # 检查张量 a 是否逐元素大于或等于标量 1
            torch.gt(a, b),               # 检查张量 a 是否逐元素大于张量 b
            torch.gt(a, 1),               # 检查张量 a 是否逐元素大于标量 1
            torch.greater(a, b),          # 检查张量 a 是否逐元素大于张量 b
            torch.isclose(a, b),          # 检查张量 a 和 b 是否在允许误差范围内相等
            torch.isfinite(a),            # 检查张量 a 是否有限
            torch.isin(a, b),             # 检查张量 a 中的元素是否在张量 b 中
            torch.isinf(a),               # 检查张量 a 是否为无穷大
            torch.isposinf(a),            # 检查张量 a 是否为正无穷大
            torch.isneginf(a),            # 检查张量 a 是否为负无穷大
            torch.isnan(a),               # 检查张量 a 是否为 NaN
            torch.isreal(a),              # 检查张量 a 是否为实数
            torch.kthvalue(a, 1),         # 返回张量 a 的第 k 小值和对应索引
            torch.le(a, b),               # 检查张量 a 是否逐元素小于或等于张量 b
            torch.le(a, 1),               # 检查张量 a 是否逐元素小于或等于标量 1
            torch.less_equal(a, b),       # 检查张量 a 是否逐元素小于或等于张量 b
            torch.lt(a, b),               # 检查张量 a 是否逐元素小于张量 b
            torch.lt(a, 1),               # 检查张量 a 是否逐元素小于标量 1
            torch.less(a, b),             # 检查张量 a 是否逐元素小于张量 b
            torch.maximum(a, b),          # 返回张量 a 和 b 逐元素的最大值
            torch.minimum(a, b),          # 返回张量 a 和 b 逐元素的最小值
            torch.fmax(a, b),             # 返回张量 a 和 b 逐元素的最大值
            torch.fmin(a, b),             # 返回张量 a 和 b 逐元素的最小值
            torch.ne(a, b),               # 检查张量 a 和 b 是否逐元素不相等
            torch.ne(a, 1),               # 检查张量 a 是否逐元素不等于标量 1
            torch.not_equal(a, b),        # 检查张量 a 和 b 是否逐元素不相
    # 定义一个方法 `forward`，该方法调用 `other_ops` 方法并返回其结果
    def forward(self):
        return self.other_ops()

    # 定义另一个方法 `other_ops`，其中包含多个 Torch 张量操作示例
    def other_ops(self):
        # 生成一个包含四个随机数的张量 a
        a = torch.randn(4)
        # 生成一个包含四个随机数的张量 b
        b = torch.randn(4)
        # 生成一个包含随机整数的张量 c，值在 [0, 8) 范围内
        c = torch.randint(0, 8, (5,), dtype=torch.int64)
        # 生成一个 4x3 的随机张量 e
        e = torch.randn(4, 3)
        # 生成一个 4x4x4 的随机张量 f
        f = torch.randn(4, 4, 4)
        # 定义一个包含索引值的列表 size
        size = [0, 1]
        # 定义一个包含维度索引的列表 dims
        dims = [0, 1]
        
        # 返回多个张量操作函数的数量，包括：
        # - atleast_1d, atleast_2d, atleast_3d: 将张量转换为至少1维、2维、3维
        # - bincount: 计算整数张量中每个值的频数
        # - block_diag: 构造一个块对角张量
        # - broadcast_tensors, broadcast_to: 广播张量到指定形状
        # - bucketize: 将值按照边界张量分桶
        # - cartesian_prod: 计算两个张量的笛卡尔积
        # - cdist: 计算两个张量之间的距离
        # - clone: 复制张量
        # - combinations: 计算张量中元素的组合
        # - corrcoef: 计算张量的相关系数矩阵
        # - cross: 计算张量的叉积
        # - cummax, cummin, cumprod, cumsum: 计算张量的累积操作
        # - diag, diag_embed, diagflat, diagonal: 计算张量的对角元素或者对角化操作
        # - diff: 计算张量元素的差分
        # - einsum: 执行张量的爱因斯坦求和约定
        # - flatten: 将张量展平为1维
        # - flip, fliplr, flipud: 执行张量的翻转操作
        # - gcd, lcm: 计算整数张量的最大公约数和最小公倍数
        # - histc, histogram: 计算张量的直方图统计信息
        # - meshgrid: 生成张量的网格坐标
        # - kron: 计算两个张量的克罗内克积
        # - rot90: 执行张量的旋转操作
        # - logcumsumexp: 计算张量的对数累积和
        # - ravel: 执行张量的扁平化操作
        # - renorm: 执行张量的重新归一化操作
        # - repeat_interleave: 执行张量的重复插值操作
        # - roll: 执行张量的滚动操作
        # - searchsorted: 在有序张量中执行二分搜索操作
        # - tensordot: 执行张量的张量积操作
        # - trace: 计算张量的迹
        # - tril, triu: 返回张量的下三角和上三角部分
        # - tril_indices, triu_indices: 返回下三角和上三角索引
        # - vander: 生成张量的范德蒙德矩阵
        # - view_as_real, view_as_complex: 执行张量的视图转换操作
        # - resolve_conj, resolve_neg: 解析复数张量的共轭和负部分
        return len(
            torch.atleast_1d(a),
            torch.atleast_2d(a),
            torch.atleast_3d(a),
            torch.bincount(c),
            torch.block_diag(a),
            torch.broadcast_tensors(a),
            torch.broadcast_to(a, (4)),
            # torch.broadcast_shapes(a),
            torch.bucketize(a, b),
            torch.cartesian_prod(a),
            torch.cdist(e, e),
            torch.clone(a),
            torch.combinations(a),
            torch.corrcoef(a),
            # torch.cov(a),
            torch.cross(e, e),
            torch.cummax(a, 0),
            torch.cummin(a, 0),
            torch.cumprod(a, 0),
            torch.cumsum(a, 0),
            torch.diag(a),
            torch.diag_embed(a),
            torch.diagflat(a),
            torch.diagonal(e),
            torch.diff(a),
            torch.einsum("iii", f),
            torch.flatten(a),
            torch.flip(e, dims),
            torch.fliplr(e),
            torch.flipud(e),
            torch.kron(a, b),
            torch.rot90(e),
            torch.gcd(c, c),
            torch.histc(a),
            torch.histogram(a),
            torch.meshgrid(a),
            torch.meshgrid(a, indexing="xy"),
            torch.lcm(c, c),
            torch.logcumsumexp(a, 0),
            torch.ravel(a),
            torch.renorm(e, 1, 0, 5),
            torch.repeat_interleave(c),
            torch.roll(a, 1, 0),
            torch.searchsorted(a, b),
            torch.tensordot(e, e),
            torch.trace(e),
            torch.tril(e),
            torch.tril_indices(3, 3),
            torch.triu(e),
            torch.triu_indices(3, 3),
            torch.vander(a),
            torch.view_as_real(torch.randn(4, dtype=torch.cfloat)),
            torch.view_as_complex(torch.randn(4, 2)).real,
            torch.resolve_conj(a),
            torch.resolve_neg(a),
        )
class SpectralOpsModule(torch.nn.Module):
    # 定义一个继承自 torch.nn.Module 的类 SpectralOpsModule
    def forward(self):
        # 前向传播函数，返回 spectral_ops 方法的结果
        return self.spectral_ops()

    def spectral_ops(self):
        # 定义 spectral_ops 方法，执行一系列频谱操作
        a = torch.randn(10)  # 创建一个形状为 (10,) 的张量 a，其中元素服从标准正态分布
        b = torch.randn(10, 8, 4, 2)  # 创建一个形状为 (10, 8, 4, 2) 的张量 b，元素服从标准正态分布
        return len(
            torch.stft(a, 8),  # 计算张量 a 的短时傅里叶变换
            torch.stft(a, torch.tensor(8)),  # 同上，传入张量表示窗口大小
            torch.istft(b, 8),  # 计算张量 b 的逆短时傅里叶变换
            torch.bartlett_window(2, dtype=torch.float),  # 创建长度为 2 的巴特利特窗口张量
            torch.blackman_window(2, dtype=torch.float),  # 创建长度为 2 的布莱克曼窗口张量
            torch.hamming_window(4, dtype=torch.float),  # 创建长度为 4 的海明窗口张量
            torch.hann_window(4, dtype=torch.float),  # 创建长度为 4 的汉宁窗口张量
            torch.kaiser_window(4, dtype=torch.float),  # 创建长度为 4 的凯泽窗口张量
        )


class BlasLapackOpsModule(torch.nn.Module):
    # 定义一个继承自 torch.nn.Module 的类 BlasLapackOpsModule
    def forward(self):
        # 前向传播函数，返回 blas_lapack_ops 方法的结果
        return self.blas_lapack_ops()

    def blas_lapack_ops(self):
        # 定义 blas_lapack_ops 方法，执行一系列 BLAS 和 LAPACK 操作
        m = torch.randn(3, 3)  # 创建一个形状为 (3, 3) 的随机张量 m
        a = torch.randn(10, 3, 4)  # 创建一个形状为 (10, 3, 4) 的随机张量 a
        b = torch.randn(10, 4, 3)  # 创建一个形状为 (10, 4, 3) 的随机张量 b
        v = torch.randn(3)  # 创建一个形状为 (3,) 的随机张量 v
        return len(
            torch.addbmm(m, a, b),  # 计算 m + torch.bmm(a, b)
            torch.addmm(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),  # 计算 torch.mm(torch.randn(2, 3), torch.randn(2, 3)) + torch.randn(3, 3)
            torch.addmv(torch.randn(2), torch.randn(2, 3), torch.randn(3)),  # 计算 torch.mv(torch.randn(2, 3), torch.randn(3)) + torch.randn(2)
            torch.addr(torch.zeros(3, 3), v, v),  # 计算 torch.outer(v, v)
            torch.baddbmm(m, a, b),  # 计算 m + torch.bmm(a, b)
            torch.bmm(a, b),  # 计算 torch.bmm(a, b)
            torch.chain_matmul(torch.randn(3, 3), torch.randn(3, 3), torch.randn(3, 3)),  # 计算 torch.mm(torch.mm(torch.randn(3, 3), torch.randn(3, 3)), torch.randn(3, 3))
            torch.dot(v, v),  # 计算 torch.dot(v, v)，即 v 的内积
            torch.ger(v, v),  # 计算 torch.outer(v, v)
            torch.inner(m, m),  # 计算 m 的迹
            torch.matmul(m, m),  # 计算 torch.mm(m, m)
            torch.matrix_power(m, 2),  # 计算 m 的 2 次方
            torch.matrix_exp(m),  # 计算矩阵 m 的指数
            torch.mm(m, m),  # 计算 torch.mm(m, m)
            torch.mv(m, v),  # 计算 torch.mv(m, v)
            torch.outer(v, v),  # 计算 torch.outer(v, v)
            torch.trapz(m, m),  # 计算 m 的梯形积分
            torch.trapezoid(m, m),  # 计算 m 的梯形积分
            torch.cumulative_trapezoid(m, m),  # 计算 m 的累积梯形积分
            torch.vdot(v, v),  # 计算 torch.dot(v, v)，即 v 的内积
        )
```