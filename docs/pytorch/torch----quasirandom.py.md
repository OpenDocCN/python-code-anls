# `.\pytorch\torch\quasirandom.py`

```py
```python`
# 允许未类型化函数的 mypy 配置
# 这是 mypy 的一个配置选项，允许函数没有类型注解
# 作用范围是当前模块中的所有代码
# 这样做有助于在使用 mypy 检查类型时避免一些错误
# mypy: allow-untyped-defs
from typing import Optional

# 导入 torch 库，用于深度学习框架中的张量操作
import torch


class SobolEngine:
    r"""
    SobolEngine 类是生成（扰动的）Sobol 序列的引擎。Sobol 序列是低差异准随机序列的一个例子。

    该实现的 Sobol 序列引擎能够生成最大维度为 21201 的序列。它使用来自 https://web.maths.unsw.edu.au/~fkuo/sobol/ 的方向数字，
    使用 D(6) 搜索标准生成最大维度为 21201 的 Sobol 序列，这是作者推荐的选择。

    参考文献：
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    参数:
        dimension (Int): 要生成序列的维度
        scramble (bool, optional): 将此设置为 ``True`` 将生成扰动的 Sobol 序列。扰动可以生成更好的 Sobol 序列。默认值: ``False``。
        seed (Int, optional): 这是扰动的种子。如果指定，随机数生成器的种子设置为此种子。否则，使用随机种子。默认值: ``None``

    示例::
        >>> # xdoctest: +SKIP("unseeded random state")
        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
        >>> soboleng.draw(3)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])
    """
    MAXBIT = 30  # 最大位数，用于定义 Sobol 序列的精度
    MAXDIM = 21201  # 最大维度，定义 Sobol 序列支持的最大维度

    def __init__(self, dimension, scramble=False, seed=None):
        # 检查维度是否合法，必须在 [1, MAXDIM] 范围内
        if dimension > self.MAXDIM or dimension < 1:
            raise ValueError(
                "Supported range of dimensionality "
                f"for SobolEngine is [1, {self.MAXDIM}]"
            )

        # 初始化种子、扰动和维度
        self.seed = seed
        self.scramble = scramble
        self.dimension = dimension

        # 使用 CPU 设备初始化 Sobol 序列状态
        cpu = torch.device("cpu")

        # 创建一个张量，存储 Sobol 序列的状态，维度为 (dimension, MAXBIT)
        self.sobolstate = torch.zeros(
            dimension, self.MAXBIT, device=cpu, dtype=torch.long
        )
        # 初始化 Sobol 序列状态
        torch._sobol_engine_initialize_state_(self.sobolstate, self.dimension)

        # 如果不扰动，创建一个全零的移位向量
        if not self.scramble:
            self.shift = torch.zeros(self.dimension, device=cpu, dtype=torch.long)
        else:
            self._scramble()  # 扰动操作

        # 创建一个拷贝的张量用于连续内存格式
        self.quasi = self.shift.clone(memory_format=torch.contiguous_format)
        # 计算第一个点的值，进行位移和归一化
        self._first_point = (self.quasi / 2**self.MAXBIT).reshape(1, -1)
        # 初始化生成的点数
        self.num_generated = 0
    def draw(
        self,
        n: int = 1,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        # 如果未指定数据类型，使用默认的数据类型
        if dtype is None:
            dtype = torch.get_default_dtype()

        # 如果是第一次生成样本
        if self.num_generated == 0:
            # 如果生成单个样本点
            if n == 1:
                # 将第一个点转换成指定的数据类型，并作为结果返回
                result = self._first_point.to(dtype)
            else:
                # 从 Sobol 序列中生成 n-1 个点，并更新状态
                result, self.quasi = torch._sobol_engine_draw(
                    self.quasi,
                    n - 1,
                    self.sobolstate,
                    self.dimension,
                    self.num_generated,
                    dtype=dtype,
                )
                # 将第一个点和生成的结果连接起来，按最后一个维度连接
                result = torch.cat((self._first_point.to(dtype), result), dim=-2)
        else:
            # 从 Sobol 序列中生成 n 个点，并更新状态
            result, self.quasi = torch._sobol_engine_draw(
                self.quasi,
                n,
                self.sobolstate,
                self.dimension,
                self.num_generated - 1,
                dtype=dtype,
            )

        # 更新生成的样本数
        self.num_generated += n

        # 如果指定了输出张量
        if out is not None:
            # 调整输出张量的大小为生成结果的大小，并复制生成的结果到输出张量
            out.resize_as_(result).copy_(result)
            return out

        # 返回生成的结果
        return result

    def draw_base2(
        self,
        m: int,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        r"""
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
        # 计算要生成的点数 n
        n = 2**m
        # 计算总共已生成的点数
        total_n = self.num_generated + n
        # 检查 n 是否为 2 的幂次方，因为 Sobol 点的平衡性质要求如此
        if not (total_n & (total_n - 1) == 0):
            raise ValueError(
                "The balance properties of Sobol' points require "
                f"n to be a power of 2. {self.num_generated} points have been "
                f"previously generated, then: n={self.num_generated}+2**{m}={total_n}. "
                "If you still want to do this, please use "
                "'SobolEngine.draw()' instead."
            )
        # 调用 draw 方法生成 Sobol 点序列，并返回结果
        return self.draw(n=n, out=out, dtype=dtype)

    def reset(self):
        r"""
        Function to reset the ``SobolEngine`` to base state.
        """
        # 将 quasi 属性复制为 shift 属性，重置已生成的点数为 0，并返回 self
        self.quasi.copy_(self.shift)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        r"""
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
        # 如果还没有生成过点，则调用 torch._sobol_engine_ff_ 函数进行快进
        if self.num_generated == 0:
            torch._sobol_engine_ff_(
                self.quasi, n - 1, self.sobolstate, self.dimension, self.num_generated
            )
        else:
            torch._sobol_engine_ff_(
                self.quasi, n, self.sobolstate, self.dimension, self.num_generated - 1
            )
        # 更新已生成的点数并返回 self
        self.num_generated += n
        return self

    def _scramble(self):
        g: Optional[torch.Generator] = None
        # 如果设置了种子，则创建一个 torch.Generator 并用该种子进行初始化
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)

        cpu = torch.device("cpu")

        # 生成 shift 向量
        shift_ints = torch.randint(
            2, (self.dimension, self.MAXBIT), device=cpu, generator=g
        )
        self.shift = torch.mv(
            shift_ints, torch.pow(2, torch.arange(0, self.MAXBIT, device=cpu))
        )

        # 生成下三角矩阵（跨维度堆叠）
        ltm_dims = (self.dimension, self.MAXBIT, self.MAXBIT)
        ltm = torch.randint(2, ltm_dims, device=cpu, generator=g).tril()

        # 调用 torch._sobol_engine_scramble_ 进行乱序处理
        torch._sobol_engine_scramble_(self.sobolstate, ltm, self.dimension)
    def __repr__(self):
        # 创建一个列表，包含对象维度信息的格式化字符串
        fmt_string = [f"dimension={self.dimension}"]
        # 如果对象的scramble属性为True，添加一个格式化字符串指示混淆为真
        if self.scramble:
            fmt_string += ["scramble=True"]
        # 如果对象的seed属性不为None，添加一个格式化字符串显示种子值
        if self.seed is not None:
            fmt_string += [f"seed={self.seed}"]
        # 返回对象类名加上格式化后的属性信息字符串
        return self.__class__.__name__ + "(" + ", ".join(fmt_string) + ")"
```