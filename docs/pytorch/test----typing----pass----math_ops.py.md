# `.\pytorch\test\typing\pass\math_ops.py`

```py
# flake8: noqa
# 导入math模块和torch模块
import math
import torch

# 生成一个包含4个随机数的张量
a = torch.randn(4)
# 生成一个包含4个随机数的张量
b = torch.randn(4)
# 生成一个包含[-1, -2, 3]的张量，数据类型为torch.int8
t = torch.tensor([-1, -2, 3], dtype=torch.int8)

# 计算张量的绝对值
torch.abs(torch.tensor([-1, -2, 3]))
torch.absolute(torch.tensor([-1, -2, 3]))

# 计算张量元素的反余弦值
torch.acos(a)
torch.arccos(a)

# 计算张量元素的反双曲余弦值
torch.acosh(a.uniform_(1, 2))

# 将张量a和标量20相加
torch.add(a, 20)
# 将张量a和形状为(4,1)的随机张量按元素相加，乘以标量10
torch.add(a, torch.randn(4, 1), alpha=10)
# 将复数张量a + 1j和标量20 + 1j按元素相加
torch.add(a + 1j, 20 + 1j)
# 将复数张量a + 1j和标量20按元素相加，乘以虚数单位1j
torch.add(a + 1j, 20, alpha=1j)

# 使用给定的张量进行元素级加权除法运算
torch.addcdiv(torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), value=0.1)

# 使用给定的张量进行元素级加权乘法运算
torch.addcmul(torch.randn(1, 3), torch.randn(3, 1), torch.randn(1, 3), value=0.1)

# 计算复数张量的幅角，并转换为角度制
torch.angle(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])) * 180 / 3.14159

# 计算张量元素的反正弦值
torch.asin(a)
torch.arcsin(a)

# 计算张量元素的反双曲正弦值
torch.asinh(a)
torch.arcsinh(a)

# 计算张量元素的反正切值
torch.atan(a)
torch.arctan(a)

# 计算张量元素的反双曲正切值
torch.atanh(a.uniform_(-1, 1))
torch.arctanh(a.uniform_(-1, 1))

# 计算张量元素的反正切值（atan2(y, x)）
torch.atan2(a, a)

# 计算整数张量的按位取反
torch.bitwise_not(t)

# 计算整数张量的按位与
torch.bitwise_and(t, torch.tensor([1, 0, 3], dtype=torch.int8))
torch.bitwise_and(torch.tensor([True, True, False]), torch.tensor([False, True, False]))

# 计算整数张量的按位或
torch.bitwise_or(t, torch.tensor([1, 0, 3], dtype=torch.int8))
torch.bitwise_or(torch.tensor([True, True, False]), torch.tensor([False, True, False]))

# 计算整数张量的按位异或
torch.bitwise_xor(t, torch.tensor([1, 0, 3], dtype=torch.int8))

# 计算张量元素的向上取整
torch.ceil(a)

# 将张量的元素限制在指定范围内
torch.clamp(a, min=-0.5, max=0.5)
torch.clamp(a, min=0.5)
torch.clamp(a, max=0.5)
torch.clip(a, min=-0.5, max=0.5)  # clamp的别名

# 计算复数张量的共轭
torch.conj(torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j]))

# 将张量b的符号复制到张量a
torch.copysign(a, 1)
torch.copysign(a, b)

# 计算张量元素的余弦值
torch.cos(a)

# 计算张量元素的双曲余弦值
torch.cosh(a)

# 将角度张量转换为弧度张量
torch.deg2rad(torch.tensor([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]]))

# 将张量元素按标量进行除法运算
torch.div(x, 0.5)
# 将矩阵p的元素按张量q的对应元素进行除法运算
torch.div(p, q)
# 将矩阵p的元素按张量q的对应元素进行除法运算，指定截断模式为"trunc"
torch.divide(p, q, rounding_mode="trunc")
# 将矩阵p的元素按张量q的对应元素进行除法运算，指定截断模式为"floor"
torch.divide(p, q, rounding_mode="floor")

# 计算张量元素的digamma函数值
torch.digamma(torch.tensor([1, 0.5]))

# 计算张量元素的误差函数值
torch.erf(torch.tensor([0, -1.0, 10.0]))

# 计算张量元素的余误差函数值
torch.erfc(torch.tensor([0, -1.0, 10.0]))

# 计算张量元素的反误差函数值
torch.erfinv(torch.tensor([0, 0.5, -1.0]))

# 计算张量元素的指数值
torch.exp(torch.tensor([0, math.log(2.0)]))

# 计算张量元素的2的幂次方值
torch.exp2(torch.tensor([0, math.log2(2.0), 3, 4]))

# 计算张量元素的exp(x) - 1值
torch.expm1(torch.tensor([0, math.log(2.0)]))

# 对张量进行按通道仿真量化（量化操作通常用于量化神经网络中的权重或激活值）
x = torch.randn(2, 2, 2)
scales = (torch.randn(2) + 1) * 0.05
zero_points = torch.zeros(2).to(torch.long)
torch.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)

# 对张量进行全局仿真量化
torch.fake_quantize_per_tensor_affine(a, 0.1, 0, 0, 255)

# 计算张量的元素级次方
# 计算随机生成的整数张量的每个元素的平方
torch.float_power(torch.randint(10, (4,)), 2)

# 计算给定张量中每个元素的指数幂，指数由第二个张量提供
torch.float_power(torch.arange(1, 5), torch.tensor([2, -3, 4, -5]))

# 计算张量的向下取整
torch.floor(a)

# 对两个张量进行逐元素的向下取整除法
torch.floor_divide(torch.tensor([4.0, 3.0]), torch.tensor([2.0, 2.0]))
torch.floor_divide(torch.tensor([4.0, 3.0]), 1.4)

# 计算张量的元素级余数
torch.fmod(torch.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
torch.fmod(torch.tensor([1, 2, 3, 4, 5]), 1.5)

# 计算张量每个元素的小数部分
torch.frac(torch.tensor([1, 2.5, -3.2]))

# 返回复数张量的虚部
torch.randn(4, dtype=torch.cfloat).imag

# 计算张量元素的 ldexp 值，将每个元素乘以 2 的指数次方
torch.ldexp(torch.tensor([1.0]), torch.tensor([1]))
torch.ldexp(torch.tensor([1.0]), torch.tensor([1, 2, 3, 4]))

# 执行线性插值，根据权重计算两个张量之间的插值
start = torch.arange(1.0, 5.0)
end = torch.empty(4).fill_(10)
torch.lerp(start, end, 0.5)
torch.lerp(start, end, torch.full_like(start, 0.5))

# 计算 gamma 函数的自然对数
torch.lgamma(torch.arange(0.5, 2, 0.5))

# 计算张量每个元素的自然对数
torch.log(torch.arange(5) + 10)

# 计算张量每个元素的以 10 为底的对数
torch.log10(torch.rand(5))

# 计算张量每个元素的自然对数 (1 + x)
torch.log1p(torch.randn(5))

# 计算张量每个元素的以 2 为底的对数
torch.log2(torch.rand(5))

# 计算两个张量每个元素的对数和的指数
torch.logaddexp(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))

# 计算两个张量每个元素的对数和的以 2 为底的指数
torch.logaddexp2(torch.tensor([-1.0]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp2(torch.tensor([-100.0, -200, -300]), torch.tensor([-1.0, -2, -3]))
torch.logaddexp2(torch.tensor([1.0, 2000, 30000]), torch.tensor([-1.0, -2, -3]))

# 计算两个布尔类型张量的逻辑与
torch.logical_and(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
r = torch.tensor([0, 1, 10, 0], dtype=torch.int8)
s = torch.tensor([4, 0, 1, 0], dtype=torch.int8)
torch.logical_and(r, s)
torch.logical_and(r.double(), s.double())
torch.logical_and(r.double(), s)
torch.logical_and(r, s, out=torch.empty(4, dtype=torch.bool))

# 计算布尔类型张量的逻辑非
torch.logical_not(torch.tensor([True, False]))
torch.logical_not(torch.tensor([0, 1, -10], dtype=torch.int8))
torch.logical_not(torch.tensor([0.0, 1.5, -10.0], dtype=torch.double))
torch.logical_not(
    torch.tensor([0.0, 1.0, -10.0], dtype=torch.double),
    out=torch.empty(3, dtype=torch.int16),
)

# 计算两个布尔类型张量的逻辑或
torch.logical_or(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
torch.logical_or(r, s)
torch.logical_or(r.double(), s.double())
torch.logical_or(r.double(), s)
torch.logical_or(r, s, out=torch.empty(4, dtype=torch.bool))

# 计算两个布尔类型张量的逻辑异或
torch.logical_xor(torch.tensor([True, False, True]), torch.tensor([True, False, False]))
torch.logical_xor(r, s)
torch.logical_xor(r.double(), s.double())
torch.logical_xor(r.double(), s)
torch.logical_xor(r, s, out=torch.empty(4, dtype=torch.bool))

# 计算张量每个元素的 logit 函数，即 log(x / (1 - x))
torch.logit(torch.rand(5), eps=1e-6)

# 计算两个张量对应元素之间的欧几里得距离
torch.hypot(torch.tensor([4.0]), torch.tensor([3.0, 4.0, 5.0]))

# 计算每个元素的修正的 0 阶贝塞尔函数
torch.i0(torch.arange(5, dtype=torch.float32))

# 计算正交归一化的不完全伽玛函数和其补函数
a1 = torch.tensor([4.0])
a2 = torch.tensor([3.0, 4.0, 5.0])
torch.igamma(a1, a2)
torch.igammac(a1, a2)

# 计算两个张量每个元素的乘法
torch.mul(torch.randn(3), 100)
# 使用 torch.randn 生成的两个矩阵进行逐元素乘法
torch.multiply(torch.randn(4, 1), torch.randn(1, 4))

# 使用 torch.randn 生成的复数张量与复数进行逐元素乘法
torch.mul(torch.randn(3) + 1j, 100 + 1j)

# 计算多变量 Gamma 函数的对数值，参数为均匀分布在 [1, 2) 区间的 2x3 张量
torch.mvlgamma(torch.empty(2, 3).uniform_(1, 2), 2)

# 将张量中的 NaN 替换为指定的数值，缺省情况下替换为 0.0
w = torch.tensor([float("nan"), float("inf"), -float("inf"), 3.14])
torch.nan_to_num(x)
torch.nan_to_num(x, nan=2.0)
torch.nan_to_num(x, nan=2.0, posinf=1.0)

# 计算张量的逐元素负数
torch.neg(torch.randn(5))

# 计算两个张量之间的最近可表示数（next representable number）
eps = torch.finfo(torch.float32).eps
torch.nextafter(torch.tensor([1, 2]), torch.tensor([2, 1])) == torch.tensor([eps + 1, 2 - eps])

# 计算多变量 Gamma 函数的导数，对于给定的参数和阶数
torch.polygamma(1, torch.tensor([1, 0.5]))
torch.polygamma(2, torch.tensor([1, 0.5]))
torch.polygamma(3, torch.tensor([1, 0.5]))
torch.polygamma(4, torch.tensor([1, 0.5]))

# 计算张量的指数次幂
torch.pow(a, 2)
torch.pow(torch.arange(1.0, 5.0), torch.arange(1.0, 5.0))

# 将张量中的弧度转换为角度
torch.rad2deg(torch.tensor([[3.142, -3.142], [6.283, -6.283], [1.570, -1.570]]))

# 提取复数张量的实部
torch.randn(4, dtype=torch.cfloat).real

# 计算张量的倒数
torch.reciprocal(a)

# 计算张量的元素级余数
torch.remainder(torch.tensor([-3.0, -2, -1, 1, 2, 3]), 2)
torch.remainder(torch.tensor([1, 2, 3, 4, 5]), 1.5)

# 将张量的每个元素四舍五入到最接近的整数
torch.round(a)

# 计算张量的逆平方根
torch.rsqrt(a)

# 计算张量的 sigmoid 函数值
torch.sigmoid(a)

# 计算张量的每个元素的符号值
torch.sign(torch.tensor([0.7, -1.2, 0.0, 2.3]))

# 计算复数张量的符号函数
torch.tensor([3 + 4j, 7 - 24j, 0, 1 + 2j]).sgn()

# 检查张量中每个元素的符号位
torch.signbit(torch.tensor([0.7, -1.2, 0.0, 2.3]))

# 计算张量的每个元素的正弦值
torch.sin(a)

# 计算张量的每个元素的 sinc 函数值
torch.sinc(a)

# 计算张量的每个元素的双曲正弦值
torch.sinh(a)

# 计算张量的每个元素的平方根
torch.sqrt(a)

# 计算张量的每个元素的平方
torch.square(a)

# 计算两个张量的逐元素减法，可指定一个缩放因子 alpha
torch.sub(torch.tensor((1, 2)), torch.tensor((0, 1)), alpha=2)
torch.sub(torch.tensor((1j, 2j)), 1j, alpha=2)
torch.sub(torch.tensor((1j, 2j)), 10, alpha=2j)

# 计算张量的每个元素的正切值
torch.tan(a)

# 计算张量的每个元素的双曲正切值
torch.tanh(a)

# 对张量的每个元素执行截断操作，保留整数部分
torch.trunc(a)

# 计算 f 和 g 张量的每个元素的 x * log(y) 的值，其中 x 为 f，y 为 g
f = torch.zeros(5,)
g = torch.tensor([-1, 0, 1, float("inf"), float("nan")])
torch.xlogy(f, g)
torch.xlogy(f, g)
torch.xlogy(f, 4)
torch.xlogy(2, g)
```