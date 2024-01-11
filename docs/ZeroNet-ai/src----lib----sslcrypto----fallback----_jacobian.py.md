# `ZeroNet\src\lib\sslcrypto\fallback\_jacobian.py`

```
# 导入 _util 模块中的 inverse 函数
from ._util import inverse

# 定义 JacobianCurve 类
class JacobianCurve:
    # 初始化函数，接受参数 p, n, a, b, g
    def __init__(self, p, n, a, b, g):
        # 初始化对象的属性
        self.p = p
        self.n = n
        self.a = a
        self.b = b
        self.g = g
        # 计算 n 的二进制表示长度
        self.n_length = len(bin(self.n).replace("0b", ""))

    # 判断点 p 是否为无穷远点
    def isinf(self, p):
        return p[0] == 0 and p[1] == 0

    # 将点 p 转换为雅可比坐标系表示
    def to_jacobian(self, p):
        return p[0], p[1], 1
    # 计算椭圆曲线上点的双倍
    def jacobian_double(self, p):
        # 如果点的 y 坐标为 0，则返回特定值
        if not p[1]:
            return 0, 0, 0
        # 计算 y 坐标的平方
        ysq = (p[1] ** 2) % self.p
        # 计算 s 值
        s = (4 * p[0] * ysq) % self.p
        # 计算 m 值
        m = (3 * p[0] ** 2 + self.a * p[2] ** 4) % self.p
        # 计算新的 x、y、z 坐标
        nx = (m ** 2 - 2 * s) % self.p
        ny = (m * (s - nx) - 8 * ysq ** 2) % self.p
        nz = (2 * p[1] * p[2]) % self.p
        return nx, ny, nz


    # 计算椭圆曲线上点的相加
    def jacobian_add(self, p, q):
        # 如果点 p 的 y 坐标为 0，则返回点 q
        if not p[1]:
            return q
        # 如果点 q 的 y 坐标为 0，则返回点 p
        if not q[1]:
            return p
        # 计算 u1、u2、s1、s2 值
        u1 = (p[0] * q[2] ** 2) % self.p
        u2 = (q[0] * p[2] ** 2) % self.p
        s1 = (p[1] * q[2] ** 3) % self.p
        s2 = (q[1] * p[2] ** 3) % self.p
        # 根据不同情况计算新的 x、y、z 坐标
        if u1 == u2:
            if s1 != s2:
                return (0, 0, 1)
            return self.jacobian_double(p)
        h = u2 - u1
        r = s2 - s1
        h2 = (h * h) % self.p
        h3 = (h * h2) % self.p
        u1h2 = (u1 * h2) % self.p
        nx = (r ** 2 - h3 - 2 * u1h2) % self.p
        ny = (r * (u1h2 - nx) - s1 * h3) % self.p
        nz = (h * p[2] * q[2]) % self.p
        return (nx, ny, nz)


    # 从雅可比坐标系转换回普通坐标系
    def from_jacobian(self, p):
        z = inverse(p[2], self.p)
        return (p[0] * z ** 2) % self.p, (p[1] * z ** 3) % self.p


    # 计算椭圆曲线上点的倍乘
    def jacobian_multiply(self, a, n, secret=False):
        # 如果点 a 的 y 坐标为 0 或 n 为 0，则返回特定值
        if a[1] == 0 or n == 0:
            return 0, 0, 1
        # 如果 n 为 1，则返回点 a
        if n == 1:
            return a
        # 如果 n 为负数或大于等于 self.n，则调整 n 的值
        if n < 0 or n >= self.n:
            return self.jacobian_multiply(a, n % self.n, secret)
        # 计算点 a 的 n 倍
        half = self.jacobian_multiply(a, n // 2, secret)
        half_sq = self.jacobian_double(half)
        # 如果 secret 为 True，则进行特定计算
        if secret:
            # 一个常量时间的实现
            half_sq_a = self.jacobian_add(half_sq, a)
            if n % 2 == 0:
                result = half_sq
            if n % 2 == 1:
                result = half_sq_a
            return result
        else:
            if n % 2 == 0:
                return half_sq
            return self.jacobian_add(half_sq, a)
    # 计算雅可比坐标系下的 Shamir's Trick，用于快速计算椭圆曲线上的点的乘法
    def jacobian_shamir(self, a, n, b, m):
        # 计算 a+b 的雅可比坐标
        ab = self.jacobian_add(a, b)
        # 对 n 进行取模运算，确保 n 在合适的范围内
        if n < 0 or n >= self.n:
            n %= self.n
        # 对 m 进行取模运算，确保 m 在合适的范围内
        if m < 0 or m >= self.n:
            m %= self.n
        # 初始化结果为无穷远点
        res = 0, 0, 1  # point on infinity
        # 从高位到低位遍历 n 的比特位
        for i in range(self.n_length - 1, -1, -1):
            # 对结果进行倍增操作
            res = self.jacobian_double(res)
            # 获取 n 的当前比特位
            has_n = n & (1 << i)
            # 获取 m 的当前比特位
            has_m = m & (1 << i)
            # 根据 n 和 m 的当前比特位进行相应的操作
            if has_n:
                if has_m == 0:
                    res = self.jacobian_add(res, a)
                if has_m != 0:
                    res = self.jacobian_add(res, ab)
            else:
                if has_m == 0:
                    res = self.jacobian_add(res, (0, 0, 1))  # Try not to leak
                if has_m != 0:
                    res = self.jacobian_add(res, b)
        # 返回计算结果
        return res


    # 快速计算椭圆曲线上的点的乘法
    def fast_multiply(self, a, n, secret=False):
        return self.from_jacobian(self.jacobian_multiply(self.to_jacobian(a), n, secret))


    # 快速计算椭圆曲线上的点的加法
    def fast_add(self, a, b):
        return self.from_jacobian(self.jacobian_add(self.to_jacobian(a), self.to_jacobian(b)))


    # 快速计算椭圆曲线上的 Shamir's Trick
    def fast_shamir(self, a, n, b, m):
        return self.from_jacobian(self.jacobian_shamir(self.to_jacobian(a), n, self.to_jacobian(b), m))


    # 检查点是否在椭圆曲线上
    def is_on_curve(self, a):
        x, y = a
        # 简单的算术检查
        if (pow(x, 3, self.p) + self.a * x + self.b) % self.p != y * y % self.p:
            return False
        # nP = 无穷远点
        return self.isinf(self.jacobian_multiply(self.to_jacobian(a), self.n))
```