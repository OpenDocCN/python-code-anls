# `ZeroNet\src\lib\sslcrypto\fallback\_util.py`

```
# 将整数转换为字节数组，指定长度
def int_to_bytes(raw, length):
    data = []
    for _ in range(length):
        data.append(raw % 256)  # 取余数，得到最低位的字节
        raw //= 256  # 整数右移8位，相当于去掉最低位的字节
    return bytes(data[::-1])  # 返回字节数组，需要反转顺序


# 将字节数组转换为整数
def bytes_to_int(data):
    raw = 0
    for byte in data:
        raw = raw * 256 + byte  # 左移8位，然后加上下一个字节的值
    return raw


# 计算勒让德符号
def legendre(a, p):
    res = pow(a, (p - 1) // 2, p)  # 计算 a 的 (p-1)/2 次方对 p 取模
    if res == p - 1:
        return -1
    else:
        return res


# 计算模逆
def inverse(a, n):
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % n, n
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % n


# 计算模素数的平方根
def square_root_mod_prime(n, p):
    if n == 0:
        return 0
    if p == 2:
        return n  # 我们不应该到达这里，但这可能会有用
    if legendre(n, p) != 1:
        raise ValueError("No square root")  # 如果 n 不是 p 的二次剩余，抛出异常
    # 优化
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)  # 如果 p % 4 == 3，直接计算平方根并返回
    # 1. 通过分解出2的幂，找到 Q 和 S，使得 p - 1 = Q * 2 ** S，其中 Q 是奇数
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    # 2. 在 Z/pZ 中搜索 z，它是二次非剩余
    z = 1
    while legendre(z, p) != -1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while True:
        if t == 0:
            return 0
        elif t == 1:
            return r
        # 使用重复平方法找到最小的 i，满足 t ** (2 ** i) = 1
        t_sq = t
        i = 0
        for i in range(1, m):
            t_sq = t_sq * t_sq % p
            if t_sq == 1:
                break
        else:
            raise ValueError("Should never get here")
        # 让 b = c ** (2 ** (m - i - 1))
        b = pow(c, 2 ** (m - i - 1), p)
        m = i
        c = b * b % p
        t = t * b * b % p
        r = r * b % p
    return r
```