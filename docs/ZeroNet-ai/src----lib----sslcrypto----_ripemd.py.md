# `ZeroNet\src\lib\sslcrypto\_ripemd.py`

```
# 版权声明，版权归 Markus Friedl 所有
#
# 在源代码和二进制形式下的再发布和使用，无论是否经过修改，都需要满足以下条件：
# 1. 源代码的再发布必须保留上述版权声明、条件列表和以下免责声明。
# 2. 二进制形式的再发布必须在文档和/或其他提供的材料中复制上述版权声明、条件列表和以下免责声明。
#
# 本软件由作者提供，不提供任何明示或暗示的担保，包括但不限于对适销性和特定用途的适用性的暗示担保。
# 无论在任何情况下，作者都不对任何直接、间接、附带、特殊、惩罚性或后果性的损害（包括但不限于替代商品或服务的采购；使用、数据或利润的损失；或业务中断）承担责任，无论是在合同、严格责任或侵权行为（包括疏忽或其他方式）的任何理论下，即使已被告知可能发生此类损害。

# 忽略 pylint 的检查
# 导入 sys 模块
import sys

# 设置摘要大小
digest_size = 20
digestsize = 20

# 定义 RIPEMD160 类
class RIPEMD160:
    """
    返回一个新的 RIPEMD160 对象。可以提供一个可选的字符串参数；
    如果提供了，这个字符串将被自动进行哈希处理。
    """
    
    # 初始化方法
    def __init__(self, arg=None):
        # 创建 RMDContext 对象
        self.ctx = RMDContext()
        # 如果提供了参数，则调用 update 方法
        if arg:
            self.update(arg)
        # 初始化摘要为空
        self.dig = None
        
    # 更新方法
    def update(self, arg):
        # 调用 RMD160Update 方法更新上下文和摘要
        RMD160Update(self.ctx, arg, len(arg))
        # 将摘要置为空
        self.dig = None
        
    # 摘要方法
    def digest(self):
        # 如果摘要不为空，则返回摘要
        if self.dig:
            return self.dig
        # 复制上下文，计算摘要，更新上下文，返回摘要
        ctx = self.ctx.copy()
        self.dig = RMD160Final(self.ctx)
        self.ctx = ctx
        return self.dig
    # 计算消息摘要的十六进制表示
    def hexdigest(self):
        # 调用 digest 方法计算消息摘要
        dig = self.digest()
        # 初始化十六进制表示的消息摘要
        hex_digest = ""
        # 遍历消息摘要的每个字节，将其转换为十六进制表示并拼接到 hex_digest 中
        for d in dig:
            hex_digest += "%02x" % d
        # 返回消息摘要的十六进制表示
        return hex_digest
    
    # 创建当前对象的深层副本
    def copy(self):
        # 导入 copy 模块
        import copy
        # 返回当前对象的深层副本
        return copy.deepcopy(self)
def new(arg=None):
    """
    Return a new RIPEMD160 object. An optional string argument
    may be provided; if present, this string will be automatically
    hashed.
    """
    # 返回一个新的 RIPEMD160 对象，如果提供了可选的字符串参数，则自动对该字符串进行哈希处理
    return RIPEMD160(arg)



#
# Private.
#

class RMDContext:
    def __init__(self):
        self.state = [0x67452301, 0xEFCDAB89, 0x98BADCFE,
                      0x10325476, 0xC3D2E1F0] # uint32
        self.count = 0 # uint64
        self.buffer = [0] * 64 # uchar
    def copy(self):
        ctx = RMDContext()
        ctx.state = self.state[:]
        ctx.count = self.count
        ctx.buffer = self.buffer[:]
        return ctx

K0 = 0x00000000
K1 = 0x5A827999
K2 = 0x6ED9EBA1
K3 = 0x8F1BBCDC
K4 = 0xA953FD4E

KK0 = 0x50A28BE6
KK1 = 0x5C4DD124
KK2 = 0x6D703EF3
KK3 = 0x7A6D76E9
KK4 = 0x00000000

def ROL(n, x):
    return ((x << n) & 0xffffffff) | (x >> (32 - n))

def F0(x, y, z):
    return x ^ y ^ z

def F1(x, y, z):
    return (x & y) | (((~x) % 0x100000000) & z)

def F2(x, y, z):
    return (x | ((~y) % 0x100000000)) ^ z

def F3(x, y, z):
    return (x & z) | (((~z) % 0x100000000) & y)

def F4(x, y, z):
    return x ^ (y | ((~z) % 0x100000000))

def R(a, b, c, d, e, Fj, Kj, sj, rj, X):
    a = ROL(sj, (a + Fj(b, c, d) + X[rj] + Kj) % 0x100000000) + e
    c = ROL(10, c)
    return a % 0x100000000, c

PADDING = [0x80] + [0] * 63

import sys
import struct

def RMD160Transform(state, block): # uint32 state[5], uchar block[64]
    x = [0] * 16
    if sys.byteorder == "little":
        x = struct.unpack("<16L", bytes(block[0:64]))
    else:
        raise ValueError("Big-endian platforms are not supported")
    a = state[0]
    b = state[1]
    c = state[2]
    d = state[3]
    e = state[4]

    # Round 1
    a, c = R(a, b, c, d, e, F0, K0, 11,  0, x)
    e, b = R(e, a, b, c, d, F0, K0, 14,  1, x)
    d, a = R(d, e, a, b, c, F0, K0, 15,  2, x)
    c, e = R(c, d, e, a, b, F0, K0, 12,  3, x)
    b, d = R(b, c, d, e, a, F0, K0,  5,  4, x)
    a, c = R(a, b, c, d, e, F0, K0,  8,  5, x)
    e, b = R(e, a, b, c, d, F0, K0,  7,  6, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F0, K0,  9,  7, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F0, K0, 11,  8, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F0, K0, 13,  9, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F0, K0, 14, 10, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F0, K0, 15, 11, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F0, K0,  6, 12, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F0, K0,  7, 13, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F0, K0,  9, 14, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F0, K0,  8, 15, x)  # 调用函数 R，更新 a 和 c 的值
    # Round 2
    e, b = R(e, a, b, c, d, F1, K1,  7,  7, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F1, K1,  6,  4, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F1, K1,  8, 13, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F1, K1, 13,  1, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F1, K1, 11, 10, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F1, K1,  9,  6, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F1, K1,  7, 15, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F1, K1, 15,  3, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F1, K1,  7, 12, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F1, K1, 12,  0, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F1, K1, 15,  9, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F1, K1,  9,  5, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F1, K1, 11,  2, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F1, K1,  7, 14, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F1, K1, 13, 11, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F1, K1, 12,  8, x)  # 调用函数 R，更新 e 和 b 的值
    # Round 3
    d, a = R(d, e, a, b, c, F2, K2, 11,  3, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F2, K2, 13, 10, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F2, K2,  6, 14, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F2, K2,  7,  4, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F2, K2, 14,  9, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F2, K2,  9, 15, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F2, K2, 13,  8, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F2, K2, 15,  1, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F2, K2, 14,  2, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F2, K2,  8,  7, x)  # 调用函数 R，更新 e 和 b 的值
    d, a = R(d, e, a, b, c, F2, K2, 13,  0, x)  # 调用函数 R，更新 d 和 a 的值
    c, e = R(c, d, e, a, b, F2, K2,  6,  6, x)  # 调用函数 R，更新 c 和 e 的值
    b, d = R(b, c, d, e, a, F2, K2,  5, 13, x)  # 调用函数 R，更新 b 和 d 的值
    a, c = R(a, b, c, d, e, F2, K2, 12, 11, x)  # 调用函数 R，更新 a 和 c 的值
    e, b = R(e, a, b, c, d, F2, K2,  7,  5, x)  # 调用函数 R，更新 e 和 b 的值
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F2, K2,  5, 12, x) # #47
    # Round 4
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F3, K3, 11,  1, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F3, K3, 12,  9, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F3, K3, 14, 11, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F3, K3, 15, 10, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F3, K3, 14,  0, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F3, K3, 15,  8, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F3, K3,  9, 12, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F3, K3,  8,  4, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F3, K3,  9, 13, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F3, K3, 14,  3, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F3, K3,  5,  7, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F3, K3,  6, 15, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F3, K3,  8, 14, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F3, K3,  6,  5, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F3, K3,  5,  6, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F3, K3, 12,  2, x) # #63
    # Round 5
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F4, K4,  9,  4, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F4, K4, 15,  0, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F4, K4,  5,  5, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F4, K4, 11,  9, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F4, K4,  6,  7, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F4, K4,  8, 12, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F4, K4, 13,  2, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F4, K4, 12, 10, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F4, K4,  5, 14, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F4, K4, 12,  1, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F4, K4, 13,  3, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F4, K4, 14,  8, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F4, K4, 11, 11, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F4, K4,  8,  6, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F4, K4,  5, 15, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 b 和 d
    b, d = R(b, c, d, e, a, F4, K4,  6, 13, x) # #79

    # 保存变量的当前值
    aa = a
    bb = b
    cc = c
    dd = d
    ee = e

    # 保存状态变量的当前值
    a = state[0]
    b = state[1]
    c = state[2]
    d = state[3]
    e = state[4]    

    # Parallel round 1
    # 调用函数 R() 进行一轮加密运算，并更新变量 a 和 c
    a, c = R(a, b, c, d, e, F4, KK0,  8,  5, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 e 和 b
    e, b = R(e, a, b, c, d, F4, KK0,  9, 14, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 d 和 a
    d, a = R(d, e, a, b, c, F4, KK0,  9,  7, x)
    # 调用函数 R() 进行一轮加密运算，并更新变量 c 和 e
    c, e = R(c, d, e, a, b, F4, KK0, 11,  0, x)
    b, d = R(b, c, d, e, a, F4, KK0, 13,  9, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F4, KK0, 15,  2, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F4, KK0, 15, 11, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F4, KK0,  5,  4, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F4, KK0,  7, 13, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F4, KK0,  7,  6, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F4, KK0,  8, 15, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F4, KK0, 11,  8, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F4, KK0, 14,  1, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F4, KK0, 14, 10, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F4, KK0, 12,  3, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F4, KK0,  6, 12, x) # #15
    # 调用函数R，对参数进行处理并赋值给a和c，同时标记为第15行
    # Parallel round 2
    e, b = R(e, a, b, c, d, F3, KK1,  9,  6, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F3, KK1, 13, 11, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F3, KK1, 15,  3, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F3, KK1,  7,  7, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F3, KK1, 12,  0, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F3, KK1,  8, 13, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F3, KK1,  9,  5, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F3, KK1, 11, 10, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F3, KK1,  7, 14, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F3, KK1,  7, 15, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F3, KK1, 12,  8, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F3, KK1,  7, 12, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F3, KK1,  6,  4, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F3, KK1, 15,  9, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F3, KK1, 13,  1, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F3, KK1, 11,  2, x) # #31
    # 调用函数R，对参数进行处理并赋值给e和b，同时标记为第31行
    # Parallel round 3
    d, a = R(d, e, a, b, c, F2, KK2,  9, 15, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F2, KK2,  7,  5, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F2, KK2, 15,  1, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F2, KK2, 11,  3, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F2, KK2,  8,  7, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F2, KK2,  6, 14, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F2, KK2,  6,  6, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    b, d = R(b, c, d, e, a, F2, KK2, 14,  9, x)
    # 调用函数R，对参数进行处理并赋值给b和d
    a, c = R(a, b, c, d, e, F2, KK2, 12, 11, x)
    # 调用函数R，对参数进行处理并赋值给a和c
    e, b = R(e, a, b, c, d, F2, KK2, 13,  8, x)
    # 调用函数R，对参数进行处理并赋值给e和b
    d, a = R(d, e, a, b, c, F2, KK2,  5, 12, x)
    # 调用函数R，对参数进行处理并赋值给d和a
    c, e = R(c, d, e, a, b, F2, KK2, 14,  2, x)
    # 调用函数R，对参数进行处理并赋值给c和e
    # 调用函数 R 处理参数，并更新变量 b, d
    b, d = R(b, c, d, e, a, F2, KK2, 13, 10, x)
    # 调用函数 R 处理参数，并更新变量 a, c
    a, c = R(a, b, c, d, e, F2, KK2, 13,  0, x)
    # 调用函数 R 处理参数，并更新变量 e, b
    e, b = R(e, a, b, c, d, F2, KK2,  7,  4, x)
    # 调用函数 R 处理参数，并更新变量 d, a
    d, a = R(d, e, a, b, c, F2, KK2,  5, 13, x) # #47
    # 并行轮 4
    c, e = R(c, d, e, a, b, F1, KK3, 15,  8, x)
    b, d = R(b, c, d, e, a, F1, KK3,  5,  6, x)
    a, c = R(a, b, c, d, e, F1, KK3,  8,  4, x)
    e, b = R(e, a, b, c, d, F1, KK3, 11,  1, x)
    d, a = R(d, e, a, b, c, F1, KK3, 14,  3, x)
    c, e = R(c, d, e, a, b, F1, KK3, 14, 11, x)
    b, d = R(b, c, d, e, a, F1, KK3,  6, 15, x)
    a, c = R(a, b, c, d, e, F1, KK3, 14,  0, x)
    e, b = R(e, a, b, c, d, F1, KK3,  6,  5, x)
    d, a = R(d, e, a, b, c, F1, KK3,  9, 12, x)
    c, e = R(c, d, e, a, b, F1, KK3, 12,  2, x)
    b, d = R(b, c, d, e, a, F1, KK3,  9, 13, x)
    a, c = R(a, b, c, d, e, F1, KK3, 12,  9, x)
    e, b = R(e, a, b, c, d, F1, KK3,  5,  7, x)
    d, a = R(d, e, a, b, c, F1, KK3, 15, 10, x)
    c, e = R(c, d, e, a, b, F1, KK3,  8, 14, x) # #63
    # 并行轮 5
    b, d = R(b, c, d, e, a, F0, KK4,  8, 12, x)
    a, c = R(a, b, c, d, e, F0, KK4,  5, 15, x)
    e, b = R(e, a, b, c, d, F0, KK4, 12, 10, x)
    d, a = R(d, e, a, b, c, F0, KK4,  9,  4, x)
    c, e = R(c, d, e, a, b, F0, KK4, 12,  1, x)
    b, d = R(b, c, d, e, a, F0, KK4,  5,  5, x)
    a, c = R(a, b, c, d, e, F0, KK4, 14,  8, x)
    e, b = R(e, a, b, c, d, F0, KK4,  6,  7, x)
    d, a = R(d, e, a, b, c, F0, KK4,  8,  6, x)
    c, e = R(c, d, e, a, b, F0, KK4, 13,  2, x)
    b, d = R(b, c, d, e, a, F0, KK4,  6, 13, x)
    a, c = R(a, b, c, d, e, F0, KK4,  5, 14, x)
    e, b = R(e, a, b, c, d, F0, KK4, 15,  0, x)
    d, a = R(d, e, a, b, c, F0, KK4, 13,  3, x)
    c, e = R(c, d, e, a, b, F0, KK4, 11,  9, x)
    b, d = R(b, c, d, e, a, F0, KK4, 11, 11, x) # #79

    # 更新状态变量 state
    t = (state[1] + cc + d) % 0x100000000
    state[1] = (state[2] + dd + e) % 0x100000000
    state[2] = (state[3] + ee + a) % 0x100000000
    state[3] = (state[4] + aa + b) % 0x100000000
    # 将 state 列表中索引为 4 的元素更新为 (state[0] + bb + c) 对 0x100000000 取模的结果
    state[4] = (state[0] + bb + c) % 0x100000000
    # 将 state 列表中索引为 0 的元素更新为 t 对 0x100000000 取模的结果
    state[0] = t % 0x100000000
# 更新 RMD160 上下文
def RMD160Update(ctx, inp, inplen):
    # 如果输入是字符串，则将其转换为 ASCII 码列表
    if type(inp) == str:
        inp = [ord(i)&0xff for i in inp]
    
    # 计算当前缓冲区中已有的字节数
    have = int((ctx.count // 8) % 64)
    inplen = int(inplen)
    # 计算还需要多少字节才能填满一个块
    need = 64 - have
    # 更新上下文中的字节数
    ctx.count += 8 * inplen
    off = 0
    # 如果输入的字节数大于等于需要的字节数
    if inplen >= need:
        # 如果缓冲区中已有数据
        if have:
            # 将输入数据填充到缓冲区中
            for i in range(need):
                ctx.buffer[have + i] = inp[i]
            # 对缓冲区中的数据进行转换
            RMD160Transform(ctx.state, ctx.buffer)
            off = need
            have = 0
        # 循环处理输入数据，每次处理 64 字节
        while off + 64 <= inplen:
            RMD160Transform(ctx.state, inp[off:]) #<---
            off += 64
    # 如果还有剩余的数据没有处理
    if off < inplen:
        # 将剩余的数据拷贝到缓冲区中
        for i in range(inplen - off):
            ctx.buffer[have + i] = inp[off + i]

# 完成 RMD160 计算，返回摘要结果
def RMD160Final(ctx):
    # 将上下文中的字节数转换为小端序的 64 位整数
    size = struct.pack("<Q", ctx.count)
    # 计算需要填充的字节数
    padlen = 64 - ((ctx.count // 8) % 64)
    if padlen < 1 + 8:
        padlen += 64
    # 使用填充数据更新上下文
    RMD160Update(ctx, PADDING, padlen - 8)
    # 使用字节数和填充数据进行最后的更新
    RMD160Update(ctx, size, 8)
    # 将最终的状态转换为小端序的 5 个 32 位整数，并返回结果
    return struct.pack("<5L", *ctx.state)


# 对比预期的哈希值和实际计算得到的哈希值
assert "37f332f68db77bd9d7edd4969571ad671cf9dd3b" == new("The quick brown fox jumps over the lazy dog").hexdigest()
assert "132072df690933835eb8b6ad0b77e7b6f14acad7" == new("The quick brown fox jumps over the lazy cog").hexdigest()
assert "9c1185a5c5e9fc54612808977ee8f548b2258d31" == new("").hexdigest()
```