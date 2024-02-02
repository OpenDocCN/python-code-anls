# `ZeroNet\src\lib\sslcrypto\fallback\ecc.py`

```py
# 导入 hmac、os 模块，以及当前目录下的 _jacobian 模块、上一级目录下的 _ecc 模块、当前目录下的 aes 模块、当前目录下的 _util 模块
import hmac
import os
from ._jacobian import JacobianCurve
from .._ecc import ECC
from .aes import aes
from ._util import int_to_bytes, bytes_to_int, inverse, square_root_mod_prime

# 定义椭圆曲线后端类
class EllipticCurveBackend:
    # 初始化方法，接受参数 p, n, a, b, g
    def __init__(self, p, n, a, b, g):
        # 将参数赋值给实例变量
        self.p, self.n, self.a, self.b, self.g = p, n, a, b, g
        # 创建 JacobianCurve 对象
        self.jacobian = JacobianCurve(p, n, a, b, g)

        # 计算公钥长度
        self.public_key_length = (len(bin(p).replace("0b", "")) + 7) // 8
        # 计算 n 的比特长度
        self.order_bitlength = len(bin(n).replace("0b", ""))

    # 定义将整数转换为字节数组的方法
    def _int_to_bytes(self, raw, len=None):
        return int_to_bytes(raw, len or self.public_key_length)

    # 定义解压点的方法，接受参数 public_key
    def decompress_point(self, public_key):
        # 解析并加载数据
        x = bytes_to_int(public_key[1:])
        # 计算 Y
        y_square = (pow(x, 3, self.p) + self.a * x + self.b) % self.p
        try:
            y = square_root_mod_prime(y_square, self.p)
        except Exception:
            raise ValueError("Invalid public key") from None
        if y % 2 != public_key[0] - 0x02:
            y = self.p - y
        return self._int_to_bytes(x), self._int_to_bytes(y)

    # 定义生成新私钥的方法
    def new_private_key(self):
        while True:
            private_key = os.urandom(self.public_key_length)
            if bytes_to_int(private_key) >= self.n:
                continue
            return private_key

    # 定义私钥转换为公钥的方法，接受参数 private_key
    def private_to_public(self, private_key):
        raw = bytes_to_int(private_key)
        x, y = self.jacobian.fast_multiply(self.g, raw)
        return self._int_to_bytes(x), self._int_to_bytes(y)

    # 定义椭圆曲线 Diffie-Hellman 密钥交换的方法，接受参数 private_key, public_key
    def ecdh(self, private_key, public_key):
        x, y = public_key
        x, y = bytes_to_int(x), bytes_to_int(y)
        private_key = bytes_to_int(private_key)
        x, _ = self.jacobian.fast_multiply((x, y), private_key, secret=True)
        return self._int_to_bytes(x)

    # 定义将主体转换为整数的方法，接受参数 subject
    def _subject_to_int(self, subject):
        return bytes_to_int(subject[:(self.order_bitlength + 7) // 8])
    # 使用私钥对给定主题进行签名，并返回签名结果
    def sign(self, subject, raw_private_key, recoverable, is_compressed, entropy):
        # 将主题转换为整数
        z = self._subject_to_int(subject)
        # 将原始私钥转换为整数
        private_key = bytes_to_int(raw_private_key)
        # 将熵转换为整数
        k = bytes_to_int(entropy)

        # 修复 k 的长度，以防止 Minerva 攻击
        ks = k + self.n
        kt = ks + self.n
        ks_len = len(bin(ks).replace("0b", "")) // 8
        kt_len = len(bin(kt).replace("0b", "")) // 8
        if ks_len == kt_len:
            k = kt
        else:
            k = ks
        # 使用 Jacobian 快速乘法计算公钥点
        px, py = self.jacobian.fast_multiply(self.g, k, secret=True)

        # 计算 r
        r = px % self.n
        if r == 0:
            # r 无效
            raise ValueError("Invalid k")

        # 计算 s
        s = (inverse(k, self.n) * (z + (private_key * r))) % self.n
        if s == 0:
            # s 无效
            raise ValueError("Invalid k")

        # 判断是否需要对 s 进行取反
        inverted = False
        if s * 2 >= self.n:
            s = self.n - s
            inverted = True
        # 将 r 和 s 转换为字节流
        rs_buf = self._int_to_bytes(r) + self._int_to_bytes(s)

        # 如果需要可恢复性签名
        if recoverable:
            # 计算恢复 ID
            recid = (py % 2) ^ inverted
            recid += 2 * int(px // self.n)
            # 如果需要压缩格式的签名
            if is_compressed:
                return bytes([31 + recid]) + rs_buf
            else:
                # 如果恢复 ID 大于等于 4，则抛出异常
                if recid >= 4:
                    raise ValueError("Too big recovery ID, use compressed address instead")
                return bytes([27 + recid]) + rs_buf
        else:
            return rs_buf
    # 从签名和主题中恢复公钥
    def recover(self, signature, subject):
        # 将主题转换为整数
        z = self._subject_to_int(subject)

        # 从签名中提取 recid、r、s
        recid = signature[0] - 27 if signature[0] < 31 else signature[0] - 31
        r = bytes_to_int(signature[1:self.public_key_length + 1])
        s = bytes_to_int(signature[self.public_key_length + 1:])

        # 验证边界
        if not 0 <= recid < 2 * (self.p // self.n + 1):
            raise ValueError("Invalid recovery ID")
        if r >= self.n:
            raise ValueError("r is out of bounds")
        if s >= self.n:
            raise ValueError("s is out of bounds")

        # 计算 r 的逆
        rinv = inverse(r, self.n)
        u1 = (-z * rinv) % self.n
        u2 = (s * rinv) % self.n

        # 恢复 R
        rx = r + (recid // 2) * self.n
        if rx >= self.p:
            raise ValueError("Rx is out of bounds")

        # 几乎是从 decompress_point 复制过来的
        ry_square = (pow(rx, 3, self.p) + self.a * rx + self.b) % self.p
        try:
            ry = square_root_mod_prime(ry_square, self.p)
        except Exception:
            raise ValueError("Invalid recovered public key") from None

        # 确保点是正确的
        if ry % 2 != recid % 2:
            # 修正 Ry 的符号
            ry = self.p - ry

        # 使用快速 Shamir 算法计算点的坐标
        x, y = self.jacobian.fast_shamir(self.g, u1, (rx, ry), u2)
        return self._int_to_bytes(x), self._int_to_bytes(y)
    # 验证签名的有效性
    def verify(self, signature, subject, public_key):
        # 将主题转换为整数
        z = self._subject_to_int(subject)

        # 将签名拆分为 r 和 s
        r = bytes_to_int(signature[:self.public_key_length])
        s = bytes_to_int(signature[self.public_key_length:])

        # 验证 r 和 s 的范围
        if r >= self.n:
            raise ValueError("r is out of bounds")
        if s >= self.n:
            raise ValueError("s is out of bounds")

        # 将公钥转换为整数列表
        public_key = [bytes_to_int(c) for c in public_key]

        # 确保公钥在曲线上
        if not self.jacobian.is_on_curve(public_key):
            raise ValueError("Public key is not on curve")

        # 计算 sinv, u1, u2
        sinv = inverse(s, self.n)
        u1 = (z * sinv) % self.n
        u2 = (r * sinv) % self.n

        # 使用快速 Shamir 算法验证签名
        x1, _ = self.jacobian.fast_shamir(self.g, u1, public_key, u2)
        if r != x1 % self.n:
            raise ValueError("Invalid signature")

        return True


    # 派生子密钥
    def derive_child(self, seed, child):
        # 第一轮
        h = hmac.new(key=b"Bitcoin seed", msg=seed, digestmod="sha512").digest()
        private_key1 = h[:32]
        x, y = self.private_to_public(private_key1)
        public_key1 = bytes([0x02 + (y[-1] % 2)]) + x
        private_key1 = bytes_to_int(private_key1)

        # 第二轮
        msg = public_key1 + self._int_to_bytes(child, 4)
        h = hmac.new(key=h[32:], msg=msg, digestmod="sha512").digest()
        private_key2 = bytes_to_int(h[:32])

        return self._int_to_bytes((private_key1 + private_key2) % self.n)


    # 获取后端
    @classmethod
    def get_backend(cls):
        return "fallback"
# 使用椭圆曲线加密算法和AES加密算法创建ECC对象
ecc = ECC(EllipticCurveBackend, aes)
```