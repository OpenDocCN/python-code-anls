# `ZeroNet\src\lib\sslcrypto\openssl\ecc.py`

```py
# 导入 ctypes 模块，用于调用 C 语言编写的库函数
import ctypes
# 导入 hmac 模块，用于生成消息认证码
import hmac
# 导入 threading 模块，用于支持多线程编程
import threading
# 从当前目录的 _ecc 模块中导入 ECC 类
from .._ecc import ECC
# 从当前目录的 aes 模块中导入 aes 函数
from .aes import aes
# 从当前目录的 library 模块中导入 lib 和 openssl_backend
from .library import lib, openssl_backend

# 设置返回类型
lib.BN_new.restype = ctypes.POINTER(ctypes.c_char)
lib.BN_bin2bn.restype = ctypes.POINTER(ctypes.c_char)
lib.BN_CTX_new.restype = ctypes.POINTER(ctypes.c_char)
lib.EC_GROUP_new_curve_GFp.restype = ctypes.POINTER(ctypes.c_char)
lib.EC_KEY_new.restype = ctypes.POINTER(ctypes.c_char)
lib.EC_POINT_new.restype = ctypes.POINTER(ctypes.c_char)
lib.EC_KEY_get0_private_key.restype = ctypes.POINTER(ctypes.c_char)
lib.EVP_PKEY_new.restype = ctypes.POINTER(ctypes.c_char)
try:
    lib.EVP_PKEY_CTX_new.restype = ctypes.POINTER(ctypes.c_char)
except AttributeError:
    pass

# 创建线程本地数据对象
thread_local = threading.local()

# 创建锁对象，用于保证 ECC 线程安全
lock = threading.Lock()

# 定义 BN 类
class BN:
    # 定义 BN 类中的 Context 类
    class Context:
        # 初始化方法
        def __init__(self):
            # 创建 BN 上下文
            self.ptr = lib.BN_CTX_new()
            self.lib = lib  # 用于最终处理

        # 析构方法
        def __del__(self):
            # 释放 BN 上下文
            self.lib.BN_CTX_free(self.ptr)

        # 类方法，获取线程安全的上下文
        @classmethod
        def get(cls):
            # 获取线程安全的上下文
            if not hasattr(thread_local, "bn_ctx"):
                thread_local.bn_ctx = cls()
            return thread_local.bn_ctx.ptr
    # 初始化方法，用于创建 BN 对象
    def __init__(self, value=None, link_only=False):
        # 如果指定了 link_only 参数，则将 value 赋值给 bn 属性，_free 属性设置为 False
        if link_only:
            self.bn = value
            self._free = False
        else:
            # 如果 value 为 None，则创建一个新的 BN 对象，_free 属性设置为 True
            if value is None:
                self.bn = lib.BN_new()
                self._free = True
            # 如果 value 是 int 类型且小于 256，则创建一个新的 BN 对象，设置其值为 value，_free 属性设置为 True
            elif isinstance(value, int) and value < 256:
                self.bn = lib.BN_new()
                lib.BN_clear(self.bn)
                lib.BN_add_word(self.bn, value)
                self._free = True
            else:
                # 如果 value 是 int 类型，则将其转换为字节数组，然后创建一个新的 BN 对象，_free 属性设置为 True
                if isinstance(value, int):
                    value = value.to_bytes(128, "big")
                self.bn = lib.BN_bin2bn(value, len(value), None)
                self._free = True

    # 析构方法，用于释放 BN 对象占用的资源
    def __del__(self):
        if self._free:
            lib.BN_free(self.bn)

    # 返回 BN 对象的字节数组表示
    def bytes(self, length=None):
        buf = ctypes.create_string_buffer((len(self) + 7) // 8)
        lib.BN_bn2bin(self.bn, buf)
        buf = bytes(buf)
        if length is None:
            return buf
        else:
            # 如果指定了 length，则在字节数组前补充 0，使其长度达到 length
            if length < len(buf):
                raise ValueError("Too little space for BN")
            return b"\x00" * (length - len(buf)) + buf

    # 返回 BN 对象的整数表示
    def __int__(self):
        value = 0
        for byte in self.bytes():
            value = value * 256 + byte
        return value

    # 返回 BN 对象的比特数
    def __len__(self):
        return lib.BN_num_bits(self.bn)

    # 返回 BN 对象关于指定模数的乘法逆元
    def inverse(self, modulo):
        result = BN()
        if not lib.BN_mod_inverse(result.bn, self.bn, modulo.bn, BN.Context.get()):
            raise ValueError("Could not compute inverse")
        return result

    # 重载 // 运算符，实现 BN 对象的整数除法
    def __floordiv__(self, other):
        if not isinstance(other, BN):
            raise TypeError("Can only divide BN by BN, not {}".format(other))
        result = BN()
        if not lib.BN_div(result.bn, None, self.bn, other.bn, BN.Context.get()):
            raise ZeroDivisionError("Division by zero")
        return result
    # 重载运算符%，实现自定义类型对象与另一个自定义类型对象的取模运算
    def __mod__(self, other):
        # 如果other不是BN类型的对象，则抛出类型错误异常
        if not isinstance(other, BN):
            raise TypeError("Can only divide BN by BN, not {}".format(other))
        # 创建一个新的BN对象
        result = BN()
        # 调用C库函数进行BN对象的取模运算
        if not lib.BN_div(None, result.bn, self.bn, other.bn, BN.Context.get()):
            raise ZeroDivisionError("Division by zero")
        return result

    # 重载运算符+，实现自定义类型对象与另一个自定义类型对象的加法运算
    def __add__(self, other):
        # 如果other不是BN类型的对象，则抛出类型错误异常
        if not isinstance(other, BN):
            raise TypeError("Can only sum BN's, not BN and {}".format(other))
        # 创建一个新的BN对象
        result = BN()
        # 调用C库函数进行BN对象的加法运算
        if not lib.BN_add(result.bn, self.bn, other.bn):
            raise ValueError("Could not sum two BN's")
        return result

    # 重载运算符-，实现自定义类型对象与另一个自定义类型对象的减法运算
    def __sub__(self, other):
        # 如果other不是BN类型的对象，则抛出类型错误异常
        if not isinstance(other, BN):
            raise TypeError("Can only subtract BN's, not BN and {}".format(other))
        # 创建一个新的BN对象
        result = BN()
        # 调用C库函数进行BN对象的减法运算
        if not lib.BN_sub(result.bn, self.bn, other.bn):
            raise ValueError("Could not subtract BN from BN")
        return result

    # 重载运算符*，实现自定义类型对象与另一个自定义类型对象的乘法运算
    def __mul__(self, other):
        # 如果other不是BN类型的对象，则抛出类型错误异常
        if not isinstance(other, BN):
            raise TypeError("Can only multiply BN by BN, not {}".format(other))
        # 创建一个新的BN对象
        result = BN()
        # 调用C库函数进行BN对象的乘法运算
        if not lib.BN_mul(result.bn, self.bn, other.bn, BN.Context.get()):
            raise ValueError("Could not multiply two BN's")
        return result

    # 重载一元运算符-，实现自定义类型对象的取负运算
    def __neg__(self):
        return BN(0) - self

    # 重载取模赋值运算符%=，实现自定义类型对象对另一个自定义类型对象的取模运算并赋值给自身
    # 通过创建临时对象实现更新当前对象并释放旧对象的脏但巧妙的方法
    def __imod__(self, other):
        res = self % other
        self.bn, res.bn = res.bn, self.bn
        return self

    # 重载加法赋值运算符+=，实现自定义类型对象与另一个自定义类型对象的加法运算并赋值给自身
    def __iadd__(self, other):
        res = self + other
        self.bn, res.bn = res.bn, self.bn
        return self

    # 重载减法赋值运算符-=，实现自定义类型对象与另一个自定义类型对象的减法运算并赋值给自身
    def __isub__(self, other):
        res = self - other
        self.bn, res.bn = res.bn, self.bn
        return self

    # 重载乘法赋值运算符*=，实现自定义类型对象与另一个自定义类型对象的乘法运算并赋值给自身
    def __imul__(self, other):
        res = self * other
        self.bn, res.bn = res.bn, self.bn
        return self
    # 定义比较方法，用于比较两个 BN 对象的大小
    def cmp(self, other):
        # 如果 other 不是 BN 类型的对象，则抛出类型错误异常
        if not isinstance(other, BN):
            raise TypeError("Can only compare BN with BN, not {}".format(other))
        # 调用底层库的 BN_cmp 方法进行比较
        return lib.BN_cmp(self.bn, other.bn)
    
    # 重载等于运算符，调用 cmp 方法比较两个对象是否相等
    def __eq__(self, other):
        return self.cmp(other) == 0
    
    # 重载小于运算符，调用 cmp 方法比较两个对象的大小关系
    def __lt__(self, other):
        return self.cmp(other) < 0
    
    # 重载大于运算符，调用 cmp 方法比较两个对象的大小关系
    def __gt__(self, other):
        return self.cmp(other) > 0
    
    # 重载不等于运算符，调用 cmp 方法比较两个对象是否不相等
    def __ne__(self, other):
        return self.cmp(other) != 0
    
    # 重载小于等于运算符，调用 cmp 方法比较两个对象的大小关系
    def __le__(self, other):
        return self.cmp(other) <= 0
    
    # 重载大于等于运算符，调用 cmp 方法比较两个对象的大小关系
    def __ge__(self, other):
        return self.cmp(other) >= 0
    
    # 重载对象的字符串表示形式
    def __repr__(self):
        return "<BN {}>".format(int(self))
    
    # 重载对象的字符串形式
    def __str__(self):
        return str(int(self))
class EllipticCurveBackend:
    # 初始化椭圆曲线后端对象
    def __init__(self, p, n, a, b, g):
        # 获取 OpenSSL 的 BN 上下文
        bn_ctx = BN.Context.get()

        # 保存 lib 对象，用于析构
        self.lib = lib

        # 初始化椭圆曲线参数
        self.p = BN(p)
        self.order = BN(n)
        self.a = BN(a)
        self.b = BN(b)
        self.h = BN((p + n // 2) // n)

        # 线程安全，创建椭圆曲线群对象
        with lock:
            self.group = lib.EC_GROUP_new_curve_GFp(self.p.bn, self.a.bn, self.b.bn, bn_ctx)
            if not self.group:
                raise ValueError("Could not create group object")
            # 将公钥转换为点，并设置生成元和阶
            generator = self._public_key_to_point(g)
            lib.EC_GROUP_set_generator(self.group, generator, self.order.bn, self.h.bn)
        if not self.group:
            raise ValueError("The curve is not supported by OpenSSL")

        # 计算公钥长度
        self.public_key_length = (len(self.p) + 7) // 8

        # 检查是否支持 EVP_PKEY_CTX
        self.is_supported_evp_pkey_ctx = hasattr(lib, "EVP_PKEY_CTX_new")


    # 析构函数，释放椭圆曲线群对象
    def __del__(self):
        self.lib.EC_GROUP_free(self.group)


    # 将私钥转换为 EC_KEY 对象
    def _private_key_to_ec_key(self, private_key):
        # 线程安全，创建 EC_KEY 对象
        eckey = lib.EC_KEY_new()
        lib.EC_KEY_set_group(eckey, self.group)
        if not eckey:
            raise ValueError("Failed to allocate EC_KEY")
        private_key = BN(private_key)
        if not lib.EC_KEY_set_private_key(eckey, private_key.bn):
            lib.EC_KEY_free(eckey)
            raise ValueError("Invalid private key")
        return eckey, private_key


    # 将公钥转换为椭圆曲线上的点
    def _public_key_to_point(self, public_key):
        x = BN(public_key[0])
        y = BN(public_key[1])
        # 创建椭圆曲线点对象，并设置坐标
        point = lib.EC_POINT_new(self.group)
        if not lib.EC_POINT_set_affine_coordinates_GFp(self.group, point, x.bn, y.bn, BN.Context.get()):
            raise ValueError("Could not set public key affine coordinates")
        return point
    # 将公钥转换为椭圆曲线密钥
    def _public_key_to_ec_key(self, public_key):
        # 线程安全
        eckey = lib.EC_KEY_new()
        lib.EC_KEY_set_group(eckey, self.group)
        if not eckey:
            raise ValueError("Failed to allocate EC_KEY")
        try:
            # EC_KEY_set_public_key_affine_coordinates 在 OpenSSL 1.0.0 中不受支持，因此我们不能使用它
            point = self._public_key_to_point(public_key)
            if not lib.EC_KEY_set_public_key(eckey, point):
                raise ValueError("Could not set point")
            lib.EC_POINT_free(point)
            return eckey
        except Exception as e:
            lib.EC_KEY_free(eckey)
            raise e from None


    # 将点转换为仿射坐标
    def _point_to_affine(self, point):
        # 转换为仿射坐标
        x = BN()
        y = BN()
        if lib.EC_POINT_get_affine_coordinates_GFp(self.group, point, x.bn, y.bn, BN.Context.get()) != 1:
            raise ValueError("Failed to convert public key to affine coordinates")
        # 转换为二进制
        if (len(x) + 7) // 8 > self.public_key_length:
            raise ValueError("Public key X coordinate is too large")
        if (len(y) + 7) // 8 > self.public_key_length:
            raise ValueError("Public key Y coordinate is too large")
        return x.bytes(self.public_key_length), y.bytes(self.public_key_length)


    # 解压点
    def decompress_point(self, public_key):
        point = lib.EC_POINT_new(self.group)
        if not point:
            raise ValueError("Could not create point")
        try:
            if not lib.EC_POINT_oct2point(self.group, point, public_key, len(public_key), BN.Context.get()):
                raise ValueError("Invalid compressed public key")
            return self._point_to_affine(point)
        finally:
            lib.EC_POINT_free(point)
    # 创建一个新的私钥
    def new_private_key(self):
        # 创建一个随机密钥
        # 线程安全
        eckey = lib.EC_KEY_new()
        lib.EC_KEY_set_group(eckey, self.group)
        lib.EC_KEY_generate_key(eckey)
        # 转换为大整数
        private_key = BN(lib.EC_KEY_get0_private_key(eckey), link_only=True)
        # 转换为二进制
        private_key_buf = private_key.bytes(self.public_key_length)
        # 清理
        lib.EC_KEY_free(eckey)
        return private_key_buf


    # 将私钥转换为公钥
    def private_to_public(self, private_key):
        eckey, private_key = self._private_key_to_ec_key(private_key)
        try:
            # 推导公钥
            point = lib.EC_POINT_new(self.group)
            try:
                if not lib.EC_POINT_mul(self.group, point, private_key.bn, None, None, BN.Context.get()):
                    raise ValueError("Failed to derive public key")
                return self._point_to_affine(point)
            finally:
                lib.EC_POINT_free(point)
        finally:
            lib.EC_KEY_free(eckey)


    # 将主题转换为大整数
    def _subject_to_bn(self, subject):
        return BN(subject[:(len(self.order) + 7) // 8])
    # 验证签名的有效性
    def verify(self, signature, subject, public_key):
        # 从签名中提取 r 的原始字节并转换为大整数
        r_raw = signature[:self.public_key_length]
        r = BN(r_raw)
        # 从签名中提取 s 的原始字节并转换为大整数
        s = BN(signature[self.public_key_length:])
        # 如果 r 大于等于椭圆曲线的阶，则引发数值错误
        if r >= self.order:
            raise ValueError("r is out of bounds")
        # 如果 s 大于等于椭圆曲线的阶，则引发数值错误
        if s >= self.order:
            raise ValueError("s is out of bounds")

        # 获取大整数的上下文
        bn_ctx = BN.Context.get()

        # 将主题转换为大整数
        z = self._subject_to_bn(subject)

        # 创建椭圆曲线上的公钥点
        pub_p = lib.EC_POINT_new(self.group)
        if not pub_p:
            raise ValueError("Could not create public key point")
        try:
            # 初始化缓冲区并将公钥转换为椭圆曲线上的点
            init_buf = b"\x04" + public_key[0] + public_key[1]
            if not lib.EC_POINT_oct2point(self.group, pub_p, init_buf, len(init_buf), bn_ctx):
                raise ValueError("Could initialize point")

            # 计算签名验证所需的中间值
            sinv = s.inverse(self.order)
            u1 = (z * sinv) % self.order
            u2 = (r * sinv) % self.order

            # 恢复公钥
            result = lib.EC_POINT_new(self.group)
            if not result:
                raise ValueError("Could not create point")
            try:
                if not lib.EC_POINT_mul(self.group, result, u1.bn, pub_p, u2.bn, bn_ctx):
                    raise ValueError("Could not recover public key")
                if BN(self._point_to_affine(result)[0]) % self.order != r:
                    raise ValueError("Invalid signature")
                return True
            finally:
                lib.EC_POINT_free(result)
        finally:
            lib.EC_POINT_free(pub_p)
    # 根据种子和子密钥派生子私钥和子公钥
    def derive_child(self, seed, child):
        # 第一轮
        # 使用HMAC-SHA512算法生成哈希值，截取前32字节作为私钥
        h = hmac.new(key=b"Bitcoin seed", msg=seed, digestmod="sha512").digest()
        private_key1 = h[:32]
        # 将私钥转换为公钥
        x, y = self.private_to_public(private_key1)
        public_key1 = bytes([0x02 + (y[-1] % 2)]) + x
        private_key1 = BN(private_key1)

        # 第二轮
        # 将子密钥转换为字节数组
        child_bytes = []
        for _ in range(4):
            child_bytes.append(child & 255)
            child >>= 8
        child_bytes = bytes(child_bytes[::-1])
        # 将公钥和子密钥组合成消息
        msg = public_key1 + child_bytes
        # 使用HMAC-SHA512算法生成哈希值，截取前32字节作为私钥
        h = hmac.new(key=h[32:], msg=msg, digestmod="sha512").digest()
        private_key2 = BN(h[:32])

        # 返回派生的子公钥
        return ((private_key1 + private_key2) % self.order).bytes(self.public_key_length)


    @classmethod
    def get_backend(cls):
        # 返回OpenSSL后端
        return openssl_backend
# 使用椭圆曲线后端和AES算法创建ECC对象
ecc = ECC(EllipticCurveBackend, aes)
```