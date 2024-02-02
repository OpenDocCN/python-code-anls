# `ZeroNet\src\lib\sslcrypto\_ecc.py`

```py
# 导入哈希算法库
import hashlib
# 导入结构处理库
import struct
# 导入哈希消息认证码库
import hmac
# 导入base58编码库
import base58

# 尝试使用ripemd160算法，如果不支持则使用自定义的_ripemd库
try:
    hashlib.new("ripemd160")
except ValueError:
    # 没有本地实现，使用自定义的_ripemd库
    from . import _ripemd
    def ripemd160(*args):
        return _ripemd.new(*args)
else:
    # 使用OpenSSL实现
    def ripemd160(*args):
        return hashlib.new("ripemd160", *args)

# ECC类，用于椭圆曲线加密
class ECC:
    # pylint: disable=line-too-long
    # 支持的椭圆曲线参数
    # name: (nid, p, n, a, b, (Gx, Gy)),
    }
    # pylint: enable=line-too-long

    # 初始化方法
    def __init__(self, backend, aes):
        self._backend = backend
        self._aes = aes

    # 获取指定椭圆曲线参数
    def get_curve(self, name):
        if name not in self.CURVES:
            raise ValueError("Unknown curve {}".format(name))
        nid, p, n, a, b, g = self.CURVES[name]
        return EllipticCurve(self._backend(p, n, a, b, g), self._aes, nid)

    # 获取后端加密算法
    def get_backend(self):
        return self._backend.get_backend()

# 椭圆曲线类
class EllipticCurve:
    # 初始化方法
    def __init__(self, backend, aes, nid):
        self._backend = backend
        self._aes = aes
        self.nid = nid

    # 编码公钥
    def _encode_public_key(self, x, y, is_compressed=True, raw=True):
        if raw:
            if is_compressed:
                return bytes([0x02 + (y[-1] % 2)]) + x
            else:
                return bytes([0x04]) + x + y
        else:
            return struct.pack("!HH", self.nid, len(x)) + x + struct.pack("!H", len(y)) + y
    # 解码公钥，根据是否部分解码选择不同的处理方式
    def _decode_public_key(self, public_key, partial=False):
        # 如果公钥为空，则抛出数值错误
        if not public_key:
            raise ValueError("No public key")

        # 如果公钥的第一个字节为0x04
        if public_key[0] == 0x04:
            # 未压缩公钥
            expected_length = 1 + 2 * self._backend.public_key_length
            # 如果是部分解码，且公钥长度小于期望长度，则抛出数值错误
            if partial:
                if len(public_key) < expected_length:
                    raise ValueError("Invalid uncompressed public key length")
            else:
                # 如果不是部分解码，且公钥长度不等于期望长度，则抛出数值错误
                if len(public_key) != expected_length:
                    raise ValueError("Invalid uncompressed public key length")
            # 获取x和y的值
            x = public_key[1:1 + self._backend.public_key_length]
            y = public_key[1 + self._backend.public_key_length:expected_length]
            # 如果是部分解码，则返回(x, y)和期望长度
            if partial:
                return (x, y), expected_length
            else:
                # 如果不是部分解码，则返回x和y
                return x, y
        # 如果公钥的第一个字节为0x02或0x03
        elif public_key[0] in (0x02, 0x03):
            # 压缩公钥
            expected_length = 1 + self._backend.public_key_length
            # 如果是部分解码，且公钥长度小于期望长度，则抛出数值错误
            if partial:
                if len(public_key) < expected_length:
                    raise ValueError("Invalid compressed public key length")
            else:
                # 如果不是部分解码，且公钥长度不等于期望长度，则抛出数值错误
                if len(public_key) != expected_length:
                    raise ValueError("Invalid compressed public key length")
            # 解压缩公钥得到x和y的值
            x, y = self._backend.decompress_point(public_key[:expected_length])
            # 对x进行合理性检查
            if x != public_key[1:expected_length]:
                raise ValueError("Incorrect compressed public key")
            # 如果是部分解码，则返回(x, y)和期望长度
            if partial:
                return (x, y), expected_length
            else:
                # 如果不是部分解码，则返回x和y
                return x, y
        # 如果公钥的前缀无效，则抛出数值错误
        else:
            raise ValueError("Invalid public key prefix")
    # 使用 OpenSSL 解码公钥
    def _decode_public_key_openssl(self, public_key, partial=False):
        # 如果公钥为空，则抛出数值错误
        if not public_key:
            raise ValueError("No public key")

        # 初始化索引 i
        i = 0

        # 从 public_key 中解包出 nid
        nid, = struct.unpack("!H", public_key[i:i + 2])
        i += 2
        # 如果 nid 不等于 self.nid，则抛出数值错误
        if nid != self.nid:
            raise ValueError("Wrong curve")

        # 从 public_key 中解包出 xlen
        xlen, = struct.unpack("!H", public_key[i:i + 2])
        i += 2
        # 如果 public_key 的长度减去 i 小于 xlen，则抛出数值错误
        if len(public_key) - i < xlen:
            raise ValueError("Too short public key")
        # 从 public_key 中获取 x
        x = public_key[i:i + xlen]
        i += xlen

        # 从 public_key 中解包出 ylen
        ylen, = struct.unpack("!H", public_key[i:i + 2])
        i += 2
        # 如果 public_key 的长度减去 i 小于 ylen，则抛出数值错误
        if len(public_key) - i < ylen:
            raise ValueError("Too short public key")
        # 从 public_key 中获取 y
        y = public_key[i:i + ylen]
        i += ylen

        # 如果 partial 为 True，则返回 (x, y) 和 i
        if partial:
            return (x, y), i
        # 否则，如果 i 小于 public_key 的长度，则抛出数值错误
        else:
            if i < len(public_key):
                raise ValueError("Too long public key")
            # 返回 x, y
            return x, y


    # 生成新的私钥
    def new_private_key(self, is_compressed=False):
        return self._backend.new_private_key() + (b"\x01" if is_compressed else b"")


    # 将私钥转换为公钥
    def private_to_public(self, private_key):
        # 根据私钥的长度确定是否压缩
        if len(private_key) == self._backend.public_key_length:
            is_compressed = False
        elif len(private_key) == self._backend.public_key_length + 1 and private_key[-1] == 1:
            is_compressed = True
            private_key = private_key[:-1]
        else:
            raise ValueError("Private key has invalid length")
        # 获取公钥的 x, y 坐标
        x, y = self._backend.private_to_public(private_key)
        # 返回编码后的公钥
        return self._encode_public_key(x, y, is_compressed=is_compressed)


    # 将私钥转换为 WIF 格式
    def private_to_wif(self, private_key):
        return base58.b58encode_check(b"\x80" + private_key)


    # 将 WIF 格式的私钥转换为原始私钥
    def wif_to_private(self, wif):
        dec = base58.b58decode_check(wif)
        # 如果解码后的第一个字节不是 0x80，则抛出数值错误
        if dec[0] != 0x80:
            raise ValueError("Invalid network (expected mainnet)")
        # 返回除去第一个字节后的内容
        return dec[1:]
    # 根据公钥生成对应的比特币地址
    def public_to_address(self, public_key):
        # 对公钥进行 SHA256 哈希运算
        h = hashlib.sha256(public_key).digest()
        # 对哈希结果进行 RIPEMD160 哈希运算
        hash160 = ripemd160(h).digest()
        # 对哈希结果进行 Base58 编码，并添加校验位
        return base58.b58encode_check(b"\x00" + hash160)


    # 根据私钥生成对应的比特币地址
    def private_to_address(self, private_key):
        # 通过私钥生成对应的公钥，然后调用 public_to_address 方法生成地址
        return self.public_to_address(self.private_to_public(private_key))


    # 根据私钥和公钥派生出共享密钥
    def derive(self, private_key, public_key):
        # 如果私钥的长度为公钥长度加1，并且最后一位为1，则去掉最后一位
        if len(private_key) == self._backend.public_key_length + 1 and private_key[-1] == 1:
            private_key = private_key[:-1]
        # 如果私钥长度不等于公钥长度，则抛出异常
        if len(private_key) != self._backend.public_key_length:
            raise ValueError("Private key has invalid length")
        # 如果公钥不是元组类型，则解码成公钥
        if not isinstance(public_key, tuple):
            public_key = self._decode_public_key(public_key)
        # 调用底层的 ECDH 方法计算共享密钥
        return self._backend.ecdh(private_key, public_key)


    # 对数据进行哈希运算
    def _digest(self, data, hash):
        # 如果哈希方法为空，则直接返回数据
        if hash is None:
            return data
        # 如果哈希方法是可调用的，则调用该方法对数据进行哈希运算
        elif callable(hash):
            return hash(data)
        # 如果哈希方法是 sha1，则使用 hashlib 计算 SHA1 哈希
        elif hash == "sha1":
            return hashlib.sha1(data).digest()
        # 如果哈希方法是 sha256，则使用 hashlib 计算 SHA256 哈希
        elif hash == "sha256":
            return hashlib.sha256(data).digest()
        # 如果哈希方法是 sha512，则使用 hashlib 计算 SHA512 哈希
        elif hash == "sha512":
            return hashlib.sha512(data).digest()
        # 如果哈希方法未知，则抛出异常
        else:
            raise ValueError("Unknown hash/derivation method")


    # 高级函数
    # 使用给定的公钥对数据进行加密，使用指定的算法和衍生方法
    def encrypt(self, data, public_key, algo="aes-256-cbc", derivation="sha256", mac="hmac-sha256", return_aes_key=False):
        # 生成临时私钥
        private_key = self.new_private_key()

        # 衍生密钥
        ecdh = self.derive(private_key, public_key)
        key = self._digest(ecdh, derivation)
        k_enc_len = self._aes.get_algo_key_length(algo)
        if len(key) < k_enc_len:
            raise ValueError("Too short digest")
        k_enc, k_mac = key[:k_enc_len], key[k_enc_len:]

        # 加密
        ciphertext, iv = self._aes.encrypt(data, k_enc, algo=algo)
        ephem_public_key = self.private_to_public(private_key)
        ephem_public_key = self._decode_public_key(ephem_public_key)
        ephem_public_key = self._encode_public_key(*ephem_public_key, raw=False)
        ciphertext = iv + ephem_public_key + ciphertext

        # 添加 MAC 标签
        if callable(mac):
            tag = mac(k_mac, ciphertext)
        elif mac == "hmac-sha256":
            h = hmac.new(k_mac, digestmod="sha256")
            h.update(ciphertext)
            tag = h.digest()
        elif mac == "hmac-sha512":
            h = hmac.new(k_mac, digestmod="sha512")
            h.update(ciphertext)
            tag = h.digest()
        elif mac is None:
            tag = b""
        else:
            raise ValueError("Unsupported MAC")

        # 如果需要返回 AES 密钥，则返回加密后的数据和 AES 密钥
        if return_aes_key:
            return ciphertext + tag, k_enc
        # 否则只返回加密后的数据和 MAC 标签
        else:
            return ciphertext + tag
    # 使用私钥对数据进行签名，可以指定哈希算法、是否可恢复、熵值
    def sign(self, data, private_key, hash="sha256", recoverable=False, entropy=None):
        # 判断私钥长度，确定是否压缩
        if len(private_key) == self._backend.public_key_length:
            is_compressed = False
        elif len(private_key) == self._backend.public_key_length + 1 and private_key[-1] == 1:
            is_compressed = True
            private_key = private_key[:-1]
        else:
            raise ValueError("Private key has invalid length")

        # 对数据进行哈希处理
        data = self._digest(data, hash)
        # 如果没有指定熵值，则进行计算
        if not entropy:
            v = b"\x01" * len(data)
            k = b"\x00" * len(data)
            k = hmac.new(k, v + b"\x00" + private_key + data, "sha256").digest()
            v = hmac.new(k, v, "sha256").digest()
            k = hmac.new(k, v + b"\x01" + private_key + data, "sha256").digest()
            v = hmac.new(k, v, "sha256").digest()
            entropy = hmac.new(k, v, "sha256").digest()
        # 调用后端方法进行签名
        return self._backend.sign(data, private_key, recoverable, is_compressed, entropy=entropy)


    # 根据签名和数据恢复公钥
    def recover(self, signature, data, hash="sha256"):
        # 检查签名是否可恢复
        if len(signature) != 1 + 2 * self._backend.public_key_length:
            raise ValueError("Cannot recover an unrecoverable signature")
        x, y = self._backend.recover(signature, self._digest(data, hash))
        is_compressed = signature[0] >= 31
        return self._encode_public_key(x, y, is_compressed=is_compressed)


    # 验证签名是否有效
    def verify(self, signature, data, public_key, hash="sha256"):
        # 如果是可恢复签名，则去掉第一个字节
        if len(signature) == 1 + 2 * self._backend.public_key_length:
            signature = signature[1:]
        # 检查签名格式是否有效
        if len(signature) != 2 * self._backend.public_key_length:
            raise ValueError("Invalid signature format")
        # 如果公钥不是元组，则解码
        if not isinstance(public_key, tuple):
            public_key = self._decode_public_key(public_key)
        # 调用后端方法进行验证
        return self._backend.verify(signature, self._digest(data, hash), public_key)
    # 根据 BIP32 协议派生子密钥
    def derive_child(self, seed, child):
        # 检查子索引是否在有效范围内
        if not 0 <= child < 2 ** 31:
            # 如果子索引无效，则抛出数值错误
            raise ValueError("Invalid child index")
        # 调用后端方法，根据种子和子索引派生子密钥
        return self._backend.derive_child(seed, child)
```