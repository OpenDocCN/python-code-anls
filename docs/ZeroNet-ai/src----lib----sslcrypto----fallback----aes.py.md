# `ZeroNet\src\lib\sslcrypto\fallback\aes.py`

```
# 导入 os 模块
import os
# 导入 pyaes 模块
import pyaes
# 从 _aes 模块中导入 AES 类
from .._aes import AES

# 定义模块中可导出的变量
__all__ = ["aes"]

# 定义 AESBackend 类
class AESBackend:
    # 定义内部方法 _get_algo_cipher_type，用于获取算法和密码类型
    def _get_algo_cipher_type(self, algo):
        # 如果算法不是以 "aes-" 开头或者不包含两个连字符，则抛出 ValueError 异常
        if not algo.startswith("aes-") or algo.count("-") != 2:
            raise ValueError("Unknown cipher algorithm {}".format(algo))
        # 获取密钥长度和密码类型
        key_length, cipher_type = algo[4:].split("-")
        # 如果密钥长度不是 "128", "192", "256" 中的一个，则抛出 ValueError 异常
        if key_length not in ("128", "192", "256"):
            raise ValueError("Unknown cipher algorithm {}".format(algo))
        # 如果密码类型不是 "cbc", "ctr", "cfb", "ofb" 中的一个，则抛出 ValueError 异常
        if cipher_type not in ("cbc", "ctr", "cfb", "ofb"):
            raise ValueError("Unknown cipher algorithm {}".format(algo))
        # 返回密码类型
        return cipher_type

    # 定义方法 is_algo_supported，用于检查算法是否受支持
    def is_algo_supported(self, algo):
        try:
            # 调用内部方法 _get_algo_cipher_type，如果没有抛出异常，则返回 True
            self._get_algo_cipher_type(algo)
            return True
        # 如果抛出 ValueError 异常，则返回 False
        except ValueError:
            return False

    # 定义方法 random，用于生成指定长度的随机字节流
    def random(self, length):
        return os.urandom(length)
    # 使用给定的密钥和算法对数据进行加密
    def encrypt(self, data, key, algo="aes-256-cbc"):
        # 获取算法对应的密码类型
        cipher_type = self._get_algo_cipher_type(algo)

        # 生成随机的初始化向量（IV）
        iv = os.urandom(16)

        # 根据密码类型选择相应的加密模式
        if cipher_type == "cbc":
            cipher = pyaes.AESModeOfOperationCBC(key, iv=iv)
        elif cipher_type == "ctr":
            # IV 实际上是一个计数器，而不是一个 IV，但几乎相同。
            # 注意：pyaes 总是使用 1 作为初始计数器！确保不直接调用 pyaes。
            
            # 在这里我们进行了两次转换：从字节数组到整数的转换，以及在 pyaes 内部从整数到字节数组的转换。
            # 这是可以修复的，但我没有注意到任何性能变化，所以我保持代码简洁。
            iv_int = 0
            for byte in iv:
                iv_int = (iv_int * 256) + byte
            counter = pyaes.Counter(iv_int)
            cipher = pyaes.AESModeOfOperationCTR(key, counter=counter)
        elif cipher_type == "cfb":
            # 将段大小从默认的 8 字节更改为 16 字节，以便与 OpenSSL 兼容
            cipher = pyaes.AESModeOfOperationCFB(key, iv, segment_size=16)
        elif cipher_type == "ofb":
            cipher = pyaes.AESModeOfOperationOFB(key, iv)

        # 创建加密器对象
        encrypter = pyaes.Encrypter(cipher)
        # 对数据进行加密
        ciphertext = encrypter.feed(data)
        ciphertext += encrypter.feed()
        # 返回加密后的数据和初始化向量
        return ciphertext, iv
    # 使用给定的密文、初始化向量和密钥对数据进行解密，使用指定的加密算法，默认为 aes-256-cbc
    def decrypt(self, ciphertext, iv, key, algo="aes-256-cbc"):
        # 获取加密算法的密码类型
        cipher_type = self._get_algo_cipher_type(algo)

        # 如果密码类型是 CBC
        if cipher_type == "cbc":
            # 使用密钥和初始化向量创建 AES CBC 模式对象
            cipher = pyaes.AESModeOfOperationCBC(key, iv=iv)
        # 如果密码类型是 CTR
        elif cipher_type == "ctr":
            # IV 实际上是一个计数器，而不是初始化向量，但几乎相同。
            # 注意：pyaes 总是使用 1 作为初始计数器！确保不直接调用 pyaes。
            
            # 我们在这里进行了两次转换：从字节数组到整数的转换，以及在 pyaes 内部从整数到字节数组的转换。
            # 可以修复这个问题，但我没有注意到任何性能变化，所以我保持代码简洁。
            iv_int = 0
            for byte in iv:
                iv_int = (iv_int * 256) + byte
            counter = pyaes.Counter(iv_int)
            # 使用密钥和计数器创建 AES CTR 模式对象
            cipher = pyaes.AESModeOfOperationCTR(key, counter=counter)
        # 如果密码类型是 CFB
        elif cipher_type == "cfb":
            # 将段大小从默认的 8 字节更改为 16 字节，以便与 OpenSSL 兼容
            # 使用密钥和初始化向量创建 AES CFB 模式对象
            cipher = pyaes.AESModeOfOperationCFB(key, iv, segment_size=16)
        # 如果密码类型是 OFB
        elif cipher_type == "ofb":
            # 使用密钥和初始化向量创建 AES OFB 模式对象
            cipher = pyaes.AESModeOfOperationOFB(key, iv)

        # 创建解密器对象
        decrypter = pyaes.Decrypter(cipher)
        # 对密文进行解密
        data = decrypter.feed(ciphertext)
        # 获取解密后的数据
        data += decrypter.feed()
        # 返回解密后的数据
        return data


    # 返回后端类型
    def get_backend(self):
        return "fallback"
# 使用AES算法创建一个AES对象，使用默认的AES后端
aes = AES(AESBackend())
```