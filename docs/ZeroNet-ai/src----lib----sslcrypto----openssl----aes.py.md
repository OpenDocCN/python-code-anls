# `ZeroNet\src\lib\sslcrypto\openssl\aes.py`

```py
# 导入 ctypes 模块
import ctypes
# 导入 threading 模块
import threading
# 从上级目录的 _aes 模块中导入 AES 类
from .._aes import AES
# 从 fallback 模块中导入 aes 函数
from ..fallback.aes import aes as fallback_aes
# 从 library 模块中导入 lib 和 openssl_backend
from .library import lib, openssl_backend

# 初始化函数
try:
    # 设置 lib.EVP_CIPHER_CTX_new 的返回类型为 ctypes.POINTER(ctypes.c_char)
    lib.EVP_CIPHER_CTX_new.restype = ctypes.POINTER(ctypes.c_char)
except AttributeError:
    # 如果出现 AttributeError 则忽略
    pass
# 设置 lib.EVP_get_cipherbyname 的返回类型为 ctypes.POINTER(ctypes.c_char)
lib.EVP_get_cipherbyname.restype = ctypes.POINTER(ctypes.c_char)

# 创建线程本地数据对象
thread_local = threading.local()

# 定义 Context 类
class Context:
    def __init__(self, ptr, do_free):
        self.lib = lib
        self.ptr = ptr
        self.do_free = do_free

    # 定义析构函数
    def __del__(self):
        # 如果需要释放内存，则调用 lib.EVP_CIPHER_CTX_free 释放内存
        if self.do_free:
            self.lib.EVP_CIPHER_CTX_free(self.ptr)

# 定义 AESBackend 类
class AESBackend:
    # 定义支持的算法列表
    ALGOS = (
        "aes-128-cbc", "aes-192-cbc", "aes-256-cbc",
        "aes-128-ctr", "aes-192-ctr", "aes-256-ctr",
        "aes-128-cfb", "aes-192-cfb", "aes-256-cfb",
        "aes-128-ofb", "aes-192-ofb", "aes-256-ofb"
    )

    # 初始化函数
    def __init__(self):
        # 检查是否支持 EVP_CIPHER_CTX_new 和 EVP_CIPHER_CTX_reset 函数
        self.is_supported_ctx_new = hasattr(lib, "EVP_CIPHER_CTX_new")
        self.is_supported_ctx_reset = hasattr(lib, "EVP_CIPHER_CTX_reset")

    # 获取上下文对象
    def _get_ctx(self):
        # 如果线程本地数据对象中没有 ctx 属性
        if not hasattr(thread_local, "ctx"):
            # 如果支持 EVP_CIPHER_CTX_new 函数，则创建新的上下文对象
            if self.is_supported_ctx_new:
                thread_local.ctx = Context(lib.EVP_CIPHER_CTX_new(), True)
            else:
                # 否则创建固定大小的上下文对象
                # 1 KiB 应该足够了。我们不知道上下文缓冲区的实际大小，因为我们不确定填充和指针大小
                thread_local.ctx = Context(ctypes.create_string_buffer(1024), False)
        # 返回上下文对象的指针
        return thread_local.ctx.ptr

    # 获取后端
    def get_backend(self):
        return openssl_backend

    # 获取密码
    def _get_cipher(self, algo):
        # 如果算法不在支持的算法列表中，则抛出 ValueError 异常
        if algo not in self.ALGOS:
            raise ValueError("Unknown cipher algorithm {}".format(algo))
        # 获取指定算法的密码
        cipher = lib.EVP_get_cipherbyname(algo.encode())
        # 如果密码不存在，则抛出 ValueError 异常
        if not cipher:
            raise ValueError("Unknown cipher algorithm {}".format(algo))
        # 返回密码
        return cipher
    # 检查算法是否受支持
    def is_algo_supported(self, algo):
        try:
            # 获取加密算法的密码对象
            self._get_cipher(algo)
            return True
        except ValueError:
            return False


    # 生成指定长度的随机字节流
    def random(self, length):
        entropy = ctypes.create_string_buffer(length)
        # 使用 OpenSSL 生成随机字节流
        lib.RAND_bytes(entropy, length)
        return bytes(entropy)


    # 使用指定的密钥和算法对数据进行加密
    def encrypt(self, data, key, algo="aes-256-cbc"):
        # 初始化加密上下文
        ctx = self._get_ctx()
        if not self.is_supported_ctx_new:
            lib.EVP_CIPHER_CTX_init(ctx)
        try:
            # 使用指定的算法初始化加密上下文
            lib.EVP_EncryptInit_ex(ctx, self._get_cipher(algo), None, None, None)

            # 生成随机的初始化向量（IV）
            iv_length = 16
            iv = self.random(iv_length)

            # 设置密钥和初始化向量（IV）
            lib.EVP_EncryptInit_ex(ctx, None, None, key, iv)

            # 实际进行加密
            block_size = 16
            output = ctypes.create_string_buffer((len(data) // block_size + 1) * block_size)
            output_len = ctypes.c_int()

            # 更新加密数据
            if not lib.EVP_CipherUpdate(ctx, output, ctypes.byref(output_len), data, len(data)):
                raise ValueError("Could not feed cipher with data")

            new_output = ctypes.byref(output, output_len.value)
            output_len2 = ctypes.c_int()
            # 完成加密
            if not lib.EVP_CipherFinal_ex(ctx, new_output, ctypes.byref(output_len2)):
                raise ValueError("Could not finalize cipher")

            # 获取加密后的数据和初始化向量（IV）
            ciphertext = output[:output_len.value + output_len2.value]
            return ciphertext, iv
        finally:
            # 重置加密上下文
            if self.is_supported_ctx_reset:
                lib.EVP_CIPHER_CTX_reset(ctx)
            else:
                lib.EVP_CIPHER_CTX_cleanup(ctx)
    # 解密函数，接受密文、初始化向量、密钥和算法参数
    def decrypt(self, ciphertext, iv, key, algo="aes-256-cbc"):
        # 初始化加密上下文
        ctx = self._get_ctx()
        # 如果不支持新的上下文初始化方法，则使用旧的方法进行初始化
        if not self.is_supported_ctx_new:
            lib.EVP_CIPHER_CTX_init(ctx)
        try:
            # 使用指定的算法初始化解密上下文
            lib.EVP_DecryptInit_ex(ctx, self._get_cipher(algo), None, None, None)

            # 确保初始化向量的长度正确
            iv_length = 16
            if len(iv) != iv_length:
                raise ValueError("Expected IV to be {} bytes, got {} bytes".format(iv_length, len(iv)))

            # 设置密钥和初始化向量
            lib.EVP_DecryptInit_ex(ctx, None, None, key, iv)

            # 实际进行解密操作
            output = ctypes.create_string_buffer(len(ciphertext))
            output_len = ctypes.c_int()

            if not lib.EVP_DecryptUpdate(ctx, output, ctypes.byref(output_len), ciphertext, len(ciphertext)):
                raise ValueError("Could not feed decipher with ciphertext")

            new_output = ctypes.byref(output, output_len.value)
            output_len2 = ctypes.c_int()
            if not lib.EVP_DecryptFinal_ex(ctx, new_output, ctypes.byref(output_len2)):
                raise ValueError("Could not finalize decipher")

            # 返回解密后的结果
            return output[:output_len.value + output_len2.value]
        finally:
            # 如果支持上下文重置，则重置上下文
            if self.is_supported_ctx_reset:
                lib.EVP_CIPHER_CTX_reset(ctx)
            else:
                lib.EVP_CIPHER_CTX_cleanup(ctx)
# 使用AES算法和指定的AES后端创建AES对象
aes = AES(AESBackend(), fallback_aes)
```