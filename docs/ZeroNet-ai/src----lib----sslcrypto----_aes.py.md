# `ZeroNet\src\lib\sslcrypto\_aes.py`

```
# 禁用 pylint 对于 import 位于模块顶层的警告

# 定义 AES 类
class AES:
    # 初始化方法，接受后端和回退参数
    def __init__(self, backend, fallback=None):
        self._backend = backend
        self._fallback = fallback

    # 获取算法的密钥长度
    def get_algo_key_length(self, algo):
        # 如果算法名称中不包含两个连字符，抛出数值错误
        if algo.count("-") != 2:
            raise ValueError("Invalid algorithm name")
        try:
            # 尝试获取算法名称中第二个连字符后的数字并转换为字节长度
            return int(algo.split("-")[1]) // 8
        except ValueError:
            # 如果转换失败，抛出数值错误
            raise ValueError("Invalid algorithm name") from None

    # 生成新的密钥
    def new_key(self, algo="aes-256-cbc"):
        # 如果后端不支持指定的算法，使用回退方法生成新的密钥
        if not self._backend.is_algo_supported(algo):
            if self._fallback is None:
                raise ValueError("This algorithm is not supported")
            return self._fallback.new_key(algo)
        # 使用后端生成指定算法的随机密钥
        return self._backend.random(self.get_algo_key_length(algo))

    # 加密数据
    def encrypt(self, data, key, algo="aes-256-cbc"):
        # 如果后端不支持指定的算法，使用回退方法加密数据
        if not self._backend.is_algo_supported(algo):
            if self._fallback is None:
                raise ValueError("This algorithm is not supported")
            return self._fallback.encrypt(data, key, algo)

        # 检查密钥长度是否符合要求，不符合则抛出数值错误
        key_length = self.get_algo_key_length(algo)
        if len(key) != key_length:
            raise ValueError("Expected key to be {} bytes, got {} bytes".format(key_length, len(key)))

        # 使用后端加密数据
        return self._backend.encrypt(data, key, algo)

    # 解密数据
    def decrypt(self, ciphertext, iv, key, algo="aes-256-cbc"):
        # 如果后端不支持指定的算法，使用回退方法解密数据
        if not self._backend.is_algo_supported(algo):
            if self._fallback is None:
                raise ValueError("This algorithm is not supported")
            return self._fallback.decrypt(ciphertext, iv, key, algo)

        # 检查密钥长度是否符合要求，不符合则抛出数值错误
        key_length = self.get_algo_key_length(algo)
        if len(key) != key_length:
            raise ValueError("Expected key to be {} bytes, got {} bytes".format(key_length, len(key)))

        # 使用后端解密数据
        return self._backend.decrypt(ciphertext, iv, key, algo)

    # 获取后端信息
    def get_backend(self):
        return self._backend.get_backend()
```