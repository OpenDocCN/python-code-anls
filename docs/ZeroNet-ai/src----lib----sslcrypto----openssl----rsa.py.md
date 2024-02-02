# `ZeroNet\src\lib\sslcrypto\openssl\rsa.py`

```py
# 禁用 pylint 对于过少公共方法的警告

# 从当前目录下的 library 模块中导入 openssl_backend
from .library import openssl_backend

# 定义 RSA 类
class RSA:
    # 定义获取后端的方法
    def get_backend(self):
        # 返回 openssl_backend
        return openssl_backend

# 创建 RSA 类的实例
rsa = RSA()
```