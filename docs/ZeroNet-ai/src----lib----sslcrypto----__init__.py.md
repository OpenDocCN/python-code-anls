# `ZeroNet\src\lib\sslcrypto\__init__.py`

```py
# 定义一个包含 "aes", "ecc", "rsa" 的列表，表示这些模块是公开的接口
__all__ = ["aes", "ecc", "rsa"]

# 尝试导入 openssl 模块中的 aes, ecc, rsa 模块
try:
    # 如果成功导入，则使用 openssl 模块中的 aes, ecc, rsa 模块
    from .openssl import aes, ecc, rsa
# 如果导入过程中出现 OSError 异常
except OSError:
    # 则使用 fallback 模块中的 aes, ecc, rsa 模块
    from .fallback import aes, ecc, rsa
```