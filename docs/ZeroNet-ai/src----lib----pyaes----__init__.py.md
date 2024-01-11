# `ZeroNet\src\lib\pyaes\__init__.py`

```
# 这是对 AES 算法和 AES 常见操作模式的纯 Python 实现

# 参见：https://en.wikipedia.org/wiki/Advanced_Encryption_Standard
# 参见：https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation

# 支持的密钥长度：
#   128 位
#   192 位
#   256 位

# 支持的操作模式：
#   ECB - 电子密码本模式
#   CBC - 密码块链接模式
#   CFB - 密码反馈模式
#   OFB - 输出反馈模式
#   CTR - 计数器模式

# 有关 API 详细信息和一般信息，请参阅 README.md

# 还有一个有用的库 PyCrypto，它是用 C 实现的加密库，带有 Python 绑定：
# https://www.dlitz.net/software/pycrypto/

# 版本号
VERSION = [1, 3, 0]

# 导入 AES 算法和各种操作模式
from .aes import AES, AESModeOfOperationCTR, AESModeOfOperationCBC, AESModeOfOperationCFB, AESModeOfOperationECB, AESModeOfOperationOFB, AESModesOfOperation, Counter
# 从blockfeeder模块中导入decrypt_stream, Decrypter, encrypt_stream, Encrypter等函数和常量
from .blockfeeder import decrypt_stream, Decrypter, encrypt_stream, Encrypter
# 从blockfeeder模块中导入PADDING_NONE, PADDING_DEFAULT等常量
from .blockfeeder import PADDING_NONE, PADDING_DEFAULT
```