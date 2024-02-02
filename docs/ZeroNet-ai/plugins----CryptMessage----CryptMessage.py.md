# `ZeroNet\plugins\CryptMessage\CryptMessage.py`

```py
# 导入 hashlib 模块，用于加密哈希算法
import hashlib
# 导入 base64 模块，用于编解码 base64 数据
import base64
# 导入 struct 模块，用于处理字节流数据的打包和解包
import struct
# 从 lib 模块中导入 sslcrypto 模块
from lib import sslcrypto
# 从 Crypt 模块中导入 Crypt 类
from Crypt import Crypt

# 使用 secp256k1 曲线初始化 ECC 对象
curve = sslcrypto.ecc.get_curve("secp256k1")

# 定义一个函数，用于对数据进行 ECIES 加密
def eciesEncrypt(data, pubkey, ciphername="aes-256-cbc"):
    # 使用 ECC 公钥对数据进行加密，返回密文和加密后的对称密钥
    ciphertext, key_e = curve.encrypt(
        data,
        base64.b64decode(pubkey),
        algo=ciphername,
        derivation="sha512",
        return_aes_key=True
    )
    return key_e, ciphertext

# 使用装饰器将函数包装成线程池加密函数
@Crypt.thread_pool_crypt.wrap
# 定义一个函数，用于对多个密文进行 ECIES 解密
def eciesDecryptMulti(encrypted_datas, privatekey):
    texts = []  # Decoded texts
    # 遍历多个密文
    for encrypted_data in encrypted_datas:
        try:
            # 对密文进行解密，并将解密后的文本添加到列表中
            text = eciesDecrypt(encrypted_data, privatekey).decode("utf8")
            texts.append(text)
        except Exception:
            # 如果解密出现异常，将 None 添加到列表中
            texts.append(None)
    return texts

# 定义一个函数，用于对密文进行 ECIES 解密
def eciesDecrypt(ciphertext, privatekey):
    # 使用 ECC 私钥对密文进行解密，返回明文
    return curve.decrypt(base64.b64decode(ciphertext), curve.wif_to_private(privatekey.encode()), derivation="sha512")

# 定义一个函数，用于解码 ECC 公钥
def decodePubkey(pubkey):
    i = 0
    # 从字节流中解析出曲线参数
    curve = struct.unpack('!H', pubkey[i:i + 2])[0]
    i += 2
    # 从字节流中解析出公钥 x 坐标
    tmplen = struct.unpack('!H', pubkey[i:i + 2])[0]
    i += 2
    pubkey_x = pubkey[i:i + tmplen]
    i += tmplen
    # 从字节流中解析出公钥 y 坐标
    tmplen = struct.unpack('!H', pubkey[i:i + 2])[0]
    i += 2
    pubkey_y = pubkey[i:i + tmplen]
    i += tmplen
    return curve, pubkey_x, pubkey_y, i

# 定义一个函数，用于从加密数据中分离出初始化向量和密文
def split(encrypted):
    # 从加密数据中提取出初始化向量
    iv = encrypted[0:16]
    # 从加密数据中提取出 ECC 公钥和密文
    curve, pubkey_x, pubkey_y, i = decodePubkey(encrypted[16:])
    ciphertext = encrypted[16 + i:-32]

    return iv, ciphertext
```