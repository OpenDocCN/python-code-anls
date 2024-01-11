# `ZeroNet\src\Crypt\CryptRsa.py`

```
# 导入 base64 和 hashlib 模块
import base64
import hashlib

# 使用私钥对数据进行签名
def sign(data, privatekey):
    # 导入 rsa 和 rsa.pkcs1 模块
    import rsa
    from rsa import pkcs1

    # 如果私钥不包含 "BEGIN RSA PRIVATE KEY"，则添加相应的标识
    if "BEGIN RSA PRIVATE KEY" not in privatekey:
        privatekey = "-----BEGIN RSA PRIVATE KEY-----\n%s\n-----END RSA PRIVATE KEY-----" % privatekey

    # 加载私钥
    priv = rsa.PrivateKey.load_pkcs1(privatekey)
    # 对数据进行 SHA-256 签名
    sign = rsa.pkcs1.sign(data, priv, 'SHA-256')
    return sign

# 使用公钥验证签名
def verify(data, publickey, sign):
    # 导入 rsa 和 rsa.pkcs1 模块
    import rsa
    from rsa import pkcs1

    # 加载公钥
    pub = rsa.PublicKey.load_pkcs1(publickey, format="DER")
    try:
        # 验证签名
        valid = rsa.pkcs1.verify(data, sign, pub)
    except pkcs1.VerificationError:
        valid = False
    return valid

# 将私钥转换为公钥
def privatekeyToPublickey(privatekey):
    # 导入 rsa 和 rsa.pkcs1 模块
    import rsa
    from rsa import pkcs1

    # 如果私钥不包含 "BEGIN RSA PRIVATE KEY"，则添加相应的标识
    if "BEGIN RSA PRIVATE KEY" not in privatekey:
        privatekey = "-----BEGIN RSA PRIVATE KEY-----\n%s\n-----END RSA PRIVATE KEY-----" % privatekey

    # 加载私钥
    priv = rsa.PrivateKey.load_pkcs1(privatekey)
    # 创建公钥
    pub = rsa.PublicKey(priv.n, priv.e)
    return pub.save_pkcs1("DER")

# 将公钥转换为 Onion 地址
def publickeyToOnion(publickey):
    # 对公钥进行 SHA-1 哈希，取前 10 个字节，然后进行 base32 编码并转换为小写
    return base64.b32encode(hashlib.sha1(publickey).digest()[:10]).lower().decode("ascii")
```