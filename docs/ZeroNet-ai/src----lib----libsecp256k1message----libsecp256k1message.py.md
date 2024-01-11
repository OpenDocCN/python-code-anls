# `ZeroNet\src\lib\libsecp256k1message\libsecp256k1message.py`

```
# 导入所需的模块
import hashlib
import base64
from coincurve import PrivateKey, PublicKey
from base58 import b58encode_check, b58decode_check
from hmac import compare_digest
from util.Electrum import format as zero_format

# 定义常量
RECID_MIN = 0
RECID_MAX = 3
RECID_UNCOMPR = 27
LEN_COMPACT_SIG = 65

# 定义自定义异常类
class SignatureError(ValueError):
    pass

# 生成公共地址和私密地址
def bitcoin_address():
    """Generate a public address and a secret address."""
    # 生成公钥和私钥对
    publickey, secretkey = key_pair()

    # 计算公共地址和私密地址
    public_address = compute_public_address(publickey)
    secret_address = compute_secret_address(secretkey)

    return (public_address, secret_address)

# 生成公钥和私钥对
def key_pair():
    """Generate a public key and a secret key."""
    # 生成私钥
    secretkey = PrivateKey()
    # 根据私钥生成公钥
    publickey = PublicKey.from_secret(secretkey.secret)
    return (publickey, secretkey)

# 将公钥转换为比特币公共地址
def compute_public_address(publickey, compressed=False):
    """Convert a public key to a public Bitcoin address."""
    # 计算公钥的哈希值
    public_plain = b'\x00' + public_digest(publickey, compressed=compressed)
    # 对哈希值进行base58编码
    return b58encode_check(public_plain)

# 将私钥转换为比特币私密地址
def compute_secret_address(secretkey):
    """Convert a secret key to a secret Bitcoin address."""
    # 计算私钥的哈希值
    secret_plain = b'\x80' + secretkey.secret
    # 对哈希值进行base58编码
    return b58encode_check(secret_plain)

# 将公钥转换为ripemd160(sha256())哈希值
def public_digest(publickey, compressed=False):
    """Convert a public key to ripemd160(sha256()) digest."""
    # 将公钥转换为十六进制格式
    publickey_hex = publickey.format(compressed=compressed)
    # 计算ripemd160(sha256())哈希值
    return hashlib.new('ripemd160', hashlib.sha256(publickey_hex).digest()).digest()

# 将比特币公共地址转换为ripemd160(sha256())哈希值
def address_public_digest(address):
    """Convert a public Bitcoin address to ripemd160(sha256()) digest."""
    # 对比特币公共地址进行base58解码
    public_plain = b58decode_check(address)
    # 检查解码后的哈希值是否符合要求
    if not public_plain.startswith(b'\x00') or len(public_plain) != 21:
        raise ValueError('Invalid public key digest')
    return public_plain[1:]

# 解码比特币私密地址
def _decode_bitcoin_secret(address):
    secret_plain = b58decode_check(address)
    # 检查解码后的私钥是否符合要求
    if not secret_plain.startswith(b'\x80') or len(secret_plain) != 33:
        raise ValueError('Invalid secret key. Uncompressed keys only.')
    # 返回从索引 1 开始到末尾的 secret_plain 列表切片
    return secret_plain[1:]
# 从签名和消息中恢复公钥
def recover_public_key(signature, message):
    """Recover public key from signature and message.
    Recovered public key guarantees a correct signature"""
    return PublicKey.from_signature_and_message(signature, message)

# 将秘密比特币地址转换为秘密密钥
def decode_secret_key(address):
    """Convert a secret Bitcoin address to a secret key."""
    return PrivateKey(_decode_bitcoin_secret(address))


# 将 Electrum 签名转换为 coincurve 签名
def coincurve_sig(electrum_signature):
    # coincurve := r + s + recovery_id
    # where (0 <= recovery_id <= 3)
    # https://github.com/bitcoin-core/secp256k1/blob/0b7024185045a49a1a6a4c5615bf31c94f63d9c4/src/modules/recovery/main_impl.h#L35
    if len(electrum_signature) != LEN_COMPACT_SIG:
        raise ValueError('Not a 65-byte compact signature.')
    # 计算 coincurve recid
    recid = (electrum_signature[0] - 27) & 3
    if not (RECID_MIN <= recid <= RECID_MAX):
        raise ValueError('Recovery ID %d is not supported.' % recid)
    recid_byte = int.to_bytes(recid, length=1, byteorder='big')
    return electrum_signature[1:] + recid_byte


# 将 coincurve 签名转换为 Electrum 签名
def electrum_sig(coincurve_signature):
    # electrum := recovery_id + r + s
    # where (27 <= recovery_id <= 30)
    # https://github.com/scintill/bitcoin-signature-tools/blob/ed3f5be5045af74a54c92d3648de98c329d9b4f7/key.cpp#L285
    if len(coincurve_signature) != LEN_COMPACT_SIG:
        raise ValueError('Not a 65-byte compact signature.')
    # 计算 Electrum recid
    recid = coincurve_signature[-1] + RECID_UNCOMPR
    if not (RECID_UNCOMPR + RECID_MIN <= recid <= RECID_UNCOMPR + RECID_MAX):
        raise ValueError('Recovery ID %d is not supported.' % recid)
    recid_byte = int.to_bytes(recid, length=1, byteorder='big')
    return recid_byte + coincurve_signature[0:-1]

# 使用秘密密钥对字节串进行签名
def sign_data(secretkey, byte_string):
    """Sign [byte_string] with [secretkey].
    Return serialized signature compatible with Electrum (ZeroNet)."""
    # 编码消息
    encoded = zero_format(byte_string)
    # 签名消息并获取 coincurve 签名
    # 使用私钥对编码后的数据进行可恢复签名
    signature = secretkey.sign_recoverable(encoded)
    # 重新序列化签名并返回
    return electrum_sig(signature)
# 验证数据的有效性，包括签名、公钥和消息
def verify_data(key_digest, electrum_signature, byte_string):
    """Verify if [electrum_signature] of [byte_string] is correctly signed and
    is signed with the secret counterpart of [key_digest].
    Raise SignatureError if the signature is forged or otherwise problematic."""
    # 重新序列化签名
    signature = coincurve_sig(electrum_signature)
    # 对消息进行编码
    encoded = zero_format(byte_string)
    # 从签名中恢复完整的公钥
    # "which guarantees a correct signature"
    publickey = recover_public_key(signature, encoded)

    # 验证消息是否由公钥正确签名
    # correct_sig = verify_sig(publickey, signature, encoded)

    # 验证公钥是否符合预期
    correct_key = verify_key(publickey, key_digest)

    if not correct_key:
        raise SignatureError('Signature is forged!')

def verify_sig(publickey, signature, byte_string):
    return publickey.verify(signature, byte_string)

def verify_key(publickey, key_digest):
    return compare_digest(key_digest, public_digest(publickey))

def recover_address(data, sign):
    sign_bytes = base64.b64decode(sign)
    is_compressed = ((sign_bytes[0] - 27) & 4) != 0
    publickey = recover_public_key(coincurve_sig(sign_bytes), zero_format(data))
    return compute_public_address(publickey, compressed=is_compressed)

__all__ = [
    'SignatureError',
    'key_pair', 'compute_public_address', 'compute_secret_address',
    'public_digest', 'address_public_digest', 'recover_public_key', 'decode_secret_key',
    'sign_data', 'verify_data', "recover_address"
]

if __name__ == "__main__":
    import base64, time, multiprocessing
    s = time.time()
    privatekey = decode_secret_key(b"5JsunC55XGVqFQj5kPGK4MWgTL26jKbnPhjnmchSNPo75XXCwtk")
    threads = []
    # 循环1000次
    for i in range(1000):
        # 将字符串"hello"转换为UTF-8编码的字节流
        data = bytes("hello", "utf8")
        # 使用给定的数据和私钥恢复地址
        address = recover_address(data, "HGbib2kv9gm9IJjDt1FXbXFczZi35u0rZR3iPUIt5GglDDCeIQ7v8eYXVNIaLoJRI4URGZrhwmsYQ9aVtRTnTfQ=")
    # 打印验证10000次所需的时间和地址
    print("- Verify x10000: %.3fs %s" % (time.time() - s, address))

    # 记录当前时间
    s = time.time()
    # 循环1000次
    for i in range(1000):
        # 解码私钥
        privatekey = decode_secret_key(b"5JsunC55XGVqFQj5kPGK4MWgTL26jKbnPhjnmchSNPo75XXCwtk")
        # 使用私钥对数据进行签名
        sign = sign_data(privatekey, b"hello")
        # 将签名结果进行Base64编码
        sign_b64 = base64.b64encode(sign)

    # 打印签名1000次所需的时间
    print("- Sign x1000: %.3fs" % (time.time() - s))
```