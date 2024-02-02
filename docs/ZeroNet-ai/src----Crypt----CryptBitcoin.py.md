# `ZeroNet\src\Crypt\CryptBitcoin.py`

```py
# 导入日志模块
import logging
# 导入 base64 编码模块
import base64
# 导入二进制转换为十六进制模块
import binascii
# 导入时间模块
import time
# 导入哈希模块
import hashlib

# 从 util.Electrum 模块中导入 dbl_format 函数
from util.Electrum import dbl_format
# 从 Config 模块中导入 config 变量
from Config import config

# 导入 util.OpensslFindPatch 模块
import util.OpensslFindPatch

# 设置 lib_verify_best 变量的初始值为 "sslcrypto"
lib_verify_best = "sslcrypto"

# 从 lib 模块中导入 sslcrypto 模块
from lib import sslcrypto
# 设置 sslcurve_native 变量的值为 sslcrypto 模块中的 ecc 模块的 get_curve 函数返回值
sslcurve_native = sslcrypto.ecc.get_curve("secp256k1")
# 设置 sslcurve_fallback 变量的值为 sslcrypto 模块中的 fallback 模块的 ecc 模块的 get_curve 函数返回值
sslcurve_fallback = sslcrypto.fallback.ecc.get_curve("secp256k1")
# 设置 sslcurve 变量的值为 sslcurve_native 变量的值
sslcurve = sslcurve_native

# 定义 loadLib 函数，接受 lib_name 和 silent 两个参数
def loadLib(lib_name, silent=False):
    # 声明全局变量 sslcurve, libsecp256k1message, lib_verify_best
    global sslcurve, libsecp256k1message, lib_verify_best
    # 如果 lib_name 为 "libsecp256k1"
    if lib_name == "libsecp256k1":
        # 记录当前时间
        s = time.time()
        # 从 lib 模块中导入 libsecp256k1message 模块
        from lib import libsecp256k1message
        # 导入 coincurve 模块
        import coincurve
        # 设置 lib_verify_best 变量的值为 "libsecp256k1"
        lib_verify_best = "libsecp256k1"
        # 如果 silent 为 False
        if not silent:
            # 记录日志信息
            logging.info(
                "Libsecpk256k1 loaded: %s in %.3fs" %
                (type(coincurve._libsecp256k1.lib).__name__, time.time() - s)
            )
    # 如果 lib_name 为 "sslcrypto"
    elif lib_name == "sslcrypto":
        # 设置 sslcurve 变量的值为 sslcurve_native 变量的值
        sslcurve = sslcurve_native
        # 如果 sslcurve_native 等于 sslcurve_fallback
        if sslcurve_native == sslcurve_fallback:
            # 记录警告日志信息
            logging.warning("SSLCurve fallback loaded instead of native")
    # 如果 lib_name 为 "sslcrypto_fallback"
    elif lib_name == "sslcrypto_fallback":
        # 设置 sslcurve 变量的值为 sslcurve_fallback 变量的值
        sslcurve = sslcurve_fallback

# 尝试执行以下代码块
try:
    # 如果 config.use_libsecp256k1 为 False
    if not config.use_libsecp256k1:
        # 抛出异常
        raise Exception("Disabled by config")
    # 调用 loadLib 函数，传入参数 "libsecp256k1"
    loadLib("libsecp256k1")
    # 设置 lib_verify_best 变量的值为 "libsecp256k1"
    lib_verify_best = "libsecp256k1"
# 捕获异常并将异常信息保存到 err 变量
except Exception as err:
    # 记录日志信息
    logging.info("Libsecp256k1 load failed: %s" % err)

# 定义 newPrivatekey 函数，不接受参数
def newPrivatekey():  # Return new private key
    # 返回 sslcurve 模块的 private_to_wif 函数对 sslcurve 模块的 new_private_key 函数的返回值进行解码后的结果
    return sslcurve.private_to_wif(sslcurve.new_private_key()).decode()

# 定义 newSeed 函数，不接受参数
def newSeed():
    # 返回 sslcurve 模块的 new_private_key 函数的返回值经过 binascii 模块的 hexlify 函数处理后的结果进行解码后的结果
    return binascii.hexlify(sslcurve.new_private_key()).decode()

# 定义 hdPrivatekey 函数，接受 seed 和 child 两个参数
def hdPrivatekey(seed, child):
    # 将 seed 编码后的结果传入 sslcurve 模块的 derive_child 函数，同时将 child 取模 100000000 的结果传入
    privatekey_bin = sslcurve.derive_child(seed.encode(), child % 100000000)
    # 返回 sslcurve 模块的 private_to_wif 函数对 privatekey_bin 的返回值进行解码后的结果
    return sslcurve.private_to_wif(privatekey_bin).decode()

# 定义 privatekeyToAddress 函数，接受 privatekey 参数
def privatekeyToAddress(privatekey):  # Return address from private key
    # 尝试执行以下代码块，捕获可能出现的异常
    try:
        # 如果私钥长度为64，则将私钥转换为二进制格式
        if len(privatekey) == 64:
            privatekey_bin = bytes.fromhex(privatekey)
        # 否则，使用sslcurve.wif_to_private方法将私钥转换为二进制格式
        else:
            privatekey_bin = sslcurve.wif_to_private(privatekey.encode())
        # 返回私钥对应的地址
        return sslcurve.private_to_address(privatekey_bin).decode()
    # 捕获任何异常，表示私钥无效
    except Exception:  # Invalid privatekey
        # 返回False
        return False
# 使用私钥对数据进行签名，返回签名结果
def sign(data, privatekey):  # Return sign to data using private key
    # 检查私钥是否为旧格式，如果是则不支持
    if privatekey.startswith("23") and len(privatekey) > 52:
        return None  # Old style private key not supported
    # 对数据进行签名并使用 base64 进行编码
    return base64.b64encode(sslcurve.sign(
        data.encode(),
        sslcurve.wif_to_private(privatekey.encode()),
        recoverable=True,
        hash=dbl_format
    )).decode()


# 使用地址和签名验证数据
def verify(data, valid_address, sign, lib_verify=None):  # Verify data using address and sign
    # 如果未指定验证库，则使用最佳验证库
    if not lib_verify:
        lib_verify = lib_verify_best

    # 如果没有签名，则返回 False
    if not sign:
        return False

    # 根据验证库进行验证
    if lib_verify == "libsecp256k1":
        # 使用 libsecp256k1 进行签名验证
        sign_address = libsecp256k1message.recover_address(data.encode("utf8"), sign).decode("utf8")
    elif lib_verify in ("sslcrypto", "sslcrypto_fallback"):
        # 使用 sslcrypto 进行签名验证
        publickey = sslcurve.recover(base64.b64decode(sign), data.encode(), hash=dbl_format)
        sign_address = sslcurve.public_to_address(publickey).decode()
    else:
        # 如果没有启用签名验证库，则抛出异常
        raise Exception("No library enabled for signature verification")

    # 如果验证地址是列表，则检查签名地址是否在列表中
    if type(valid_address) is list:  # Any address in the list
        return sign_address in valid_address
    else:  # One possible address
        # 如果验证地址是单个地址，则检查签名地址是否与之相等
        return sign_address == valid_address
```