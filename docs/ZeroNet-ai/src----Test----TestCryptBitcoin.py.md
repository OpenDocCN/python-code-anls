# `ZeroNet\src\Test\TestCryptBitcoin.py`

```py
# 导入 CryptBitcoin 模块
from Crypt import CryptBitcoin

# 定义测试类 TestCryptBitcoin
class TestCryptBitcoin:
    # 测试私钥签名方法
    def testSign(self, crypt_bitcoin_lib):
        # 设置私钥
        privatekey = "5K9S6dVpufGnroRgFrT6wsKiz2mJRYsC73eWDmajaHserAp3F1C"
        privatekey_bad = "5Jbm9rrusXyApAoM8YoM4Rja337zMMoBUMRJ1uijiguU2aZRnwC"

        # 通过私钥获取地址
        address = crypt_bitcoin_lib.privatekeyToAddress(privatekey)
        # 断言地址与预期地址相等
        assert address == "1MpDMxFeDUkiHohxx9tbGLeEGEuR4ZNsJz"

        # 通过错误的私钥获取地址
        address_bad = crypt_bitcoin_lib.privatekeyToAddress(privatekey_bad)
        # 断言地址与预期地址不相等
        assert address_bad != "1MpDMxFeDUkiHohxx9tbGLeEGEuR4ZNsJz"

        # 对文本进行签名
        data_len_list = list(range(0, 300, 10))
        data_len_list += [1024, 2048, 1024 * 128, 1024 * 1024, 1024 * 2048]
        for data_len in data_len_list:
            data = data_len * "!"
            sign = crypt_bitcoin_lib.sign(data, privatekey)

            # 断言签名验证通过
            assert crypt_bitcoin_lib.verify(data, address, sign)
            # 断言签名验证不通过
            assert not crypt_bitcoin_lib.verify("invalid" + data, address, sign)

        # 使用错误的私钥进行签名
        sign_bad = crypt_bitcoin_lib.sign("hello", privatekey_bad)
        # 断言签名验证不通过
        assert not crypt_bitcoin_lib.verify("hello", address, sign_bad)

    # 测试验证方法
    def testVerify(self, crypt_bitcoin_lib):
        # 验证未压缩的签名
        sign_uncompressed = b'G6YkcFTuwKMVMHI2yycGQIFGbCZVNsZEZvSlOhKpHUt/BlADY94egmDAWdlrbbFrP9wH4aKcEfbLO8sa6f63VU0='
        assert crypt_bitcoin_lib.verify("1NQUem2M4cAqWua6BVFBADtcSP55P4QobM#web/gitcenter", "19Bir5zRm1yo4pw9uuxQL8xwf9b7jqMpR", sign_uncompressed)

        # 验证压缩的签名
        sign_compressed = b'H6YkcFTuwKMVMHI2yycGQIFGbCZVNsZEZvSlOhKpHUt/BlADY94egmDAWdlrbbFrP9wH4aKcEfbLO8sa6f63VU0='
        assert crypt_bitcoin_lib.verify("1NQUem2M4cAqWua6BVFBADtcSP55P4QobM#web/gitcenter", "1KH5BdNnqxh2KRWMMT8wUXzUgz4vVQ4S8p", sign_compressed)

    # 测试生成新私钥方法
    def testNewPrivatekey(self):
        # 断言生成的新私钥不相等
        assert CryptBitcoin.newPrivatekey() != CryptBitcoin.newPrivatekey()
        # 断言私钥转地址方法正常运行
        assert CryptBitcoin.privatekeyToAddress(CryptBitcoin.newPrivatekey())
    # 定义一个测试函数，用于测试生成新的种子
    def testNewSeed(self):
        # 断言生成的两个种子不相等
        assert CryptBitcoin.newSeed() != CryptBitcoin.newSeed()
        # 断言生成的私钥对应的地址与通过生成的种子和索引生成的私钥对应的地址相等
        assert CryptBitcoin.privatekeyToAddress(
            CryptBitcoin.hdPrivatekey(CryptBitcoin.newSeed(), 0)
        )
        # 断言生成的私钥对应的地址与通过生成的种子和索引生成的私钥对应的地址相等
        assert CryptBitcoin.privatekeyToAddress(
            CryptBitcoin.hdPrivatekey(CryptBitcoin.newSeed(), 2**256)
        )
```