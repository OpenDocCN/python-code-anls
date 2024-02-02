# `ZeroNet\plugins\CryptMessage\CryptMessagePlugin.py`

```py
# 导入 base64 模块
import base64
# 导入 os 模块
import os
# 导入 gevent 模块
import gevent
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Crypt 模块中导入 CryptBitcoin, CryptHash 类
from Crypt import CryptBitcoin, CryptHash
# 从 Config 模块中导入 config 变量
from Config import config
# 导入 sslcrypto 模块
import sslcrypto
# 从当前目录中的 CryptMessage 模块中导入所有内容
from . import CryptMessage

# 获取椭圆曲线 secp256k1
curve = sslcrypto.ecc.get_curve("secp256k1")

# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # - Actions -

    # 返回用户在站点上的公钥
    # 返回：公钥
    def actionUserPublickey(self, to, index=0):
        self.response(to, self.user.getEncryptPublickey(self.site.address, index))

    # 使用公钥或用户站点的唯一公钥加密文本
    # 返回：使用 base64 编码的加密文本
    def actionEciesEncrypt(self, to, text, publickey=0, return_aes_key=False):
        if type(publickey) is int:  # 使用用户的公钥进行加密
            publickey = self.user.getEncryptPublickey(self.site.address, publickey)
        aes_key, encrypted = CryptMessage.eciesEncrypt(text.encode("utf8"), publickey)
        if return_aes_key:
            self.response(to, [base64.b64encode(encrypted).decode("utf8"), base64.b64encode(aes_key).decode("utf8")])
        else:
            self.response(to, base64.b64encode(encrypted).decode("utf8"))

    # 使用私钥或用户站点的唯一私钥解密文本
    # 返回：解密后的文本或解密后的文本列表
    def actionEciesDecrypt(self, to, param, privatekey=0):
        if type(privatekey) is int:  # 使用用户的私钥进行解密
            privatekey = self.user.getEncryptPrivatekey(self.site.address, privatekey)

        if type(param) == list:
            encrypted_texts = param
        else:
            encrypted_texts = [param]

        texts = CryptMessage.eciesDecryptMulti(encrypted_texts, privatekey)

        if type(param) == list:
            self.response(to, texts)
        else:
            self.response(to, texts[0])

    # 使用 AES 加密文本
    # 返回：初始化向量（Iv）、AES 密钥、加密文本
    # 使用AES加密文本
    # 参数to: 目标
    # 参数text: 要加密的文本
    # 参数key: 加密密钥，如果没有指定则生成一个新的密钥
    def actionAesEncrypt(self, to, text, key=None):
        if key:
            key = base64.b64decode(key)  # 如果有指定密钥，则解码为二进制格式
        else:
            key = sslcrypto.aes.new_key()  # 否则生成一个新的AES密钥

        if text:
            encrypted, iv = sslcrypto.aes.encrypt(text.encode("utf8"), key)  # 如果有文本，则使用AES加密
        else:
            encrypted, iv = b"", b""  # 否则返回空的加密文本和初始向量

        res = [base64.b64encode(item).decode("utf8") for item in [key, iv, encrypted]]  # 将加密后的密钥、初始向量和文本进行base64编码并存储在列表中
        self.response(to, res)  # 返回加密结果

    # 解密使用AES加密的文本
    # 参数to: 目标
    # 参数args: 要解密的文本和密钥
    def actionAesDecrypt(self, to, *args):
        if len(args) == 3:  # 如果参数个数为3，表示单个解密
            encrypted_texts = [(args[0], args[1])]
            keys = [args[2]]
        else:  # 否则为批量解密
            encrypted_texts, keys = args

        texts = []  # 存储解密后的文本
        for iv, encrypted_text in encrypted_texts:
            encrypted_text = base64.b64decode(encrypted_text)  # 解码加密文本
            iv = base64.b64decode(iv)  # 解码初始向量
            text = None
            for key in keys:
                try:
                    decrypted = sslcrypto.aes.decrypt(encrypted_text, iv, base64.b64decode(key))  # 使用AES解密
                    if decrypted and decrypted.decode("utf8"):  # 如果解密成功且为有效文本
                        text = decrypted.decode("utf8")  # 存储解密后的文本
                except Exception as err:
                    pass
            texts.append(text)  # 将解密后的文本存储在列表中

        if len(args) == 3:
            self.response(to, texts[0])  # 如果参数个数为3，返回单个解密后的文本
        else:
            self.response(to, texts)  # 否则返回批量解密后的文本列表

    # 使用ECDSA对数据进行签名
    # 参数to: 目标
    # 参数data: 要签名的数据
    # 参数privatekey: 私钥，如果没有指定则使用用户的私钥
    def actionEcdsaSign(self, to, data, privatekey=None):
        if privatekey is None:  # 如果没有指定私钥，则使用用户的私钥进行签名
            privatekey = self.user.getAuthPrivatekey(self.site.address)

        self.response(to, CryptBitcoin.sign(data, privatekey))  # 返回数据签名结果

    # 使用ECDSA验证数据的签名
    # 参数to: 目标
    # 参数data: 要验证的数据
    # 参数address: 地址或地址数组
    # 参数signature: 签名
    def actionEcdsaVerify(self, to, data, address, signature):
        self.response(to, CryptBitcoin.verify(data, address, signature))  # 返回数据验证结果
    # 获取给定私钥的公钥
    def actionEccPrivToPub(self, to, privatekey):
        # 调用 curve.wif_to_private() 方法将私钥编码为字节流，再调用 curve.private_to_public() 方法获取公钥
        self.response(to, curve.private_to_public(curve.wif_to_private(privatekey.encode())))
    
    # 获取给定公钥的地址
    def actionEccPubToAddr(self, to, publickey):
        # 将十六进制的公钥转换为字节流，然后调用 curve.public_to_address() 方法获取地址
        self.response(to, curve.public_to_address(bytes.fromhex(publickey)))
# 将 UserPlugin 类注册到 PluginManager 的 User 插件中
@PluginManager.registerTo("User")
class UserPlugin(object):
    # 获取加密私钥
    def getEncryptPrivatekey(self, address, param_index=0):
        # 如果参数索引小于0或大于1000，则抛出异常
        if param_index < 0 or param_index > 1000:
            raise Exception("Param_index out of range")

        # 获取地址对应的站点数据
        site_data = self.getSiteData(address)

        # 如果站点数据中存在证书，则根据证书提供商获取索引
        if site_data.get("cert"):
            index = param_index + self.getAddressAuthIndex(site_data["cert"])
        else:
            index = param_index

        # 如果站点数据中不存在加密私钥，则生成新的加密私钥并保存到站点数据中
        if "encrypt_privatekey_%s" % index not in site_data:
            address_index = self.getAddressAuthIndex(address)
            crypt_index = address_index + 1000 + index
            site_data["encrypt_privatekey_%s" % index] = CryptBitcoin.hdPrivatekey(self.master_seed, crypt_index)
            self.log.debug("New encrypt privatekey generated for %s:%s" % (address, index))
        # 返回站点数据中的加密私钥
        return site_data["encrypt_privatekey_%s" % index]

    # 获取加密公钥
    def getEncryptPublickey(self, address, param_index=0):
        # 如果参数索引小于0或大于1000，则抛出异常
        if param_index < 0 or param_index > 1000:
            raise Exception("Param_index out of range")

        # 获取地址对应的站点数据
        site_data = self.getSiteData(address)

        # 如果站点数据中存在证书，则根据证书提供商获取索引
        if site_data.get("cert"):
            index = param_index + self.getAddressAuthIndex(site_data["cert"])
        else:
            index = param_index

        # 如果站点数据中不存在加密公钥，则根据加密私钥生成新的加密公钥并保存到站点数据中
        if "encrypt_publickey_%s" % index not in site_data:
            privatekey = self.getEncryptPrivatekey(address, param_index).encode()
            publickey = curve.private_to_public(curve.wif_to_private(privatekey) + b"\x01")
            site_data["encrypt_publickey_%s" % index] = base64.b64encode(publickey).decode("utf8")
        # 返回站点数据中的加密公钥
        return site_data["encrypt_publickey_%s" % index]


# 将 ActionsPlugin 类注册到 PluginManager 的 Actions 插件中
@PluginManager.registerTo("Actions")
class ActionsPlugin:
    publickey = "A3HatibU4S6eZfIQhVs2u7GLN5G9wXa9WwlkyYIfwYaj"
    privatekey = "5JBiKFYBm94EUdbxtnuLi6cvNcPzcKymCUHBDf2B6aq19vvG3rL"
    # 定义一个包含非 ASCII 字符的 UTF-8 文本
    utf8_text = '\xc1rv\xedzt\xfbr\xf5t\xfck\xf6rf\xfar\xf3g\xe9p'
    
    # 获取基准测试，可以选择是否在线
    def getBenchmarkTests(self, online=False):
        # 如果父类有 getBenchmarkTests 方法，则调用父类的方法获取测试
        if hasattr(super(), "getBenchmarkTests"):
            tests = super().getBenchmarkTests(online)
        else:
            tests = []  # 否则创建一个空列表
    
        # 对文本进行加密，作为预热
        aes_key, encrypted = CryptMessage.eciesEncrypt(self.utf8_text.encode("utf8"), self.publickey)
        # 添加一系列测试到 tests 列表中
        tests.extend([
            {"func": self.testCryptEciesEncrypt, "kwargs": {}, "num": 100, "time_standard": 1.2},
            {"func": self.testCryptEciesDecrypt, "kwargs": {}, "num": 500, "time_standard": 1.3},
            {"func": self.testCryptEciesDecryptMulti, "kwargs": {}, "num": 5, "time_standard": 0.68},
            {"func": self.testCryptAesEncrypt, "kwargs": {}, "num": 10000, "time_standard": 0.27},
            {"func": self.testCryptAesDecrypt, "kwargs": {}, "num": 10000, "time_standard": 0.25}
        ])
        return tests  # 返回测试列表
    
    # 测试加密函数
    def testCryptEciesEncrypt(self, num_run=1):
        for i in range(num_run):
            aes_key, encrypted = CryptMessage.eciesEncrypt(self.utf8_text.encode("utf8"), self.publickey)
            assert len(aes_key) == 32  # 断言 AES 密钥长度为 32
            yield "."
    
    # 测试解密函数
    def testCryptEciesDecrypt(self, num_run=1):
        aes_key, encrypted = CryptMessage.eciesEncrypt(self.utf8_text.encode("utf8"), self.publickey)
        for i in range(num_run):
            assert len(aes_key) == 32  # 断言 AES 密钥长度为 32
            decrypted = CryptMessage.eciesDecrypt(base64.b64encode(encrypted), self.privatekey)
            assert decrypted == self.utf8_text.encode("utf8"), "%s != %s" % (decrypted, self.utf8_text.encode("utf8"))  # 断言解密后的文本与原文相同
            yield "."
    # 测试使用 ECIES 解密多个加密消息
    def testCryptEciesDecryptMulti(self, num_run=1):
        # 生成测试信息
        yield "x 100 (%s threads) " % config.threads_crypt
        # 使用 ECIES 加密文本，并返回加密后的 AES 密钥和密文
        aes_key, encrypted = CryptMessage.eciesEncrypt(self.utf8_text.encode("utf8"), self.publickey)

        # 创建线程列表
        threads = []
        # 循环指定次数
        for i in range(num_run):
            # 确保 AES 密钥长度为 32
            assert len(aes_key) == 32
            # 向线程列表中添加一个线程，用于解密多个加密消息
            threads.append(gevent.spawn(
                CryptMessage.eciesDecryptMulti, [base64.b64encode(encrypted)] * 100, self.privatekey
            ))

        # 遍历线程列表
        for thread in threads:
            # 获取线程的结果
            res = thread.get()
            # 断言解密后的文本与原文相同
            assert res[0] == self.utf8_text, "%s != %s" % (res[0], self.utf8_text)
            # 断言解密后的文本都相同
            assert res[0] == res[-1], "%s != %s" % (res[0], res[-1])
            # 产生一个点，表示测试通过
            yield "."
        # 等待所有线程结束
        gevent.joinall(threads)

    # 测试使用 AES 加密文本
    def testCryptAesEncrypt(self, num_run=1):
        # 循环指定次数
        for i in range(num_run):
            # 生成一个随机的 AES 密钥
            key = os.urandom(32)
            # 使用 AES 加密文本
            encrypted = sslcrypto.aes.encrypt(self.utf8_text.encode("utf8"), key)
            # 产生一个点，表示测试通过
            yield "."

    # 测试使用 AES 解密文本
    def testCryptAesDecrypt(self, num_run=1):
        # 生成一个随机的 AES 密钥
        key = os.urandom(32)
        # 使用 AES 加密文本，并返回加密后的文本和初始化向量
        encrypted_text, iv = sslcrypto.aes.encrypt(self.utf8_text.encode("utf8"), key)

        # 循环指定次数
        for i in range(num_run):
            # 使用 AES 解密文本
            decrypted = sslcrypto.aes.decrypt(encrypted_text, iv, key).decode("utf8")
            # 断言解密后的文本与原文相同
            assert decrypted == self.utf8_text
            # 产生一个点，表示测试通过
            yield "."
```