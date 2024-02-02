# `ZeroNet\plugins\CryptMessage\Test\TestCrypt.py`

```py
# 导入 pytest 模块
import pytest
# 导入 base64 模块
import base64
# 从 CryptMessage 模块中导入 CryptMessage 类
from CryptMessage import CryptMessage

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
class TestCrypt:
    # 设置公钥和私钥
    publickey = "A3HatibU4S6eZfIQhVs2u7GLN5G9wXa9WwlkyYIfwYaj"
    privatekey = "5JBiKFYBm94EUdbxtnuLi6cvNcPzcKymCUHBDf2B6aq19vvG3rL"
    # 设置 UTF-8 编码的文本
    utf8_text = '\xc1rv\xedzt\xfbr\xf5t\xfck\xf6rf\xfar\xf3g\xe9'
    # 设置经 ECIES 加密后的文本
    ecies_encrypted_text = "R5J1RFIDOzE5bnWopvccmALKACCk/CRcd/KSE9OgExJKASyMbZ57JVSUenL2TpABMmcT+wAgr2UrOqClxpOWvIUwvwwupXnMbRTzthhIJJrTRW3sCJVaYlGEMn9DAcvbflgEkQX/MVVdLV3tWKySs1Vk8sJC/y+4pGYCrZz7vwDNEEERaqU="

    # 参数化测试用例，text 参数为 b"hello" 和 UTF-8 编码的文本
    @pytest.mark.parametrize("text", [b"hello", '\xc1rv\xedzt\xfbr\xf5t\xfck\xf6rf\xfar\xf3g\xe9'.encode("utf8")])
    # 参数化测试用例，text_repeat 参数为 1, 10, 128, 1024
    @pytest.mark.parametrize("text_repeat", [1, 10, 128, 1024])
    # 测试 ECIES 加密函数
    def testEncryptEcies(self, text, text_repeat):
        # 将文本重复 text_repeat 次
        text_repeated = text * text_repeat
        # 使用 ECIES 加密文本
        aes_key, encrypted = CryptMessage.eciesEncrypt(text_repeated, self.publickey)
        # 断言 AES 密钥长度为 32
        assert len(aes_key) == 32
        # 断言加密后的文本长度，由于长度不确定，注释掉该断言
        # assert len(encrypted) == 134 + int(len(text) / 16) * 16  # Not always true
        # 使用 ECIES 解密函数解密文本，并断言解密后的文本与原文相同
        assert CryptMessage.eciesDecrypt(base64.b64encode(encrypted), self.privatekey) == text_repeated

    # 测试 ECIES 解密函数
    def testDecryptEcies(self, user):
        # 使用 ECIES 解密函数解密经过 ECIES 加密的文本，并断言解密后的文本与原文 b"hello" 相同
        assert CryptMessage.eciesDecrypt(self.ecies_encrypted_text, self.privatekey) == b"hello"
    # 测试公钥的方法，接受一个 UI WebSocket 对象作为参数
    def testPublickey(self, ui_websocket):
        # 调用 UI WebSocket 对象的 testAction 方法，获取用户公钥
        pub = ui_websocket.testAction("UserPublickey", 0)
        # 断言公钥长度为 44，表示压缩并经过 base64 编码的公钥
        assert len(pub) == 44  

        # 断言指定索引的公钥与索引为 0 的公钥不相同
        assert ui_websocket.testAction("UserPublickey", 1) != ui_websocket.testAction("UserPublickey", 0)

        # 断言相同索引的公钥相同
        assert ui_websocket.testAction("UserPublickey", 2) == ui_websocket.testAction("UserPublickey", 2)

        # 修改站点数据中的证书信息，获取索引为 0 的公钥
        site_data = ui_websocket.user.getSiteData(ui_websocket.site.address)
        site_data["cert"] = None
        pub1 = ui_websocket.testAction("UserPublickey", 0)

        # 再次修改站点数据中的证书信息，获取索引为 0 的公钥
        site_data = ui_websocket.user.getSiteData(ui_websocket.site.address)
        site_data["cert"] = "zeroid.bit"
        pub2 = ui_websocket.testAction("UserPublickey", 0)
        # 断言两次获取的公钥不相同
        assert pub1 != pub2

    # 测试 ECIES 加密解密的方法，接受一个 UI WebSocket 对象作为参数
    def testEcies(self, ui_websocket):
        # 获取用户公钥
        pub = ui_websocket.testAction("UserPublickey")

        # 使用公钥加密字符串 "hello"
        encrypted = ui_websocket.testAction("EciesEncrypt", "hello", pub)
        # 断言加密后的字符串长度为 180
        assert len(encrypted) == 180

        # 使用错误的私钥索引尝试解密
        decrypted = ui_websocket.testAction("EciesDecrypt", encrypted, 123)
        # 断言解密结果不等于 "hello"
        assert decrypted != "hello"

        # 使用正确的私钥索引进行解密
        decrypted = ui_websocket.testAction("EciesDecrypt", encrypted)
        # 断言解密结果等于 "hello"
        assert decrypted == "hello"

        # 尝试解密错误的文本
        decrypted = ui_websocket.testAction("EciesDecrypt", "baad")
        # 断言解密结果为 None
        assert decrypted is None

        # 批量解密
        decrypted = ui_websocket.testAction("EciesDecrypt", [encrypted, "baad", encrypted])
        # 断言批量解密结果为 ["hello", None, "hello"]
        assert decrypted == ["hello", None, "hello"]
    # 测试使用 UTF-8 编码的 Ecies 加密和解密
    def testEciesUtf8(self, ui_websocket):
        # 调用 EciesEncrypt 方法对 UTF-8 文本进行加密
        ui_websocket.actionEciesEncrypt(0, self.utf8_text)
        # 获取加密后的结果
        encrypted = ui_websocket.ws.getResult()

        # 调用 EciesDecrypt 方法对加密后的数据进行解密
        ui_websocket.actionEciesDecrypt(0, encrypted)
        # 断言解密后的结果与原始 UTF-8 文本相等
        assert ui_websocket.ws.getResult() == self.utf8_text

    # 测试使用 AES 加密的 Ecies 加密和解密
    def testEciesAes(self, ui_websocket):
        # 调用 EciesEncrypt 方法对 "hello" 文本进行加密，并返回 AES 密钥
        ui_websocket.actionEciesEncrypt(0, "hello", return_aes_key=True)
        # 获取 Ecies 加密后的结果和 AES 密钥
        ecies_encrypted, aes_key = ui_websocket.ws.getResult()

        # 使用 Ecies 解密
        ui_websocket.actionEciesDecrypt(0, ecies_encrypted)
        # 断言 Ecies 解密后的结果与原始文本 "hello" 相等
        assert ui_websocket.ws.getResult() == "hello"

        # 使用 AES 解密
        aes_iv, aes_encrypted = CryptMessage.split(base64.b64decode(ecies_encrypted))

        # 调用 AesDecrypt 方法对 AES 加密的数据进行解密
        ui_websocket.actionAesDecrypt(0, base64.b64encode(aes_iv), base64.b64encode(aes_encrypted), aes_key)
        # 断言 AES 解密后的结果与原始文本 "hello" 相等
        assert ui_websocket.ws.getResult() == "hello"

    # 测试使用长公钥的 Ecies 加密和解密
    def testEciesAesLongpubkey(self, ui_websocket):
        privatekey = "5HwVS1bTFnveNk9EeGaRenWS1QFzLFb5kuncNbiY3RiHZrVR6ok"

        ecies_encrypted, aes_key = ["lWiXfEikIjw1ac3J/RaY/gLKACALRUfksc9rXYRFyKDSaxhwcSFBYCgAdIyYlY294g/6VgAf/68PYBVMD3xKH1n7Zbo+ge8b4i/XTKmCZRJvy0eutMKWckYCMVcxgIYNa/ZL1BY1kvvH7omgzg1wBraoLfdbNmVtQgdAZ9XS8PwRy6OB2Q==", "Rvlf7zsMuBFHZIGHcbT1rb4If+YTmsWDv6kGwcvSeMM="]

        # 使用 Ecies 解密
        ui_websocket.actionEciesDecrypt(0, ecies_encrypted, privatekey)
        # 断言 Ecies 解密后的结果与原始文本 "hello" 相等
        assert ui_websocket.ws.getResult() == "hello"

        # 使用 AES 解密
        aes_iv, aes_encrypted = CryptMessage.split(base64.b64decode(ecies_encrypted))

        # 调用 AesDecrypt 方法对 AES 加密的数据进行解密
        ui_websocket.actionAesDecrypt(0, base64.b64encode(aes_iv), base64.b64encode(aes_encrypted), aes_key)
        # 断言 AES 解密后的结果与原始文本 "hello" 相等
        assert ui_websocket.ws.getResult() == "hello"
    # 测试 AES 加密功能
    def testAes(self, ui_websocket):
        # 调用 AES 加密方法，对 "hello" 进行加密
        ui_websocket.actionAesEncrypt(0, "hello")
        # 获取加密后的密钥、初始化向量和加密结果
        key, iv, encrypted = ui_websocket.ws.getResult()

        # 断言密钥长度为 44
        assert len(key) == 44
        # 断言初始化向量长度为 24
        assert len(iv) == 24
        # 断言加密结果长度为 24
        assert len(encrypted) == 24

        # 单个解密
        ui_websocket.actionAesDecrypt(0, iv, encrypted, key)
        # 断言解密结果为 "hello"
        assert ui_websocket.ws.getResult() == "hello"

        # 批量解密
        ui_websocket.actionAesEncrypt(0, "hello")
        key2, iv2, encrypted2 = ui_websocket.ws.getResult()

        # 断言两组密钥、初始化向量和加密结果不相等
        assert [key, iv, encrypted] != [key2, iv2, encrypted2]

        # 使用两个正确的密钥进行解密
        ui_websocket.actionAesDecrypt(0, [[iv, encrypted], [iv, encrypted], [iv, "baad"], [iv2, encrypted2]], [key])
        # 断言解密结果为 ["hello", "hello", None, None]
        assert ui_websocket.ws.getResult() == ["hello", "hello", None, None]

        # 使用三个密钥进行解密
        ui_websocket.actionAesDecrypt(0, [[iv, encrypted], [iv, encrypted], [iv, "baad"], [iv2, encrypted2]], [key, key2])
        # 断言解密结果为 ["hello", "hello", None, "hello"]
        assert ui_websocket.ws.getResult() == ["hello", "hello", None, "hello"]

    # 测试 AES 加密功能（UTF-8 编码）
    def testAesUtf8(self, ui_websocket):
        # 调用 AES 加密方法，对 UTF-8 编码的文本进行加密
        ui_websocket.actionAesEncrypt(0, self.utf8_text)
        # 获取加密后的密钥、初始化向量和加密结果
        key, iv, encrypted = ui_websocket.ws.getResult()

        # 使用密钥和初始化向量进行解密
        ui_websocket.actionAesDecrypt(0, iv, encrypted, key)
        # 断言解密结果为 UTF-8 编码的文本
        assert ui_websocket.ws.getResult() == self.utf8_text
```