# `ZeroNet\src\Test\TestCryptConnection.py`

```py
# 导入 os 模块
import os

# 从 Config 模块中导入 config 对象
from Config import config
# 从 Crypt 模块中导入 CryptConnection 类

class TestCryptConnection:
    # 定义测试方法 testSslCert
    def testSslCert(self):
        # 删除旧的证书文件
        if os.path.isfile("%s/cert-rsa.pem" % config.data_dir):
            os.unlink("%s/cert-rsa.pem" % config.data_dir)
        if os.path.isfile("%s/key-rsa.pem" % config.data_dir):
            os.unlink("%s/key-rsa.pem" % config.data_dir)

        # 生成新的证书
        CryptConnection.manager.loadCerts()

        # 断言 "tls-rsa" 在 CryptConnection.manager.crypt_supported 中
        assert "tls-rsa" in CryptConnection.manager.crypt_supported
        # 断言 CryptConnection.manager.selectCrypt(["tls-rsa", "unknown"]) 返回 "tls-rsa"，应该选择已知的加密方式
        assert CryptConnection.manager.selectCrypt(["tls-rsa", "unknown"]) == "tls-rsa"

        # 检查 openssl 证书生成
        # 断言 "%s/cert-rsa.pem" % config.data_dir 文件存在
        assert os.path.isfile("%s/cert-rsa.pem" % config.data_dir)
        # 断言 "%s/key-rsa.pem" % config.data_dir 文件存在
        assert os.path.isfile("%s/key-rsa.pem" % config.data_dir)
```