# `ZeroNet\src\Test\TestUser.py`

```py
# 导入 pytest 模块
import pytest

# 从 Crypt 模块中导入 CryptBitcoin 类
from Crypt import CryptBitcoin

# 使用 resetSettings 修饰器来重置设置
@pytest.mark.usefixtures("resetSettings")
class TestUser:
    # 测试用户地址
    def testAddress(self, user):
        # 断言用户的主地址
        assert user.master_address == "15E5rhcAUD69WbiYsYARh4YHJ4sLm2JEyc"
        # 设置地址索引
        address_index = 1458664252141532163166741013621928587528255888800826689784628722366466547364755811
        # 断言用户地址的认证索引
        assert user.getAddressAuthIndex("15E5rhcAUD69WbiYsYARh4YHJ4sLm2JEyc") == address_index

    # 根据地址索引重新生成私钥
    def testNewSite(self, user):
        # 获取新站点数据
        address, address_index, site_data = user.getNewSiteData()  # Create a new random site
        # 断言根据用户主种子和地址索引生成的私钥与站点数据中的私钥相等
        assert CryptBitcoin.hdPrivatekey(user.master_seed, address_index) == site_data["privatekey"]

        # 重置用户数据
        user.sites = {}

        # 站点地址和认证地址不同
        assert user.getSiteData(address)["auth_address"] != address
        # 重新生成站点的认证私钥
        assert user.getSiteData(address)["auth_privatekey"] == site_data["auth_privatekey"]

    # 测试认证地址
    def testAuthAddress(self, user):
        # 没有证书的认证地址
        auth_address = user.getAuthAddress("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr")
        # 断言认证地址
        assert auth_address == "1MyJgYQjeEkR9QD66nkfJc9zqi9uUy5Lr2"
        # 获取认证私钥
        auth_privatekey = user.getAuthPrivatekey("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr")
        # 断言根据认证私钥生成的地址与认证地址相等
        assert CryptBitcoin.privatekeyToAddress(auth_privatekey) == auth_address
    # 测试用户证书功能
    def testCert(self, user):
        # 获取用户的认证地址
        cert_auth_address = user.getAuthAddress("1iD5ZQJMNXu43w1qLB8sfdHVKppVMduGz")  # 将站点添加到用户的注册表中
        # 添加证书
        user.addCert(cert_auth_address, "zeroid.bit", "faketype", "fakeuser", "fakesign")
        # 设置证书
        user.setCert("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr", "zeroid.bit")

        # 通过使用证书，认证地址应该与证书提供者相同
        assert user.getAuthAddress("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr") == cert_auth_address
        # 获取认证私钥
        auth_privatekey = user.getAuthPrivatekey("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr")
        # 断言认证私钥对应的地址与认证地址相同
        assert CryptBitcoin.privatekeyToAddress(auth_privatekey) == cert_auth_address

        # 测试删除站点数据
        assert "1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr" in user.sites
        # 删除站点数据
        user.deleteSiteData("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr")
        # 断言站点数据已被删除
        assert "1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr" not in user.sites

        # 重新创建站点应该生成正常的唯一认证地址
        # 断言获取的认证地址与证书认证地址不同
        assert not user.getAuthAddress("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr") == cert_auth_address
        # 断言获取的认证地址与特定地址相同
        assert user.getAuthAddress("1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr") == "1MyJgYQjeEkR9QD66nkfJc9zqi9uUy5Lr2"
```