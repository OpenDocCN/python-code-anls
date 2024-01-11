# `ZeroNet\src\User\User.py`

```
# 导入日志模块
import logging
# 导入 JSON 模块
import json
# 导入时间模块
import time
# 导入二进制转换模块
import binascii

# 导入协程模块
import gevent

# 导入自定义工具模块
import util
# 导入加密模块
from Crypt import CryptBitcoin
# 导入插件管理模块
from Plugin import PluginManager
# 导入配置模块
from Config import config
# 导入辅助函数模块
from util import helper
# 导入调试模块
from Debug import Debug

# 标记为可接受插件
@PluginManager.acceptPlugins
# 用户类
class User(object):
    # 初始化方法
    def __init__(self, master_address=None, master_seed=None, data={}):
        # 如果有主密钥，则设置主密钥和主地址
        if master_seed:
            self.master_seed = master_seed
            self.master_address = CryptBitcoin.privatekeyToAddress(self.master_seed)
        # 如果有主地址，则设置主地址和主密钥
        elif master_address:
            self.master_address = master_address
            self.master_seed = data.get("master_seed")
        # 否则生成新的主密钥和主地址
        else:
            self.master_seed = CryptBitcoin.newSeed()
            self.master_address = CryptBitcoin.privatekeyToAddress(self.master_seed)
        # 设置站点数据
        self.sites = data.get("sites", {})
        # 设置证书数据
        self.certs = data.get("certs", {})
        # 设置用户设置数据
        self.settings = data.get("settings", {})
        # 延迟保存线程
        self.delayed_save_thread = None

        # 设置日志记录器
        self.log = logging.getLogger("User:%s" % self.master_address)

    # 保存到 data/users.json
    @util.Noparallel(queue=True, ignore_class=True)
    def save(self):
        # 记录开始时间
        s = time.time()
        # 读取 users.json 文件内容
        users = json.load(open("%s/users.json" % config.data_dir))
        # 如果主地址不在 users 中，则创建
        if self.master_address not in users:
            users[self.master_address] = {}  # Create if not exist
        user_data = users[self.master_address]
        # 如果有主密钥，则设置主密钥
        if self.master_seed:
            user_data["master_seed"] = self.master_seed
        # 设置站点数据
        user_data["sites"] = self.sites
        # 设置证书数据
        user_data["certs"] = self.certs
        # 设置用户设置数据
        user_data["settings"] = self.settings
        # 原子性写入 users.json 文件
        helper.atomicWrite("%s/users.json" % config.data_dir, helper.jsonDumps(users).encode("utf8"))
        # 记录保存时间
        self.log.debug("Saved in %.3fs" % (time.time() - s))
        # 清空延迟保存线程
        self.delayed_save_thread = None

    # 延迟保存方法
    def saveDelayed(self):
        # 如果没有延迟保存线程，则创建延迟保存线程
        if not self.delayed_save_thread:
            self.delayed_save_thread = gevent.spawn_later(5, self.save)
    # 获取地址的授权索引，将地址编码为十六进制，再转换为整数返回
    def getAddressAuthIndex(self, address):
        return int(binascii.hexlify(address.encode()), 16)

    # 生成授权地址
    @util.Noparallel()
    def generateAuthAddress(self, address):
        s = time.time()
        # 将地址转换为授权索引
        address_id = self.getAddressAuthIndex(address)  # Convert site address to int
        # 使用主种子和地址索引生成授权私钥
        auth_privatekey = CryptBitcoin.hdPrivatekey(self.master_seed, address_id)
        # 将生成的授权地址和授权私钥存储到sites字典中
        self.sites[address] = {
            "auth_address": CryptBitcoin.privatekeyToAddress(auth_privatekey),
            "auth_privatekey": auth_privatekey
        }
        # 延迟保存数据
        self.saveDelayed()
        # 记录日志
        self.log.debug("Added new site: %s in %.3fs" % (address, time.time() - s))
        # 返回生成的站点数据
        return self.sites[address]

    # 获取用户站点数据
    # 返回: {"auth_address": "xxx", "auth_privatekey": "xxx"}
    def getSiteData(self, address, create=True):
        # 如果地址不在sites字典中
        if address not in self.sites:  # Generate new BIP32 child key based on site address
            # 如果不需要创建新用户，则返回空数据
            if not create:
                return {"auth_address": None, "auth_privatekey": None}  # Dont create user yet
            # 否则生成新的授权地址
            self.generateAuthAddress(address)
        # 返回地址对应的站点数据
        return self.sites[address]

    # 删除站点数据
    def deleteSiteData(self, address):
        # 如果地址在sites字典中
        if address in self.sites:
            # 删除对应的站点数据
            del(self.sites[address])
            # 延迟保存数据
            self.saveDelayed()
            # 记录日志
            self.log.debug("Deleted site: %s" % address)

    # 设置站点设置
    def setSiteSettings(self, address, settings):
        # 获取地址对应的站点数据
        site_data = self.getSiteData(address)
        # 更新站点设置
        site_data["settings"] = settings
        # 延迟保存数据
        self.saveDelayed()
        # 返回更新后的站点数据
        return site_data

    # 获取新的唯一站点数据
    # 返回: [site_address, bip32_index, {"auth_address": "xxx", "auth_privatekey": "xxx", "privatekey": "xxx"}]
    # 获取新的站点数据
    def getNewSiteData(self):
        # 导入随机模块
        import random
        # 生成一个随机的 BIP32 索引
        bip32_index = random.randrange(2 ** 256) % 100000000
        # 使用主种子和 BIP32 索引生成站点私钥
        site_privatekey = CryptBitcoin.hdPrivatekey(self.master_seed, bip32_index)
        # 将站点私钥转换为站点地址
        site_address = CryptBitcoin.privatekeyToAddress(site_privatekey)
        # 如果站点地址已存在于 self.sites 中，则抛出异常
        if site_address in self.sites:
            raise Exception("Random error: site exist!")
        # 保存站点数据
        self.getSiteData(site_address)
        # 将站点私钥存储到 self.sites 中
        self.sites[site_address]["privatekey"] = site_privatekey
        # 保存数据
        self.save()
        # 返回站点地址、BIP32 索引和站点地址对应的数据
        return site_address, bip32_index, self.sites[site_address]

    # 从站点地址获取 BIP32 地址
    # 返回：BIP32 授权地址
    def getAuthAddress(self, address, create=True):
        # 获取站点地址对应的证书
        cert = self.getCert(address)
        # 如果证书存在，则返回证书中的授权地址
        if cert:
            return cert["auth_address"]
        # 如果证书不存在，则根据站点地址获取站点数据，并返回其中的授权地址
        else:
            return self.getSiteData(address, create)["auth_address"]

    # 从站点地址获取授权私钥
    def getAuthPrivatekey(self, address, create=True):
        # 获取站点地址对应的证书
        cert = self.getCert(address)
        # 如果证书存在，则返回证书中的授权私钥
        if cert:
            return cert["auth_privatekey"]
        # 如果证书不存在，则根据站点地址获取站点数据，并返回其中的授权私钥
        else:
            return self.getSiteData(address, create)["auth_privatekey"]

    # 为用户添加证书
    # 向证书字典中添加证书信息
    def addCert(self, auth_address, domain, auth_type, auth_user_name, cert_sign):
        # 通过认证地址查找私钥
        auth_privatekey = [site["auth_privatekey"] for site in list(self.sites.values()) if site["auth_address"] == auth_address][0]
        # 创建证书节点
        cert_node = {
            "auth_address": auth_address,
            "auth_privatekey": auth_privatekey,
            "auth_type": auth_type,
            "auth_user_name": auth_user_name,
            "cert_sign": cert_sign
        }
        # 检查是否已经存在该域名的证书，并且不是相同的证书
        if self.certs.get(domain) and self.certs[domain] != cert_node:
            return False
        # 如果已经存在相同的证书，则返回 None
        elif self.certs.get(domain) == cert_node:  # Same, not updated
            return None
        else:  # 不存在该证书，添加
            self.certs[domain] = cert_node
            self.save()
            return True

    # 从用户中删除证书
    def deleteCert(self, domain):
        del self.certs[domain]

    # 为站点设置活跃的证书
    def setCert(self, address, domain):
        site_data = self.getSiteData(address)
        if domain:
            site_data["cert"] = domain
        else:
            if "cert" in site_data:
                del site_data["cert"]
        self.saveDelayed()
        return site_data

    # 获取站点地址的证书
    # 返回: { "auth_address":.., "auth_privatekey":.., "auth_type": "web", "auth_user_name": "nofish", "cert_sign":.. } 或 None
    def getCert(self, address):
        site_data = self.getSiteData(address, create=False)
        if not site_data or "cert" not in site_data:
            return None  # 站点没有证书
        return self.certs.get(site_data["cert"])

    # 获取站点地址的证书用户名
    # 返回: user@certprovider.bit 或 None
    # 获取证书的用户ID
    def getCertUserId(self, address):
        # 调用getSiteData方法获取指定地址的站点数据，如果站点数据不存在或者不包含证书信息，则返回None
        site_data = self.getSiteData(address, create=False)
        if not site_data or "cert" not in site_data:
            return None  # 站点没有证书
        # 从self.certs中获取站点数据中的证书信息
        cert = self.certs.get(site_data["cert"])
        if cert:
            # 返回证书的认证用户名和站点数据中的证书信息拼接而成的字符串
            return cert["auth_user_name"] + "@" + site_data["cert"]
```