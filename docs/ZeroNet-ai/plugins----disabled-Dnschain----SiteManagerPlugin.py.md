# `ZeroNet\plugins\disabled-Dnschain\SiteManagerPlugin.py`

```
# 导入所需的模块
import logging, json, os, re, sys, time
import gevent
from Plugin import PluginManager
from Config import config
from util import Http
from Debug import Debug

# 不支持重新加载
allow_reload = False 

# 获取名为"DnschainPlugin"的日志记录器
log = logging.getLogger("DnschainPlugin")

# 将SiteManagerPlugin类注册到PluginManager的SiteManager中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # DNS缓存文件路径
    dns_cache_path = "%s/dns_cache.json" % config.data_dir
    # DNS缓存
    dns_cache = None

    # 检查是否为有效地址
    def isAddress(self, address):
        # 如果是域名，则返回True
        if self.isDomain(address): 
            return True
        else:
            return super(SiteManagerPlugin, self).isAddress(address)

    # 检查是否为域名
    # 返回：如果是域名则返回True
    def isDomain(self, address):
        return re.match(r"(.*?)([A-Za-z0-9_-]+\.[A-Za-z0-9]+)$", address)

    # 从data/dns_cache.json加载DNS条目
    def loadDnsCache(self):
        # 如果DNS缓存文件存在，则加载文件内容到dns_cache
        if os.path.isfile(self.dns_cache_path):
            self.dns_cache = json.load(open(self.dns_cache_path))
        else:
            self.dns_cache = {}
        # 记录调试信息
        log.debug("Loaded dns cache, entries: %s" % len(self.dns_cache))

    # 将DNS条目保存到data/dns_cache.json
    def saveDnsCache(self):
        # 将dns_cache内容以可读的格式写入到dns_cache_path文件中
        json.dump(self.dns_cache, open(self.dns_cache_path, "wb"), indent=2)

    # 使用dnschain.net解析域名
    # 返回：地址或None
    # 通过dnschain.net解析域名的DNS信息
    def resolveDomainDnschainNet(self, domain):
        try:
            # 匹配域名
            match = self.isDomain(domain)
            # 获取子域名和顶级域名
            sub_domain = match.group(1).strip(".")
            top_domain = match.group(2)
            # 如果没有子域名，则设置为@
            if not sub_domain: sub_domain = "@"
            address = None
            # 设置超时时间为5秒
            with gevent.Timeout(5, Exception("Timeout: 5s")):
                # 发送HTTP请求获取域名对应的数据
                res = Http.get("https://api.dnschain.net/v1/namecoin/key/%s" % top_domain).read()
                # 解析返回的JSON数据
                data = json.loads(res)["data"]["value"]
                # 如果数据中包含"zeronet"字段
                if "zeronet" in data:
                    # 遍历"zeronet"字段中的键值对
                    for key, val in data["zeronet"].items():
                        # 将解析结果存入DNS缓存，并设置缓存时间为5小时
                        self.dns_cache[key+"."+top_domain] = [val, time.time()+60*60*5] # Cache for 5 hours
                    # 保存DNS缓存
                    self.saveDnsCache()
                    # 返回子域名对应的地址
                    return data["zeronet"].get(sub_domain)
            # 如果未找到对应地址，则返回None
            return address
        except Exception as err:
            # 记录错误日志
            log.debug("Dnschain.net %s resolve error: %s" % (domain, Debug.formatException(err)))


    # 通过dnschain.info解析域名的DNS信息
    # 返回：地址或None
    def resolveDomainDnschainInfo(self, domain):
        try:
            # 匹配域名
            match = self.isDomain(domain)
            # 获取子域名和顶级域名
            sub_domain = match.group(1).strip(".")
            top_domain = match.group(2)
            # 如果没有子域名，则设置为@
            if not sub_domain: sub_domain = "@"
            address = None
            # 设置超时时间为5秒
            with gevent.Timeout(5, Exception("Timeout: 5s")):
                # 发送HTTP请求获取域名对应的数据
                res = Http.get("https://dnschain.info/bit/d/%s" % re.sub(r"\.bit$", "", top_domain)).read()
                # 解析返回的JSON数据
                data = json.loads(res)["value"]
                # 遍历"zeronet"字段中的键值对
                for key, val in data["zeronet"].items():
                    # 将解析结果存入DNS缓存，并设置缓存时间为5小时
                    self.dns_cache[key+"."+top_domain] = [val, time.time()+60*60*5] # Cache for 5 hours
                # 保存DNS缓存
                self.saveDnsCache()
                # 返回子域名对应的地址
                return data["zeronet"].get(sub_domain)
            # 如果未找到对应地址，则返回None
            return address
        except Exception as err:
            # 记录错误日志
            log.debug("Dnschain.info %s resolve error: %s" % (domain, Debug.formatException(err)))


    # 解析域名
    # 返回：地址或者 None
    def resolveDomain(self, domain):
        # 将域名转换为小写
        domain = domain.lower()
        # 如果 DNS 缓存为空，则加载 DNS 缓存
        if self.dns_cache == None:
            self.loadDnsCache()
        # 如果域名中的点数小于 2，表示顶级请求，需要在前面添加 @.
        if domain.count(".") < 2: 
            domain = "@."+domain
    
        # 获取域名在 DNS 缓存中的详细信息
        domain_details = self.dns_cache.get(domain)
        # 如果在缓存中找到域名信息，并且未过期，则返回地址
        if domain_details and time.time() < domain_details[1]: 
            return domain_details[0]
        else:
            # 使用 dnschain 解析 DNS
            thread_dnschain_info = gevent.spawn(self.resolveDomainDnschainInfo, domain)
            thread_dnschain_net = gevent.spawn(self.resolveDomainDnschainNet, domain)
            gevent.joinall([thread_dnschain_net, thread_dnschain_info]) # 等待完成
    
            # 如果两个线程都成功返回结果
            if thread_dnschain_info.value and thread_dnschain_net.value: 
                # 如果两个线程返回的值相同，则返回该值
                if thread_dnschain_info.value == thread_dnschain_net.value: 
                    return thread_dnschain_info.value 
                else:
                    log.error("Dns %s missmatch: %s != %s" % (domain, thread_dnschain_info.value, thread_dnschain_net.value))
    
            # 解析过程中出现问题
            if domain_details: # 解析失败，但是在缓存中有记录
                domain_details[1] = time.time()+60*60 # 1 小时后再尝试解析
                return domain_details[0]
            else: # 在缓存中未找到记录
                self.dns_cache[domain] = [None, time.time()+60] # 1 分钟后再尝试解析
                return None
    
    
    # 返回或创建站点并开始下载站点文件
    # 返回：站点或者如果 DNS 解析失败则返回 None
    # 检查是否需要获取指定地址的所有文件，默认为 True
    def need(self, address, all_file=True):
        # 如果地址是一个域名
        if self.isDomain(address): # Its looks like a domain
            # 解析域名得到对应的地址
            address_resolved = self.resolveDomain(address)
            # 如果成功解析，将地址替换为解析后的地址
            if address_resolved:
                address = address_resolved
            else:
                return None  # 如果解析失败，返回 None
        
        # 调用父类的 need 方法获取地址对应的内容
        return super(SiteManagerPlugin, self).need(address, all_file)


    # 返回: Site 对象，如果未找到则返回 None
    def get(self, address):
        # 如果站点列表未加载
        if self.sites == None: # Not loaded yet
            self.load()  # 加载站点列表
        # 如果地址是一个域名
        if self.isDomain(address): # Its looks like a domain
            # 解析域名得到对应的地址
            address_resolved = self.resolveDomain(address)
            # 如果成功解析，表示找到了对应的域名
            if address_resolved: # Domain found
                # 获取解析后的地址对应的站点对象
                site = self.sites.get(address_resolved)
                # 如果站点对象存在
                if site:
                    # 获取站点对象的域名设置
                    site_domain = site.settings.get("domain")
                    # 如果站点对象的域名与地址不一致，更新站点对象的域名设置
                    if site_domain != address:
                        site.settings["domain"] = address
            else: # Domain not found
                # 获取地址对应的站点对象
                site = self.sites.get(address)

        else: # Access by site address
            # 获取地址对应的站点对象
            site = self.sites.get(address)
        return site  # 返回获取到的站点对象
```