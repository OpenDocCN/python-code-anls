# `ZeroNet\plugins\disabled-ZeronameLocal\UiRequestPlugin.py`

```py
# 导入正则表达式模块
import re
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 初始化方法
    def __init__(self, *args, **kwargs):
        # 从 Site 模块中导入 SiteManager 类
        from Site import SiteManager
        # 实例化 SiteManager 类并赋值给 self.site_manager
        self.site_manager = SiteManager.site_manager
        # 调用父类的初始化方法
        super(UiRequestPlugin, self).__init__(*args, **kwargs)

    # 媒体请求方法
    def actionSiteMedia(self, path):
        # 使用正则表达式匹配路径中的媒体地址和内部路径
        match = re.match(r"/media/(?P<address>[A-Za-z0-9-]+\.[A-Za-z0-9\.-]+)(?P<inner_path>/.*|$)", path)
        if match: # 如果匹配成功
            # 获取地址部分
            domain = match.group("address")
            # 通过 site_manager 解析域名，获取地址
            address = self.site_manager.resolveDomain(domain)
            if address: # 如果地址存在
                # 重新构建路径
                path = "/media/"+address+match.group("inner_path")
        # 调用父类的 actionSiteMedia 方法，获取包装框架输出
        return super(UiRequestPlugin, self).actionSiteMedia(path)

    # 检查媒体请求是否允许的方法
    def isMediaRequestAllowed(self, site_address, referer):
        # 从引用中移除站点地址
        referer_path = re.sub("http[s]{0,1}://.*?/", "/", referer).replace("/media", "")
        # 移除 http 参数
        referer_path = re.sub(r"\?.*", "", referer_path)

        if self.isProxyRequest(): # 如果是代理请求
            # 允许 /zero 访问
            referer = re.sub("^http://zero[/]+", "http://", referer)
            # 获取引用站点地址
            referer_site_address = re.match("http[s]{0,1}://(.*?)(/|$)", referer).group(1)
        else: # 如果不是代理请求
            # 匹配请求路径
            referer_site_address = re.match(r"/(?P<address>[A-Za-z0-9\.-]+)(?P<inner_path>/.*|$)", referer_path).group("address")

        if referer_site_address == site_address: # 如果引用站点地址与简单地址匹配
            return True
        elif self.site_manager.resolveDomain(referer_site_address) == site_address: # 如果引用站点地址与 DNS 匹配
            return True
        else: # 无效的引用
            return False
```