# `ZeroNet\plugins\Newsfeed\NewsfeedPlugin.py`

```py
# 导入时间模块
import time
# 导入正则表达式模块
import re

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 DbQuery 模块中导入 DbQuery 类
from Db.DbQuery import DbQuery
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 helper 函数
from util import helper
# 从 util.Flag 模块中导入 flag 对象
from util.Flag import flag

# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 格式化站点信息
    def formatSiteInfo(self, site, create_user=True):
        # 调用父类的 formatSiteInfo 方法，获取站点信息
        site_info = super(UiWebsocketPlugin, self).formatSiteInfo(site, create_user=create_user)
        # 获取用户关注的站点的动态
        feed_following = self.user.sites.get(site.address, {}).get("follow", None)
        # 如果用户没有关注动态，则将 feed_follow_num 设为 None
        if feed_following == None:
            site_info["feed_follow_num"] = None
        else:
            # 否则，将 feed_follow_num 设为关注动态的数量
            site_info["feed_follow_num"] = len(feed_following)
        return site_info

    # 执行关注动态的操作
    def actionFeedFollow(self, to, feeds):
        # 设置用户关注的站点的动态
        self.user.setFeedFollow(self.site.address, feeds)
        # 保存用户信息
        self.user.save()
        # 响应操作结果
        self.response(to, "ok")

    # 执行列出关注动态的操作
    def actionFeedListFollow(self, to):
        # 获取用户关注的站点的动态
        feeds = self.user.sites.get(self.site.address, {}).get("follow", {})
        # 响应用户关注的站点的动态
        self.response(to, feeds)

    # 解析搜索关键词
    @flag.admin
    def parseSearch(self, search):
        # 使用正则表达式分割搜索关键词
        parts = re.split("(site|type):", search)
        # 如果找到了过滤器
        if len(parts) > 1:
            search_text = parts[0]
            parts = [part.strip() for part in parts]
            # 将过滤器组合成字典
            filters = dict(zip(parts[1::2], parts[2::2]))
        else:
            search_text = search
            filters = {}
        return [search_text, filters]

# 将 UserPlugin 类注册到 PluginManager 的 User 插件中
@PluginManager.registerTo("User")
class UserPlugin(object):
    # 设置用户关注的动态
    def setFeedFollow(self, address, feeds):
        # 获取站点数据
        site_data = self.getSiteData(address)
        # 设置用户关注的动态
        site_data["follow"] = feeds
        # 保存用户信息
        self.save()
        return site_data
```