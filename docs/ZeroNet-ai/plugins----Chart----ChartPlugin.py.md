# `ZeroNet\plugins\Chart\ChartPlugin.py`

```py
# 导入时间模块
import time
# 导入迭代工具模块
import itertools

# 导入协程模块
import gevent

# 从Config模块中导入config变量
from Config import config
# 从util模块中导入helper函数
from util import helper
# 从util.Flag模块中导入flag变量
from util.Flag import flag
# 从Plugin模块中导入PluginManager类
from Plugin import PluginManager
# 从当前目录下的ChartDb模块中导入ChartDb类
from .ChartDb import ChartDb
# 从当前目录下的ChartCollector模块中导入ChartCollector类
from .ChartCollector import ChartCollector

# 如果当前作用域中不存在名为"db"的变量，则创建ChartDb对象并赋值给db变量
if "db" not in locals().keys():  # Share on reloads
    db = ChartDb()
    # 创建一个延迟10分钟执行的协程，用于执行db的archive方法
    gevent.spawn_later(10 * 60, db.archive)
    # 每6小时执行一次db的archive方法
    helper.timer(60 * 60 * 6, db.archive)
    # 创建ChartCollector对象并赋值给collector变量
    collector = ChartCollector(db)

# 将SiteManagerPlugin类注册到PluginManager的"SiteManager"中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # 加载方法
    def load(self, *args, **kwargs):
        # 调用父类的load方法
        back = super(SiteManagerPlugin, self).load(*args, **kwargs)
        # 设置collector的初始lastValues值为sites中的值
        collector.setInitialLastValues(self.sites.values())
        return back

    # 删除方法
    def delete(self, address, *args, **kwargs):
        # 删除指定地址的站点信息
        db.deleteSite(address)
        return super(SiteManagerPlugin, self).delete(address, *args, **kwargs)

# 将UiWebsocketPlugin类注册到PluginManager的"UiWebsocket"中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 标记为管理员权限的actionChartDbQuery方法
    @flag.admin
    def actionChartDbQuery(self, to, query, params=None):
        # 如果处于debug或verbose模式，则记录当前时间
        if config.debug or config.verbose:
            s = time.time()
        # 初始化结果集
        rows = []
        try:
            # 如果查询语句不是以SELECT开头，则抛出异常
            if not query.strip().upper().startswith("SELECT"):
                raise Exception("Only SELECT query supported")
            # 执行数据库查询
            res = db.execute(query, params)
        except Exception as err:  # 响应错误给客户端
            # 记录错误日志
            self.log.error("ChartDbQuery error: %s" % err)
            return {"error": str(err)}
        # 将结果转换为字典形式
        for row in res:
            rows.append(dict(row))
        # 如果处于verbose模式且查询时间超过0.1秒，则记录慢查询日志
        if config.verbose and time.time() - s > 0.1:  # Log slow query
            self.log.debug("Slow query: %s (%.3fs)" % (query, time.time() - s))
        return rows

    # 标记为管理员权限的actionChartGetPeerLocations方法
    @flag.admin
    def actionChartGetPeerLocations(self, to):
        # 初始化peers字典
        peers = {}
        # 遍历所有站点的peers，并更新到peers字典中
        for site in self.server.sites.values():
            peers.update(site.peers)
        # 获取peer的位置信息
        peer_locations = self.getPeerLocations(peers)
        return peer_locations
```