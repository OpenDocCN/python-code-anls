# `ZeroNet\plugins\Stats\StatsPlugin.py`

```
# 导入时间模块
import time
# 导入 HTML 模块
import html
# 导入操作系统模块
import os
# 导入 JSON 模块
import json
# 导入系统模块
import sys
# 导入迭代工具模块
import itertools

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 变量
from Config import config
# 从 util 模块中导入 helper 函数
from util import helper
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 Db 模块中导入 Db 类
from Db import Db

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):

    # 格式化表格行
    def formatTableRow(self, row, class_name=""):
        back = []
        # 遍历行中的格式和值
        for format, val in row:
            # 如果值为空，则格式化为 "n/a"
            if val is None:
                formatted = "n/a"
            # 如果格式为 "since"，则计算时间差
            elif format == "since":
                if val:
                    formatted = "%.0f" % (time.time() - val)
                else:
                    formatted = "n/a"
            # 否则按照指定格式进行格式化
            else:
                formatted = format % val
            back.append("<td>%s</td>" % formatted)
        # 返回格式化后的表格行
        return "<tr class='%s'>%s</tr>" % (class_name, "".join(back))

    # 获取对象大小
    def getObjSize(self, obj, hpy=None):
        # 如果传入了 hpy 参数，则返回对象大小
        if hpy:
            return float(hpy.iso(obj).domisize) / 1024
        # 否则返回 0
        else:
            return 0
    # 导入 main 模块
    import main
    # 从 Crypt 模块中导入 CryptConnection
    from Crypt import CryptConnection

    # 输出程序版本号
    yield "rev%s | " % config.rev
    # 输出文件服务器的外部 IP 地址列表
    yield "%s | " % main.file_server.ip_external_list
    # 输出文件服务器的端口号
    yield "Port: %s | " % main.file_server.port
    # 输出文件服务器支持的网络类型
    yield "Network: %s | " % main.file_server.supported_ip_types
    # 输出文件服务器的端口是否已打开
    yield "Opened: %s | " % main.file_server.port_opened
    # 输出加密连接是否支持，以及是否支持 TLSv1.3
    yield "Crypt: %s, TLSv1.3: %s | " % (CryptConnection.manager.crypt_supported, CryptConnection.ssl.HAS_TLSv1_3)
    # 输出文件服务器接收和发送的数据量
    yield "In: %.2fMB, Out: %.2fMB  | " % (
        float(main.file_server.bytes_recv) / 1024 / 1024,
        float(main.file_server.bytes_sent) / 1024 / 1024
    )
    # 输出文件服务器的对等 ID
    yield "Peerid: %s  | " % main.file_server.peer_id
    # 输出文件服务器的时间校正
    yield "Time: %.2fs | " % main.file_server.getTimecorrection()
    # 输出调试模式下的块数量
    yield "Blocks: %s" % Debug.num_block

    # 尝试获取系统资源使用情况
    try:
        # 导入 psutil 模块
        import psutil
        # 获取当前进程对象
        process = psutil.Process(os.getpid())
        # 获取内存使用情况
        mem = process.get_memory_info()[0] / float(2 ** 20)
        yield "Mem: %.2fMB | " % mem
        # 输出当前进程的线程数
        yield "Threads: %s | " % len(process.threads())
        # 输出当前进程的 CPU 使用情况
        yield "CPU: usr %.2fs sys %.2fs | " % process.cpu_times()
        # 输出当前进程打开的文件数
        yield "Files: %s | " % len(process.open_files())
        # 输出当前进程的网络连接数
        yield "Sockets: %s | " % len(process.connections())
        # 输出是否开启了计算大小的链接
        yield "Calc size <a href='?size=1'>on</a> <a href='?size=0'>off</a>"
    except Exception:
        # 如果获取系统资源使用情况出现异常，则忽略
        pass
    # 输出换行符
    yield "<br>"
    # 渲染跟踪器信息并生成 HTML 格式的字符串
    def renderTrackers(self):
        # 输出标题
        yield "<br><br><b>Trackers:</b><br>"
        # 输出表格头部
        yield "<table class='trackers'><tr> <th>address</th> <th>request</th> <th>successive errors</th> <th>last_request</th></tr>"
        # 导入 SiteAnnouncer 模块，遍历全局统计信息并输出到表格中
        from Site import SiteAnnouncer  # importing at the top of the file breaks plugins
        for tracker_address, tracker_stat in sorted(SiteAnnouncer.global_stats.items()):
            # 格式化输出每一行数据
            yield self.formatTableRow([
                ("%s", tracker_address),
                ("%s", tracker_stat["num_request"]),
                ("%s", tracker_stat["num_error"]),
                ("%.0f min ago", min(999, (time.time() - tracker_stat["time_request"]) / 60))
            ])
        # 输出表格尾部
        yield "</table>"

        # 如果存在名为 "AnnounceShare" 的插件，则输出共享跟踪器信息
        if "AnnounceShare" in PluginManager.plugin_manager.plugin_names:
            yield "<br><br><b>Shared trackers:</b><br>"
            yield "<table class='trackers'><tr> <th>address</th> <th>added</th> <th>found</th> <th>latency</th> <th>successive errors</th> <th>last_success</th></tr>"
            # 导入 AnnounceSharePlugin 模块，遍历共享跟踪器信息并输出到表格中
            from AnnounceShare import AnnounceSharePlugin
            for tracker_address, tracker_stat in sorted(AnnounceSharePlugin.tracker_storage.getTrackers().items()):
                # 格式化输出每一行数据
                yield self.formatTableRow([
                    ("%s", tracker_address),
                    ("%.0f min ago", min(999, (time.time() - tracker_stat["time_added"]) / 60)),
                    ("%.0f min ago", min(999, (time.time() - tracker_stat.get("time_found", 0)) / 60)),
                    ("%.3fs", tracker_stat["latency"]),
                    ("%s", tracker_stat["num_error"]),
                    ("%.0f min ago", min(999, (time.time() - tracker_stat["time_success"]) / 60)),
                ])
            # 输出表格尾部
            yield "</table>"
    # 渲染 Tor hidden services 的信息并生成 HTML 格式的字符串
    def renderTor(self):
        # 导入 main 模块
        import main
        # 生成包含 Tor hidden services 状态信息的字符串
        yield "<br><br><b>Tor hidden services (status: %s):</b><br>" % main.file_server.tor_manager.status
        # 遍历 main.file_server.tor_manager.site_onions 字典，生成包含站点地址和 onion 地址的字符串
        for site_address, onion in list(main.file_server.tor_manager.site_onions.items()):
            yield "- %-34s: %s<br>" % (site_address, onion)

    # 渲染数据库统计信息并生成 HTML 格式的字符串
    def renderDbStats(self):
        # 生成包含数据库信息的字符串
        yield "<br><br><b>Db</b>:<br>"
        # 遍历已打开的数据库列表
        for db in Db.opened_dbs:
            # 获取数据库中所有表的名称
            tables = [row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()]
            # 统计每个表的行数
            table_rows = {}
            for table in tables:
                table_rows[table] = db.execute("SELECT COUNT(*) AS c FROM %s" % table).fetchone()["c"]
            # 获取数据库文件大小
            db_size = os.path.getsize(db.db_path) / 1024.0 / 1024.0
            # 生成包含数据库信息的字符串
            yield "- %.3fs: %s %.3fMB, table rows: %s<br>" % (
                time.time() - db.last_query_time, db.db_path, db_size, json.dumps(table_rows, sort_keys=True)
            )
    # 定义一个生成器函数，用于渲染大文件信息
    def renderBigfiles(self):
        # 生成 HTML 标签，显示大文件标题
        yield "<br><br><b>Big files</b>:<br>"
        # 遍历服务器中的所有站点
        for site in list(self.server.sites.values()):
            # 如果站点设置中没有包含大文件信息，则跳过当前站点
            if not site.settings.get("has_bigfile"):
                continue
            # 创建空字典用于存储大文件信息
            bigfiles = {}
            # 生成 HTML 标签，点击链接显示站点的大文件信息
            yield """<a href="#" onclick='document.getElementById("bigfiles_%s").style.display="initial"; return false'>%s</a><br>""" % (site.address, site.address)
            # 遍历站点中的所有对等节点
            for peer in list(site.peers.values()):
                # 如果对等节点的时间片段字段没有更新，则跳过当前对等节点
                if not peer.time_piecefields_updated:
                    continue
                # 遍历对等节点的片段字段，将大文件信息存储到 bigfiles 字典中
                for sha512, piecefield in peer.piecefields.items():
                    if sha512 not in bigfiles:
                        bigfiles[sha512] = []
                    bigfiles[sha512].append(peer)

            # 生成 HTML 标签，显示站点的大文件信息
            yield "<div id='bigfiles_%s' style='display: none'>" % site.address
            # 遍历 bigfiles 字典，显示大文件的哈希值和对应的对等节点信息
            for sha512, peers in bigfiles.items():
                yield "<br> - " + sha512 + " (hash id: %s)<br>" % site.content_manager.hashfield.getHashId(sha512)
                yield "<table>"
                for peer in peers:
                    yield "<tr><td>" + peer.key + "</td><td>" + peer.piecefields[sha512].tostring() + "</td></tr>"
                yield "</table>"
            # 生成 HTML 标签，结束显示站点的大文件信息
            yield "</div>"
    # 渲染发送的请求信息
    def renderRequests(self):
        # 导入 main 模块
        import main
        # 生成一个左浮动的 div 标签
        yield "<div style='float: left'>"
        # 生成一个换行标签和粗体的“Sent commands”文本
        yield "<br><br><b>Sent commands</b>:<br>"
        # 生成一个表格标签
        yield "<table>"
        # 遍历已发送的命令统计信息，按照字节数从大到小排序
        for stat_key, stat in sorted(main.file_server.stat_sent.items(), key=lambda i: i[1]["bytes"], reverse=True):
            # 生成一个包含命令名、次数和字节数的表格行
            yield "<tr><td>%s</td><td style='white-space: nowrap'>x %s =</td><td>%.0fkB</td></tr>" % (stat_key, stat["num"], stat["bytes"] / 1024)
        # 生成表格结束标签
        yield "</table>"
        # 生成一个结束左浮动的 div 标签
        yield "</div>"

        # 生成一个左浮动、左边距为 20%、最大宽度为 50% 的 div 标签
        yield "<div style='float: left; margin-left: 20%; max-width: 50%'>"
        # 生成一个换行标签和粗体的“Received commands”文本
        yield "<br><br><b>Received commands</b>:<br>"
        # 生成一个表格标签
        yield "<table>"
        # 遍历已接收的命令统计信息，按照字节数从大到小排序
        for stat_key, stat in sorted(main.file_server.stat_recv.items(), key=lambda i: i[1]["bytes"], reverse=True):
            # 生成一个包含命令名、次数和字节数的表格行
            yield "<tr><td>%s</td><td style='white-space: nowrap'>x %s =</td><td>%.0fkB</td></tr>" % (stat_key, stat["num"], stat["bytes"] / 1024)
        # 生成表格结束标签
        yield "</table>"
        # 生成一个结束左浮动的 div 标签
        yield "</div>"
        # 生成一个清除浮动的 div 标签
        yield "<div style='clear: both'></div>"

    # /Stats entry point
    # 对 renderRequests 方法的响应进行编码
    @helper.encodeResponse
    # 定义一个名为 actionStats 的方法，用于生成统计信息页面
    def actionStats(self):
        import gc  # 导入垃圾回收模块

        self.sendHeader()  # 发送 HTTP 头部信息

        # 如果插件管理器中包含 Multiuser 插件，并且不是在本地多用户模式下，则禁用该函数并返回提示信息
        if "Multiuser" in PluginManager.plugin_manager.plugin_names and not config.multiuser_local:
            yield "This function is disabled on this proxy"
            return

        s = time.time()  # 记录当前时间

        # 添加内联样式
        yield """
        <style>
         * { font-family: monospace }
         table td, table th { text-align: right; padding: 0px 10px }
         .connections td { white-space: nowrap }
         .serving-False { opacity: 0.3 }
        </style>
        """

        # 定义要渲染的内容列表
        renderers = [
            self.renderHead(),  # 渲染头部信息
            self.renderConnectionsTable(),  # 渲染连接表格
            self.renderTrackers(),  # 渲染跟踪器信息
            self.renderTor(),  # 渲染 Tor 信息
            self.renderDbStats(),  # 渲染数据库统计信息
            self.renderSites(),  # 渲染站点信息
            self.renderBigfiles(),  # 渲染大文件信息
            self.renderRequests()  # 渲染请求信息
        ]

        # 遍历渲染内容列表，并逐个生成页面内容
        for part in itertools.chain(*renderers):
            yield part

        # 如果处于调试模式，则渲染内存信息
        if config.debug:
            for part in self.renderMemory():
                yield part

        gc.collect()  # 执行隐式垃圾回收
        yield "Done in %.1f" % (time.time() - s)  # 返回处理完成所花费的时间

    @helper.encodeResponse  # 对生成的响应进行编码
    # 定义一个名为 actionDumpobj 的方法，用于获取对象信息并输出
    def actionDumpobj(self):

        # 导入垃圾回收模块和系统模块
        import gc
        import sys

        # 发送 HTTP 头部信息
        self.sendHeader()

        # 如果插件管理器中包含 "Multiuser"，并且不是在本地多用户模式下，则输出提示信息并返回
        if "Multiuser" in PluginManager.plugin_manager.plugin_names and not config.multiuser_local:
            yield "This function is disabled on this proxy"
            return

        # 如果不是在调试模式下，则输出提示信息并返回
        if not config.debug:
            yield "Not in debug mode"
            return

        # 获取 URL 参数中的 class 值
        class_filter = self.get.get("class")

        # 输出一段 CSS 样式
        yield """
        <style>
         * { font-family: monospace; white-space: pre }
         table * { text-align: right; padding: 0px 10px }
        </style>
        """

        # 获取当前 Python 程序中的所有对象
        objs = gc.get_objects()
        # 遍历所有对象
        for obj in objs:
            # 获取对象的类型
            obj_type = str(type(obj))
            # 如果对象类型不是 "<type 'instance'>"，或者对象的类名不等于 class_filter，则继续下一次循环
            if obj_type != "<type 'instance'>" or obj.__class__.__name__ != class_filter:
                continue
            # 输出对象的大小和部分内容
            yield "%.1fkb %s... " % (float(sys.getsizeof(obj)) / 1024, html.escape(str(obj)))
            # 遍历对象的属性，并输出属性名和属性值
            for attr in dir(obj):
                yield "- %s: %s<br>" % (attr, html.escape(str(getattr(obj, attr))))
            yield "<br>"

        # 执行隐式垃圾回收
        gc.collect()  # Implicit grabage collection

    # 对响应进行编码
    @helper.encodeResponse
    @helper.encodeResponse
    # 定义名为 actionGcCollect 的方法，用于执行垃圾回收并返回结果
    def actionGcCollect(self):
        # 导入垃圾回收模块
        import gc
        # 发送 HTTP 头部信息
        self.sendHeader()
        # 返回执行垃圾回收的结果
        yield str(gc.collect())

    # /About 入口点
    # 对响应进行编码
    @helper.encodeResponse
    # 定义名为 actionEnv 的方法，用于输出环境信息
    def actionEnv(self):
        # 导入 main 模块
        import main

        # 发送 HTTP 头部信息
        self.sendHeader()

        # 输出一段 CSS 样式
        yield """
        <style>
         * { font-family: monospace; white-space: pre; }
         h2 { font-size: 100%; margin-bottom: 0px; }
         small { opacity: 0.5; }
         table { border-collapse: collapse; }
         td { padding-right: 10px; }
        </style>
        """

        # 如果插件管理器中包含 "Multiuser"，并且不是在本地多用户模式下，则输出提示信息并返回
        if "Multiuser" in PluginManager.plugin_manager.plugin_names and not config.multiuser_local:
            yield "This function is disabled on this proxy"
            return

        # 调用 main 模块中的 testEnv 方法，并以 HTML 格式输出结果
        yield from main.actions.testEnv(format="html")
# 将 ActionsPlugin 类注册到 PluginManager 的 Actions 插件中
@PluginManager.registerTo("Actions")
class ActionsPlugin:
    # 格式化表格，根据不同的格式返回不同的结果
    def formatTable(self, *rows, format="text"):
        # 如果格式为 html，则调用 formatTableHtml 方法
        if format == "html":
            return self.formatTableHtml(*rows)
        # 否则调用 formatTableText 方法
        else:
            return self.formatTableText(*rows)

    # 格式化表格标题，根据不同的格式返回不同的结果
    def formatHead(self, title, format="text"):
        # 如果格式为 html，则返回带有标题的 h2 标签
        if format == "html":
            return "<h2>%s</h2>" % title
        # 否则返回带有标题的列表项
        else:
            return "\n* %s\n" % title

    # 以 HTML 格式格式化表格
    def formatTableHtml(self, *rows):
        # 生成 HTML 表格的起始标签
        yield "<table>"
        # 遍历每一行
        for row in rows:
            # 生成 HTML 表格的行标签
            yield "<tr>"
            # 遍历每一列
            for col in row:
                # 生成 HTML 表格的单元格标签，并对内容进行 HTML 转义
                yield "<td>%s</td>" % html.escape(str(col))
            # 生成 HTML 表格的行结束标签
            yield "</tr>"
        # 生成 HTML 表格的结束标签
        yield "</table>"

    # 以文本格式格式化表格
    def formatTableText(self, *rows):
        # 遍历每一行
        for row in rows:
            # 生成空格
            yield " "
            # 遍历每一列
            for col in row:
                # 生成带有内容的文本行
                yield " " + str(col)
            # 生成换行符
            yield "\n"
```