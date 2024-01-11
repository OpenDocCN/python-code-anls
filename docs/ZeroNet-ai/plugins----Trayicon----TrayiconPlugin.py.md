# `ZeroNet\plugins\Trayicon\TrayiconPlugin.py`

```
# 导入所需的模块
import os
import sys
import atexit

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 对象
from Config import config
# 从 Translate 模块中导入 Translate 类
from Translate import Translate

# 设置是否允许重新加载插件的标志
allow_reload = False  # No source reload supported in this plugin

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果当前作用域中不存在下划线变量，则创建一个 Translate 对象并赋值给下划线变量
if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")

# 将 ActionsPlugin 类注册到 PluginManager 的 "Actions" 中
@PluginManager.registerTo("Actions")
class ActionsPlugin(object):

    # 退出方法
    def quit(self):
        # 关闭图标
        self.icon.die()
        # 设置退出服务器事件为 True
        self.quit_servers_event.set(True)

    # 退出服务器方法
    def quitServers(self):
        # 停止主 UI 服务器
        self.main.ui_server.stop()
        # 停止文件服务器
        self.main.file_server.stop()

    # 打开网站方法
    def opensite(self, url):
        # 导入 webbrowser 模块
        import webbrowser
        # 在浏览器中打开指定的 URL
        webbrowser.open(url, new=0)

    # 获取 IP 标题方法
    def titleIp(self):
        # 获取 IP 标题
        title = "!IP: %s " % ", ".join(self.main.file_server.ip_external_list)
        # 如果有任何一个端口打开，则添加 "(active)"，否则添加 "(passive)"
        if any(self.main.file_server.port_opened):
            title += _["(active)"]
        else:
            title += _["(passive)"]
        return title

    # 获取连接数标题方法
    def titleConnections(self):
        # 获取连接数标题
        title = _["Connections: %s"] % len(self.main.file_server.connections)
        return title

    # 获取传输标题方法
    def titleTransfer(self):
        # 获取传输标题
        title = _["Received: %.2f MB | Sent: %.2f MB"] % (
            float(self.main.file_server.bytes_recv) / 1024 / 1024,
            float(self.main.file_server.bytes_sent) / 1024 / 1024
        )
        return title

    # 获取控制台标题方法
    def titleConsole(self):
        # 获取控制台标题
        translate = _["Show console window"]
        if self.console:
            return "+" + translate
        else:
            return translate

    # 切换控制台方法
    def toggleConsole(self):
        # 如果控制台已经打开，则隐藏控制台并将控制台标志设置为 False
        if self.console:
            notificationicon.hideConsole()
            self.console = False
        # 如果控制台未打开，则显示控制台并将控制台标志设置为 True
        else:
            notificationicon.showConsole()
            self.console = True

    # 获取自动运行路径方法
    def getAutorunPath(self):
        return "%s\\zeronet.cmd" % winfolders.get(winfolders.STARTUP)
    # 格式化自动运行脚本
    def formatAutorun(self):
        # 复制命令行参数
        args = sys.argv[:]
    
        # 如果不是冻结状态，即未打包成可执行文件
        if not getattr(sys, 'frozen', False):  # Not frozen
            # 在参数列表的开头插入 Python 解释器的路径
            args.insert(0, sys.executable)
            # 获取当前工作目录
            cwd = os.getcwd()
        else:
            # 获取 Python 解释器的目录
            cwd = os.path.dirname(sys.executable)
    
        # 忽略的命令行参数
        ignored_args = [
            "--open_browser", "default_browser",
            "--dist_type", "bundle_win64"
        ]
    
        # 如果是 Windows 平台
        if sys.platform == 'win32':
            # 对参数列表中的每个参数进行处理，将其用双引号括起来，同时排除忽略的参数
            args = ['"%s"' % arg for arg in args if arg and arg not in ignored_args]
        # 将参数列表连接成字符串
        cmd = " ".join(args)
    
        # 在自动运行时不打开浏览器
        cmd = cmd.replace("start.py", "zeronet.py").strip()
        cmd += ' --open_browser ""'
    
        # 返回格式化后的自动运行脚本
        return "\r\n".join([
            '@echo off',
            'chcp 65001 > nul',
            'set PYTHONIOENCODING=utf-8',
            'cd /D \"%s\"' % cwd,
            'start "" %s' % cmd
        ])
    
    # 检查自动运行是否已启用
    def isAutorunEnabled(self):
        # 获取自动运行脚本的路径
        path = self.getAutorunPath()
        # 判断自动运行脚本是否存在，并且内容与格式化后的自动运行脚本一致
        return os.path.isfile(path) and open(path, "rb").read().decode("utf8") == self.formatAutorun()
    
    # 获取自动运行的标题
    def titleAutorun(self):
        # 获取翻译后的文本
        translate = _["Start ZeroNet when Windows starts"]
        # 如果自动运行已启用，则返回带有加号的翻译文本
        if self.isAutorunEnabled():
            return "+" + translate
        else:
            # 否则返回翻译文本
            return translate
    
    # 切换自动运行状态
    def toggleAutorun(self):
        # 如果自动运行已启用，则删除自动运行脚本
        if self.isAutorunEnabled():
            os.unlink(self.getAutorunPath())
        else:
            # 否则创建自动运行脚本并写入格式化后的自动运行脚本内容
            open(self.getAutorunPath(), "wb").write(self.formatAutorun().encode("utf8"))
```