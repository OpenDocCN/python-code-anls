# `ZeroNet\plugins\UiFileManager\UiFileManagerPlugin.py`

```py
# 导入所需的模块
import io
import os
import re
import urllib

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 对象
from Config import config
# 从 Translate 模块中导入 Translate 类
from Translate import Translate

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果当前作用域中不存在下划线变量，则创建一个 Translate 对象并赋值给下划线变量
if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")

# 将 UiFileManagerPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiFileManagerPlugin(object):
    # 定义 actionWrapper 方法，接受路径和额外的头部信息作为参数
    def actionWrapper(self, path, extra_headers=None):
        # 使用正则表达式匹配路径中的内容
        match = re.match("/list/(.*?)(/.*|)$", path)
        # 如果没有匹配到内容，则调用父类的 actionWrapper 方法并返回结果
        if not match:
            return super().actionWrapper(path, extra_headers)

        # 如果额外的头部信息不存在，则创建一个空字典
        if not extra_headers:
            extra_headers = {}

        # 获取请求地址和内部路径
        request_address, inner_path = match.groups()

        # 获取脚本的 nonce 值
        script_nonce = self.getScriptNonce()

        # 发送头部信息
        self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)

        # 获取请求的站点对象
        site = self.server.site_manager.need(request_address)

        # 如果站点对象不存在，则调用父类的 actionWrapper 方法并返回结果
        if not site:
            return super().actionWrapper(path, extra_headers)

        # 将请求参数编码为 URL 格式
        request_params = urllib.parse.urlencode(
            {"address": site.address, "site": request_address, "inner_path": inner_path.strip("/")}
        )

        # 检查内容是否已加载
        is_content_loaded = "content.json" in site.content_manager.contents

        # 返回一个迭代器，其中包含渲染后的结果
        return iter([super().renderWrapper(
            site, path, "uimedia/plugins/uifilemanager/list.html?%s" % request_params,
            "List", extra_headers, show_loadingscreen=not is_content_loaded, script_nonce=script_nonce
        )])
    # 定义处理 UI 媒体文件的方法，接受路径和其他参数
    def actionUiMedia(self, path, *args, **kwargs):
        # 如果路径以指定字符串开头
        if path.startswith("/uimedia/plugins/uifilemanager/"):
            # 替换路径中的字符串，得到文件路径
            file_path = path.replace("/uimedia/plugins/uifilemanager/", plugin_dir + "/media/")
            # 如果处于调试模式并且文件路径以特定后缀结尾
            if config.debug and (file_path.endswith("all.js") or file_path.endswith("all.css")):
                # 如果处于调试模式，将 *.css 合并到 all.css，将 *.js 合并到 all.js
                from Debug import DebugMedia
                DebugMedia.merge(file_path)

            # 如果文件路径以 .js 结尾
            if file_path.endswith("js"):
                # 读取文件内容并进行 JS 模式的数据转换，编码为 utf8
                data = _.translateData(open(file_path).read(), mode="js").encode("utf8")
            # 如果文件路径以 .html 结尾
            elif file_path.endswith("html"):
                # 如果存在特定参数
                if self.get.get("address"):
                    # 获取站点管理器中指定地址的站点
                    site = self.server.site_manager.need(self.get.get("address"))
                    # 如果站点内容管理器中不包含 content.json
                    if "content.json" not in site.content_manager.contents:
                        # 需要 content.json 文件
                        site.needFile("content.json")
                # 读取文件内容并进行 HTML 模式的数据转换，编码为 utf8
                data = _.translateData(open(file_path).read(), mode="html").encode("utf8")
            else:
                # 以二进制形式读取文件内容
                data = open(file_path, "rb").read()

            # 调用 actionFile 方法，传入文件路径、文件对象和文件大小
            return self.actionFile(file_path, file_obj=io.BytesIO(data), file_size=len(data))
        else:
            # 调用父类的 actionUiMedia 方法
            return super().actionUiMedia(path)
    # 定义处理 404 错误的方法，接受一个路径参数
    def error404(self, path=""):
        # 如果路径不以 "index.html" 结尾且不以 "/" 结尾，则调用父类的 error404 方法处理
        if not path.endswith("index.html") and not path.endswith("/"):
            return super().error404(path)

        # 解析路径，如果解析失败，则调用父类的 error404 方法处理
        path_parts = self.parsePath(path)
        if not path_parts:
            return super().error404(path)

        # 获取请求地址对应的站点
        site = self.server.site_manager.get(path_parts["request_address"])

        # 如果站点不存在或者站点内容管理器中没有 "content.json" 文件，则调用父类的 error404 方法处理
        if not site or not site.content_manager.contents.get("content.json"):
            return super().error404(path)

        # 如果请求的内部路径在站点内容管理器的 "content.json" 文件中的文件列表中，则调用父类的 error404 方法处理
        if path_parts["inner_path"] in site.content_manager.contents.get("content.json").get("files", {}):
            return super().error404(path)

        # 发送 200 状态码的响应头
        self.sendHeader(200)
        # 重定向路径为 "/list" + 将 "/media/" 替换为 "/" 的路径
        path_redirect = "/list" + re.sub("^/media/", "/", path)
        # 记录调试信息，指出未找到 index.html 文件，重定向到指定路径
        self.log.debug("Index.html not found: %s, redirecting to: %s" % (path, path_redirect))
        # 格式化重定向响应
        return self.formatRedirect(path_redirect)
```