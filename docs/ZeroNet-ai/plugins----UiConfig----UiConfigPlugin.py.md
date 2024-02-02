# `ZeroNet\plugins\UiConfig\UiConfigPlugin.py`

```py
# 导入所需的模块
import io
import os

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 对象
from Config import config
# 从 Translate 模块中导入 Translate 类
from Translate import Translate
# 从 util.Flag 模块中导入 flag 对象
from util.Flag import flag

# 获取当前文件所在目录的路径
plugin_dir = os.path.dirname(__file__)

# 如果 "_" 不在当前作用域中
if "_" not in locals():
    # 创建 Translate 对象，并将其赋值给 "_"
    _ = Translate(plugin_dir + "/languages/")

# 将 UiRequestPlugin 类注册到 PluginManager 的 "UiRequest" 中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 定义 actionWrapper 方法
    def actionWrapper(self, path, extra_headers=None):
        # 如果去除两端斜杠后的路径不是 "Config"
        if path.strip("/") != "Config":
            # 调用父类的 actionWrapper 方法，并返回其结果
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

        # 如果 extra_headers 为空
        if not extra_headers:
            # 创建一个空字典并赋值给 extra_headers
            extra_headers = {}

        # 获取 script_nonce
        script_nonce = self.getScriptNonce()

        # 发送头部信息
        self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)
        # 获取站点信息
        site = self.server.site_manager.get(config.homepage)
        # 返回一个迭代器，其中包含调用父类的 renderWrapper 方法的结果
        return iter([super(UiRequestPlugin, self).renderWrapper(
            site, path, "uimedia/plugins/uiconfig/config.html",
            "Config", extra_headers, show_loadingscreen=False, script_nonce=script_nonce
        )])

    # 定义 actionUiMedia 方法
    def actionUiMedia(self, path, *args, **kwargs):
        # 如果路径以 "/uimedia/plugins/uiconfig/" 开头
        if path.startswith("/uimedia/plugins/uiconfig/"):
            # 构建文件路径
            file_path = path.replace("/uimedia/plugins/uiconfig/", plugin_dir + "/media/")
            # 如果处于调试模式并且文件路径以 "all.js" 或 "all.css" 结尾
            if config.debug and (file_path.endswith("all.js") or file_path.endswith("all.css")):
                # 如果处于调试模式，将 *.css 合并为 all.css，将 *.js 合并为 all.js
                from Debug import DebugMedia
                DebugMedia.merge(file_path)

            # 如果文件路径以 "js" 结尾
            if file_path.endswith("js"):
                # 读取文件内容并进行 js 模式的翻译，然后编码为 utf8
                data = _.translateData(open(file_path).read(), mode="js").encode("utf8")
            # 如果文件路径以 "html" 结尾
            elif file_path.endswith("html"):
                # 读取文件内容并进行 html 模式的翻译，然后编码为 utf8
                data = _.translateData(open(file_path).read(), mode="html").encode("utf8")
            else:
                # 读取文件内容的二进制数据
                data = open(file_path, "rb").read()

            # 调用 actionFile 方法，返回文件路径、文件对象和文件大小
            return self.actionFile(file_path, file_obj=io.BytesIO(data), file_size=len(data))
        else:
            # 调用父类的 actionUiMedia 方法，并返回其结果
            return super(UiRequestPlugin, self).actionUiMedia(path)

# 将 UiWebsocket 类注册到 PluginManager 的 "UiWebsocket" 中
@PluginManager.registerTo("UiWebsocket")
# 定义一个名为 UiWebsocketPlugin 的类
class UiWebsocketPlugin(object):
    # 使用装饰器 flag.admin，将 actionConfigList 方法标记为管理员权限
    @flag.admin
    # 定义 actionConfigList 方法，接受参数 to
    def actionConfigList(self, to):
        # 初始化一个空字典 back
        back = {}
        # 获取配置参数的所有值，包括已生效和待生效的
        config_values = vars(config.arguments)
        config_values.update(config.pending_changes)
        # 遍历配置参数的键值对
        for key, val in config_values.items():
            # 如果键不在允许修改的键列表中，则跳过
            if key not in config.keys_api_change_allowed:
                continue
            # 判断当前参数是否为待生效状态
            is_pending = key in config.pending_changes
            # 如果参数值为 None 且为待生效状态，则使用默认值
            if val is None and is_pending:
                val = config.parser.get_default(key)
            # 将参数名作为键，值、默认值和是否为待生效状态作为值，添加到字典 back 中
            back[key] = {
                "value": val,
                "default": config.parser.get_default(key),
                "pending": is_pending
            }
        # 返回整理好的参数字典
        return back
```