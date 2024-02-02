# `ZeroNet\plugins\TranslateSite\TranslateSitePlugin.py`

```py
# 导入时间模块
import time

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Translate 模块中导入 translate 函数
from Translate import translate

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 处理站点媒体请求的方法
    def actionSiteMedia(self, path, **kwargs):
        # 获取文件名
        file_name = path.split("/")[-1].lower()
        # 如果文件名为空，即路径以 / 结尾，则将文件名设置为 index.html
        if not file_name:  # Path ends with /
            file_name = "index.html"
        # 获取文件扩展名
        extension = file_name.split(".")[-1]

        # 如果扩展名为 html，则始终替换 html 文件中的翻译变量
        if extension == "html":
            should_translate = True
        # 如果扩展名为 js 并且当前语言不是英语，则需要进行翻译
        elif extension == "js" and translate.lang != "en":
            should_translate = True
        else:
            should_translate = False

        # 如果需要进行翻译
        if should_translate:
            # 解析路径
            path_parts = self.parsePath(path)
            # 设置 header_length 参数为 False
            kwargs["header_length"] = False
            # 调用父类的 actionSiteMedia 方法获取文件生成器
            file_generator = super(UiRequestPlugin, self).actionSiteMedia(path, **kwargs)
            # 如果文件生成器中包含 __next__ 方法，表示找到文件并返回了生成器
            if "__next__" in dir(file_generator):  # File found and generator returned
                # 获取站点对象
                site = self.server.sites.get(path_parts["address"])
                # 如果站点不存在或者站点内容管理器中不包含 content.json，则直接返回文件生成器
                if not site or not site.content_manager.contents.get("content.json"):
                    return file_generator
                # 对文件进行修补并返回修补后的文件生成器
                return self.actionPatchFile(site, path_parts["inner_path"], file_generator)
            else:
                return file_generator

        # 如果不需要进行翻译，则直接调用父类的 actionSiteMedia 方法
        else:
            return super(UiRequestPlugin, self).actionSiteMedia(path, **kwargs)

    # 处理 UI 媒体请求的方法
    def actionUiMedia(self, path):
        # 调用父类的 actionUiMedia 方法获取文件生成器
        file_generator = super(UiRequestPlugin, self).actionUiMedia(path)
        # 如果当前语言不是英语并且路径以 .js 结尾
        if translate.lang != "en" and path.endswith(".js"):
            # 记录当前时间
            s = time.time()
            # 将文件生成器中的数据拼接成字节流
            data = b"".join(list(file_generator))
            # 对数据进行翻译并编码为 utf8 格式
            data = translate.translateData(data.decode("utf8"))
            # 记录修补后的文件信息
            self.log.debug("Patched %s (%s bytes) in %.3fs" % (path, len(data), time.time() - s))
            # 返回修补后的数据生成器
            return iter([data.encode("utf8")])
        else:
            return file_generator
    # 定义一个方法，用于对文件进行补丁操作
    def actionPatchFile(self, site, inner_path, file_generator):
        # 获取站点内容管理器中的content.json文件内容
        content_json = site.content_manager.contents.get("content.json")
        # 根据当前语言获取对应的语言文件路径
        lang_file = "languages/%s.json" % translate.lang
        # 初始化语言文件存在标识为False
        lang_file_exist = False
        # 如果是自己的站点，检查语言文件是否存在（允许在不登录的情况下添加新的语言）
        if site.settings.get("own"):
            if site.storage.isFile(lang_file):
                lang_file_exist = True
        # 如果不是自己的站点，content.json中的引用足够（稍后会等待下载）
        else:
            if lang_file in content_json.get("files", {}):
                lang_file_exist = True

        # 如果语言文件不存在或者inner_path不在content.json的translate列表中
        if not lang_file_exist or inner_path not in content_json.get("translate", []):
            # 遍历文件生成器
            for part in file_generator:
                # 如果inner_path以.html结尾
                if inner_path.endswith(".html"):
                    # 替换部分内容，将lang参数替换为translate.lang的编码形式
                    yield part.replace(b"lang={lang}", b"lang=" + translate.lang.encode("utf8"))  # lang get parameter to .js file to avoid cache
                else:
                    yield part
        else:
            # 记录开始时间
            s = time.time()
            # 将文件生成器中的内容拼接成字节流，然后解码成utf8格式
            data = b"".join(list(file_generator)).decode("utf8")

            # 如果content.json中的files中存在lang_file
            site.needFile(lang_file, priority=10)
            try:
                # 如果inner_path以"js"结尾，使用translate.translateData方法对data进行翻译
                if inner_path.endswith("js"):
                    data = translate.translateData(data, site.storage.loadJson(lang_file), "js")
                else:
                    # 否则，使用translate.translateData方法对data进行翻译
                    data = translate.translateData(data, site.storage.loadJson(lang_file), "html")
            except Exception as err:
                # 捕获异常并记录错误日志
                site.log.error("Error loading translation file %s: %s" % (lang_file, err))

            # 记录补丁操作的信息
            self.log.debug("Patched %s (%s bytes) in %.3fs" % (inner_path, len(data), time.time() - s))
            # 返回经过编码的data
            yield data.encode("utf8")
```