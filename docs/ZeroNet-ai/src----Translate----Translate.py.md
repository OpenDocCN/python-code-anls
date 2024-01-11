# `ZeroNet\src\Translate\Translate.py`

```
import os  # 导入操作系统模块
import json  # 导入 JSON 模块
import logging  # 导入日志模块
import inspect  # 导入检查模块
import re  # 导入正则表达式模块
import html  # 导入 HTML 模块
import string  # 导入字符串模块

from Config import config  # 从 Config 模块导入 config 变量

translates = []  # 创建一个空列表用于存储翻译对象


class EscapeProxy(dict):
    # 自动转义访问的字符串值
    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if type(val) in (str, str):  # 如果值的类型是字符串
            return html.escape(val)  # 对值进行 HTML 转义
        elif type(val) is dict:  # 如果值的类型是字典
            return EscapeProxy(val)  # 创建一个新的转义字典
        elif type(val) is list:  # 如果值的类型是列表
            return EscapeProxy(enumerate(val))  # 将列表转换为字典
        else:
            return val  # 返回原始值


class Translate(dict):
    def __init__(self, lang_dir=None, lang=None):
        if not lang_dir:  # 如果未提供语言目录
            lang_dir = os.path.dirname(__file__) + "/languages/"  # 使用默认语言目录
        if not lang:  # 如果未提供语言
            lang = config.language  # 使用配置文件中的语言
        self.lang = lang  # 设置语言
        self.lang_dir = lang_dir  # 设置语言目录
        self.setLanguage(lang)  # 调用设置语言方法
        self.formatter = string.Formatter()  # 创建字符串格式化对象

        if config.debug:  # 如果处于调试模式
            # 自动重新加载 FileRequest
            from Debug import DebugReloader
            DebugReloader.watcher.addCallback(self.load)  # 添加回调函数来重新加载翻译文件

        translates.append(self)  # 将当前翻译对象添加到翻译列表中

    def setLanguage(self, lang):
        self.lang = re.sub("[^a-z-]", "", lang)  # 使用正则表达式清理语言代码
        self.lang_file = self.lang_dir + "%s.json" % lang  # 构建语言文件路径
        self.load()  # 调用加载方法加载语言文件

    def __repr__(self):
        return "<translate %s>" % self.lang  # 返回翻译对象的字符串表示形式
    # 加载语言文件
    def load(self):
        # 如果语言为英文
        if self.lang == "en":
            # 初始化空数据字典
            data = {}
            dict.__init__(self, data)
            # 清空数据
            self.clear()
        # 如果语言文件存在
        elif os.path.isfile(self.lang_file):
            try:
                # 读取语言文件中的数据
                data = json.load(open(self.lang_file, encoding="utf8"))
                # 记录日志，显示加载的翻译文件和条目数
                logging.debug("Loaded translate file: %s (%s entries)" % (self.lang_file, len(data)))
            except Exception as err:
                # 记录错误日志，显示加载翻译文件出错
                logging.error("Error loading translate file %s: %s" % (self.lang_file, err))
                # 初始化空数据字典
                data = {}
            # 初始化数据字典
            dict.__init__(self, data)
        # 如果语言文件不存在
        else:
            # 初始化空数据字典
            data = {}
            dict.__init__(self, data)
            # 清空数据
            self.clear()
            # 记录日志，显示翻译文件不存在
            logging.debug("Translate file not exists: %s" % self.lang_file)

    # 格式化字符串
    def format(self, s, kwargs, nested=False):
        # 将当前对象添加到参数字典中
        kwargs["_"] = self
        # 如果是嵌套格式化
        if nested:
            # 使用格式化器格式化字符串，并再次格式化结果
            back = self.formatter.vformat(s, [], kwargs)  # PY3 TODO: Change to format_map
            return self.formatter.vformat(back, [], kwargs)
        # 如果不是嵌套格式化
        else:
            # 使用格式化器格式化字符串
            return self.formatter.vformat(s, [], kwargs)

    # 格式化本地变量
    def formatLocals(self, s, nested=False):
        # 获取调用者的本地变量作为参数字典
        kwargs = inspect.currentframe().f_back.f_locals
        return self.format(s, kwargs, nested=nested)

    # 调用函数
    def __call__(self, s, kwargs=None, nested=False, escape=True):
        # 如果没有指定参数字典，则使用调用者的本地变量
        if not kwargs:
            kwargs = inspect.currentframe().f_back.f_locals
        # 如果需要转义
        if escape:
            # 使用转义代理包装参数字典
            kwargs = EscapeProxy(kwargs)
        # 格式化字符串
        return self.format(s, kwargs, nested=nested)

    # 处理缺失的键
    def __missing__(self, key):
        return key

    # 处理复数形式
    def pluralize(self, value, single, multi):
        # 如果数值大于1，使用复数形式
        if value > 1:
            return self[multi].format(value)
        # 否则使用单数形式
        else:
            return self[single].format(value)
    # 将给定数据中的文本进行翻译，使用指定的翻译表和模式
    def translateData(self, data, translate_table=None, mode="js"):
        # 如果没有指定翻译表，则使用默认的翻译表
        if not translate_table:
            translate_table = self

        # 初始化模式列表
        patterns = []
        # 遍历翻译表中的键值对
        for key, val in list(translate_table.items()):
            # 如果键以"_("开头，表示需要特殊处理
            if key.startswith("_("):  # Problematic string: only match if called between _(" ") function
                # 对键进行处理，去除"_("和")"，并替换", "为'", "'
                key = key.replace("_(", "").replace(")", "").replace(", ", '", "')
                # 将处理后的键作为新的键，对应的值以"|"开头
                translate_table[key] = "|" + val
            # 将处理后的键加入模式列表
            patterns.append(re.escape(key))

        # 定义替换函数
        def replacer(match):
            # 获取匹配到的键对应的值
            target = translate_table[match.group(1)]
            # 如果模式为"js"
            if mode == "js":
                # 如果目标值不为空且以"|"开头，表示严格字符串匹配
                if target and target[0] == "|":  # Strict string match
                    # 只有当匹配在_(" ")函数之间调用时才进行替换
                    if match.string[match.start() - 2] == "_":  # Only if the match if called between _(" ") function
                        return '"' + target[1:] + '"'
                    else:
                        return '"' + match.group(1) + '"'
                # 否则直接返回目标值
                return '"' + target + '"'
            # 如果模式不为"js"，直接返回目标值
            else:
                return match.group(0)[0] + target + match.group(0)[-1]

        # 根据模式选择不同的匹配模式
        if mode == "html":
            pattern = '[">](' + "|".join(patterns) + ')["<]'
        else:
            pattern = '"(' + "|".join(patterns) + ')"'
        # 使用替换函数对数据进行替换
        data = re.sub(pattern, replacer, data)

        # 如果模式为"html"，替换特定字符串
        if mode == "html":
            data = data.replace("lang={lang}", "lang=%s" % self.lang)  # lang get parameter to .js file to avoid cache

        # 返回替换后的数据
        return data
# 创建一个名为 translate 的 Translate 对象实例
translate = Translate()
```