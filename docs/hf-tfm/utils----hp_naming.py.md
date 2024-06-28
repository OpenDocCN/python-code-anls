# `.\utils\hp_naming.py`

```
    # 复制标准库 copy 和 re
    import copy
    import re

# 试验短命名器类
class TrialShortNamer:
    # 类变量 PREFIX 初始化为 "hp"
    PREFIX = "hp"
    # 类变量 DEFAULTS 初始化为空字典
    DEFAULTS = {}
    # 类变量 NAMING_INFO 初始化为 None
    NAMING_INFO = None

    # 类方法，设置类变量 PREFIX 和 DEFAULTS，并调用 build_naming_info 方法
    @classmethod
    def set_defaults(cls, prefix, defaults):
        cls.PREFIX = prefix
        cls.DEFAULTS = defaults
        cls.build_naming_info()

    # 静态方法，为单词生成短名称
    @staticmethod
    def shortname_for_word(info, word):
        # 如果单词长度为0，返回空字符串
        if len(word) == 0:
            return ""
        # 初始化 short_word 为 None
        short_word = None
        # 如果单词中包含数字，抛出异常
        if any(char.isdigit() for char in word):
            raise Exception(f"Parameters should not contain numbers: '{word}' contains a number")
        # 如果单词已经在 info 的 "short_word" 中，直接返回其短名称
        if word in info["short_word"]:
            return info["short_word"][word]
        # 尝试生成单词的短前缀，避免与已有的短名称冲突
        for prefix_len in range(1, len(word) + 1):
            prefix = word[:prefix_len]
            if prefix in info["reverse_short_word"]:
                continue
            else:
                short_word = prefix
                break

        # 如果未能生成短前缀，则采用备用方法生成唯一的短名称
        if short_word is None:
            # 备用方法：将数字转换为字母
            def int_to_alphabetic(integer):
                s = ""
                while integer != 0:
                    s = chr(ord("A") + integer % 10) + s
                    integer //= 10
                return s

            i = 0
            while True:
                sword = word + "#" + int_to_alphabetic(i)
                if sword in info["reverse_short_word"]:
                    continue
                else:
                    short_word = sword
                    break

        # 将生成的短名称存储在 info 中，并更新反向映射
        info["short_word"][word] = short_word
        info["reverse_short_word"][short_word] = word
        return short_word

    # 静态方法，为参数名生成短名称
    @staticmethod
    def shortname_for_key(info, param_name):
        # 将参数名分割成单词列表
        words = param_name.split("_")

        # 为每个单词生成短名称部分
        shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for word in words]

        # 尝试创建无分隔符的短名称，若存在冲突则使用分隔符分隔单词
        separators = ["", "_"]

        for separator in separators:
            shortname = separator.join(shortname_parts)
            if shortname not in info["reverse_short_param"]:
                info["short_param"][param_name] = shortname
                info["reverse_short_param"][shortname] = param_name
                return shortname

        # 如果无法避免冲突，则返回原参数名
        return param_name

    @staticmethod
    def add_new_param_name(info, param_name):
        # 使用 TrialShortNamer 提供的方法生成 param_name 的短名称，并添加到 info 字典中
        short_name = TrialShortNamer.shortname_for_key(info, param_name)
        # 将 param_name 和其对应的 short_name 存储在 info 字典的 "short_param" 和 "reverse_short_param" 中
        info["short_param"][param_name] = short_name
        info["reverse_short_param"][short_name] = param_name

    @classmethod
    def build_naming_info(cls):
        # 如果 NAMING_INFO 已经存在，则直接返回，避免重复构建
        if cls.NAMING_INFO is not None:
            return

        # 初始化一个空的命名信息字典
        info = {
            "short_word": {},
            "reverse_short_word": {},
            "short_param": {},
            "reverse_short_param": {},
        }

        # 获取类的默认参数列表
        field_keys = list(cls.DEFAULTS.keys())

        # 为每个参数调用 add_new_param_name 方法，构建参数名和短名称的映射关系
        for k in field_keys:
            cls.add_new_param_name(info, k)

        # 将构建好的命名信息保存到类的 NAMING_INFO 属性中
        cls.NAMING_INFO = info

    @classmethod
    def shortname(cls, params):
        # 确保命名信息已经构建
        cls.build_naming_info()
        # 断言类的 PREFIX 属性不为空
        assert cls.PREFIX is not None
        # 创建一个名称列表，起始部分为类的 PREFIX 属性的拷贝
        name = [copy.copy(cls.PREFIX)]

        # 遍历传入的参数字典
        for k, v in params.items():
            # 如果参数 k 不在默认参数列表中，则抛出异常
            if k not in cls.DEFAULTS:
                raise Exception(f"You should provide a default value for the param name {k} with value {v}")
            # 如果参数 v 等于默认值，则不将其添加到名称中
            if v == cls.DEFAULTS[k]:
                continue

            # 根据参数名 k 获取其短名称
            key = cls.NAMING_INFO["short_param"][k]

            # 如果参数值是布尔类型，则转换为整数形式
            if isinstance(v, bool):
                v = 1 if v else 0

            # 确定连接参数名和参数值的分隔符
            sep = "" if isinstance(v, (int, float)) else "-"
            # 构建参数名和参数值的字符串表示，并添加到名称列表中
            e = f"{key}{sep}{v}"
            name.append(e)

        # 返回连接后的名称字符串，使用下划线连接各部分
        return "_".join(name)

    @classmethod
    def parse_repr(cls, repr):
        # 截取 repr 字符串，去除前缀部分，得到实际的参数表示部分
        repr = repr[len(cls.PREFIX) + 1 :]
        # 如果 repr 为空字符串，则初始化值列表为空
        if repr == "":
            values = []
        else:
            # 否则，按下划线分割 repr 字符串，得到值列表
            values = repr.split("_")

        # 初始化参数字典
        parameters = {}

        # 遍历值列表中的每个值
        for value in values:
            # 如果值中包含 "-" 符号，则按照该符号分割键和值
            if "-" in value:
                p_k, p_v = value.split("-")
            else:
                # 否则，提取键部分，去除数字和小数点，并转换为字符串
                p_k = re.sub("[0-9.]", "", value)
                # 提取值部分，去除非数字和小数点字符，并转换为浮点数
                p_v = float(re.sub("[^0-9.]", "", value))

            # 根据短参数名 p_k 获取原始参数名，并将其与值 p_v 存储到参数字典中
            key = cls.NAMING_INFO["reverse_short_param"][p_k]
            parameters[key] = p_v

        # 对于每个默认参数，如果其在参数字典中不存在，则添加默认值
        for k in cls.DEFAULTS:
            if k not in parameters:
                parameters[k] = cls.DEFAULTS[k]

        # 返回解析得到的参数字典
        return parameters
```