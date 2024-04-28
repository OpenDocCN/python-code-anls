# `.\transformers\utils\hp_naming.py`

```
# 版权声明和许可信息
#
# 版权所有 2020 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
#

# 导入必要的库
import copy
import re

# 定义一个类 TrialShortNamer
class TrialShortNamer:
    # 类变量 PREFIX 和 DEFAULTS
    PREFIX = "hp"
    DEFAULTS = {}
    NAMING_INFO = None

    # 设置默认值的类方法
    @classmethod
    def set_defaults(cls, prefix, defaults):
        cls.PREFIX = prefix
        cls.DEFAULTS = defaults
        cls.build_naming_info()

    # 根据单词生成简短名称的静态方法
    @staticmethod
    def shortname_for_word(info, word):
        if len(word) == 0:
            return ""
        short_word = None
        if any(char.isdigit() for char in word):
            raise Exception(f"Parameters should not contain numbers: '{word}' contains a number")
        if word in info["short_word"]:
            return info["short_word"][word]
        for prefix_len in range(1, len(word) + 1):
            prefix = word[:prefix_len]
            if prefix in info["reverse_short_word"]:
                continue
            else:
                short_word = prefix
                break

        if short_word is None:
            # Paranoid fallback
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

        info["short_word"][word] = short_word
        info["reverse_short_word"][short_word] = word
        return short_word

    # 根据参数名生成简短名称的静态方法
    @staticmethod
    def shortname_for_key(info, param_name):
        words = param_name.split("_")

        shortname_parts = [TrialShortNamer.shortname_for_word(info, word) for word in words]

        # 尝试创建一个无分隔符的简短名称，但如果存在冲突，则必须回退到带分隔符的简短名称
        separators = ["", "_"]

        for separator in separators:
            shortname = separator.join(shortname_parts)
            if shortname not in info["reverse_short_param"]:
                info["short_param"][param_name] = shortname
                info["reverse_short_param"][shortname] = param_name
                return shortname

        return param_name

    # 下一个静态方法
    @staticmethod
    # 添加新参数名称到信息字典中
    def add_new_param_name(info, param_name):
        # 根据参数名获取短名称
        short_name = TrialShortNamer.shortname_for_key(info, param_name)
        # 将参数名和对应的短名称添加到信息字典中
        info["short_param"][param_name] = short_name
        info["reverse_short_param"][short_name] = param_name

    # 构建命名信息
    @classmethod
    def build_naming_info(cls):
        # 如果命名信息已存在，则直接返回
        if cls.NAMING_INFO is not None:
            return

        # 初始化信息字典
        info = {
            "short_word": {},
            "reverse_short_word": {},
            "short_param": {},
            "reverse_short_param": {},
        }

        # 获取字段键列表
        field_keys = list(cls.DEFAULTS.keys())

        # 遍历字段键列表，为每个字段添加新参数名称
        for k in field_keys:
            cls.add_new_param_name(info, k)

        # 将构建好的信息字典赋值给类属性
        cls.NAMING_INFO = info

    # 获取短名称
    @classmethod
    def shortname(cls, params):
        # 构建命名信息
        cls.build_naming_info()
        # 断言前缀不为空
        assert cls.PREFIX is not None
        # 初始化名称列表
        name = [copy.copy(cls.PREFIX)]

        # 遍历参数字典
        for k, v in params.items():
            # 如果参数不在默认值中，则抛出异常
            if k not in cls.DEFAULTS:
                raise Exception(f"You should provide a default value for the param name {k} with value {v}")
            # 如果值等于默认值，则不添加到名称中
            if v == cls.DEFAULTS[k]:
                continue

            # 获取参数对应的短名称
            key = cls.NAMING_INFO["short_param"][k]

            # 处理布尔类型值
            if isinstance(v, bool):
                v = 1 if v else 0

            # 根据值类型添加分隔符
            sep = "" if isinstance(v, (int, float)) else "-"
            e = f"{key}{sep}{v}"
            name.append(e)

        # 返回拼接后的名称
        return "_".join(name)

    # 解析表示形式
    @classmethod
    def parse_repr(cls, repr):
        # 去除前缀后的表示形式
        repr = repr[len(cls.PREFIX) + 1 :]
        # 如果表示形式为空，则初始化值列表为空
        if repr == "":
            values = []
        else:
            # 否则根据下划线分割表示形式
            values = repr.split("_")

        # 初始化参数字典
        parameters = {}

        # 遍历值列表
        for value in values:
            # 如果值中包含分隔符
            if "-" in value:
                p_k, p_v = value.split("-")
            else:
                # 否则根据正则表达式提取键和值
                p_k = re.sub("[0-9.]", "", value)
                p_v = float(re.sub("[^0-9.]", "", value))

            # 根据短参数名称获取键
            key = cls.NAMING_INFO["reverse_short_param"][p_k]

            # 将键值对添加到参数字典中
            parameters[key] = p_v

        # 遍历默认值，如果参数字典中不存在该键，则添加默认值
        for k in cls.DEFAULTS:
            if k not in parameters:
                parameters[k] = cls.DEFAULTS[k]

        # 返回参数字典
        return parameters
```