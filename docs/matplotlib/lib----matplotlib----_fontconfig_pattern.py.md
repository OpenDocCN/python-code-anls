# `D:\src\scipysrc\matplotlib\lib\matplotlib\_fontconfig_pattern.py`

```py
# 定义一个用于解析 fontconfig 模式的函数，接受一个 pattern 参数
@lru_cache
def parse_fontconfig_pattern(pattern):
    """
    使用 lru_cache 装饰器缓存函数结果，避免重复解析相同的 pattern。
    在测试期间，这个函数是一个性能瓶颈，因为在重置 rcParams 时会被多次调用，
    用于验证默认字体。
    缓存大小在测试套件期间不会超过几十个条目。
    """
    Parse a fontconfig *pattern* into a dict that can initialize a
    `.font_manager.FontProperties` object.
    """
    # 创建一个用于解析 fontconfig 样式模式的解析器对象
    parser = _make_fontconfig_parser()
    # 尝试解析给定的 pattern
    try:
        parse = parser.parseString(pattern)
    # 捕获解析过程中可能发生的 ParseException 异常
    except ParseException as err:
        # 在 pyparsing 3 中，explain 成为一个普通方法 (err.explain(0))。
        # 抛出解析错误的详细信息，用于调试
        raise ValueError("\n" + ParseException.explain(err, 0)) from None
    # 重置解析器的缓存状态
    parser.resetCache()
    # 初始化属性字典
    props = {}
    # 如果解析结果中包含 "families" 键
    if "families" in parse:
        # 将解析结果中的 "families" 列表中的每个元素进行反转义处理，作为 "family" 属性的值
        props["family"] = [*map(_family_unescape, parse["families"])]
    # 如果解析结果中包含 "sizes" 键
    if "sizes" in parse:
        # 将解析结果中的 "sizes" 列表直接作为 "size" 属性的值
        props["size"] = [*parse["sizes"]]
    # 遍历解析结果中的 "properties" 列表
    for prop in parse.get("properties", []):
        # 如果属性长度为 1，则根据 _CONSTANTS 字典进行映射处理
        if len(prop) == 1:
            prop = _CONSTANTS[prop[0]]
        # 否则，将属性名及其对应的值进行反转义处理，添加到 props 字典中
        k, *v = prop
        props.setdefault(k, []).extend(map(_value_unescape, v))
    # 返回初始化完成的属性字典
    return props
def generate_fontconfig_pattern(d):
    """Convert a `.FontProperties` to a fontconfig pattern string."""
    # 创建一个包含键值对的列表，每个键是字体属性名，值是对应属性的值
    kvs = [(k, getattr(d, f"get_{k}")())
           for k in ["style", "variant", "weight", "stretch", "file", "size"]]
    
    # 将字体族名称作为第一个条目，不加前导关键字。其他条目（必然是标量）作为key=value形式，跳过值为None的条目。
    # 拼接字体族名称，并对其他属性进行格式化，形成字体配置模式字符串
    return (",".join(_family_escape(f) for f in d.get_family())
            + "".join(f":{k}={_value_escape(str(v))}"
                      for k, v in kvs if v is not None))
```