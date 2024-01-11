# `ZeroNet\src\util\SafeRe.py`

```
# 导入 re 模块，用于正则表达式操作
import re

# 自定义异常类，用于表示不安全的模式
class UnsafePatternError(Exception):
    pass

# 用于缓存编译后的正则表达式模式
cached_patterns = {}

# 判断模式是否安全的函数
def isSafePattern(pattern):
    # 如果模式长度超过255个字符，则抛出异常
    if len(pattern) > 255:
        raise UnsafePatternError("Pattern too long: %s characters in %s" % (len(pattern), pattern))

    # 查找模式中是否存在不安全的部分，如果存在则抛出异常
    unsafe_pattern_match = re.search(r"[^\.][\*\{\+]", pattern)  # Always should be "." before "*{+" characters to avoid ReDoS
    if unsafe_pattern_match:
        raise UnsafePatternError("Potentially unsafe part of the pattern: %s in %s" % (unsafe_pattern_match.group(0), pattern))

    # 查找模式中是否存在重复的字符，如果超过10次则抛出异常
    repetitions = re.findall(r"\.[\*\{\+]", pattern)
    if len(repetitions) >= 10:
        raise UnsafePatternError("More than 10 repetitions of %s in %s" % (repetitions[0], pattern))

    # 如果模式安全，则返回 True
    return True

# 匹配模式的函数
def match(pattern, *args, **kwargs):
    # 从缓存中获取已编译的模式
    cached_pattern = cached_patterns.get(pattern)
    if cached_pattern:
        # 如果存在则直接使用已编译的模式进行匹配
        return cached_pattern.match(*args, **kwargs)
    else:
        # 如果不存在则判断模式是否安全，如果安全则编译模式并缓存，然后进行匹配
        if isSafePattern(pattern):
            cached_patterns[pattern] = re.compile(pattern)
            return cached_patterns[pattern].match(*args, **kwargs)
```