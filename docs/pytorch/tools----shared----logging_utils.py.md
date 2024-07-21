# `.\pytorch\tools\shared\logging_utils.py`

```py
# 根据给定数量和单词生成合适的复数形式的字符串
def pluralize(count: int, singular_word: str, plural_word: str = "") -> str:
    if count == 1:
        return f"{count} {singular_word}"  # 如果数量为1，返回单数形式的字符串
    if not plural_word:
        plural_word = f"{singular_word}s"  # 如果未提供复数形式，默认在单词后加上's'构成复数形式
    return f"{count} {plural_word}"  # 返回合适的复数形式的字符串


# 将给定的秒数转换为易读的时间格式字符串
def duration_to_str(seconds: float) -> str:
    if seconds < 0.00001:
        return "0s"  # 如果秒数小于0.00001，返回字符串"0s"
    elif seconds < 60:
        return f"{seconds:.1f}s"  # 如果秒数小于60秒，返回带有一位小数的秒数字符串
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"  # 如果秒数小于3600秒（1小时），返回带有一位小数的分钟数字符串
    else:
        return f"{seconds / 3600:.1f}h"  # 否则，返回带有一位小数的小时数字符串
```