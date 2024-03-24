# `.\lucidrains\naturalspeech2-pytorch\naturalspeech2_pytorch\utils\expand\time_norm.py`

```
import re
import inflect
from num2words import num2words
from num_to_words import num_to_word

# 定义一个时间扩展器类
class TimeExpander:
    def __init__(self):
        # 初始化 inflect 引擎
        self._inflect = inflect.engine()
        # 获取时间正则表达式
        self._time_re = self._get_time_regex()

    # 获取时间正则表达式的私有方法
    def _get_time_regex(self):
        return re.compile(
            r"""\b
            ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
            :
            ([0-5][0-9])                            # minutes
            \s*(a\\.m\\.|am|pm|p\\.m\\.|a\\.m|p\\.m)? # am/pm
            \b""",
            re.IGNORECASE | re.X,
        )

    # 将数字扩展为单词的私有方法
    def _expand_num(self, n: int, language: str) -> str:
        try:
            if language == 'en':
                return self._inflect.number_to_words(n)
            else:
                return num2words(n, lang=language)
        except:
            try:
                return num_to_word(n, lang=language)
            except:
                raise NotImplementedError("language not implemented")

    # 扩展时间的私有方法
    def _expand_time(self, match: "re.Match", language: str) -> str:
        hour = int(match.group(1))
        past_noon = hour >= 12
        time = []
        if hour > 12:
            hour -= 12
        elif hour == 0:
            hour = 12
            past_noon = True
        time.append(self._expand_num(hour, language))

        minute = int(match.group(6))
        if minute > 0:
            if minute < 10:
                time.append("oh")
            time.append(self._expand_num(minute, language))

        am_pm = match.group(7)
        if am_pm is not None:
            time.extend(list(am_pm.replace(".", "")))

        return " ".join(time)

    # 扩展时间的公共方法
    def expand_time(self, text: str, language: str) -> str:
        return re.sub(self._time_re, lambda match: self._expand_time(match, language), text)
```