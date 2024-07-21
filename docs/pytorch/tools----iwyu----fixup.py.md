# `.\pytorch\tools\iwyu\fixup.py`

```
import re  # 导入正则表达式模块
import sys  # 导入系统相关模块


QUOTE_INCLUDE_RE = re.compile(r'^#include "(.*)"')  # 匹配双引号包裹的 #include 语句
ANGLE_INCLUDE_RE = re.compile(r"^#include <(.*)>")  # 匹配尖括号包裹的 #include 语句

# 标准 C 头文件到 C++ 头文件的映射表
STD_C_HEADER_MAP = {
    "<assert.h>": "<cassert>",
    "<complex.h>": "<ccomplex>",
    "<ctype.h>": "<cctype>",
    "<errno.h>": "<cerrno>",
    "<fenv.h>": "<cfenv>",
    "<float.h>": "<cfloat>",
    "<inttypes.h>": "<cinttypes>",
    "<iso646.h>": "<ciso646>",
    "<limits.h>": "<climits>",
    "<locale.h>": "<clocale>",
    "<math.h>": "<cmath>",
    "<setjmp.h>": "<csetjmp>",
    "<signal.h>": "<csignal>",
    "<stdalign.h>": "<cstdalign>",
    "<stdarg.h>": "<cstdarg>",
    "<stdbool.h>": "<cstdbool>",
    "<stddef.h>": "<cstddef>",
    "<stdint.h>": "<cstdint>",
    "<stdio.h>": "<cstdio>",
    "<stdlib.h>": "<cstdlib>",
    "<string.h>": "<cstring>",
    "<tgmath.h>": "<ctgmath>",
    "<time.h>": "<ctime>",
    "<uchar.h>": "<cuchar>",
    "<wchar.h>": "<cwchar>",
    "<wctype.h>": "<cwctype>",
}


def main() -> None:
    for line in sys.stdin:
        # 将双引号包裹的 #include 转换为尖括号包裹的形式
        match = QUOTE_INCLUDE_RE.match(line)
        if match is not None:
            # 输出转换后的 #include 语句
            print(f"#include <{match.group(1)}>{line[match.end(0):]}", end="")
            continue

        match = ANGLE_INCLUDE_RE.match(line)
        if match is not None:
            path = f"<{match.group(1)}>"
            new_path = STD_C_HEADER_MAP.get(path, path)  # 获取对应的 C++ 头文件路径，如果没有则保持不变
            tail = line[match.end(0):]
            if len(tail) > 1:
                tail = " " + tail
            # 输出转换后的 #include 语句
            print(f"#include {new_path}{tail}", end="")
            continue

        # 输出原始的 #include 行
        print(line, end="")


if __name__ == "__main__":
    main()
```