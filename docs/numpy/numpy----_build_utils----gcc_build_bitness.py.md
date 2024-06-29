# `.\numpy\numpy\_build_utils\gcc_build_bitness.py`

```
# 导入正则表达式和子进程模块
import re
from subprocess import run, PIPE

# 主函数入口
def main():
    # 运行 gcc 命令并获取输出结果
    res = run(['gcc', '-v'], check=True, text=True, capture_output=True)
    # 从 gcc 输出的错误信息中匹配目标平台信息
    target = re.search(r'^Target: (.*)$', res.stderr, flags=re.M).groups()[0]
    # 检查目标平台信息以确定 Mingw-w64 的位数
    if target.startswith('i686'):
        # 如果是 32 位
        print('32')
    elif target.startswith('x86_64'):
        # 如果是 64 位
        print('64')
    else:
        # 如果无法检测到 Mingw-w64 的位数，则抛出运行时异常
        raise RuntimeError('Could not detect Mingw-w64 bitness')

# 如果当前脚本作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```