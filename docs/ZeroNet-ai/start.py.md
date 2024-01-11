# `ZeroNet\start.py`

```
#!/usr/bin/env python3
# 设置脚本的解释器为 Python3

# Included modules
# 导入系统模块
import sys

# ZeroNet Modules
# 导入 ZeroNet 模块
import zeronet

# 定义主函数
def main():
    # 如果命令行参数中没有 "--open_browser"，则添加该参数和默认浏览器参数到命令行参数中
    if "--open_browser" not in sys.argv:
        sys.argv = [sys.argv[0]] + ["--open_browser", "default_browser"] + sys.argv[1:]
    # 启动 ZeroNet
    zeronet.start()

# 如果当前脚本被直接执行，则调用主函数
if __name__ == '__main__':
    main()
```