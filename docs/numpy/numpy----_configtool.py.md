# `.\numpy\numpy\_configtool.py`

```
# 导入必要的模块
import argparse  # 导入命令行参数解析模块
from pathlib import Path  # 导入处理路径的模块
import sys  # 导入系统相关的模块

# 从当前包中导入版本号
from .version import __version__
# 从当前包的子模块中导入获取包含路径的函数
from .lib._utils_impl import get_include

# 主函数入口
def main() -> None:
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加参数：打印版本信息并退出
    parser.add_argument(
        "--version",
        action="version",  # 设置动作为打印版本信息
        version=__version__,  # 设置版本号
        help="Print the version and exit.",  # 参数的帮助信息
    )
    
    # 添加参数：获取使用 NumPy 头文件时需要的编译标志
    parser.add_argument(
        "--cflags",
        action="store_true",  # 设置动作为存储 True
        help="Compile flag needed when using the NumPy headers.",  # 参数的帮助信息
    )
    
    # 添加参数：打印存储 `numpy.pc` 的 pkgconfig 目录
    parser.add_argument(
        "--pkgconfigdir",
        action="store_true",  # 设置动作为存储 True
        help=("Print the pkgconfig directory in which `numpy.pc` is stored "  # 参数的帮助信息
              "(useful for setting $PKG_CONFIG_PATH)."),
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有传递任何参数，则打印帮助信息
    if not sys.argv[1:]:
        parser.print_help()
    
    # 如果传递了 --cflags 参数，则打印 NumPy 头文件的包含路径
    if args.cflags:
        print("-I" + get_include())
    
    # 如果传递了 --pkgconfigdir 参数，则打印存储 numpy.pc 的 pkgconfig 目录
    if args.pkgconfigdir:
        _path = Path(get_include()) / '..' / 'lib' / 'pkgconfig'  # 构建路径对象表示 pkgconfig 目录
        print(_path.resolve())  # 打印路径的绝对路径


# 如果作为主程序运行，则调用主函数
if __name__ == "__main__":
    main()
```