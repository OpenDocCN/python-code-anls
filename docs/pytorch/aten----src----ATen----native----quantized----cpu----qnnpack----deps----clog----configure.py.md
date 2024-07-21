# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\deps\clog\configure.py`

```py
# 导入 confu 模块，用于配置解析
import confu

# 使用 confu 提供的标准解析器创建一个解析器对象，用于处理 clog 的配置脚本
parser = confu.standard_parser("clog configuration script")

# 主函数定义，接受命令行参数作为输入
def main(args):
    # 解析命令行参数并获取选项对象
    options = parser.parse_args(args)
    
    # 根据选项对象创建一个构建对象
    build = confu.Build.from_options(options)
    
    # 导出头文件 "clog.h" 到 "include" 目录下
    build.export_cpath("include", ["clog.h"])
    
    # 使用 build 对象的选项来设置源目录为 "src"，额外包含目录也设置为 "src"，
    # 然后编译生成静态库 "clog"，使用 "clog.c" 文件作为源文件
    with build.options(source_dir="src", extra_include_dirs="src"):
        build.static_library("clog", build.cc("clog.c"))
    
    # 使用 build 对象的选项来设置源目录为 "test"，
    # 定义依赖关系为使用 googletest 框架进行单元测试，仅当目标平台为 Android 时才使用 "log" 库
    with build.options(
        source_dir="test",
        deps={(build, build.deps.googletest): all, "log": build.target.is_android},
    ):
        build.unittest("clog-test", build.cxx("clog.cc"))
    
    # 返回构建对象，用于进一步生成构建文件
    return build

# 如果脚本被直接执行，则调用 main 函数，并生成构建文件
if __name__ == "__main__":
    import sys

    main(sys.argv[1:]).generate()
```