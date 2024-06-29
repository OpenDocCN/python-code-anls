# `.\numpy\numpy\f2py\diagnose.py`

```py
#!/usr/bin/env python3
# 导入必要的模块
import os  # 导入操作系统接口模块
import sys  # 导入系统特定的参数和函数模块
import tempfile  # 导入临时文件和目录创建模块


# 定义一个函数，用于执行系统命令并打印输出
def run_command(cmd):
    # 打印即将执行的命令
    print('Running %r:' % (cmd))
    # 执行系统命令
    os.system(cmd)
    # 打印分隔线
    print('------')


# 定义主程序函数
def run():
    _path = os.getcwd()  # 获取当前工作目录并保存
    os.chdir(tempfile.gettempdir())  # 切换工作目录到系统临时目录
    # 打印分隔线
    print('------')
    # 打印操作系统名称
    print('os.name=%r' % (os.name))
    # 打印分隔线
    print('------')
    # 打印系统平台信息
    print('sys.platform=%r' % (sys.platform))
    # 打印分隔线
    print('------')
    # 打印Python解释器版本信息
    print('sys.version:')
    print(sys.version)
    # 打印分隔线
    print('------')
    # 打印Python解释器安装路径前缀
    print('sys.prefix:')
    print(sys.prefix)
    # 打印分隔线
    print('------')
    # 打印Python模块搜索路径
    print('sys.path=%r' % (':'.join(sys.path)))
    # 打印分隔线
    print('------')

    # 尝试导入新版numpy模块
    try:
        import numpy
        has_newnumpy = 1
    except ImportError as e:
        # 若导入失败，打印错误信息
        print('Failed to import new numpy:', e)
        has_newnumpy = 0

    # 尝试导入f2py2e模块
    try:
        from numpy.f2py import f2py2e
        has_f2py2e = 1
    except ImportError as e:
        # 若导入失败，打印错误信息
        print('Failed to import f2py2e:', e)
        has_f2py2e = 0

    # 尝试导入numpy的distutils模块
    try:
        import numpy.distutils
        has_numpy_distutils = 2
    except ImportError:
        try:
            import numpy_distutils
            has_numpy_distutils = 1
        except ImportError as e:
            # 若导入失败，打印错误信息
            print('Failed to import numpy_distutils:', e)
            has_numpy_distutils = 0

    # 若成功导入新版numpy模块，打印其版本信息和文件路径
    if has_newnumpy:
        try:
            print('Found new numpy version %r in %s' %
                  (numpy.__version__, numpy.__file__))
        except Exception as msg:
            # 若获取信息失败，打印错误信息
            print('error:', msg)
            print('------')

    # 若成功导入f2py2e模块，打印其版本信息和文件路径
    if has_f2py2e:
        try:
            print('Found f2py2e version %r in %s' %
                  (f2py2e.__version__.version, f2py2e.__file__))
        except Exception as msg:
            # 若获取信息失败，打印错误信息
            print('error:', msg)
            print('------')

    os.chdir(_path)  # 恢复原始工作目录
# 如果脚本作为主程序运行，则执行run函数
if __name__ == "__main__":
    run()
```