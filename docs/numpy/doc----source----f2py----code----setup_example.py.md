# `.\numpy\doc\source\f2py\code\setup_example.py`

```
# 从 numpy.distutils.core 模块导入 Extension 类
from numpy.distutils.core import Extension

# 创建名为 ext1 的 Extension 对象，指定名称为 'scalar'，源文件为 'scalar.f'
ext1 = Extension(name = 'scalar',
                 sources = ['scalar.f'])

# 创建名为 ext2 的 Extension 对象，指定名称为 'fib2'，源文件为 'fib2.pyf' 和 'fib1.f'
ext2 = Extension(name = 'fib2',
                 sources = ['fib2.pyf', 'fib1.f'])

# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 从 numpy.distutils.core 模块导入 setup 函数
    from numpy.distutils.core import setup
    
    # 调用 setup 函数，设置项目名称为 'f2py_example'，描述为 "F2PY Users Guide examples"
    # 作者为 "Pearu Peterson"，作者邮箱为 "pearu@cens.ioc.ee"，扩展模块为 [ext1, ext2]
    setup(name = 'f2py_example',
          description       = "F2PY Users Guide examples",
          author            = "Pearu Peterson",
          author_email      = "pearu@cens.ioc.ee",
          ext_modules = [ext1, ext2]
          )
# End of setup_example.py
```