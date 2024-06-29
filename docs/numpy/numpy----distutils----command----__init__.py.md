# `.\numpy\numpy\distutils\command\__init__.py`

```
# 导入distutils.command模块
"""distutils.command

Package containing implementation of all the standard Distutils
commands.

"""
# 删除测试对象属性不可写
def test_na_writable_attributes_deletion():
    # 创建NA对象
    a = np.NA(2)
    # 待删除的属性列表
    attr =  ['payload', 'dtype']
    # 遍历属性列表
    for s in attr:
        # 断言删除属性时出现属性错误
        assert_raises(AttributeError, delattr, a, s)

# 版本信息
__revision__ = "$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $"

# 所有distutils命令列表
distutils_all = [  #'build_py',
                   'clean',
                   'install_clib',
                   'install_scripts',
                   'bdist',
                   'bdist_dumb',
                   'bdist_wininst',
                ]

# 动态导入distutils.command模块中的指定命令
__import__('distutils.command', globals(), locals(), distutils_all)

# 导出的模块列表
__all__ = ['build',
           'config_compiler',
           'config',
           'build_src',
           'build_py',
           'build_ext',
           'build_clib',
           'build_scripts',
           'install',
           'install_data',
           'install_headers',
           'install_lib',
           'bdist_rpm',
           'sdist',
          ] + distutils_all
```