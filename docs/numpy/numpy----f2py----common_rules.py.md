# `.\numpy\numpy\f2py\common_rules.py`

```
"""
Build common block mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 导入版本信息
from . import __version__
# 设置 f2py_version 变量为当前模块的版本号
f2py_version = __version__.version

# 导入各种辅助功能函数
from .auxfuncs import (
    hasbody, hascommon, hasnote, isintent_hide, outmess, getuseblocks
)
# 导入模块 capi_maps 和 func2subr
from . import capi_maps
from . import func2subr
# 导入 crackfortran 模块中的 rmbadname 函数
from .crackfortran import rmbadname


# 定义函数 findcommonblocks，用于查找共用块
def findcommonblocks(block, top=1):
    # 初始化返回结果列表
    ret = []
    # 如果当前块有共用块
    if hascommon(block):
        # 遍历共用块中的每对键值对
        for key, value in block['common'].items():
            # 从变量字典中获取相关变量，并添加到 ret 列表中
            vars_ = {v: block['vars'][v] for v in value}
            ret.append((key, value, vars_))
    # 如果当前块有子块
    elif hasbody(block):
        # 遍历每个子块并递归调用 findcommonblocks，将结果追加到 ret 列表中
        for b in block['body']:
            ret = ret + findcommonblocks(b, 0)
    # 如果是顶层调用，则处理 ret 列表，去重并返回结果
    if top:
        tret = []
        names = []
        for t in ret:
            if t[0] not in names:
                names.append(t[0])
                tret.append(t)
        return tret
    return ret


# 定义函数 buildhooks，用于构建钩子函数
def buildhooks(m):
    # 初始化返回结果字典和各种钩子列表
    ret = {'commonhooks': [], 'initcommonhooks': [],
           'docs': ['"COMMON blocks:\\n"']}
    fwrap = ['']

    # 定义函数 fadd，用于向 fwrap 列表中添加内容
    def fadd(line, s=fwrap):
        s[0] = '%s\n      %s' % (s[0], line)
    
    chooks = ['']

    # 定义函数 cadd，用于向 chooks 列表中添加内容
    def cadd(line, s=chooks):
        s[0] = '%s\n%s' % (s[0], line)
    
    ihooks = ['']

    # 定义函数 iadd，用于向 ihooks 列表中添加内容
    def iadd(line, s=ihooks):
        s[0] = '%s\n%s' % (s[0], line)
    
    doc = ['']

    # 定义函数 dadd，用于向 doc 列表中添加内容
    def dadd(line, s=doc):
        s[0] = '%s\n%s' % (s[0], line)
    
    # 设置返回字典中的各种钩子列表和文档内容
    ret['commonhooks'] = chooks
    ret['initcommonhooks'] = ihooks
    ret['latexdoc'] = doc[0]
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    return ret, fwrap[0]
```