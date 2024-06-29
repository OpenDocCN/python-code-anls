# `.\numpy\numpy\f2py\f90mod_rules.py`

```
"""
Build F90 module support for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 定义版本信息字符串，从 "$Revision: 1.27 $" 提取版本号
__version__ = "$Revision: 1.27 $"[10:-1]

# 定义 f2py_version 字符串
f2py_version = 'See `f2py -v`'

# 导入 NumPy 库，并重命名为 np
import numpy as np

# 从当前包中导入 capi_maps、func2subr、undo_rmbadname、undo_rmbadname1 模块
from . import capi_maps
from . import func2subr
from .crackfortran import undo_rmbadname, undo_rmbadname1

# 从 auxfuncs 模块中导入所有函数和对象
from .auxfuncs import *

# 定义空字典 options
options = {}

# 定义函数 findf90modules，查找模块中的 Fortran 90 模块
def findf90modules(m):
    # 如果 m 是模块，则返回列表包含 m 自身
    if ismodule(m):
        return [m]
    # 如果 m 没有主体或者主体为空，则返回空列表
    if not hasbody(m):
        return []
    ret = []
    # 遍历 m 的主体
    for b in m['body']:
        # 如果 b 是模块，则添加到 ret 列表中
        if ismodule(b):
            ret.append(b)
        else:
            # 否则递归查找 b 中的 Fortran 90 模块并加入 ret 列表
            ret = ret + findf90modules(b)
    return ret

# 定义 fgetdims1 字符串，包含 Fortran 代码片段
fgetdims1 = """\
      external f2pysetdata
      logical ns
      integer r,i
      integer(%d) s(*)
      ns = .FALSE.
      if (allocated(d)) then
         do i=1,r
            if ((size(d,i).ne.s(i)).and.(s(i).ge.0)) then
               ns = .TRUE.
            end if
         end do
         if (ns) then
            deallocate(d)
         end if
      end if
      if ((.not.allocated(d)).and.(s(1).ge.1)) then""" % np.intp().itemsize

# 定义 fgetdims2 字符串，包含 Fortran 代码片段
fgetdims2 = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
      end if
      flag = 1
      call f2pysetdata(d,allocated(d))"""

# 定义 fgetdims2_sa 字符串，包含 Fortran 代码片段
fgetdims2_sa = """\
      end if
      if (allocated(d)) then
         do i=1,r
            s(i) = size(d,i)
         end do
         !s(r) must be equal to len(d(1))
      end if
      flag = 2
      call f2pysetdata(d,allocated(d))"""

# 定义函数 buildhooks，用于构建钩子
def buildhooks(pymod):
    # 从当前包中导入 rules 模块
    from . import rules
    # 定义返回的字典 ret，包含各种钩子和文档
    ret = {'f90modhooks': [], 'initf90modhooks': [], 'body': [],
           'need': ['F_FUNC', 'arrayobject.h'],
           'separatorsfor': {'includes0': '\n', 'includes': '\n'},
           'docs': ['"Fortran 90/95 modules:\\n"'],
           'latexdoc': []}
    # 定义空字符串 fhooks
    fhooks = ['']

    # 定义函数 fadd，用于向 fhooks 添加行
    def fadd(line, s=fhooks):
        s[0] = '%s\n      %s' % (s[0], line)
    
    # 定义空列表 doc
    doc = ['']

    # 定义函数 dadd，用于向 doc 添加行
    def dadd(line, s=doc):
        s[0] = '%s\n%s' % (s[0], line)

    # 使用 getuseblocks 函数获取 pymod 的 use 块信息
    usenames = getuseblocks(pymod)
    
    # 设置 ret 的一些属性为空字符串或空列表
    ret['routine_defs'] = ''
    ret['doc'] = []
    ret['docshort'] = []
    ret['latexdoc'] = doc[0]
    
    # 如果 docs 的长度小于等于 1，则将其设置为空字符串
    if len(ret['docs']) <= 1:
        ret['docs'] = ''
    
    # 返回 ret 和 fhooks 的第一个元素
    return ret, fhooks[0]
```