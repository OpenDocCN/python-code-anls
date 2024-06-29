# `.\numpy\numpy\f2py\use_rules.py`

```
"""
Build 'use others module data' mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
__version__ = "$Revision: 1.3 $"[10:-1]

# 设置 f2py2e 的版本号信息
f2py_version = 'See `f2py -v`'

# 导入 auxfuncs 模块中的若干函数
from .auxfuncs import (
    applyrules, dictappend, gentitle, hasnote, outmess
)

# 定义 usemodule_rules 字典，包含针对不同部分的模板和需求
usemodule_rules = {
    'body': """
#begintitle#
static char doc_#apiname#[] = \"\\\nVariable wrapper signature:\\n\\
\t #name# = get_#name#()\\n\\
Arguments:\\n\\
#docstr#\";
extern F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#);
static PyObject *#apiname#(PyObject *capi_self, PyObject *capi_args) {
/*#decl#*/
\tif (!PyArg_ParseTuple(capi_args, \"\")) goto capi_fail;
printf(\"c: %d\\n\",F_MODFUNC(#usemodulename#,#USEMODULENAME#,#realname#,#REALNAME#));
\treturn Py_BuildValue(\"\");
capi_fail:
\treturn NULL;
}
""",
    'method': '\t{\"get_#name#\",#apiname#,METH_VARARGS|METH_KEYWORDS,doc_#apiname#},',
    'need': ['F_MODFUNC']
}

################

# 定义 buildusevars 函数，用于构建模块的使用变量钩子
def buildusevars(m, r):
    ret = {}  # 初始化返回的字典
    outmess(
        '\t\tBuilding use variable hooks for module "%s" (feature only for F90/F95)...\n' % (m['name']))
    varsmap = {}  # 初始化变量映射字典
    revmap = {}  # 初始化反向映射字典

    # 如果规则字典中包含映射关系，处理重复和变量映射
    if 'map' in r:
        for k in r['map'].keys():
            if r['map'][k] in revmap:
                outmess('\t\t\tVariable "%s<=%s" is already mapped by "%s". Skipping.\n' % (
                    r['map'][k], k, revmap[r['map'][k]]))
            else:
                revmap[r['map'][k]] = k

    # 如果只允许特定映射，并且规则中声明了，只处理符合条件的变量
    if 'only' in r and r['only']:
        for v in r['map'].keys():
            if r['map'][v] in m['vars']:
                if revmap[r['map'][v]] == v:
                    varsmap[v] = r['map'][v]
                else:
                    outmess('\t\t\tIgnoring map "%s=>%s". See above.\n' %
                            (v, r['map'][v]))
            else:
                outmess(
                    '\t\t\tNo definition for variable "%s=>%s". Skipping.\n' % (v, r['map'][v]))
    else:
        for v in m['vars'].keys():
            if v in revmap:
                varsmap[v] = revmap[v]
            else:
                varsmap[v] = v

    # 根据映射关系构建使用变量的字典
    for v in varsmap.keys():
        ret = dictappend(ret, buildusevar(v, varsmap[v], m['vars'], m['name']))

    return ret

# 定义 buildusevar 函数，用于构建单个变量的包装器函数
def buildusevar(name, realname, vars, usemodulename):
    outmess('\t\t\tConstructing wrapper function for variable "%s=>%s"...\n' % (
        name, realname))
    ret = {}  # 初始化返回的字典
    # 创建一个字典 vrd，包含各种与名称相关的键值对
    vrd = {'name': name,
           'realname': realname,
           'REALNAME': realname.upper(),  # 将 realname 转换为大写保存到 REALNAME 键中
           'usemodulename': usemodulename,
           'USEMODULENAME': usemodulename.upper(),  # 将 usemodulename 转换为大写保存到 USEMODULENAME 键中
           'texname': name.replace('_', '\\_'),  # 将名称中的下划线替换为转义后的斜线保存到 texname 键中
           'begintitle': gentitle('%s=>%s' % (name, realname)),  # 使用 gentitle 函数生成名称和真实名称的标题保存到 begintitle 键中
           'endtitle': gentitle('end of %s=>%s' % (name, realname)),  # 使用 gentitle 函数生成名称和真实名称结尾的标题保存到 endtitle 键中
           'apiname': '#modulename#_use_%s_from_%s' % (realname, usemodulename)  # 生成 API 名称保存到 apiname 键中
           }
    
    # 创建一个数字映射的字典 nummap
    nummap = {0: 'Ro', 1: 'Ri', 2: 'Rii', 3: 'Riii', 4: 'Riv',
              5: 'Rv', 6: 'Rvi', 7: 'Rvii', 8: 'Rviii', 9: 'Rix'}
    
    # 将 texnamename 键设置为初始的名称
    vrd['texnamename'] = name
    
    # 替换名称中的数字为对应的罗马数字
    for i in nummap.keys():
        vrd['texnamename'] = vrd['texnamename'].replace(repr(i), nummap[i])
    
    # 如果 realname 在 vars 中有注释，则将注释保存到 vrd 的 note 键中
    if hasnote(vars[realname]):
        vrd['note'] = vars[realname]['note']
    
    # 将 vrd 添加到一个新创建的字典 rd 中
    rd = dictappend({}, vrd)

    # 打印 name、realname 和 vars[realname] 的内容
    print(name, realname, vars[realname])
    
    # 应用 usemodule_rules 到 rd 上，并返回结果
    ret = applyrules(usemodule_rules, rd)
    return ret
```