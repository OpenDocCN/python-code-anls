# `.\numpy\numpy\f2py\func2subr.py`

```
"""
Rules for building C/API module with f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 导入必要的模块
import copy

# 从本地模块中导入辅助函数
from .auxfuncs import (
    getfortranname, isexternal, isfunction, isfunction_wrap, isintent_in,
    isintent_out, islogicalfunction, ismoduleroutine, isscalar,
    issubroutine, issubroutine_wrap, outmess, show
)

# 从_isocbind模块导入isoc_kindmap变量
from ._isocbind import isoc_kindmap

# 将变量映射为适合Fortran的定义
def var2fixfortran(vars, a, fa=None, f90mode=None):
    # 如果fa未指定，则默认为a
    if fa is None:
        fa = a
    # 如果a不在变量字典中，则显示变量列表并输出消息，返回空字符串
    if a not in vars:
        show(vars)
        outmess('var2fixfortran: No definition for argument "%s".\n' % a)
        return ''
    # 如果变量定义中没有类型说明，则显示变量信息并输出消息，返回空字符串
    if 'typespec' not in vars[a]:
        show(vars[a])
        outmess('var2fixfortran: No typespec for argument "%s".\n' % a)
        return ''
    # 获取变量的类型定义
    vardef = vars[a]['typespec']
    # 如果变量类型是'type'并且有typename属性，则在类型定义中添加typename
    if vardef == 'type' and 'typename' in vars[a]:
        vardef = '%s(%s)' % (vardef, vars[a]['typename'])
    selector = {}
    lk = ''
    # 如果变量具有'kindselector'属性，则将其作为kindselector
    if 'kindselector' in vars[a]:
        selector = vars[a]['kindselector']
        lk = 'kind'
    # 如果变量具有'charselector'属性，则将其作为charselector
    elif 'charselector' in vars[a]:
        selector = vars[a]['charselector']
        lk = 'len'
    # 如果selector中包含'*'，根据f90mode添加变量定义
    if '*' in selector:
        if f90mode:
            if selector['*'] in ['*', ':', '(*)']:
                vardef = '%s(len=*)' % (vardef)
            else:
                vardef = '%s(%s=%s)' % (vardef, lk, selector['*'])
        else:
            if selector['*'] in ['*', ':']:
                vardef = '%s*(%s)' % (vardef, selector['*'])
            else:
                vardef = '%s*%s' % (vardef, selector['*'])
    else:
        # 如果selector中包含'len'属性，则添加长度限定
        if 'len' in selector:
            vardef = '%s(len=%s' % (vardef, selector['len'])
            if 'kind' in selector:
                vardef = '%s,kind=%s)' % (vardef, selector['kind'])
            else:
                vardef = '%s)' % (vardef)
        # 如果selector中包含'kind'属性，则添加kind属性
        elif 'kind' in selector:
            vardef = '%s(kind=%s)' % (vardef, selector['kind'])

    # 将最终的变量定义与fa连接起来
    vardef = '%s %s' % (vardef, fa)
    # 如果变量具有'dimension'属性，则添加维度信息
    if 'dimension' in vars[a]:
        vardef = '%s(%s)' % (vardef, ','.join(vars[a]['dimension']))
    return vardef

# 检查是否需要使用iso_c_binding模块
def useiso_c_binding(rout):
    useisoc = False
    # 遍历函数变量字典，检查每个变量的kindselector是否在isoc_kindmap中
    for key, value in rout['vars'].items():
        kind_value = value.get('kindselector', {}).get('kind')
        if kind_value in isoc_kindmap:
            return True
    return useisoc

# 创建函数包装器
def createfuncwrapper(rout, signature=0):
    # 断言函数是外部函数
    assert isfunction(rout)

    extra_args = []
    vars = rout['vars']
    # 遍历路由字典中的参数列表
    for a in rout['args']:
        # 获取当前参数在变量字典中的信息
        v = rout['vars'][a]
        # 遍历参数的维度信息，对于每个维度进行处理
        for i, d in enumerate(v.get('dimension', [])):
            # 如果维度标识为冒号，生成新的维度变量名
            if d == ':':
                dn = 'f2py_%s_d%s' % (a, i)
                # 创建新的维度变量字典
                dv = dict(typespec='integer', intent=['hide'])
                dv['='] = 'shape(%s, %s)' % (a, i)
                # 将新的维度变量名添加到额外参数列表
                extra_args.append(dn)
                # 在变量字典中添加新的维度变量信息
                vars[dn] = dv
                # 更新参数的维度信息为新的维度变量名
                v['dimension'][i] = dn

    # 将额外的参数列表添加到路由参数列表末尾
    rout['args'].extend(extra_args)
    # 检查是否需要生成接口
    need_interface = bool(extra_args)

    # 初始化返回结果列表
    ret = ['']

    # 定义用于将行添加到返回结果列表的函数
    def add(line, ret=ret):
        ret[0] = '%s\n      %s' % (ret[0], line)

    # 获取路由的名称
    name = rout['name']
    # 获取路由的 Fortran 名称
    fortranname = getfortranname(rout)
    # 检查是否为模块化路由
    f90mode = ismoduleroutine(rout)
    # 生成新的路由名称
    newname = '%sf2pywrap' % (name)

    # 如果新的路由名称不在变量字典中，则复制原有路由名称的变量信息
    if newname not in vars:
        vars[newname] = vars[name]
        # 更新参数列表，将原有路由名称替换为新的路由名称
        args = [newname] + rout['args'][1:]
    else:
        args = [newname] + rout['args']

    # 将变量字典中的信息转换为固定格式的 Fortran 代码
    l_tmpl = var2fixfortran(vars, name, '@@@NAME@@@', f90mode)
    
    # 如果第一行的类型声明为 character*(*)，根据 Fortran 标准设置长度
    if l_tmpl[:13] == 'character*(*)':
        if f90mode:
            l_tmpl = 'character(len=10)' + l_tmpl[13:]
        else:
            l_tmpl = 'character*10' + l_tmpl[13:]
        # 更新字符选择器字典中的信息
        charselect = vars[name]['charselector']
        if charselect.get('*', '') == '(*)':
            charselect['*'] = '10'

    # 将第一行转换为新路由名称的声明
    l1 = l_tmpl.replace('@@@NAME@@@', newname)
    rl = None

    # 检查是否使用 ISO C 绑定
    useisoc = useiso_c_binding(rout)
    # 将参数列表转换为字符串
    sargs = ', '.join(args)
    if f90mode:
        # 修正问题编号为 gh-23598 的警告
        # 重新生成参数列表，移除模块名称和路由名称
        sargs = sargs.replace(f"{name}, ", '')
        args = [arg for arg in args if arg != name]
        # 更新路由参数列表
        rout['args'] = args
        # 添加新的子程序声明到返回结果列表中
        add('subroutine f2pywrap_%s_%s (%s)' %
            (rout['modulename'], name, sargs))
        # 如果没有签名信息，添加模块使用声明到返回结果列表中
        if not signature:
            add('use %s, only : %s' % (rout['modulename'], fortranname))
        # 如果使用 ISO C 绑定，添加使用声明到返回结果列表中
        if useisoc:
            add('use iso_c_binding')
    else:
        # 添加新的子程序声明到返回结果列表中
        add('subroutine f2pywrap%s (%s)' % (name, sargs))
        # 如果使用 ISO C 绑定，添加使用声明到返回结果列表中
        if useisoc:
            add('use iso_c_binding')
        # 如果不需要生成接口，添加外部函数声明到返回结果列表中
        if not need_interface:
            add('external %s' % (fortranname))
            # 生成新的声明语句并赋值给 rl
            rl = l_tmpl.replace('@@@NAME@@@', '') + ' ' + fortranname

    # 如果需要生成接口，处理保存的接口信息
    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            # 添加不包含 '__user__' 的 use 声明到返回结果列表中
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    # 更新参数列表，移除第一个参数
    args = args[1:]
    # 初始化已生成参数列表为空
    dumped_args = []
    # 遍历参数列表
    for a in args:
        # 如果变量为外部变量，添加外部声明到返回结果列表中
        if isexternal(vars[a]):
            add('external %s' % (a))
            # 将已处理的参数添加到已生成参数列表中
            dumped_args.append(a)
    # 遍历参数列表
    for a in args:
        # 如果参数已在已生成参数列表中，跳过当前参数
        if a in dumped_args:
            continue
        # 如果变量为标量，添加变量声明到返回结果列表中
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            # 将已处理的参数添加到已生成参数列表中
            dumped_args.append(a)
    # 遍历参数列表
    for a in args:
        # 如果参数已在已生成参数列表中，跳过当前参数
        if a in dumped_args:
            continue
        # 如果变量为输入参数，添加变量声明到返回结果列表中
        if isintent_in(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            # 将已处理的参数添加到已生成参数列表中
            dumped_args.append(a)
    # 遍历参数列表args中的每个参数a
    for a in args:
        # 如果参数a已经在dumped_args中，则跳过当前循环
        if a in dumped_args:
            continue
        # 调用函数var2fixfortran，并将返回值添加到结果集中
        add(var2fixfortran(vars, a, f90mode=f90mode))

    # 将变量l1添加到结果集中
    add(l1)
    # 如果参数rl不为None，则将其添加到结果集中
    if rl is not None:
        add(rl)

    # 如果需要接口定义
    if need_interface:
        # 如果处于f90模式下
        if f90mode:
            # f90模块已经定义了所需的接口，不需要额外操作
            pass
        else:
            # 添加interface关键字到结果集中
            add('interface')
            # 添加保存的接口内容到结果集中（去掉左侧空格）
            add(rout['saved_interface'].lstrip())
            # 添加end interface到结果集中

            add('end interface')

    # 将args中不在extra_args中的参数用', '连接成字符串
    sargs = ', '.join([a for a in args if a not in extra_args])

    # 如果不需要生成函数签名
    if not signature:
        # 如果rout是逻辑函数
        if islogicalfunction(rout):
            # 生成逻辑函数的赋值语句，添加到结果集中
            add('%s = .not.(.not.%s(%s))' % (newname, fortranname, sargs))
        else:
            # 生成一般函数的调用语句，添加到结果集中
            add('%s = %s(%s)' % (newname, fortranname, sargs))
    
    # 如果处于f90模式下
    if f90mode:
        # 添加f90模式下的子程序结束语句到结果集中
        add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
    else:
        # 添加一般的结束语句到结果集中
        add('end')

    # 返回结果集的第一个元素
    return ret[0]
# 定义一个函数，用于创建包装子程序的包装器
def createsubrwrapper(rout, signature=0):

    # 断言rout确实是一个子程序
    assert issubroutine(rout)

    # 用于存储额外参数的列表
    extra_args = []

    # 从rout中获取变量字典
    vars = rout['vars']

    # 遍历rout中的参数
    for a in rout['args']:
        # 获取参数对应的变量
        v = rout['vars'][a]
        # 遍历该变量的维度信息
        for i, d in enumerate(v.get('dimension', [])):
            # 如果维度为':'
            if d == ':':
                # 创建新的维度名
                dn = 'f2py_%s_d%s' % (a, i)
                # 创建维度变量的字典
                dv = dict(typespec='integer', intent=['hide'])
                dv['='] = 'shape(%s, %s)' % (a, i)
                # 添加到额外参数列表中
                extra_args.append(dn)
                # 将维度变量添加到变量字典中
                vars[dn] = dv
                # 更新原始参数的维度信息
                v['dimension'][i] = dn

    # 将额外参数添加到参数列表中
    rout['args'].extend(extra_args)

    # 是否需要接口的标志
    need_interface = bool(extra_args)

    # 初始化返回结果的列表
    ret = ['']

    # 定义一个内部函数，用于将文本行添加到返回结果中
    def add(line, ret=ret):
        ret[0] = '%s\n      %s' % (ret[0], line)

    # 获取子程序的名称
    name = rout['name']

    # 获取子程序的Fortran名称
    fortranname = getfortranname(rout)

    # 检查是否为Fortran 90模式的子程序
    f90mode = ismoduleroutine(rout)

    # 获取子程序的参数列表
    args = rout['args']

    # 检查是否使用ISO_C_BINDING
    useisoc = useiso_c_binding(rout)

    # 将参数列表转换为字符串形式
    sargs = ', '.join(args)

    # 根据Fortran模式不同，添加不同的子程序定义行
    if f90mode:
        add('subroutine f2pywrap_%s_%s (%s)' %
            (rout['modulename'], name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not signature:
            add('use %s, only : %s' % (rout['modulename'], fortranname))
    else:
        add('subroutine f2pywrap%s (%s)' % (name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add('external %s' % (fortranname))

    # 如果需要接口，添加保存的接口信息
    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)

    # 已导出的参数列表
    dumped_args = []

    # 遍历参数列表，如果是外部变量，则添加external声明
    for a in args:
        if isexternal(vars[a]):
            add('external %s' % (a))
            dumped_args.append(a)

    # 继续遍历参数列表，如果是标量，则添加到返回结果中
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)

    # 最后一次遍历参数列表，将变量添加到返回结果中
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))

    # 如果需要接口，根据Fortran模式添加接口声明
    if need_interface:
        if f90mode:
            # 对于Fortran 90模式，接口已经定义，不需要再次声明
            pass
        else:
            # 添加接口声明
            add('interface')
            for line in rout['saved_interface'].split('\n'):
                if line.lstrip().startswith('use ') and '__user__' in line:
                    continue
                add(line)
            add('end interface')

    # 更新参数列表字符串形式，排除额外参数
    sargs = ', '.join([a for a in args if a not in extra_args])

    # 如果不是签名模式，添加调用Fortran子程序的行
    if not signature:
        add('call %s(%s)' % (fortranname, sargs))

    # 根据Fortran模式添加结束子程序的行
    if f90mode:
        add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
    else:
        add('end')

    # 返回处理后的文本结果
    return ret[0]
    # 如果是函数包装，则执行以下代码
    if isfunction_wrap(rout):
        # 获取 Fortran 函数名称
        fortranname = getfortranname(rout)
        # 获取函数名
        name = rout['name']
        # 输出提示信息
        outmess('\t\tCreating wrapper for Fortran function "%s"("%s")...\n' % (
            name, fortranname))
        # 复制函数对象
        rout = copy.copy(rout)
        # 获取函数名
        fname = name
        rname = fname
        # 如果函数对象中有'result'属性，则将其赋值给rname
        if 'result' in rout:
            rname = rout['result']
            rout['vars'][fname] = rout['vars'][rname]
        # 获取函数变量
        fvar = rout['vars'][fname]
        # 如果函数变量不是输出变量，则将其设为输出变量
        if not isintent_out(fvar):
            if 'intent' not in fvar:
                fvar['intent'] = []
            fvar['intent'].append('out')
            flag = 1
            for i in fvar['intent']:
                if i.startswith('out='):
                    flag = 0
                    break
            if flag:
                fvar['intent'].append('out=%s' % (rname))
        # 将函数名添加到参数列表的最前面
        rout['args'][:] = [fname] + rout['args']
        # 返回函数对象和创建的函数包装器
        return rout, createfuncwrapper(rout)
    # 如果是子例程包装，则执行以下代码
    if issubroutine_wrap(rout):
        # 获取 Fortran 子例程名称
        fortranname = getfortranname(rout)
        # 获取子例程名
        name = rout['name']
        # 输出提示信息
        outmess('\t\tCreating wrapper for Fortran subroutine "%s"("%s")...\n'
                % (name, fortranname))
        # 复制子例程对象
        rout = copy.copy(rout)
        # 返回子例程对象和创建的子例程包装器
        return rout, createsubrwrapper(rout)
    # 返回原始对象和空字符串
    return rout, ''
```