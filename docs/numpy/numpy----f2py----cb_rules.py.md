# `.\numpy\numpy\f2py\cb_rules.py`

```
"""
Build call-back mechanism for f2py2e.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""
# 导入模块中的版本信息
from . import __version__
# 导入辅助函数模块
from .auxfuncs import (
    applyrules, debugcapi, dictappend, errmess, getargs, hasnote, isarray,
    iscomplex, iscomplexarray, iscomplexfunction, isfunction, isintent_c,
    isintent_hide, isintent_in, isintent_inout, isintent_nothide,
    isintent_out, isoptional, isrequired, isscalar, isstring,
    isstringfunction, issubroutine, l_and, l_not, l_or, outmess, replace,
    stripcomma, throw_error
)
# 导入C函数模块
from . import cfuncs

# 获取当前模块的f2py版本信息
f2py_version = __version__.version


################## Rules for callback function ##############

# 回调函数规则的定义
cb_routine_rules = {
    'cbtypedefs': 'typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);',
    'body': """
#begintitle#
typedef struct {
    PyObject *capi;
    PyTupleObject *args_capi;
    int nofargs;
    jmp_buf jmpbuf;
} #name#_t;

#if defined(F2PY_THREAD_LOCAL_DECL) && !defined(F2PY_USE_PYTHON_TLS)

static F2PY_THREAD_LOCAL_DECL #name#_t *_active_#name# = NULL;

static #name#_t *swap_active_#name#(#name#_t *ptr) {
    #name#_t *prev = _active_#name#;
    _active_#name# = ptr;
    return prev;
}

static #name#_t *get_active_#name#(void) {
    return _active_#name#;
}

#else

static #name#_t *swap_active_#name#(#name#_t *ptr) {
    char *key = "__f2py_cb_#name#";
    return (#name#_t *)F2PySwapThreadLocalCallbackPtr(key, ptr);
}

static #name#_t *get_active_#name#(void) {
    char *key = "__f2py_cb_#name#";
    return (#name#_t *)F2PyGetThreadLocalCallbackPtr(key);
}

#endif

/*typedef #rctype#(*#name#_typedef)(#optargs_td##args_td##strarglens_td##noargs#);*/
#static# #rctype# #callbackname# (#optargs##args##strarglens##noargs#) {
    #name#_t cb_local = { NULL, NULL, 0 };
    #name#_t *cb = NULL;
    PyTupleObject *capi_arglist = NULL;
    PyObject *capi_return = NULL;
    PyObject *capi_tmp = NULL;
    PyObject *capi_arglist_list = NULL;
    int capi_j,capi_i = 0;
    int capi_longjmp_ok = 1;
#decl#
#ifdef F2PY_REPORT_ATEXIT
f2py_cb_start_clock();
#endif
    cb = get_active_#name#();
    if (cb == NULL) {
        capi_longjmp_ok = 0;
        cb = &cb_local;
    }
    capi_arglist = cb->args_capi;
    CFUNCSMESS(\"cb:Call-back function #name# (maxnofargs=#maxnofargs#(-#nofoptargs#))\\n\");
    CFUNCSMESSPY(\"cb:#name#_capi=\",cb->capi);
    if (cb->capi==NULL) {
        capi_longjmp_ok = 0;
        cb->capi = PyObject_GetAttrString(#modulename#_module,\"#argname#\");
        CFUNCSMESSPY(\"cb:#name#_capi=\",cb->capi);
    }
    if (cb->capi==NULL) {
        PyErr_SetString(#modulename#_error,\"cb: Callback #argname# not defined (as an argument or module #modulename# attribute).\\n\");
        goto capi_fail;
    }
    if (F2PyCapsule_Check(cb->capi)) {
    #name#_typedef #name#_cptr;
    #name#_cptr = F2PyCapsule_AsVoidPtr(cb->capi);
    #returncptr#(*#name#_cptr)(#optargs_nm##args_nm##strarglens_nm#);
    #return#
    }

    if (capi_arglist==NULL) {
        // 如果回调函数的参数列表为空指针，则执行以下操作
        capi_longjmp_ok = 0;
        // 从模块中获取名为 "#argname#_extra_args" 的属性
        capi_tmp = PyObject_GetAttrString(#modulename#_module,\"#argname#_extra_args\");
        // 如果成功获取到了属性
        if (capi_tmp) {
            // 将属性转换为元组对象
            capi_arglist = (PyTupleObject *)PySequence_Tuple(capi_tmp);
            // 释放临时对象
            Py_DECREF(capi_tmp);
            // 如果转换失败
            if (capi_arglist==NULL) {
                // 抛出异常，指示转换失败
                PyErr_SetString(#modulename#_error,\"Failed to convert #modulename#.#argname#_extra_args to tuple.\\n\");
                // 跳转到错误处理标签
                goto capi_fail;
            }
        } else {
            // 清除之前的异常状态
            PyErr_Clear();
            // 如果未获取到属性，则创建一个空的元组对象
            capi_arglist = (PyTupleObject *)Py_BuildValue(\"()\");
        }
    }
    // 如果参数列表仍为空
    if (capi_arglist == NULL) {
        // 抛出异常，指示回调函数的参数列表未设置
        PyErr_SetString(#modulename#_error,\"Callback #argname# argument list is not set.\\n\");
        // 跳转到错误处理标签
        goto capi_fail;
    }
#setdims#
#ifdef PYPY_VERSION
#define CAPI_ARGLIST_SETITEM(idx, value) PyList_SetItem((PyObject *)capi_arglist_list, idx, value)
    // 如果是在 PyPy 环境下，将 capi_arglist 转换为 PyList 对象
    capi_arglist_list = PySequence_List((PyObject *)capi_arglist);
    // 如果转换失败，跳转到 capi_fail 标签处处理异常
    if (capi_arglist_list == NULL) goto capi_fail;
#else
#define CAPI_ARGLIST_SETITEM(idx, value) PyTuple_SetItem((PyObject *)capi_arglist, idx, value)
#endif
#pyobjfrom#
#undef CAPI_ARGLIST_SETITEM
#ifdef PYPY_VERSION
    // 在 PyPy 环境下，输出 capi_arglist_list 的内容到日志
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist_list);
#else
    // 在非 PyPy 环境下，输出 capi_arglist 的内容到日志
    CFUNCSMESSPY(\"cb:capi_arglist=\",capi_arglist);
#endif
    // 输出调试信息到日志，指示正在调用 Python 函数
    CFUNCSMESS(\"cb:Call-back calling Python function #argname#.\\n\");
#ifdef F2PY_REPORT_ATEXIT
    // 如果定义了 F2PY_REPORT_ATEXIT 宏，则开始计时
    f2py_cb_start_call_clock();
#endif
#ifdef PYPY_VERSION
    // 在 PyPy 环境下，调用 Python 函数 cb->capi，传入 capi_arglist_list 作为参数
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist_list);
    // 减少 capi_arglist_list 的引用计数
    Py_DECREF(capi_arglist_list);
    // 置空 capi_arglist_list 指针
    capi_arglist_list = NULL;
#else
    // 在非 PyPy 环境下，调用 Python 函数 cb->capi，传入 capi_arglist 作为参数
    capi_return = PyObject_CallObject(cb->capi,(PyObject *)capi_arglist);
#endif
#ifdef F2PY_REPORT_ATEXIT
    // 如果定义了 F2PY_REPORT_ATEXIT 宏，则停止计时
    f2py_cb_stop_call_clock();
#endif
    // 输出 capi_return 的内容到日志
    CFUNCSMESSPY(\"cb:capi_return=\",capi_return);
    // 如果 capi_return 为空指针，输出错误信息并跳转到 capi_fail 处
    if (capi_return == NULL) {
        fprintf(stderr,\"capi_return is NULL\\n\");
        goto capi_fail;
    }
    // 如果 capi_return 为 Py_None，减少其引用计数，并重新创建一个空元组对象
    if (capi_return == Py_None) {
        Py_DECREF(capi_return);
        capi_return = Py_BuildValue(\"()\");
    }
    // 如果 capi_return 不是元组对象，将其封装为一个元组对象
    else if (!PyTuple_Check(capi_return)) {
        capi_return = Py_BuildValue(\"(N)\",capi_return);
    }
    // 获取 capi_return 的元素个数
    capi_j = PyTuple_Size(capi_return);
    // 初始化 capi_i 为 0
    capi_i = 0;
#frompyobj#
    // 输出成功调用函数的信息到日志
    CFUNCSMESS(\"cb:#name#:successful\\n\");
    // 减少 capi_return 的引用计数
    Py_DECREF(capi_return);
#ifdef F2PY_REPORT_ATEXIT
    // 如果定义了 F2PY_REPORT_ATEXIT 宏，则停止计时
    f2py_cb_stop_clock();
#endif
    // 跳转到 capi_return_pt 处，返回执行点
    goto capi_return_pt;
capi_fail:
    // 输出调用失败的信息到 stderr
    fprintf(stderr,\"Call-back #name# failed.\\n\");
    // 减少 capi_return 和 capi_arglist_list 的引用计数
    Py_XDECREF(capi_return);
    Py_XDECREF(capi_arglist_list);
    // 如果允许使用 longjmp，则跳转到 cb->jmpbuf 指定的位置
    if (capi_longjmp_ok) {
        longjmp(cb->jmpbuf,-1);
    }
capi_return_pt:
    // 返回空语句
    ;
#return#
}
#endtitle#
    {  # Init
        # 初始化字典，包含用于不同用途的分隔符和声明标记
        'separatorsfor': {'decl': '\n',
                          'args': ',', 'optargs': '', 'pyobjfrom': '\n', 'freemem': '\n',
                          'args_td': ',', 'optargs_td': '',
                          'args_nm': ',', 'optargs_nm': '',
                          'frompyobj': '\n', 'setdims': '\n',
                          'docstrsigns': '\\n"\n"',
                          'latexdocstrsigns': '\n',
                          'latexdocstrreq': '\n', 'latexdocstropt': '\n',
                          'latexdocstrout': '\n', 'latexdocstrcbs': '\n',
                          },
        # 声明标记和从Python对象转换的标记
        'decl': '/*decl*/', 'pyobjfrom': '/*pyobjfrom*/', 'frompyobj': '/*frompyobj*/',
        'args': [], 'optargs': '', 'return': '', 'strarglens': '', 'freemem': '/*freemem*/',
        'args_td': [], 'optargs_td': '', 'strarglens_td': '',
        'args_nm': [], 'optargs_nm': '', 'strarglens_nm': '',
        'noargs': '',
        'setdims': '/*setdims*/',
        # 文档字符串相关标记和信息
        'docstrsigns': '', 'latexdocstrsigns': '',
        'docstrreq': '    Required arguments:',
        'docstropt': '    Optional arguments:',
        'docstrout': '    Return objects:',
        'docstrcbs': '    Call-back functions:',
        'docreturn': '', 'docsign': '', 'docsignopt': '',
        'latexdocstrreq': '\\noindent Required arguments:',
        'latexdocstropt': '\\noindent Optional arguments:',
        'latexdocstrout': '\\noindent Return objects:',
        'latexdocstrcbs': '\\noindent Call-back functions:',
        'routnote': {hasnote: '--- #note#', l_not(hasnote): ''},
    }, {  # Function
        # 函数声明
        'decl': '    #ctype# return_value = 0;',
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},
            # 从Python对象转换为C对象时的调试信息
            '''\
    if (capi_j>capi_i) {
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,
          "#ctype#_from_pyobj failed in converting return_value of"
          " call-back function #name# to C #ctype#\\n");
    } else {
        fprintf(stderr,"Warning: call-back function #name# did not provide"
                       " return value (index=%d, type=#ctype#)\\n",capi_i);
    }''',
            {debugcapi:
             '    fprintf(stderr,"#showvalueformat#.\\n",return_value);'}
        ],
        # 从Python对象获取标量值的相关函数和调试信息
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'}, 'GETSCALARFROMPYTUPLE'],
        'return': '    return return_value;',
        '_check': l_and(isfunction, l_not(isstringfunction), l_not(iscomplexfunction))
    },
    {  # String function
        # 字符串函数相关信息
        'pyobjfrom': {debugcapi: '    fprintf(stderr,"debug-capi:cb:#name#:%d:\\n",return_value_len);'},
        'args': '#ctype# return_value,int return_value_len',
        'args_nm': 'return_value,&return_value_len',
        'args_td': '#ctype# ,int',
        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->\\"");'},
            """\
    if (capi_j>capi_i) {
        GETSTRFROMPYTUPLE(capi_return,capi_i++,return_value,return_value_len);
        # 从Python元组中获取字符串时的调试信息和错误处理
    } else {
        # 如果条件不满足，打印警告信息，说明回调函数 #name# 没有提供返回值
        # (%d, type=#ctype#) 中 %d 为索引，#ctype# 为参数类型
        fprintf(stderr,"Warning: call-back function #name# did not provide"
                       " return value (index=%d, type=#ctype#)\\n",capi_i);
    }""",
            {debugcapi:
             '    fprintf(stderr,"#showvalueformat#\\".\\n",return_value);'}
        ],
        # 需要的依赖项列表
        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
                 'string.h', 'GETSTRFROMPYTUPLE'],
        # 返回值表达式
        'return': 'return;',
        # 检查函数是否为字符串处理函数
        '_check': isstringfunction
    },
    {  # 复杂函数
        # 可选参数的默认值定义
        'optargs': """
#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *return_value
#endif


#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *return_value  // 如果未定义 F2PY_CB_RETURNCOMPLEX，则定义 return_value 指向 #ctype# 类型的指针
#endif



""",
        'optargs_nm': """
#ifndef F2PY_CB_RETURNCOMPLEX
return_value
#endif
""",


#ifndef F2PY_CB_RETURNCOMPLEX
return_value  // 如果未定义 F2PY_CB_RETURNCOMPLEX，则直接使用 return_value
#endif



        'optargs_td': """
#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *
#endif
""",


#ifndef F2PY_CB_RETURNCOMPLEX
#ctype# *  // 如果未定义 F2PY_CB_RETURNCOMPLEX，则定义 #ctype# 类型的指针
#endif



        'decl': """
#ifdef F2PY_CB_RETURNCOMPLEX
    #ctype# return_value = {0, 0};
#endif
""",


#ifdef F2PY_CB_RETURNCOMPLEX
    #ctype# return_value = {0, 0};  // 如果定义了 F2PY_CB_RETURNCOMPLEX，则定义并初始化 return_value 为 {0, 0}
#endif



        'frompyobj': [
            {debugcapi: '    CFUNCSMESS("cb:Getting return_value->");'},


            // 如果 debugcapi 被定义，则输出调试信息 "cb:Getting return_value->"



            """\
    if (capi_j>capi_i) {
#ifdef F2PY_CB_RETURNCOMPLEX
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,&return_value,#ctype#,
          \"#ctype#_from_pyobj failed in converting return_value of call-back\"
          \" function #name# to C #ctype#\\n\");
#else
        GETSCALARFROMPYTUPLE(capi_return,capi_i++,return_value,#ctype#,
          \"#ctype#_from_pyobj failed in converting return_value of call-back\"
          \" function #name# to C #ctype#\\n\");
#endif
    } else {
        fprintf(stderr,
                \"Warning: call-back function #name# did not provide\"
                \" return value (index=%d, type=#ctype#)\\n\",capi_i);
    }""",


    // 根据条件检查获取来自 Python 元组的标量值，并根据定义的类型进行转换，如未提供则输出警告信息



            {debugcapi: """\
#ifdef F2PY_CB_RETURNCOMPLEX
    fprintf(stderr,\"#showvalueformat#.\\n\",(return_value).r,(return_value).i);
#else
    fprintf(stderr,\"#showvalueformat#.\\n\",(*return_value).r,(*return_value).i);
#endif
"""}
        ],


#ifdef F2PY_CB_RETURNCOMPLEX
    // 如果定义了 F2PY_CB_RETURNCOMPLEX，则输出复杂数值的格式信息
    fprintf(stderr,"#showvalueformat#.\\n",(return_value).r,(return_value).i);
#else
    // 否则输出普通数值的格式信息
    fprintf(stderr,"#showvalueformat#.\\n",(*return_value).r,(*return_value).i);
#endif



        'return': """
#ifdef F2PY_CB_RETURNCOMPLEX
    return return_value;
#else
    return;
#endif
""",


#ifdef F2PY_CB_RETURNCOMPLEX
    return return_value;  // 如果定义了 F2PY_CB_RETURNCOMPLEX，则返回 return_value
#else
    return;  // 否则返回空
#endif



        'need': ['#ctype#_from_pyobj', {debugcapi: 'CFUNCSMESS'},
                 'string.h', 'GETSCALARFROMPYTUPLE', '#ctype#'],


        // 需要包含的头文件和宏定义
        #ctype#_from_pyobj  // 根据 #ctype# 从 Python 对象获取数据的函数
        CFUNCSMESS  // 如果 debugcapi 被定义，则输出调试信息的宏
        string.h  // C 标准库中的字符串操作头文件
        GETSCALARFROMPYTUPLE  // 从 Python 元组中获取标量值的宏
        #ctype#  // 定义的 #ctype# 类型



        '_check': iscomplexfunction
    },


        // 检查是否为复杂函数的标志



    {'docstrout': '        #pydocsignout#',
     'latexdocstrout': ['\\item[]{{}\\verb@#pydocsignout#@{}}',
                        {hasnote: '--- #note#'}],


    // 输出文档字符串的格式化字符串
    #pydocsignout#  // 根据定义的格式输出文档字符串
    // 如果有注释，则添加注释内容



     'docreturn': '#rname#,',
     '_check': isfunction},


     // 文档中的返回值说明
     #rname#,  // 根据定义的返回值名称格式化输出



    {'_check': issubroutine, 'return': 'return;'}
]


    // 检查是否为子例程的标志
    return;  // 返回空
    {
        'args': {
            l_and(isscalar, isintent_c): '#ctype# #varname_i#',
            l_and(isscalar, l_not(isintent_c)): '#ctype# *#varname_i#_cb_capi',
            isarray: '#ctype# *#varname_i#',
            isstring: '#ctype# #varname_i#'
        },
        'args_nm': {
            l_and(isscalar, isintent_c): '#varname_i#',
            l_and(isscalar, l_not(isintent_c)): '#varname_i#_cb_capi',
            isarray: '#varname_i#',
            isstring: '#varname_i#'
        },
        'args_td': {
            l_and(isscalar, isintent_c): '#ctype#',
            l_and(isscalar, l_not(isintent_c)): '#ctype# *',
            isarray: '#ctype# *',
            isstring: '#ctype#'
        },
        'need': {l_or(isscalar, isarray, isstring): '#ctype#'},
        # 在多参数情况下未经测试
        'strarglens': {isstring: ',int #varname_i#_cb_len'},
        'strarglens_td': {isstring: ',int'},  # 在多参数情况下未经测试
        # 在多参数情况下未经测试
        'strarglens_nm': {isstring: ',#varname_i#_cb_len'},
    },
    {  # Scalars
        'decl': {l_not(isintent_c): '    #ctype# #varname_i#=(*#varname_i#_cb_capi);'},
        'error': {l_and(isintent_c, isintent_out,
                        throw_error('intent(c,out) is forbidden for callback scalar arguments')):
                  ''},
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
                      {isintent_out:
                       '    if (capi_j>capi_i)\n        GETSCALARFROMPYTUPLE(capi_return,capi_i++,#varname_i#_cb_capi,#ctype#,"#ctype#_from_pyobj failed in converting argument #varname# of call-back function #name# to C #ctype#\\n");'},
                      {l_and(debugcapi, l_and(l_not(iscomplex), isintent_c)):
                          '    fprintf(stderr,"#showvalueformat#.\\n",#varname_i#);'},
                      {l_and(debugcapi, l_and(l_not(iscomplex), l_not( isintent_c))):
                          '    fprintf(stderr,"#showvalueformat#.\\n",*#varname_i#_cb_capi);'},
                      {l_and(debugcapi, l_and(iscomplex, isintent_c)):
                          '    fprintf(stderr,"#showvalueformat#.\\n",(#varname_i#).r,(#varname_i#).i);'},
                      {l_and(debugcapi, l_and(iscomplex, l_not( isintent_c))):
                          '    fprintf(stderr,"#showvalueformat#.\\n",(*#varname_i#_cb_capi).r,(*#varname_i#_cb_capi).i);'},
                      ],
        'need': [{isintent_out: ['#ctype#_from_pyobj', 'GETSCALARFROMPYTUPLE']},
                 {debugcapi: 'CFUNCSMESS'}],
        '_check': isscalar
    }, {
        'pyobjfrom': [{isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1(#varname_i#)))
            goto capi_fail;"""},
                      {isintent_inout: """\


注释：

{
    # 定义函数参数格式，根据数据类型和意图生成参数字符串
    'args': {
        l_and(isscalar, isintent_c): '#ctype# #varname_i#',
        l_and(isscalar, l_not(isintent_c)): '#ctype# *#varname_i#_cb_capi',
        isarray: '#ctype# *#varname_i#',
        isstring: '#ctype# #varname_i#'
    },
    # 定义函数参数名称格式，根据数据类型和意图生成参数名称字符串
    'args_nm': {
        l_and(isscalar, isintent_c): '#varname_i#',
        l_and(isscalar, l_not(isintent_c)): '#varname_i#_cb_capi',
        isarray: '#varname_i#',
        isstring: '#varname_i#'
    },
    # 定义函数参数类型格式，根据数据类型和意图生成参数类型字符串
    'args_td': {
        l_and(isscalar, isintent_c): '#ctype#',
        l_and(isscalar, l_not(isintent_c)): '#ctype# *',
        isarray: '#ctype# *',
        isstring: '#ctype#'
    },
    # 需要的数据类型，对于标量、数组或字符串类型都需要
    'need': {l_or(isscalar, isarray, isstring): '#ctype#'},
    # 在处理字符串参数时，未经测试多个参数情况下的参数长度
    'strarglens': {isstring: ',int #varname_i#_cb_len'},
    # 在处理字符串参数时，未经测试多个参数情况下的参数长度
    'strarglens_td': {isstring: ',int'},
    # 在处理字符串参数时，未经测试多个参数情况下的参数长度
    'strarglens_nm': {isstring: ',#varname_i#_cb_len'},
},
{  # 标量
    # 如果不是意图参数，生成标量变量声明
    'decl': {l_not(isintent_c): '    #ctype# #varname_i#=(*#varname_i#_cb_capi);'},
    # 如果参数是意图输出且禁止使用意图输出的回调标量参数，生成错误消息
    'error': {l_and(isintent_c, isintent_out,
                    throw_error('intent(c,out) is forbidden for callback scalar arguments')):
              ''},
    # 从Python对象中提取标量数据
    'frompyobj': [
        {debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
        {isintent_out:
         '    if (capi_j>capi_i)\n        GETSCALARFROMPYTUPLE(capi_return,capi_i++,#varname_i#_cb_capi,#ctype#,"#ctype#_from_pyobj failed in converting argument #varname# of call-back function #name# to C #ctype#\\n");'},
        {l_and(debugcapi, l_and(l_not(iscomplex), isintent_c)):
          '    fprintf(stderr,"#showvalueformat#.\\n",#varname_i#);'},
        {l_and(debugcapi, l_and(l_not(iscomplex), l_not( isintent_c))):
          '    fprintf(stderr,"#showvalueformat#.\\n",*#varname_i#_cb_capi);'},
        {l_and(debugcapi, l_and(iscomplex, isintent_c)):
          '    fprintf(stderr,"#showvalueformat#.\\n",(#varname_i#).r,(#varname_i#).i);'},
        {l_and(debugcapi, l_and(iscomplex, l_not( isintent_c))):
          '    fprintf(stderr,"#showvalueformat#.\\n",(*#varname_i#_cb_capi).r,(*#varname_i#_cb_capi).i);'},
    ],
    # 需要的辅助函数和宏定义，包括输出参数和调试信息
    'need': [
        {isintent_out: ['#ctype#_from_pyobj', 'GETSCALARFROMPYTUPLE']},
        {debugcapi: 'CFUNCSMESS'}
    ],
    # 检查是否为标量变量
    '_check': isscalar
}, {
    # 从Python对象生成标量数据
    'pyobjfrom': [
        {isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1(#varname_i#)))
            goto capi_fail;"""},
        {isintent_inout: """\
    if (cb->nofargs>capi_i)
        # 如果回调参数数量大于当前索引 c-api_i
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyarr_from_p_#ctype#1(#varname_i#_cb_capi)))
            # 尝试将 pyarr_from_p_#ctype#1(#varname_i#_cb_capi) 设置到 C API 参数列表中的 c-api_i 索引处，如果失败则跳转到 capi_fail
            goto capi_fail;



        'need': [{isintent_in: 'pyobj_from_#ctype#1'},
                 # 需要 'pyobj_from_#ctype#1' 函数支持的参数类型
                 {isintent_inout: 'pyarr_from_p_#ctype#1'},
                 # 需要 'pyarr_from_p_#ctype#1' 函数支持的参数类型
                 {iscomplex: '#ctype#'}],
                 # 需要类型为 '#ctype#' 的复杂参数



        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->\\"");'},
                      # 调试信息：输出字符串 "cb:Getting #varname#->\""
                      """    if (capi_j>capi_i)
        GETSTRFROMPYTUPLE(capi_return,capi_i++,#varname_i#,#varname_i#_cb_len);""",
                      # 如果 capi_j 大于 capi_i，从 Python 元组中获取字符串数据到 #varname_i# 和 #varname_i#_cb_len
                      {debugcapi:
                       '    fprintf(stderr,"#showvalueformat#\\":%d:.\\n",#varname_i#,#varname_i#_cb_len);'},
                       # 调试信息：输出格式化字符串 "#showvalueformat#\":%d:.\n"，并输出 #varname_i# 和 #varname_i#_cb_len 的值
                      ],



        'need': ['#ctype#', 'GETSTRFROMPYTUPLE',
                 # 需要类型为 '#ctype#' 的参数，以及 'GETSTRFROMPYTUPLE' 函数
                 {debugcapi: 'CFUNCSMESS'}, 'string.h'],
                 # 需要调试信息：'CFUNCSMESS'，以及 'string.h' 头文件支持



        'pyobjfrom': [
            {debugcapi:
             ('    fprintf(stderr,"debug-capi:cb:#varname#=#showvalueformat#:'
              '%d:\\n",#varname_i#,#varname_i#_cb_len);')},
             # 调试信息：输出字符串 "debug-capi:cb:#varname#=#showvalueformat#:%d:\n"，并输出 #varname_i# 和 #varname_i#_cb_len 的值
            {isintent_in: """\
    if (cb->nofargs>capi_i)
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len)))
            goto capi_fail;"""},
            # 如果回调参数数量大于当前索引 c-api_i，尝试将 pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len) 设置到 C API 参数列表中的 c-api_i 索引处，如果失败则跳转到 capi_fail
            {isintent_inout: """\
    if (cb->nofargs>capi_i) {
        int #varname_i#_cb_dims[] = {#varname_i#_cb_len};
        if (CAPI_ARGLIST_SETITEM(capi_i++,pyarr_from_p_#ctype#1(#varname_i#,#varname_i#_cb_dims)))
            goto capi_fail;
    }"""}],
    # 如果回调参数数量大于当前索引 c-api_i，将 pyarr_from_p_#ctype#1(#varname_i#,#varname_i#_cb_dims) 或 pyobj_from_#ctype#1size(#varname_i#,#varname_i#_cb_len) 设置到 C API 参数列表中的 c-api_i 索引处



        'need': [{isintent_in: 'pyobj_from_#ctype#1size'},
                 # 需要 'pyobj_from_#ctype#1size' 函数支持的参数类型
                 {isintent_inout: 'pyarr_from_p_#ctype#1'}],
                 # 需要 'pyarr_from_p_#ctype#1' 函数支持的参数类型



    # Array ...
    # 数组声明部分



    {
        'decl': '    npy_intp #varname_i#_Dims[#rank#] = {#rank*[-1]#};',
        # 声明一个 numpy 数组对象的维度，维度大小为 #rank#
        'setdims': '    #cbsetdims#;',
        # 设置数组的维度信息，具体设置方法在 #cbsetdims# 中定义
        '_check': isarray,
        # 检查是否为数组对象
        '_depend': ''
        # 依赖关系为空
    },



    {
        'pyobjfrom': [{debugcapi: '    fprintf(stderr,"debug-capi:cb:#varname#\\n");'},
                      # 调试信息：输出字符串 "debug-capi:cb:#varname#\\n"
                      {isintent_c: """\
    if (cb->nofargs>capi_i) {
        /* tmp_arr will be inserted to capi_arglist_list that will be
           destroyed when leaving callback function wrapper together
           with tmp_arr. */
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,
          #rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,#elsize#,
          NPY_ARRAY_CARRAY,NULL);
# 定义一个多行字符串，包含多个代码块，每个代码块用于生成回调函数的特定部分
"""
""",
# 当回调函数中的参数数量大于当前处理的参数索引时执行以下代码块
                       l_not(isintent_c): """\
    if (cb->nofargs>capi_i) {
        /* tmp_arr 将被插入到 capi_arglist_list 中，在离开回调函数包装器时会一同被销毁。 */
        PyArrayObject *tmp_arr = (PyArrayObject *)PyArray_New(&PyArray_Type,
          #rank#,#varname_i#_Dims,#atype#,NULL,(char*)#varname_i#,#elsize#,
          NPY_ARRAY_FARRAY,NULL);
""",
# 如果条件不满足 isintent_c 的规则，则执行此代码块
                       },
# 当 isarray 为真且 isintent_nothide 为真且 isintent_in 或 isintent_inout 其中之一为真时执行以下代码块
                      """
        if (tmp_arr==NULL)
            goto capi_fail;
        if (CAPI_ARGLIST_SETITEM(capi_i++,(PyObject *)tmp_arr))
            goto capi_fail;
}"""],
        '_check': l_and(isarray, isintent_nothide, l_or(isintent_in, isintent_inout)),
        '_optional': '',
    }, {
        'frompyobj': [{debugcapi: '    CFUNCSMESS("cb:Getting #varname#->");'},
# 如果 debugcapi 为真，则输出调试信息到标准错误流
                      """    if (capi_j>capi_i) {
        PyArrayObject *rv_cb_arr = NULL;
        if ((capi_tmp = PyTuple_GetItem(capi_return,capi_i++))==NULL) goto capi_fail;
        rv_cb_arr =  array_from_pyobj(#atype#,#varname_i#_Dims,#rank#,F2PY_INTENT_IN""",
# 如果 capi_j 大于 capi_i，则执行以下代码块
                      {isintent_c: '|F2PY_INTENT_C'},
# 如果 isintent_c 为真，则在参数列表中包含 F2PY_INTENT_C
                      """,capi_tmp);
        if (rv_cb_arr == NULL) {
            fprintf(stderr,\"rv_cb_arr is NULL\\n\");
            goto capi_fail;
        }
        MEMCOPY(#varname_i#,PyArray_DATA(rv_cb_arr),PyArray_NBYTES(rv_cb_arr));
        if (capi_tmp != (PyObject *)rv_cb_arr) {
            Py_DECREF(rv_cb_arr);
        }
    }""",
# 复制 PyArray 对象的数据到 #varname_i# 中，并根据需要减少对象的引用计数
                      {debugcapi: '    fprintf(stderr,"<-.\\n");'},
# 如果 debugcapi 为真，则输出调试信息到标准错误流
                      ],
        'need': ['MEMCOPY', {iscomplexarray: '#ctype#'}],
        '_check': l_and(isarray, isintent_out)
# 当 isarray 和 isintent_out 同时为真时执行以下代码块
    }, {
        'docreturn': '#varname#,',
# 返回 #varname#，
        '_check': isintent_out
# 当 isintent_out 为真时执行以下代码块
    }
]

################## 构建回调模块 #############
cb_map = {}

# 构建回调函数列表，将回调函数映射到模块名称的字典中
def buildcallbacks(m):
    cb_map[m['name']] = []
    for bi in m['body']:
        if bi['block'] == 'interface':
            for b in bi['body']:
                if b:
                    buildcallback(b, m['name'])
                else:
                    errmess('warning: empty body for %s\n' % (m['name']))

# 构建回调函数，将每个回调函数添加到对应模块名称的回调函数列表中
def buildcallback(rout, um):
    from . import capi_maps

    outmess('    Constructing call-back function "cb_%s_in_%s"\n' %
            (rout['name'], um))
    args, depargs = getargs(rout)
    capi_maps.depargs = depargs
    var = rout['vars']
    vrd = capi_maps.cb_routsign2map(rout, um)
    rd = dictappend({}, vrd)
    cb_map[um].append([rout['name'], rd['name']])
    for r in cb_rout_rules:
        if ('_check' in r and r['_check'](rout)) or ('_check' not in r):
            ar = applyrules(r, vrd, rout)
            rd = dictappend(rd, ar)
    savevrd = {}
    # 对于参数列表 args 中的每个参数 a，使用索引 i 枚举处理
    for i, a in enumerate(args):
        # 调用 capi_maps.cb_sign2map 函数，获取参数 a 的映射结果 vrd
        vrd = capi_maps.cb_sign2map(a, var[a], index=i)
        # 将参数 a 的映射结果 vrd 存储到 savevrd 字典中
        savevrd[a] = vrd
        # 遍历规则列表 cb_arg_rules
        for r in cb_arg_rules:
            # 如果规则字典 r 中包含 '_depend' 键，则跳过本次循环
            if '_depend' in r:
                continue
            # 如果规则字典 r 中包含 '_optional' 键，并且参数 a 是可选的，则跳过本次循环
            if '_optional' in r and isoptional(var[a]):
                continue
            # 如果规则字典 r 中包含 '_check' 键且其对应的检查函数返回 True，或者规则字典 r 中没有 '_check' 键
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                # 应用规则 r 到 vrd 和 var[a] 上，将结果添加到 rd 字典中
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                # 如果规则字典 r 中包含 '_break' 键，则跳出当前循环
                if '_break' in r:
                    break
    # 对于参数列表 args 中的每个参数 a
    for a in args:
        # 从 savevrd 字典中获取参数 a 的映射结果 vrd
        vrd = savevrd[a]
        # 再次遍历规则列表 cb_arg_rules
        for r in cb_arg_rules:
            # 如果规则字典 r 中包含 '_depend' 键，则跳过本次循环
            if '_depend' in r:
                continue
            # 如果规则字典 r 中不包含 '_optional' 键，或者包含 '_optional' 键且参数 a 是必需的
            if ('_optional' not in r) or ('_optional' in r and isrequired(var[a])):
                continue
            # 如果规则字典 r 中包含 '_check' 键且其对应的检查函数返回 True，或者规则字典 r 中没有 '_check' 键
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                # 应用规则 r 到 vrd 和 var[a] 上，将结果添加到 rd 字典中
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                # 如果规则字典 r 中包含 '_break' 键，则跳出当前循环
                if '_break' in r:
                    break
    # 对于依赖参数列表 depargs 中的每个参数 a
    for a in depargs:
        # 从 savevrd 字典中获取参数 a 的映射结果 vrd
        vrd = savevrd[a]
        # 再次遍历规则列表 cb_arg_rules
        for r in cb_arg_rules:
            # 如果规则字典 r 中不包含 '_depend' 键，则跳过本次循环
            if '_depend' not in r:
                continue
            # 如果规则字典 r 中包含 '_optional' 键，则跳过本次循环
            if '_optional' in r:
                continue
            # 如果规则字典 r 中包含 '_check' 键且其对应的检查函数返回 True，或者规则字典 r 中没有 '_check' 键
            if ('_check' in r and r['_check'](var[a])) or ('_check' not in r):
                # 应用规则 r 到 vrd 和 var[a] 上，将结果添加到 rd 字典中
                ar = applyrules(r, vrd, var[a])
                rd = dictappend(rd, ar)
                # 如果规则字典 r 中包含 '_break' 键，则跳出当前循环
                if '_break' in r:
                    break
    # 如果 rd 字典中包含 'args' 键和 'optargs' 键
    if 'args' in rd and 'optargs' in rd:
        # 如果 'optargs' 键对应的值是列表类型
        if isinstance(rd['optargs'], list):
            # 将 'optargs' 键对应的列表与另一个列表合并，赋值回 'optargs' 键
            rd['optargs'] = rd['optargs'] + [```
#ifndef F2PY_CB_RETURNCOMPLEX
,  # 如果未定义 F2PY_CB_RETURNCOMPLEX，则插入逗号
#endif
"""]
            rd['optargs_nm'] = rd['optargs_nm'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,  # 如果未定义 F2PY_CB_RETURNCOMPLEX，则插入逗号
#endif
"""]
            rd['optargs_td'] = rd['optargs_td'] + ["""
#ifndef F2PY_CB_RETURNCOMPLEX
,  # 如果未定义 F2PY_CB_RETURNCOMPLEX，则插入逗号
#endif
"""]
    if isinstance(rd['docreturn'], list):
        rd['docreturn'] = stripcomma(
            replace('#docreturn#', {'docreturn': rd['docreturn']}))
        # 如果 rd['docreturn'] 是列表类型，则将其用 stripcomma 函数处理，替换 '#docreturn#'
    optargs = stripcomma(replace('#docsignopt#',
                                 {'docsignopt': rd['docsignopt']}
                                 ))
    if optargs == '':
        rd['docsignature'] = stripcomma(
            replace('#docsign#', {'docsign': rd['docsign']}))
        # 如果 optargs 是空字符串，则使用 stripcomma 函数处理 rd['docsignature']，替换 '#docsign#'
    else:
        rd['docsignature'] = replace('#docsign#[#docsignopt#]',
                                     {'docsign': rd['docsign'],
                                      'docsignopt': optargs,
                                      })
        # 否则，使用 replace 函数替换 '#docsign#[#docsignopt#]' 的内容
    rd['latexdocsignature'] = rd['docsignature'].replace('_', '\\_')
    rd['latexdocsignature'] = rd['latexdocsignature'].replace(',', ', ')
    rd['docstrsigns'] = []
    rd['latexdocstrsigns'] = []
    for k in ['docstrreq', 'docstropt', 'docstrout', 'docstrcbs']:
        if k in rd and isinstance(rd[k], list):
            rd['docstrsigns'] = rd['docstrsigns'] + rd[k]
        k = 'latex' + k
        if k in rd and isinstance(rd[k], list):
            rd['latexdocstrsigns'] = rd['latexdocstrsigns'] + rd[k][0:1] +\
                ['\\begin{description}'] + rd[k][1:] +\
                ['\\end{description}']
        # 合并文档字符串相关的列表
    if 'args' not in rd:
        rd['args'] = ''
        rd['args_td'] = ''
        rd['args_nm'] = ''
        # 如果 rd 中没有 'args' 键，则初始化为空字符串
    if not (rd.get('args') or rd.get('optargs') or rd.get('strarglens')):
        rd['noargs'] = 'void'
        # 如果 rd 中 'args'、'optargs' 或 'strarglens' 为空，则设置 'noargs' 为 'void'
    ar = applyrules(cb_routine_rules, rd)
    cfuncs.callbacks[rd['name']] = ar['body']
    if isinstance(ar['need'], str):
        ar['need'] = [ar['need']]
        # 如果 ar['need'] 是字符串类型，则转换为列表
    if 'need' in rd:
        for t in cfuncs.typedefs.keys():
            if t in rd['need']:
                ar['need'].append(t)
        # 遍历 rd['need']，将其对应的键添加到 ar['need'] 中
    cfuncs.typedefs_generated[rd['name'] + '_typedef'] = ar['cbtypedefs']
    ar['need'].append(rd['name'] + '_typedef')
    cfuncs.needs[rd['name']] = ar['need']
    # 设置回调函数和相关的 typedef 和需要的键
    capi_maps.lcb2_map[rd['name']] = {'maxnofargs': ar['maxnofargs'],
                                      'nofoptargs': ar['nofoptargs'],
                                      'docstr': ar['docstr'],
                                      'latexdocstr': ar['latexdocstr'],
                                      'argname': rd['argname']
                                      }
    outmess('      %s\n' % (ar['docstrshort']))
    return
################## Build call-back function #############
```