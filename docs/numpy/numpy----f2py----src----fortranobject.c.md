# `.\numpy\numpy\f2py\src\fortranobject.c`

```
/*
  This file implements: FortranObject, array_from_pyobj, copy_ND_array

  Author: Pearu Peterson <pearu@cens.ioc.ee>
  $Revision: 1.52 $
  $Date: 2005/07/11 07:44:20 $
*/

// 设置字典 dict 中名为 name 的项为 obj，若 obj 为 NULL 则打印错误信息并返回 -1
int
F2PyDict_SetItemString(PyObject *dict, char *name, PyObject *obj)
{
    if (obj == NULL) {
        fprintf(stderr, "Error loading %s\n", name);
        // 检查是否有 Python 异常发生，如果有则打印异常信息并清除异常状态
        if (PyErr_Occurred()) {
            PyErr_Print();
            PyErr_Clear();
        }
        return -1;
    }
    return PyDict_SetItemString(dict, name, obj);
}

/*
 * Python-only fallback for thread-local callback pointers
 */
// 更新线程本地回调指针
void *
F2PySwapThreadLocalCallbackPtr(char *key, void *ptr)
{
    PyObject *local_dict, *value;
    void *prev;

    // 获取当前线程状态的字典
    local_dict = PyThreadState_GetDict();
    if (local_dict == NULL) {
        // 如果获取失败，触发致命错误
        Py_FatalError(
                "F2PySwapThreadLocalCallbackPtr: PyThreadState_GetDict "
                "failed");
    }

    // 获取字典中 key 对应的值
    value = PyDict_GetItemString(local_dict, key);
    if (value != NULL) {
        // 如果找到值，将其转换为 void 指针类型
        prev = PyLong_AsVoidPtr(value);
        if (PyErr_Occurred()) {
            // 如果转换出错，触发致命错误
            Py_FatalError(
                    "F2PySwapThreadLocalCallbackPtr: PyLong_AsVoidPtr failed");
        }
    }
    else {
        prev = NULL;
    }

    // 将新的指针 ptr 转换为 PyLong 对象
    value = PyLong_FromVoidPtr((void *)ptr);
    if (value == NULL) {
        // 如果转换失败，触发致命错误
        Py_FatalError(
                "F2PySwapThreadLocalCallbackPtr: PyLong_FromVoidPtr failed");
    }

    // 将新的值设置回字典中
    if (PyDict_SetItemString(local_dict, key, value) != 0) {
        // 如果设置失败，触发致命错误
        Py_FatalError(
                "F2PySwapThreadLocalCallbackPtr: PyDict_SetItemString failed");
    }

    // 释放 PyLong 对象的引用
    Py_DECREF(value);

    // 返回先前的值 prev
    return prev;
}

// 获取线程本地回调指针
void *
F2PyGetThreadLocalCallbackPtr(char *key)
{
    PyObject *local_dict, *value;
    void *prev;

    // 获取当前线程状态的字典
    local_dict = PyThreadState_GetDict();
    if (local_dict == NULL) {
        // 如果获取失败，触发致命错误
        Py_FatalError(
                "F2PyGetThreadLocalCallbackPtr: PyThreadState_GetDict failed");
    }

    // 获取字典中 key 对应的值
    value = PyDict_GetItemString(local_dict, key);
    if (value != NULL) {
        // 如果找到值，将其转换为 void 指针类型
        prev = PyLong_AsVoidPtr(value);
        if (PyErr_Occurred()) {
            // 如果转换出错，触发致命错误
            Py_FatalError(
                    "F2PyGetThreadLocalCallbackPtr: PyLong_AsVoidPtr failed");
        }
    }
    else {
        prev = NULL;
    }

    // 返回先前的值 prev
    return prev;
}

// 根据给定的类型编号和元素大小，创建一个 PyArray_Descr 对象
static PyArray_Descr *
get_descr_from_type_and_elsize(const int type_num, const int elsize)  {
  // 根据类型编号创建 PyArray_Descr 对象
  PyArray_Descr * descr = PyArray_DescrFromType(type_num);
  if (type_num == NPY_STRING) {
    // 对于字符串类型，PyArray_DescrFromType 返回的描述符 elsize 为 0
    // 需要用 PyArray_DESCR_REPLACE 更新描述符并设置正确的 elsize
    PyArray_DESCR_REPLACE(descr);
    if (descr == NULL) {
      return NULL;
    }
    PyDataType_SET_ELSIZE(descr, elsize);
  }
  return descr;
}
    // 初始化指针变量，确保在后续使用中能正常赋值
    PyFortranObject *fp = NULL;
    // Python 对象指针，用于暂存临时变量
    PyObject *v = NULL;
    // 如果初始化函数不为空，则调用该函数进行 F90 模块对象的初始化
    if (init != NULL) { /* Initialize F90 module objects */
        (*(init))();
    }
    // 分配内存并创建 PyFortranObject 对象实例
    fp = PyObject_New(PyFortranObject, &PyFortran_Type);
    // 如果分配失败，则返回空指针
    if (fp == NULL) {
        return NULL;
    }
    // 创建一个新的空字典对象，并将其赋值给 fp 对象的 dict 属性
    if ((fp->dict = PyDict_New()) == NULL) {
        Py_DECREF(fp);
        return NULL;
    }
    // 初始化 fp 对象的长度属性为 0
    fp->len = 0;
    // 遍历 defs 数组，计算其中非空项的数量并赋值给 fp 对象的 len 属性
    while (defs[fp->len].name != NULL) {
        fp->len++;
    }
    // 如果 defs 数组长度为 0，则跳转到错误处理部分
    if (fp->len == 0) {
        goto fail;
    }
    // 将 defs 数组赋值给 fp 对象的 defs 属性
    fp->defs = defs;
    // 遍历 fp 对象的 defs 数组
    for (i = 0; i < fp->len; i++) {
        // 如果当前项的 rank 属性为 -1，表示是 Fortran 过程
        if (fp->defs[i].rank == -1) { /* Is Fortran routine */
            // 创建一个新的 Python 对象作为属性，并赋值给 v
            v = PyFortranObject_NewAsAttr(&(fp->defs[i]));
            // 如果创建失败，则跳转到错误处理部分
            if (v == NULL) {
                goto fail;
            }
            // 将 v 对象以字符串形式存入 fp 对象的 dict 属性
            PyDict_SetItemString(fp->dict, fp->defs[i].name, v);
            // 减少 v 对象的引用计数
            Py_XDECREF(v);
        }
        // 如果当前项的 data 属性不为空，表示是 Fortran 变量或数组（非 allocatable）
        else if ((fp->defs[i].data) != NULL) { /* Is Fortran variable or array (not allocatable) */
            // 根据类型和元素大小获取 PyArray_Descr 对象
            PyArray_Descr *descr = get_descr_from_type_and_elsize(fp->defs[i].type,
                                                                   fp->defs[i].elsize);
            // 如果获取失败，则跳转到错误处理部分
            if (descr == NULL) {
                goto fail;
            }
            // 根据描述符创建一个新的数组对象，并赋值给 v
            v = PyArray_NewFromDescr(&PyArray_Type, descr, fp->defs[i].rank,
                                     fp->defs[i].dims.d, NULL, fp->defs[i].data,
                                     NPY_ARRAY_FARRAY, NULL);
            // 如果创建失败，则减少描述符的引用计数，并跳转到错误处理部分
            if (v == NULL) {
                Py_DECREF(descr);
                goto fail;
            }
            // 将 v 对象以字符串形式存入 fp 对象的 dict 属性
            PyDict_SetItemString(fp->dict, fp->defs[i].name, v);
            // 减少 v 对象的引用计数
            Py_XDECREF(v);
        }
    }
    // 返回 fp 对象的 PyObject 指针类型
    return (PyObject *)fp;
fail:
    // 释放 fp 指针指向的对象，并返回 NULL，用于错误处理
    Py_XDECREF(fp);
    return NULL;
}

PyObject *
PyFortranObject_NewAsAttr(FortranDataDef *defs)
{ /* used for calling F90 module routines */
    // 初始化 fp 为 NULL
    PyFortranObject *fp = NULL;
    // 创建一个新的 PyFortranObject 对象，并将其类型设置为 PyFortran_Type
    fp = PyObject_New(PyFortranObject, &PyFortran_Type);
    // 如果创建对象失败，则调用失败处理块
    if (fp == NULL)
        return NULL;
    // 创建一个新的字典对象并将其赋值给 fp->dict
    if ((fp->dict = PyDict_New()) == NULL) {
        PyObject_Del(fp);
        // 如果创建字典失败，则调用失败处理块
        return NULL;
    }
    // 设置 fp->len 为 1
    fp->len = 1;
    // 将 defs 赋值给 fp->defs
    fp->defs = defs;
    // 根据 defs->rank 的值设置 "__name__" 键在 fp->dict 中的值
    if (defs->rank == -1) {
      PyDict_SetItemString(fp->dict, "__name__", PyUnicode_FromFormat("function %s", defs->name));
    } else if (defs->rank == 0) {
      PyDict_SetItemString(fp->dict, "__name__", PyUnicode_FromFormat("scalar %s", defs->name));
    } else {
      PyDict_SetItemString(fp->dict, "__name__", PyUnicode_FromFormat("array %s", defs->name));
    }
    // 返回创建的 PyFortranObject 对象
    return (PyObject *)fp;
}

/* Fortran methods */

static void
fortran_dealloc(PyFortranObject *fp)
{
    // 释放 fp->dict 指向的对象
    Py_XDECREF(fp->dict);
    // 释放 fp 指向的对象
    PyObject_Del(fp);
}

/* Returns number of bytes consumed from buf, or -1 on error. */
static Py_ssize_t
format_def(char *buf, Py_ssize_t size, FortranDataDef def)
{
    char *p = buf;
    int i;
    npy_intp n;

    // 将数组格式的描述写入 buf 中，返回写入的字节数或者 -1 表示出错
    n = PyOS_snprintf(p, size, "array(%" NPY_INTP_FMT, def.dims.d[0]);
    if (n < 0 || n >= size) {
        return -1;
    }
    p += n;
    size -= n;

    // 循环将数组维度信息写入 buf 中
    for (i = 1; i < def.rank; i++) {
        n = PyOS_snprintf(p, size, ",%" NPY_INTP_FMT, def.dims.d[i]);
        if (n < 0 || n >= size) {
            return -1;
        }
        p += n;
        size -= n;
    }

    // 检查 buf 是否还有足够空间写入字符 ')'
    if (size <= 0) {
        return -1;
    }

    *p++ = ')';
    size--;

    // 如果 def.data 为 NULL，则在 buf 中添加字符串 ", not allocated"
    if (def.data == NULL) {
        static const char notalloc[] = ", not allocated";
        if ((size_t)size < sizeof(notalloc)) {
            return -1;
        }
        memcpy(p, notalloc, sizeof(notalloc));
        p += sizeof(notalloc);
        size -= sizeof(notalloc);
    }

    // 返回 buf 中的有效字节数
    return p - buf;
}

static PyObject *
fortran_doc(FortranDataDef def)
{
    char *buf, *p;
    PyObject *s = NULL;
    Py_ssize_t n, origsize, size = 100;

    // 根据 def.doc 的大小调整 size
    if (def.doc != NULL) {
        size += strlen(def.doc);
    }
    origsize = size;
    // 分配内存给 buf
    buf = p = (char *)PyMem_Malloc(size);
    if (buf == NULL) {
        return PyErr_NoMemory();
    }

    // 根据 def.rank 的值生成文档信息
    if (def.rank == -1) {
        if (def.doc) {
            // 将 def.doc 的内容复制到 buf 中
            n = strlen(def.doc);
            if (n > size) {
                goto fail;
            }
            memcpy(p, def.doc, n);
            p += n;
            size -= n;
        }
        else {
            // 如果没有 def.doc，则生成默认的描述信息
            n = PyOS_snprintf(p, size, "%s - no docs available", def.name);
            if (n < 0 || n >= size) {
                goto fail;
            }
            p += n;
            size -= n;
        }
    }
    else {
        // 根据数据类型定义创建一个 PyArray_Descr 结构体指针 d
        PyArray_Descr *d = PyArray_DescrFromType(def.type);
        // 使用 PyArray_Descr 结构体中的数据类型字符和定义的名称格式化输出到 p 所指向的 buf 中
        n = PyOS_snprintf(p, size, "%s : '%c'-", def.name, d->type);
        // 释放 PyArray_Descr 结构体指针 d
        Py_DECREF(d);
        // 如果格式化失败或者输出的字符数超过了 size，跳转到失败处理
        if (n < 0 || n >= size) {
            goto fail;
        }
        // 更新 p 的位置到新的可写入位置
        p += n;
        // 更新 size 的可用大小
        size -= n;

        // 如果 def.data 为 NULL，则调用 format_def 函数格式化输出到 p 所指向的 buf 中
        if (def.data == NULL) {
            n = format_def(p, size, def);
            if (n < 0) {
                goto fail;
            }
            p += n;
            size -= n;
        }
        // 如果 def.rank 大于 0，则调用 format_def 函数格式化输出到 p 所指向的 buf 中
        else if (def.rank > 0) {
            n = format_def(p, size, def);
            if (n < 0) {
                goto fail;
            }
            p += n;
            size -= n;
        }
        // 否则，输出字符串 "scalar" 到 p 所指向的 buf 中
        else {
            n = strlen("scalar");
            // 如果 size 小于需要写入的字符串长度，跳转到失败处理
            if (size < n) {
                goto fail;
            }
            // 将字符串 "scalar" 复制到 p 所指向的 buf 中
            memcpy(p, "scalar", n);
            // 更新 p 的位置到新的可写入位置
            p += n;
            // 更新 size 的可用大小
            size -= n;
        }
    }
    // 如果 size 小于等于 1，跳转到失败处理
    if (size <= 1) {
        goto fail;
    }
    // 在 p 所指向的 buf 末尾写入换行符 '\n'
    *p++ = '\n';
    // 更新 size 的可用大小
    size--;

    /* p 现在指向 buf 中字符串的末尾的下一个位置 */

    // 使用 buf 和 p 指向的字符串长度创建一个 Python Unicode 对象 s
    s = PyUnicode_FromStringAndSize(buf, p - buf);

    // 释放动态分配的 buf 内存
    PyMem_Free(buf);
    // 返回创建的 Python Unicode 对象 s
    return s;
fail:
    /* 打印错误消息到标准错误流，指出文档字符串长度超出允许的大小 */
    fprintf(stderr,
            "fortranobject.c: fortran_doc: len(p)=%zd>%zd=size:"
            " too long docstring required, increase size\n",
            p - buf, origsize);
    /* 释放之前分配的内存 */
    PyMem_Free(buf);
    /* 返回空指针，表示函数执行失败 */
    return NULL;
}

static FortranDataDef *save_def; /* 保存可分配数组的指针 */

static void
set_data(char *d, npy_intp *f)
{           /* 来自Fortran的回调函数 */
    /* 如果 f 不为零，说明在Fortran中 d 已经被分配了内存 */
    if (*f) /* In fortran f=allocated(d) */
        save_def->data = d;
    else
        save_def->data = NULL;
    /* printf("set_data: d=%p,f=%d\n",d,*f); */
}

static PyObject *
fortran_getattr(PyFortranObject *fp, char *name)
{
    int i, j, k, flag;
    /* 检查对象字典是否为空 */
    if (fp->dict != NULL) {
        /* 尝试从对象字典中获取名为 'name' 的项 */
        PyObject *v = _PyDict_GetItemStringWithError(fp->dict, name);
        /* 如果获取失败并且发生了错误，则返回空指针 */
        if (v == NULL && PyErr_Occurred()) {
            return NULL;
        }
        /* 如果获取成功，则增加引用计数并返回该项 */
        else if (v != NULL) {
            Py_INCREF(v);
            return v;
        }
    }
    /* 在对象的定义列表中查找与 'name' 匹配的项 */
    for (i = 0, j = 1; i < fp->len && (j = strcmp(name, fp->defs[i].name));
         i++)
        ;
    /* 如果找到了匹配项 */
    if (j == 0)
        /* 检查定义是否为F90可分配数组 */
        if (fp->defs[i].rank != -1) { /* F90 allocatable array */
            /* 如果回调函数不为空，则进行数组维度重置 */
            if (fp->defs[i].func == NULL)
                return NULL;
            /* 将数组维度设置为-1 */
            for (k = 0; k < fp->defs[i].rank; ++k) fp->defs[i].dims.d[k] = -1;
            /* 保存当前定义以便后续使用 */
            save_def = &fp->defs[i];
            /* 调用回调函数设置数据 */
            (*(fp->defs[i].func))(&fp->defs[i].rank, fp->defs[i].dims.d,
                                  set_data, &flag);
            /* 根据返回的标志检查数组状态 */
            if (flag == 2)
                k = fp->defs[i].rank + 1;
            else
                k = fp->defs[i].rank;
            /* 如果数据不为空，则创建新的PyArray对象 */
            if (fp->defs[i].data != NULL) { /* array is allocated */
                PyObject *v = PyArray_New(
                        &PyArray_Type, k, fp->defs[i].dims.d, fp->defs[i].type,
                        NULL, fp->defs[i].data, 0, NPY_ARRAY_FARRAY, NULL);
                /* 如果创建失败，则返回空指针 */
                if (v == NULL)
                    return NULL;
                /* 增加新对象的引用计数并返回 */
                /* Py_INCREF(v); */
                return v;
            }
            else { /* 如果数据为空，则返回Py_None */
                Py_RETURN_NONE;
            }
        }
    /* 如果 'name' 是特殊属性 '__dict__'，返回对象字典 */
    if (strcmp(name, "__dict__") == 0) {
        Py_INCREF(fp->dict);
        return fp->dict;
    }
    /* 如果 'name' 是特殊属性 '__doc__' */
    if (strcmp(name, "__doc__") == 0) {
        /* 创建一个空的PyUnicode对象 */
        PyObject *s = PyUnicode_FromString(""), *s2, *s3;
        /* 遍历对象定义列表并获取其文档字符串 */
        for (i = 0; i < fp->len; i++) {
            s2 = fortran_doc(fp->defs[i]);
            s3 = PyUnicode_Concat(s, s2);
            Py_DECREF(s2);
            Py_DECREF(s);
            s = s3;
        }
        /* 将生成的文档字符串添加到对象字典中 */
        if (PyDict_SetItemString(fp->dict, name, s))
            return NULL;
        /* 返回文档字符串对象 */
        return s;
    }
    /* 如果 'name' 是特殊属性 '_cpointer'，且对象只有一个定义 */
    if ((strcmp(name, "_cpointer") == 0) && (fp->len == 1)) {
        /* 创建一个F2PyCapsule对象 */
        PyObject *cobj =
                F2PyCapsule_FromVoidPtr((void *)(fp->defs[0].data), NULL);
        /* 将F2PyCapsule对象添加到对象字典中 */
        if (PyDict_SetItemString(fp->dict, name, cobj))
            return NULL;
        /* 返回F2PyCapsule对象 */
        return cobj;
    }
    /* 如果 'name' 不是以上任何特殊属性，则使用通用的属性获取方法 */
    PyObject *str, *ret;
    /* 从 'name' 创建一个PyUnicode对象 */
    str = PyUnicode_FromString(name);
    /* 调用通用的属性获取方法获取属性值 */
    ret = PyObject_GenericGetAttr((PyObject *)fp, str);
    /* 释放PyUnicode对象 */
    Py_DECREF(str);
    /* 返回获取的属性值 */
    return ret;
}

static int
# 设置 Fortran 对象的属性值
fortran_setattr(PyFortranObject *fp, char *name, PyObject *v)
{
    int i, j, flag;
    PyArrayObject *arr = NULL;

    // 遍历 fp 对象的定义列表，查找属性名为 name 的定义
    for (i = 0, j = 1; i < fp->len && (j = strcmp(name, fp->defs[i].name)); i++)
        ;

    // 如果找到属性名为 name 的定义
    if (j == 0) {
        // 如果属性是一个可分配的数组
        if (fp->defs[i].rank == -1) {
            // 抛出异常，不能覆盖 Fortran 程序
            PyErr_SetString(PyExc_AttributeError, "over-writing fortran routine");
            return -1; // 返回错误标志
        }
        // 如果属性有关联的函数
        if (fp->defs[i].func != NULL) { /* is allocatable array */
            npy_intp dims[F2PY_MAX_DIMS];
            int k;
            save_def = &fp->defs[i];

            // 如果新值 v 不是 Py_None，设置新值（如果需要重新分配，请参阅生成的 f2py 代码以获取更多细节）
            if (v != Py_None) {
                for (k = 0; k < fp->defs[i].rank; k++) dims[k] = -1;
                // 从 Python 对象 v 创建 NumPy 数组 arr
                if ((arr = array_from_pyobj(fp->defs[i].type, dims,
                                            fp->defs[i].rank, F2PY_INTENT_IN,
                                            v)) == NULL)
                    return -1;
                // 调用属性关联的函数来设置数据
                (*(fp->defs[i].func))(&fp->defs[i].rank, PyArray_DIMS(arr),
                                      set_data, &flag);
            }
            else { // 如果 v 是 Py_None，进行释放操作
                for (k = 0; k < fp->defs[i].rank; k++) dims[k] = 0;
                // 调用属性关联的函数来释放数据
                (*(fp->defs[i].func))(&fp->defs[i].rank, dims, set_data,
                                      &flag);
                for (k = 0; k < fp->defs[i].rank; k++) dims[k] = -1;
            }
            // 将维度信息复制回属性定义中
            memcpy(fp->defs[i].dims.d, dims,
                   fp->defs[i].rank * sizeof(npy_intp));
        }
        else { // 如果属性不是可分配的数组
            // 从 Python 对象 v 创建 NumPy 数组 arr
            if ((arr = array_from_pyobj(fp->defs[i].type, fp->defs[i].dims.d,
                                        fp->defs[i].rank, F2PY_INTENT_IN,
                                        v)) == NULL)
                return -1;
        }
        
        // 如果属性有关联的数据
        if (fp->defs[i].data != NULL) {
            // 计算 Fortran 数组的总大小
            npy_intp s = PyArray_MultiplyList(fp->defs[i].dims.d,
                                              PyArray_NDIM(arr));
            if (s == -1)
                s = PyArray_MultiplyList(PyArray_DIMS(arr), PyArray_NDIM(arr));
            // 将 Python 对象的数据复制到 Fortran 数组中
            if (s < 0 || (memcpy(fp->defs[i].data, PyArray_DATA(arr),
                                 s * PyArray_ITEMSIZE(arr))) == NULL) {
                if ((PyObject *)arr != v) {
                    Py_DECREF(arr);
                }
                return -1;
            }
            if ((PyObject *)arr != v) {
                Py_DECREF(arr);
            }
        }
        else
            return (fp->defs[i].func == NULL ? -1 : 0);
        return 0; // 操作成功
    }
    
    // 如果找不到属性名为 name 的定义，并且 fp 对象的字典为空，则创建一个新的字典
    if (fp->dict == NULL) {
        fp->dict = PyDict_New();
        if (fp->dict == NULL)
            return -1;
    }
    # 如果 v 为 NULL，则执行以下操作
    if (v == NULL) {
        # 尝试从字典 fp->dict 中删除键名为 name 的项
        int rv = PyDict_DelItemString(fp->dict, name);
        # 如果删除操作返回值 rv 小于 0，表示删除失败
        if (rv < 0)
            # 设置错误信息，指明试图删除不存在的 Fortran 属性
            PyErr_SetString(PyExc_AttributeError,
                            "delete non-existing fortran attribute");
        # 返回删除操作的结果值 rv
        return rv;
    }
    # 如果 v 不为 NULL，则执行以下操作
    else
        # 向字典 fp->dict 中设置键名为 name 的项的值为 v
        return PyDict_SetItemString(fp->dict, name, v);
static PyObject *
fortran_call(PyFortranObject *fp, PyObject *arg, PyObject *kw)
{
    int i = 0;
    /* 检查调用的 Fortran 对象是否为 Fortran 程序 */
    if (fp->defs[i].rank == -1) { /* is Fortran routine */
        if (fp->defs[i].func == NULL) {
            PyErr_Format(PyExc_RuntimeError, "no function to call");
            return NULL;
        }
        else if (fp->defs[i].data == NULL)
            /* 调用没有数据的 Fortran 子例程 */
            return (*((fortranfunc)(fp->defs[i].func)))((PyObject *)fp, arg,
                                                        kw, NULL);
        else
            /* 调用带有数据的 Fortran 子例程 */
            return (*((fortranfunc)(fp->defs[i].func)))(
                    (PyObject *)fp, arg, kw, (void *)fp->defs[i].data);
    }
    /* 如果不是 Fortran 对象，返回类型错误 */
    PyErr_Format(PyExc_TypeError, "this fortran object is not callable");
    return NULL;
}

static PyObject *
fortran_repr(PyFortranObject *fp)
{
    PyObject *name = NULL, *repr = NULL;
    // 获取对象的名称属性
    name = PyObject_GetAttrString((PyObject *)fp, "__name__");
    PyErr_Clear();
    // 根据对象名称生成对象的字符串表示
    if (name != NULL && PyUnicode_Check(name)) {
        repr = PyUnicode_FromFormat("<fortran %U>", name);
    }
    else {
        repr = PyUnicode_FromString("<fortran object>");
    }
    Py_XDECREF(name);
    return repr;
}

PyTypeObject PyFortran_Type = {
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "fortran",
        .tp_basicsize = sizeof(PyFortranObject),
        .tp_dealloc = (destructor)fortran_dealloc,
        .tp_getattr = (getattrfunc)fortran_getattr,
        .tp_setattr = (setattrfunc)fortran_setattr,
        .tp_repr = (reprfunc)fortran_repr,
        .tp_call = (ternaryfunc)fortran_call,
};

/************************* f2py_report_atexit *******************************/

#ifdef F2PY_REPORT_ATEXIT
static int passed_time = 0;
static int passed_counter = 0;
static int passed_call_time = 0;
static struct timeb start_time;
static struct timeb stop_time;
static struct timeb start_call_time;
static struct timeb stop_call_time;
static int cb_passed_time = 0;
static int cb_passed_counter = 0;
static int cb_passed_call_time = 0;
static struct timeb cb_start_time;
static struct timeb cb_stop_time;
static struct timeb cb_start_call_time;
static struct timeb cb_stop_call_time;

extern void
f2py_start_clock(void)
{
    // 开始计时
    ftime(&start_time);
}
extern void
f2py_start_call_clock(void)
{
    // 停止当前计时并开始调用计时
    f2py_stop_clock();
    ftime(&start_call_time);
}
extern void
f2py_stop_clock(void)
{
    // 停止计时并计算经过的时间
    ftime(&stop_time);
    passed_time += 1000 * (stop_time.time - start_time.time);
    passed_time += stop_time.millitm - start_time.millitm;
}
extern void
f2py_stop_call_clock(void)
{
    // 停止调用计时并计算经过的时间和调用次数
    ftime(&stop_call_time);
    passed_call_time += 1000 * (stop_call_time.time - start_call_time.time);
    passed_call_time += stop_call_time.millitm - start_call_time.millitm;
    passed_counter += 1;
    // 重新开始总计时
    f2py_start_clock();
}

extern void
f2py_cb_start_clock(void)
{
    // 开始回调计时
    ftime(&cb_start_time);
}
extern void
static int f2py_report_on_exit_been_here = 0;
extern void
f2py_report_on_exit(int exit_flag, void *name)
{
    // 如果已经在此函数中打印过一次报告，则只输出名称并返回
    if (f2py_report_on_exit_been_here) {
        fprintf(stderr, "             %s\n", (char *)name);
        return;
    }
    // 标记已经在此函数中打印过一次报告
    f2py_report_on_exit_been_here = 1;

    // 输出性能报告表头
    fprintf(stderr, "                      /-----------------------\\\n");
    fprintf(stderr, "                     < F2PY performance report >\n");
    fprintf(stderr, "                      \\-----------------------/\n");

    // 输出总体调用时间报告
    fprintf(stderr, "Overall time spent in ...\n");
    fprintf(stderr, "(a) wrapped (Fortran/C) functions           : %8d msec\n",
            passed_call_time);
    fprintf(stderr, "(b) f2py interface,           %6d calls  : %8d msec\n",
            passed_counter, passed_time);
    fprintf(stderr, "(c) call-back (Python) functions            : %8d msec\n",
            cb_passed_call_time);
    fprintf(stderr, "(d) f2py call-back interface, %6d calls  : %8d msec\n",
            cb_passed_counter, cb_passed_time);

    // 输出实际耗时较长的 wrapped 函数的时间
    fprintf(stderr,
            "(e) wrapped (Fortran/C) functions (actual) : %8d msec\n\n",
            passed_call_time - cb_passed_call_time - cb_passed_time);

    // 输出退出时的提示信息和模块名称
    fprintf(stderr,
            "Use -DF2PY_REPORT_ATEXIT_DISABLE to disable this message.\n");
    fprintf(stderr, "Exit status: %d\n", exit_flag);
    fprintf(stderr, "Modules    : %s\n", (char *)name);
}
/*
 * File: array_from_pyobj.c
 *
 * Description:
 * ------------
 * Provides array_from_pyobj function that returns a contiguous array
 * object with the given dimensions and required storage order, either
 * in row-major (C) or column-major (Fortran) order. The function
 * array_from_pyobj is very flexible about its Python object argument
 * that can be any number, list, tuple, or array.
 *
 * array_from_pyobj is used in f2py generated Python extension
 * modules.
 *
 * Author: Pearu Peterson <pearu@cens.ioc.ee>
 * Created: 13-16 January 2002
 * $Id: fortranobject.c,v 1.52 2005/07/11 07:44:20 pearu Exp $
 */

// 静态函数声明：检查并修正数组维度
static int check_and_fix_dimensions(const PyArrayObject* arr,
                                    const int rank,
                                    npy_intp *dims,
                                    const char *errmess);

// 查找第一个负维度的索引
static int
find_first_negative_dimension(const int rank, const npy_intp *dims)
{
    int i;
    for (i = 0; i < rank; ++i) {
        if (dims[i] < 0) {
            return i;
        }
    }
    return -1;
}

#ifdef DEBUG_COPY_ND_ARRAY
// 输出数组维度信息
void
dump_dims(int rank, npy_intp const *dims)
{
    int i;
    printf("[");
    for (i = 0; i < rank; ++i) {
        printf("%3" NPY_INTP_FMT, dims[i]);
    }
    printf("]\n");
}

// 输出数组属性信息
void
dump_attrs(const PyArrayObject *obj)
{
    const PyArrayObject_fields *arr = (const PyArrayObject_fields *)obj;
    int rank = PyArray_NDIM(arr);
    npy_intp size = PyArray_Size((PyObject *)arr);
    printf("\trank = %d, flags = %d, size = %" NPY_INTP_FMT "\n", rank,
           arr->flags, size);
    printf("\tstrides = ");
    dump_dims(rank, arr->strides);
    printf("\tdimensions = ");
    dump_dims(rank, arr->dimensions);
}
#endif

// 定义宏：交换两个数组对象的数据及属性
#define SWAPTYPE(a, b, t) \
    {                     \
        t c;              \
        c = (a);          \
        (a) = (b);        \
        (b) = c;          \
    }

// 交换两个数组对象的数据及属性
static int
swap_arrays(PyArrayObject *obj1, PyArrayObject *obj2)
{
    PyArrayObject_fields *arr1 = (PyArrayObject_fields *)obj1,
                         *arr2 = (PyArrayObject_fields *)obj2;
    SWAPTYPE(arr1->data, arr2->data, char *);
    SWAPTYPE(arr1->nd, arr2->nd, int);
    SWAPTYPE(arr1->dimensions, arr2->dimensions, npy_intp *);
    SWAPTYPE(arr1->strides, arr2->strides, npy_intp *);
    SWAPTYPE(arr1->base, arr2->base, PyObject *);
    SWAPTYPE(arr1->descr, arr2->descr, PyArray_Descr *);
    SWAPTYPE(arr1->flags, arr2->flags, int);
    /* SWAPTYPE(arr1->weakreflist,arr2->weakreflist,PyObject*); */
    return 0;
}

// 定义宏：判断数组对象是否与指定数据类型兼容
#define ARRAY_ISCOMPATIBLE(arr,type_num)                                \
    ((PyArray_ISINTEGER(arr) && PyTypeNum_ISINTEGER(type_num)) ||     \
     (PyArray_ISFLOAT(arr) && PyTypeNum_ISFLOAT(type_num)) ||         \
     (PyArray_ISCOMPLEX(arr) && PyTypeNum_ISCOMPLEX(type_num)) ||     \
     (PyArray_ISBOOL(arr) && PyTypeNum_ISBOOL(type_num)) ||           \
     (PyArray_ISSTRING(arr) && PyTypeNum_ISSTRING(type_num)))

// 获取对象的元素大小
static int
get_elsize(PyObject *obj) {
  /*
    /* 
    get_elsize 函数根据输入的 Python 对象确定数组元素的大小。如果成功，则返回元素大小，否则返回 -1。

    支持的输入类型包括：numpy.ndarray, bytes, str, tuple, list.
    */

    if (PyArray_Check(obj)) {
        // 如果输入对象是 numpy 数组，则返回数组元素的大小
        return PyArray_ITEMSIZE((PyArrayObject *)obj);
    } else if (PyBytes_Check(obj)) {
        // 如果输入对象是字节对象（bytes），则返回其大小
        return PyBytes_GET_SIZE(obj);
    } else if (PyUnicode_Check(obj)) {
        // 如果输入对象是 Unicode 对象（str），则返回其长度
        return PyUnicode_GET_LENGTH(obj);
    } else if (PySequence_Check(obj)) {
        // 如果输入对象是序列（tuple 或 list）
        PyObject* fast = PySequence_Fast(obj, "f2py:fortranobject.c:get_elsize");
        if (fast != NULL) {
            // 获取序列的长度
            Py_ssize_t i, n = PySequence_Fast_GET_SIZE(fast);
            int sz, elsize = 0;
            // 遍历序列中的每个元素
            for (i=0; i<n; i++) {
                // 递归调用 get_elsize 函数，获取每个元素的大小
                sz = get_elsize(PySequence_Fast_GET_ITEM(fast, i) /* borrowed */);
                // 更新 elsize 为最大的元素大小
                if (sz > elsize) {
                    elsize = sz;
                }
            }
            Py_DECREF(fast);
            return elsize;
        }
    }
    // 如果无法确定输入对象的类型或者其他情况，返回 -1
    return -1;
    /*
    */
extern PyArrayObject *
ndarray_from_pyobj(const int type_num,
                   const int elsize_,
                   npy_intp *dims,
                   const int rank,
                   const int intent,
                   PyObject *obj,
                   const char *errmess) {
    /*
     * 从 Python 对象中创建一个具有指定元素类型和形状的数组，并考虑数组的使用意图。
     *
     * - 元素类型由 type_num 和 elsize 定义
     * - 形状由 dims 和 rank 定义
     *
     * ndarray_from_pyobj 用于将 Python 对象参数转换为具有给定类型和形状的 numpy ndarray，
     * 以便将数据传递给与 Fortran 或 C 函数交互的场合。
     *
     * 如果 errmess 不为 NULL，则包含在此函数内部触发异常的错误消息前缀。
     *
     * 负的 elsize 值表示 elsize 在运行时从 Python 对象中确定。
     *
     * 字符串类型 (type_num == NPY_STRING) 没有固定的元素大小，默认情况下类型对象将其设置为 0。
     * 因此，对于字符串类型，必须使用 elsize 参数。对于其他类型，elsize 值被忽略。
     *
     * NumPy 将固定宽度字符串的类型定义为 dtype('S<width>')。此外，还有 dtype('c')，显示为 dtype('S1')
     * （它们具有相同的 type_num 值），但实际上不同（.char 属性为 'S' 或 'c'）。
     *
     * 在 Fortran 中，字符数组和字符串是不同的概念。Fortran 类型、NumPy dtypes 和 type_num-elsize 对之间的关系定义如下：
     *
     * character*5 foo     | dtype('S5')  | elsize=5, shape=()
     * character(5) foo    | dtype('S1')  | elsize=1, shape=(5)
     * character*5 foo(n)  | dtype('S5')  | elsize=5, shape=(n,)
     * character(5) foo(n) | dtype('S1')  | elsize=1, shape=(5, n)
     * character*(*) foo   | dtype('S')   | elsize=-1, shape=()
     *
     * 引用计数注意事项
     * -----------------
     *
     * 如果调用者将数组返回给 Python，则必须使用 Py_BuildValue("N",arr)。否则，如果 obj!=arr，则调用者必须调用 Py_DECREF(arr)。
     *
     * 使用意图（intent）（缓存、输出等）的注意事项
     * ------------------------------------------
     *
     * 当返回 intent(cache) 数组时，不要期望数据是正确的。
     *
     */
    char mess[F2PY_MESSAGE_BUFFER_SIZE]; // 用于存储错误消息的缓冲区
    PyArrayObject *arr = NULL; // 初始化一个空的 PyArrayObject 对象
    int elsize = (elsize_ < 0 ? get_elsize(obj) : elsize_); // 确定元素大小，如果 elsize_ 小于 0，则从 obj 中获取

    // 如果 elsize 仍然小于 0，表示无法确定元素大小
    if (elsize < 0) {
      // 如果有错误消息，将其复制到 mess 中
      if (errmess != NULL) {
        strcpy(mess, errmess);
      }
      // 将错误消息格式化添加到 mess 中
      sprintf(mess + strlen(mess),
              " -- failed to determine element size from %s",
              Py_TYPE(obj)->tp_name);
      // 设置 Python 异常并返回空指针
      PyErr_SetString(PyExc_SystemError, mess);
      return NULL;
    }
    PyArray_Descr * descr = get_descr_from_type_and_elsize(type_num, elsize);  // 根据数据类型和元素大小获取描述符，返回新引用
    if (descr == NULL) {
      return NULL;
    }
    elsize = PyDataType_ELSIZE(descr);  // 获取描述符中的元素大小
    if ((intent & F2PY_INTENT_HIDE)
        || ((intent & F2PY_INTENT_CACHE) && (obj == Py_None))
        || ((intent & F2PY_OPTIONAL) && (obj == Py_None))
        ) {
        /* intent(cache), optional, intent(hide) */
        int ineg = find_first_negative_dimension(rank, dims);  // 查找维度数组中第一个负数的索引
        if (ineg >= 0) {
            int i;
            strcpy(mess, "failed to create intent(cache|hide)|optional array"
                   "-- must have defined dimensions but got (");
            for(i = 0; i < rank; ++i)
                sprintf(mess + strlen(mess), "%" NPY_INTP_FMT ",", dims[i]);  // 格式化拼接维度信息到错误消息中
            strcat(mess, ")");
            PyErr_SetString(PyExc_ValueError, mess);  // 设置值错误异常并传入错误消息
            Py_DECREF(descr);  // 减少描述符的引用计数
            return NULL;
        }
        arr = (PyArrayObject *)                                      \
          PyArray_NewFromDescr(&PyArray_Type, descr, rank, dims,
                               NULL, NULL, !(intent & F2PY_INTENT_C), NULL);  // 根据描述符创建新的数组对象
        if (arr == NULL) {
          Py_DECREF(descr);  // 创建失败时释放描述符的引用
          return NULL;
        }
        if (PyArray_ITEMSIZE(arr) != elsize) {
          strcpy(mess, "failed to create intent(cache|hide)|optional array");
          sprintf(mess+strlen(mess)," -- expected elsize=%d got %" NPY_INTP_FMT, elsize, (npy_intp)PyArray_ITEMSIZE(arr));  // 格式化拼接预期和实际的元素大小到错误消息中
          PyErr_SetString(PyExc_ValueError,mess);  // 设置值错误异常并传入错误消息
          Py_DECREF(arr);  // 减少数组对象的引用计数
          return NULL;
        }
        if (!(intent & F2PY_INTENT_CACHE)) {
          PyArray_FILLWBYTE(arr, 0);  // 如果没有缓存意图，则用零填充数组
        }
        return arr;  // 返回创建的数组对象
    }

    if ((intent & F2PY_INTENT_INOUT) || (intent & F2PY_INTENT_INPLACE) ||
        (intent & F2PY_INTENT_CACHE)) {
        PyErr_Format(PyExc_TypeError,
                     "failed to initialize intent(inout|inplace|cache) "
                     "array, input '%s' object is not an array",
                     Py_TYPE(obj)->tp_name);  // 格式化错误消息，指示输入对象不是数组
        Py_DECREF(descr);  // 减少描述符的引用计数
        return NULL;
    }
    {
        // 执行F2PY_REPORT_ON_ARRAY_COPY_FROMANY宏，可能生成有关数组复制的报告
        F2PY_REPORT_ON_ARRAY_COPY_FROMANY;
        // 将给定的Python对象转换为NumPy数组对象
        arr = (PyArrayObject *)PyArray_FromAny(
                obj, descr, 0, 0,
                // 根据intent标志选择数组存储方式：C顺序或Fortran顺序，并强制类型转换
                ((intent & F2PY_INTENT_C) ? NPY_ARRAY_CARRAY
                                          : NPY_ARRAY_FARRAY) |
                        NPY_ARRAY_FORCECAST,
                NULL);
        // 警告：在NPY_STRING的情况下，PyArray_FromAny可能会重置descr->elsize，
        // 例如dtype('S0')可能变成dtype('S1')。
        if (arr == NULL) {
            // 如果转换失败，释放描述符对象并返回NULL
            Py_DECREF(descr);
            return NULL;
        }
        // 如果数组类型不是NPY_STRING且数组项大小不等于期望的elsize
        if (type_num != NPY_STRING && PyArray_ITEMSIZE(arr) != elsize) {
            // 这是内部的健全性检查：在函数开头已将elsize设置为descr->elsize。
            // 设置错误信息说明预期的elsize和实际获取的数组项大小不匹配
            strcpy(mess, "failed to initialize intent(in) array");
            sprintf(mess + strlen(mess),
                    " -- expected elsize=%d got %" NPY_INTP_FMT, elsize,
                    (npy_intp)PyArray_ITEMSIZE(arr));
            // 设置异常，并释放数组对象后返回NULL
            PyErr_SetString(PyExc_ValueError, mess);
            Py_DECREF(arr);
            return NULL;
        }
        // 检查并修正数组的维度，如果出错则释放数组对象并返回NULL
        if (check_and_fix_dimensions(arr, rank, dims, errmess)) {
            Py_DECREF(arr);
            return NULL;
        }
        // 如果以上检查都通过，返回转换后的NumPy数组对象
        return arr;
    }
}

extern PyArrayObject *
array_from_pyobj(const int type_num,
                 npy_intp *dims,
                 const int rank,
                 const int intent,
                 PyObject *obj) {
  /*
    Same as ndarray_from_pyobj but with elsize determined from type,
    if possible. Provided for backward compatibility.
   */
  // 根据给定的类型编号获取对应的数组描述符
  PyArray_Descr* descr = PyArray_DescrFromType(type_num);
  // 获取描述符的元素大小
  int elsize = PyDataType_ELSIZE(descr);
  // 减少描述符的引用计数，防止内存泄漏
  Py_DECREF(descr);
  // 调用ndarray_from_pyobj函数创建并返回NumPy数组对象
  return ndarray_from_pyobj(type_num, elsize, dims, rank, intent, obj, NULL);
}

/*****************************************/
/* Helper functions for array_from_pyobj */
/*****************************************/

static int
check_and_fix_dimensions(const PyArrayObject* arr, const int rank,
                         npy_intp *dims, const char *errmess)
{
    /*
     * This function fills in blanks (that are -1's) in dims list using
     * the dimensions from arr. It also checks that non-blank dims will
     * match with the corresponding values in arr dimensions.
     *
     * Returns 0 if the function is successful.
     *
     * If an error condition is detected, an exception is set and 1 is
     * returned.
     */
    // 定义错误消息的字符数组
    char mess[F2PY_MESSAGE_BUFFER_SIZE];
    // 计算arr的总元素数
    const npy_intp arr_size =
            (PyArray_NDIM(arr)) ? PyArray_Size((PyObject *)arr) : 1;
#ifdef DEBUG_COPY_ND_ARRAY
    // 调试模式下打印数组的属性
    dump_attrs(arr);
    printf("check_and_fix_dimensions:init: dims=");
    // 调试模式下打印维度信息
    dump_dims(rank, dims);
#endif
    # 如果要求的维度rank大于数组arr的维度数，进行以下处理：[1,2] -> [[1],[2]]; 1 -> [[1]]
    # 这段代码用于调整数组的维度匹配要求的rank
    if (rank > PyArray_NDIM(arr)) { /* [1,2] -> [[1],[2]]; 1 -> [[1]]  */
        npy_intp new_size = 1;  // 初始化新数组的总大小为1
        int free_axe = -1;  // 自由轴的索引，初始化为-1表示无自由轴
        int i;
        npy_intp d;
        // 填充dims中为-1或0的维度，并检查维度是否符合要求，计算新的总大小
        for (i = 0; i < PyArray_NDIM(arr); ++i) {
            d = PyArray_DIM(arr, i);  // 获取数组arr在第i维的大小
            if (dims[i] >= 0) {  // 如果dims中第i维要求大于等于0
                if (d > 1 && dims[i] != d) {  // 如果数组arr在第i维的大小大于1且与要求的dims[i]不符
                    PyErr_Format(
                            PyExc_ValueError,
                            "%d-th dimension must be fixed to %" NPY_INTP_FMT
                            " but got %" NPY_INTP_FMT "\n",
                            i, dims[i], d);
                    return 1;  // 报错并返回1，表示处理失败
                }
                if (!dims[i])
                    dims[i] = 1;  // 如果dims[i]为0，则设置为1
            }
            else {
                dims[i] = d ? d : 1;  // 如果dims[i]小于0，则设为数组arr在第i维的大小，如果为0则设为1
            }
            new_size *= dims[i];  // 计算新数组的总大小
        }
        // 对于超出数组arr维度数的维度rank，进行额外处理
        for (i = PyArray_NDIM(arr); i < rank; ++i)
            if (dims[i] > 1) {  // 如果dims中第i维要求大于1
                PyErr_Format(PyExc_ValueError,
                             "%d-th dimension must be %" NPY_INTP_FMT
                             " but got 0 (not defined).\n",
                             i, dims[i]);
                return 1;  // 报错并返回1，表示处理失败
            }
            else if (free_axe < 0)
                free_axe = i;  // 记录第一个自由轴的索引
            else
                dims[i] = 1;  // 其余维度设为1
        // 如果存在自由轴free_axe
        if (free_axe >= 0) {
            dims[free_axe] = arr_size / new_size;  // 计算自由轴的大小
            new_size *= dims[free_axe];  // 更新新数组的总大小
        }
        // 如果新的总大小与原数组的大小不匹配，报错
        if (new_size != arr_size) {
            PyErr_Format(PyExc_ValueError,
                         "unexpected array size: new_size=%" NPY_INTP_FMT
                         ", got array with arr_size=%" NPY_INTP_FMT
                         " (maybe too many free indices)\n",
                         new_size, arr_size);
            return 1;  // 报错并返回1，表示处理失败
        }
    }
    # 如果排名（维度数）等于数组的维度数
    else if (rank == PyArray_NDIM(arr)) {
        # 计算新的数组大小为1
        npy_intp new_size = 1;
        int i;
        npy_intp d;
        # 遍历数组的每一个维度
        for (i = 0; i < rank; ++i) {
            # 获取数组在第i维度上的大小
            d = PyArray_DIM(arr, i);
            # 如果给定的维度值大于等于0
            if (dims[i] >= 0) {
                # 如果数组的维度大于1且不等于给定的维度值
                if (d > 1 && d != dims[i]) {
                    # 如果错误消息不为空，则复制到mess中
                    if (errmess != NULL) {
                        strcpy(mess, errmess);
                    }
                    # 将错误消息格式化加入到mess中
                    sprintf(mess + strlen(mess),
                            " -- %d-th dimension must be fixed to %"
                            NPY_INTP_FMT " but got %" NPY_INTP_FMT,
                            i, dims[i], d);
                    # 设置Python异常并返回1表示出错
                    PyErr_SetString(PyExc_ValueError, mess);
                    return 1;
                }
                # 如果给定的维度值为0，则将其设为1
                if (!dims[i])
                    dims[i] = 1;
            }
            else
                # 否则，将数组当前维度大小赋值给给定的维度值
                dims[i] = d;
            # 计算新的数组大小
            new_size *= dims[i];
        }
        # 如果新的数组大小与原数组大小不一致
        if (new_size != arr_size) {
            # 抛出格式化的异常，指出意外的数组大小
            PyErr_Format(PyExc_ValueError,
                         "unexpected array size: new_size=%" NPY_INTP_FMT
                         ", got array with arr_size=%" NPY_INTP_FMT "\n",
                         new_size, arr_size);
            # 返回1表示出错
            return 1;
        }
    }
    }
#ifdef DEBUG_COPY_ND_ARRAY
    // 如果定义了 DEBUG_COPY_ND_ARRAY 宏，则输出调试信息
    printf("check_and_fix_dimensions:end: dims=");
    // 打印调试信息，显示维度信息
    dump_dims(rank, dims);
#endif
    // 返回操作成功的标志
    return 0;
}

/* End of file: array_from_pyobj.c */

/************************* copy_ND_array *******************************/

extern int
copy_ND_array(const PyArrayObject *arr, PyArrayObject *out)
{
    // 报告在从 arr 复制数组到 out 时的状态
    F2PY_REPORT_ON_ARRAY_COPY_FROMARR;
    // 调用 NumPy 提供的函数将 arr 复制到 out 中
    return PyArray_CopyInto(out, (PyArrayObject *)arr);
}

/********************* Various utility functions ***********************/

extern int
f2py_describe(PyObject *obj, char *buf) {
  /*
    Write the description of a Python object to buf. The caller must
    provide buffer with size sufficient to write the description.

    Return 1 on success.
  */
  // 本地缓冲区，用于存储描述信息
  char localbuf[F2PY_MESSAGE_BUFFER_SIZE];
  // 根据不同类型的 Python 对象生成描述信息
  if (PyBytes_Check(obj)) {
    sprintf(localbuf, "%d-%s", (npy_int)PyBytes_GET_SIZE(obj), Py_TYPE(obj)->tp_name);
  } else if (PyUnicode_Check(obj)) {
    sprintf(localbuf, "%d-%s", (npy_int)PyUnicode_GET_LENGTH(obj), Py_TYPE(obj)->tp_name);
  } else if (PyArray_CheckScalar(obj)) {
    PyArrayObject* arr = (PyArrayObject*)obj;
    sprintf(localbuf, "%c%" NPY_INTP_FMT "-%s-scalar", PyArray_DESCR(arr)->kind, PyArray_ITEMSIZE(arr), Py_TYPE(obj)->tp_name);
  } else if (PyArray_Check(obj)) {
    int i;
    PyArrayObject* arr = (PyArrayObject*)obj;
    strcpy(localbuf, "(");
    for (i=0; i<PyArray_NDIM(arr); i++) {
      if (i) {
        strcat(localbuf, " ");
      }
      sprintf(localbuf + strlen(localbuf), "%" NPY_INTP_FMT ",", PyArray_DIM(arr, i));
    }
    sprintf(localbuf + strlen(localbuf), ")-%c%" NPY_INTP_FMT "-%s", PyArray_DESCR(arr)->kind, PyArray_ITEMSIZE(arr), Py_TYPE(obj)->tp_name);
  } else if (PySequence_Check(obj)) {
    sprintf(localbuf, "%d-%s", (npy_int)PySequence_Length(obj), Py_TYPE(obj)->tp_name);
  } else {
    sprintf(localbuf, "%s instance", Py_TYPE(obj)->tp_name);
  }
  // 将本地缓冲区的内容复制到目标缓冲区 buf 中
  strcpy(buf, localbuf);
  // 返回成功标志
  return 1;
}

extern npy_intp
f2py_size_impl(PyArrayObject* var, ...)
{
  // 初始大小为 0
  npy_intp sz = 0;
  npy_intp dim;
  npy_intp rank;
  va_list argp;
  va_start(argp, var);
  // 获取可变参数中的第一个参数
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      // 如果 dim 为 -1，则返回数组的总元素个数
      sz = PyArray_SIZE(var);
    }
  else
    {
      // 否则，获取数组的维度数
      rank = PyArray_NDIM(var);
      // 如果 dim 在有效范围内，则返回对应维度的大小
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        // 否则输出错误信息并返回 0
        fprintf(stderr, "f2py_size: 2nd argument value=%" NPY_INTP_FMT
                " fails to satisfy 1<=value<=%" NPY_INTP_FMT
                ". Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  // 返回计算出的大小
  return sz;
}

/*********************************************/
/* Compatibility functions for Python >= 3.0 */
/*********************************************/

PyObject *
F2PyCapsule_FromVoidPtr(void *ptr, void (*dtor)(PyObject *))
{
    // 创建一个 Python Capsule 对象
    PyObject *ret = PyCapsule_New(ptr, NULL, dtor);
    // 如果创建失败，清空错误信息
    if (ret == NULL) {
        PyErr_Clear();
    }
    // 返回创建的 Python 对象
    return ret;
}

void *
F2PyCapsule_AsVoidPtr(PyObject *obj)
{
    # 使用 PyCapsule_GetPointer 函数从 Python Capsule 对象中获取指针
    void *ret = PyCapsule_GetPointer(obj, NULL);
    
    # 检查获取的指针是否为空
    if (ret == NULL) {
        # 如果为空指针，则清除当前的 Python 异常状态
        PyErr_Clear();
    }
    
    # 返回获取到的指针
    return ret;
}

int
F2PyCapsule_Check(PyObject *ptr)
{
    // 检查给定指针是否是一个 PyCapsule 对象
    return PyCapsule_CheckExact(ptr);
}

#ifdef __cplusplus
}
#endif
/************************* EOF fortranobject.c *******************************/
```