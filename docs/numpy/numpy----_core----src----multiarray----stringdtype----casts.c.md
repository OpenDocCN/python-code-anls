# `.\numpy\numpy\_core\src\multiarray\stringdtype\casts.c`

```
/*
定义了一个宏函数 ANY_TO_STRING_RESOLVE_DESCRIPTORS，用于处理类型转换描述符的解析和处理。
此宏函数展开后，会生成一个静态函数 any_to_string_SAFE_resolve_descriptors 和一个静态函数 any_to_string_SAME_KIND_resolve_descriptors。

any_to_string_SAFE_resolve_descriptors 函数：
    参数:
        - PyObject *NPY_UNUSED(self): 指向 Python 对象的指针，未使用，表示不使用该参数。
        - PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]): 指向包含数据类型元信息的数组的指针，未使用，表示不使用该参数。
        - PyArray_Descr *given_descrs[2]: 包含两个指向数组数据类型描述符的指针的数组。
        - PyArray_Descr *loop_descrs[2]: 用于存储处理后的数据类型描述符的数组。
        - npy_intp *NPY_UNUSED(view_offset): 指向整数数组的指针，未使用，表示不使用该参数。
    函数逻辑：
        - 检查给定的第二个描述符 given_descrs[1] 是否为 NULL。
        - 如果 given_descrs[1] 为 NULL，则创建一个新的字符串数据类型描述符 new，并将其存储在 loop_descrs[1] 中。
        - 如果 given_descrs[1] 不为 NULL，则增加其引用计数，并将其存储在 loop_descrs[1] 中。
        - 增加给定的第一个描述符 given_descrs[0] 的引用计数，并将其存储在 loop_descrs[0] 中。
        - 返回 NPY_SAFE_CASTING。

any_to_string_SAME_KIND_resolve_descriptors 函数：
    参数和逻辑与 any_to_string_SAFE_resolve_descriptors 函数类似，区别在于最后返回的是 NPY_SAME_KIND_CASTING。

static NPY_CASTING
string_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                     PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                     PyArray_Descr *given_descrs[2],
                                     PyArray_Descr *loop_descrs[2],
                                     npy_intp *view_offset)
{
*/

/*
上述代码段中的注释主要是解释了宏定义 ANY_TO_STRING_RESOLVE_DESCRIPTORS 的作用和展开后的两个静态函数 any_to_string_SAFE_resolve_descriptors 和 any_to_string_SAME_KIND_resolve_descriptors 的参数及逻辑。
*/
    # 检查给定描述符列表中的第二个描述符是否为 NULL
    if (given_descrs[1] == NULL) {
        # 如果第二个描述符为 NULL，则使用第一个描述符创建一个新的字符串类型描述符
        loop_descrs[1] = stringdtype_finalize_descr(given_descrs[0]);
    }
    else {
        # 如果第二个描述符不为 NULL，则增加其引用计数，并直接使用该描述符
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    # 增加第一个描述符的引用计数，并直接使用该描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    # 将循环中使用的第一个和第二个描述符分别转换为字符串类型描述符对象
    PyArray_StringDTypeObject *descr0 = (PyArray_StringDTypeObject *)loop_descrs[0];
    PyArray_StringDTypeObject *descr1 = (PyArray_StringDTypeObject *)loop_descrs[1];

    # 如果第一个描述符有 NA 对象而第二个描述符没有，则转换是不安全的
    if ((descr0->na_object != NULL) && (descr1->na_object == NULL)) {
        // 从带有 NA 的 dtype 转换到没有 NA 的 dtype 是不安全的，因为会丢失信息
        // 从没有 NA 的 dtype 转换到带有 NA 的 dtype 是安全的，因为源数据没有 NA 可能丢失
        return NPY_UNSAFE_CASTING;
    }

    # 视图（view）仅在共享分配器（allocator）的描述符之间合法（例如同一对象）
    if (descr0->allocator == descr1->allocator) {
        // 如果描述符共享相同的分配器，则视图偏移量设置为 0
        *view_offset = 0;
    };

    # 默认情况下，允许无需类型转换的赋值（casting）
    return NPY_NO_CASTING;
}

static int
string_to_string(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入和输出的字符串描述符
    PyArray_StringDTypeObject *idescr = (PyArray_StringDTypeObject *)context->descriptors[0];
    PyArray_StringDTypeObject *odescr = (PyArray_StringDTypeObject *)context->descriptors[1];
    // 检查输入和输出是否包含空值
    int in_has_null = idescr->na_object != NULL;
    int out_has_null = odescr->na_object != NULL;
    // 获取输入的空值名称
    const npy_static_string *in_na_name = &idescr->na_name;
    // 获取数据的维度
    npy_intp N = dimensions[0];
    // 获取输入和输出数据指针
    char *in = data[0];
    char *out = data[1];
    // 获取输入和输出的步长
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    // 分配输入和输出的字符串分配器
    npy_string_allocator *allocators[2] = {NULL, NULL};
    NpyString_acquire_allocators(2, context->descriptors, allocators);
    npy_string_allocator *iallocator = allocators[0];
    npy_string_allocator *oallocator = allocators[1];


    while (N--) {
        // 读取输入字符串
        const npy_packed_static_string *s = (npy_packed_static_string *)in;
        // 获取输出字符串
        npy_packed_static_string *os = (npy_packed_static_string *)out;
        // 如果输入和输出不共享内存
        if (!NpyString_share_memory(s, iallocator, os, oallocator)) {
            // 如果输入包含空值但输出不包含，并且输入字符串是空值
            if (in_has_null && !out_has_null && NpyString_isnull(s)) {
                // 执行不安全的转换，将空值包装为输出字符串
                if (NpyString_pack(oallocator, os, in_na_name->buf,
                                   in_na_name->size) < 0) {
                    // 内存错误处理
                    npy_gil_error(PyExc_MemoryError,
                              "Failed to pack string in string to string "
                              "cast.");
                    goto fail;
                }
            }
            // 否则，执行自由和复制操作
            else if (free_and_copy(iallocator, oallocator, s, os,
                                   "string to string cast") < 0) {
                goto fail;
            }
        }

        // 更新输入和输出指针位置
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocators(2, allocators);

    return 0;

fail:

    // 失败时释放字符串分配器并返回错误码
    NpyString_release_allocators(2, allocators);

    return -1;
}

static PyType_Slot s2s_slots[] = {
        // 解析描述符方法槽
        {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
        // 字符串到字符串的循环处理方法槽
        {NPY_METH_strided_loop, &string_to_string},
        // 不对齐的字符串到字符串循环处理方法槽
        {NPY_METH_unaligned_strided_loop, &string_to_string},
        // 终止槽
        {0, NULL}};

static char *s2s_name = "cast_StringDType_to_StringDType";

// unicode to string

static int
unicode_to_string(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取描述符数组
    PyArray_Descr *const *descrs = context->descriptors;
    // 获取输出描述符
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)descrs[1];

    // 获取输出字符串的分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    // 计算最大输入大小
    long max_in_size = (descrs[0]->elsize) / sizeof(Py_UCS4);

    // 获取数据的维度
    npy_intp N = dimensions[0];
    // 获取输入和输出数据指针
    Py_UCS4 *in = (Py_UCS4 *)data[0];
    char *out = data[1];
    # 计算输入数组的第一个维度步长，除以每个元素的字节数得到 UCS4 编码的步长
    npy_intp in_stride = strides[0] / sizeof(Py_UCS4);
    # 输出数组的步长，即每个元素之间的间隔
    npy_intp out_stride = strides[1];

    # 循环处理每个元素
    while (N--) {
        # 初始化输出字节数和代码点数
        size_t out_num_bytes = 0;
        size_t num_codepoints = 0;
        
        # 检查输入的 UTF-8 字符串，获取其中的代码点数和输出的字节数
        if (utf8_size(in, max_in_size, &num_codepoints, &out_num_bytes) ==
            -1) {
            # 如果出现无效的 Unicode 代码点，抛出类型错误并跳转到处理失败的标签
            npy_gil_error(PyExc_TypeError, "Invalid unicode code point found");
            goto fail;
        }
        
        # 用于存储输出字符串的静态字符串结构体
        npy_static_string out_ss = {0, NULL};
        
        # 加载新的字符串到静态字符串结构体中，进行 Unicode 到字符串的转换
        if (load_new_string((npy_packed_static_string *)out,
                            &out_ss, out_num_bytes, allocator,
                            "unicode to string cast") == -1) {
            # 如果加载字符串失败，跳转到处理失败的标签
            goto fail;
        }
        
        // 忽略常量以填充缓冲区

        # 将输出缓冲区转换为字符指针
        char *out_buf = (char *)out_ss.buf;
        
        # 遍历每个代码点
        for (size_t i = 0; i < num_codepoints; i++) {
            # 获取当前代码点
            Py_UCS4 code = in[i];

            # 用于存储 UTF-8 字节的数组
            char utf8_c[4] = {0};

            # 将 UCS4 编码的代码点转换为 UTF-8 字符串
            size_t num_bytes = ucs4_code_to_utf8_char(code, utf8_c);

            # 将 UTF-8 字符串复制到输出缓冲区
            strncpy(out_buf, utf8_c, num_bytes);

            # 更新输出缓冲区指针，移动到下一个字符的位置
            out_buf += num_bytes;
        }

        # 将输出缓冲区指针重置到字符串的起始位置
        out_buf -= out_num_bytes;

        # 更新输入和输出数组指针，以处理下一个元素
        in += in_stride;
        out += out_stride;
    }

    # 释放分配器使用的内存资源
    NpyString_release_allocator(allocator);

    # 返回成功标志
    return 0;
// 释放 NpyString 分配的内存空间的分配器
NpyString_release_allocator(allocator);

// 返回 -1，表示函数执行失败
return -1;
}

// 定义 PyType_Slot 数组，包含解析描述符和相应的函数指针
static PyType_Slot u2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAME_KIND_resolve_descriptors},
                                  {NPY_METH_strided_loop, &unicode_to_string},
                                  {0, NULL}};

// 定义字符串 u2s_name，用于类型转换从 Unicode 到 StringDType
static char *u2s_name = "cast_Unicode_to_StringDType";

// 将字符串转换为固定宽度数据类型时的解析描述符处理函数
static NPY_CASTING
string_to_fixed_width_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        // 如果给定描述符为 NULL，则设置类型错误并返回 -1
        // 表示当前不支持从 StringDType 到固定宽度数据类型的转换
        PyErr_SetString(
                PyExc_TypeError,
                "Casting from StringDType to a fixed-width dtype with an "
                "unspecified size is not currently supported, specify "
                "an explicit size for the output dtype instead.");
        return (NPY_CASTING)-1;
    }
    else {
        // 增加给定描述符的引用计数，并设置循环描述符
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    // 增加给定描述符的引用计数，并设置循环描述符
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // 返回相同类型的类型转换标志
    return NPY_SAME_KIND_CASTING;
}

// 加载可空字符串的函数，从静态字符串中加载到动态分配的字符串中
static int
load_nullable_string(const npy_packed_static_string *ps,
                     npy_static_string *s,
                     int has_null,
                     int has_string_na,
                     const npy_static_string *default_string,
                     const npy_static_string *na_name,
                     npy_string_allocator *allocator,
                     char *context)
{
    // 使用 NpyString_load 函数加载字符串，如果加载失败则报错并返回 -1
    int is_null = NpyString_load(allocator, ps, s);
    if (is_null == -1) {
        npy_gil_error(PyExc_MemoryError,
                      "Failed to load string in %s", context);
        return -1;
    }
    else if (is_null) {
        if (has_null && !has_string_na) {
            // 如果字符串为 NULL，且支持 NULL，但没有设置字符串 NA，则使用 NA 名称
            // 否则使用默认字符串
            *s = *na_name;
        }
        else {
            *s = *default_string;
        }
    }
    return 0;
}

// 字符串到 Unicode 的转换函数，处理 PyArrayMethod_Context 上下文和数据
static int
string_to_unicode(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取字符串类型的描述符
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 检查是否有 NULL 值对象
    int has_null = descr->na_object != NULL;
    // 检查是否有字符串 NA
    int has_string_na = descr->has_string_na;
    // 默认字符串
    const npy_static_string *default_string = &descr->default_string;
    // NA 名称
    const npy_static_string *na_name = &descr->na_name;
    // 获取数据的维度大小
    npy_intp N = dimensions[0];
    // 输入数据的指针
    char *in = data[0];
    // 输出数据的指针
    Py_UCS4 *out = (Py_UCS4 *)data[1];
    // 输入数据的步长
    npy_intp in_stride = strides[0];
    // 输出数据的步长，转换为 Py_UCS4 的大小
    npy_intp out_stride = strides[1] / sizeof(Py_UCS4);
    // 计算每个输出的最大 UCS4 字符数
    size_t max_out_size = (context->descriptors[1]->elsize) / sizeof(Py_UCS4);

    // 循环处理输入数据，N 表示迭代次数
    while (N--) {
        // 从输入中读取一个静态字符串结构体 ps
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        // 初始化一个静态字符串结构体 s
        npy_static_string s = {0, NULL};
        
        // 加载可空字符串，转换为 Unicode 字符串 s
        if (load_nullable_string(ps, &s, has_null, has_string_na,
                                 default_string, na_name, allocator,
                                 "in string to unicode cast") == -1) {
            // 如果加载失败，跳转到失败处理标签
            goto fail;
        }

        // 转换为无符号字符指针
        unsigned char *this_string = (unsigned char *)(s.buf);
        // 字符串字节数
        size_t n_bytes = s.size;
        // 总字节数
        size_t tot_n_bytes = 0;

        // 如果字符串长度为0，填充输出数组为0
        if (n_bytes == 0) {
            for (int i=0; i < max_out_size; i++) {
                out[i] = (Py_UCS4)0;
            }
        }
        else {
            int i = 0;
            // 将 UTF-8 字符转换为 UCS4 编码，填充输出数组直到达到最大输出长度或者处理完所有字节
            for (; i < max_out_size && tot_n_bytes < n_bytes; i++) {
                int num_bytes = utf8_char_to_ucs4_code(this_string, &out[i]);

                // 移动到下一个字符
                this_string += num_bytes;
                tot_n_bytes += num_bytes;
            }
            // 如果未填充满最大输出长度，剩余部分填充为0
            for(; i < max_out_size; i++) {
                out[i] = (Py_UCS4)0;
            }
        }

        // 更新输入和输出指针，以及释放分配器资源
        in += in_stride;
        out += out_stride;
    }

    // 释放分配器资源
    NpyString_release_allocator(allocator);

    // 返回成功状态
    return 0;
// 释放分配的内存，并返回失败代码
fail:
    NpyString_release_allocator(allocator);
    return -1;
}

// 定义一个静态的 PyType_Slot 数组，包含解析描述符和字符串到Unicode的函数指针
static PyType_Slot s2u_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_fixed_width_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_unicode},
        {0, NULL}};

// 定义一个静态的字符串，用于表示函数名称
static char *s2u_name = "cast_StringDType_to_Unicode";

// string to bool

// 解析描述符的函数，将字符串转换为布尔值
static NPY_CASTING
string_to_bool_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *NPY_UNUSED(view_offset))
{
    // 如果给定的第二个描述符为空，则使用 NPY_BOOL 类型创建一个新的描述符
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = PyArray_DescrNewFromType(NPY_BOOL);
    }
    else {
        // 否则增加给定描述符的引用计数并使用它
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    // 增加给定描述符的引用计数并使用它
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // 返回不安全的强制转换类型
    return NPY_UNSAFE_CASTING;
}

// 将字符串转换为布尔值的函数
static int
string_to_bool(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取描述符对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 检查是否有 NULL 对象
    int has_null = descr->na_object != NULL;
    // 检查是否有字符串 NA
    int has_string_na = descr->has_string_na;
    // 检查是否有 NaN NA
    int has_nan_na = descr->has_nan_na;
    // 获取默认字符串
    const npy_static_string *default_string = &descr->default_string;

    // 获取数据维度
    npy_intp N = dimensions[0];
    // 输入数据指针
    char *in = data[0];
    // 输出数据指针
    char *out = data[1];

    // 输入数据步长
    npy_intp in_stride = strides[0];
    // 输出数据步长
    npy_intp out_stride = strides[1];

    // 遍历数据进行转换
    while (N--) {
        // 获取紧凑静态字符串对象
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        // 初始化静态字符串对象
        npy_static_string s = {0, NULL};
        // 加载字符串到静态字符串对象中
        int is_null = NpyString_load(allocator, ps, &s);
        // 如果加载失败则报错并跳转到失败标签
        if (is_null == -1) {
            npy_gil_error(PyExc_MemoryError, "Failed to load string in string to bool cast");
            goto fail;
        }
        // 如果字符串为空
        else if (is_null) {
            // 如果存在 NULL 对象且不存在字符串 NA
            if (has_null && !has_string_na) {
                // 如果存在 NaN NA，则将其视为真值，按照 Python 的规则
                if (has_nan_na) {
                    *out = NPY_TRUE;
                }
                // 否则将其视为假值
                else {
                    *out = NPY_FALSE;
                }
            }
            // 如果不存在 NULL 对象，则将默认字符串的大小作为条件进行判断
            else {
                *out = (npy_bool)(default_string->size == 0);
            }
        }
        // 如果字符串长度为零，则将输出设置为假值
        else if (s.size == 0) {
            *out = NPY_FALSE;
        }
        // 否则将输出设置为真值
        else {
            *out = NPY_TRUE;
        }

        // 更新输入和输出指针
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocator(allocator);

    return 0;

// 失败时释放字符串分配器并返回失败代码
fail:
    NpyString_release_allocator(allocator);
    return -1;
}
// 定义 PyType_Slot 结构体数组 s2b_slots，用于描述字符串到布尔类型转换的方法
static PyType_Slot s2b_slots[] = {
    // 解析描述符方法，使用 string_to_bool_resolve_descriptors 函数
    {NPY_METH_resolve_descriptors, &string_to_bool_resolve_descriptors},
    // 使用 string_to_bool 函数进行跨步循环方法
    {NPY_METH_strided_loop, &string_to_bool},
    // 数组结束标记
    {0, NULL}
};

// 字符串 "cast_StringDType_to_Bool"，用于标识字符串到布尔类型转换
static char *s2b_name = "cast_StringDType_to_Bool";

// 布尔到字符串的转换

// 定义 bool_to_string 函数，将布尔值转换为字符串
static int
bool_to_string(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取数组维度大小
    npy_intp N = dimensions[0];
    // 输入数据指针
    char *in = data[0];
    // 输出数据指针
    char *out = data[1];

    // 输入和输出的步长
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    // 获取描述符对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[1];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    // 循环处理每个元素
    while (N--) {
        // 输出数据转换为静态字符串指针
        npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
        // 返回值字符串指针初始化为空
        char *ret_val = NULL;
        // 返回值字符串大小初始化为0
        size_t size = 0;
        
        // 根据布尔值设置返回值和大小
        if ((npy_bool)(*in) == NPY_TRUE) {
            ret_val = "True";
            size = 4;
        }
        else if ((npy_bool)(*in) == NPY_FALSE) {
            ret_val = "False";
            size = 5;
        }
        else {
            // 如果布尔值无效，则引发运行时错误
            npy_gil_error(PyExc_RuntimeError,
                          "invalid value encountered in bool to string cast");
            // 转到错误处理标签
            goto fail;
        }
        
        // 尝试将返回值打包到输出静态字符串指针中
        if (NpyString_pack(allocator, out_pss, ret_val, size) < 0) {
            // 如果打包失败，则引发内存错误
            npy_gil_error(PyExc_MemoryError,
                          "Failed to pack string in bool to string cast");
            // 转到错误处理标签
            goto fail;
        }
        
        // 更新输入和输出指针位置
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocator(allocator);

    // 返回成功状态
    return 0;

fail:
    // 失败时释放字符串分配器
    NpyString_release_allocator(allocator);

    // 返回失败状态
    return -1;
}

// 定义 PyType_Slot 结构体数组 b2s_slots，用于描述布尔到字符串类型转换的方法
static PyType_Slot b2s_slots[] = {
    // 解析描述符方法，使用 any_to_string_SAFE_resolve_descriptors 函数
    {NPY_METH_resolve_descriptors, &any_to_string_SAFE_resolve_descriptors},
    // 使用 bool_to_string 函数进行跨步循环方法
    {NPY_METH_strided_loop, &bool_to_string},
    // 数组结束标记
    {0, NULL}
};

// 字符串 "cast_Bool_to_StringDType"，用于标识布尔到字符串类型转换
static char *b2s_name = "cast_Bool_to_StringDType";

// 字符串和 (u)int 数据类型之间的转换

// 定义 load_non_nullable_string 函数，加载非空字符串并进行类型转换
static int
load_non_nullable_string(char *in, int has_null, const npy_static_string *default_string,
                         npy_static_string *string_to_load, npy_string_allocator *allocator,
                         int has_gil)
{
    // 输入静态字符串指针转换为打包的静态字符串指针
    const npy_packed_static_string *ps = (npy_packed_static_string *)in;
    // 加载静态字符串并返回是否为空
    int isnull = NpyString_load(allocator, ps, string_to_load);
    
    // 如果加载失败
    if (isnull == -1) {
        // 错误消息
        char *msg = "Failed to load string for conversion to a non-nullable type";
        // 如果有全局解释器锁
        if (has_gil) {
            // 设置内存错误异常消息
            PyErr_SetString(PyExc_MemoryError, msg);
        }
        else {
            // 否则，引发内存错误
            npy_gil_error(PyExc_MemoryError, msg);
        }
        // 返回加载失败状态
        return -1;
    }

    // 加载成功，返回成功状态
    return 0;
}
    else if (isnull) {
        // 如果数组中有空值
        if (has_null) {
            // 错误消息
            char *msg = "Arrays with missing data cannot be converted to a non-nullable type";
            // 如果已经获取了全局解释器锁
            if (has_gil)
            {
                // 设置 Python 异常对象为 ValueError，并传入错误消息
                PyErr_SetString(PyExc_ValueError, msg);
            }
            else {
                // 否则，调用 numpy 的 GIL 错误处理函数，设置 Python 异常为 ValueError，并传入错误消息
                npy_gil_error(PyExc_ValueError, msg);
            }
            // 返回 -1，表示操作失败
            return -1;
        }
        // 如果数组中没有空值，则将默认字符串复制到要加载的字符串中
        *string_to_load = *default_string;
    }
    // 操作成功，返回 0
    return 0;
// 这是一个静态函数，将非空字符串转换为 Python 字符串对象。
// 如果转换过程中遇到问题，返回 NULL。
static PyObject *
non_nullable_string_to_pystring(char *in, int has_null, const npy_static_string *default_string,
                                npy_string_allocator *allocator)
{
    // 创建一个静态字符串结构体 s，初始化为空
    npy_static_string s = {0, NULL};
    // 调用 load_non_nullable_string 函数，将输入的非空字符串转换为 s.buf 中的字符串
    if (load_non_nullable_string(in, has_null, default_string, &s, allocator, 1) == -1) {
        return NULL;
    }
    // 使用 s.buf 和 s.size 创建一个新的 Python Unicode 字符串对象
    PyObject *val_obj = PyUnicode_FromStringAndSize(s.buf, s.size);
    // 如果创建过程中出错，返回 NULL
    if (val_obj == NULL) {
        return NULL;
    }
    // 返回创建的 Python 字符串对象
    return val_obj;
}

// 这是一个静态函数，将字符串转换为 Python 长整型对象。
// 如果转换过程中遇到问题，返回 NULL。
static PyObject *
string_to_pylong(char *in, int has_null,
                 const npy_static_string *default_string,
                 npy_string_allocator *allocator)
{
    // 调用 non_nullable_string_to_pystring 函数，将输入的字符串转换为 Python 字符串对象
    PyObject *val_obj = non_nullable_string_to_pystring(
            in, has_null, default_string, allocator);
    // 如果转换过程中出错，返回 NULL
    if (val_obj == NULL) {
        return NULL;
    }
    // 将 Python 字符串对象解释为十进制整数
    PyObject *pylong_value = PyLong_FromUnicodeObject(val_obj, 10);
    // 减少对 val_obj 的引用计数
    Py_DECREF(val_obj);
    // 返回创建的 Python 长整型对象
    return pylong_value;
}

// 这是一个静态函数，将字符串转换为无符号长长整型数。
// 如果转换过程中出错，返回 -1；否则返回 0。
static npy_longlong
stringbuf_to_uint(char *in, npy_ulonglong *value, int has_null,
                  const npy_static_string *default_string,
                  npy_string_allocator *allocator)
{
    // 调用 string_to_pylong 函数，将输入的字符串转换为 Python 长整型对象
    PyObject *pylong_value =
            string_to_pylong(in, has_null, default_string, allocator);
    // 如果转换过程中出错，返回 -1
    if (pylong_value == NULL) {
        return -1;
    }
    // 将 Python 长整型对象转换为无符号长长整型数
    *value = PyLong_AsUnsignedLongLong(pylong_value);
    // 如果转换过程中出错，释放 Python 长整型对象并返回 -1
    if (*value == (unsigned long long)-1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    // 释放 Python 长整型对象并返回 0
    Py_DECREF(pylong_value);
    return 0;
}

// 这是一个静态函数，将字符串转换为有符号长长整型数。
// 如果转换过程中出错，返回 -1；否则返回 0。
static npy_longlong
stringbuf_to_int(char *in, npy_longlong *value, int has_null,
                 const npy_static_string *default_string,
                 npy_string_allocator *allocator)
{
    // 调用 string_to_pylong 函数，将输入的字符串转换为 Python 长整型对象
    PyObject *pylong_value =
            string_to_pylong(in, has_null, default_string, allocator);
    // 如果转换过程中出错，返回 -1
    if (pylong_value == NULL) {
        return -1;
    }
    // 将 Python 长整型对象转换为有符号长长整型数
    *value = PyLong_AsLongLong(pylong_value);
    // 如果转换过程中出错，释放 Python 长整型对象并返回 -1
    if (*value == -1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    // 释放 Python 长整型对象并返回 0
    Py_DECREF(pylong_value);
    return 0;
}

// 这是一个静态函数，将 Python 对象转换为字符串。
// 如果转换过程中出错，返回 -1；否则返回 0。
static int
pyobj_to_string(PyObject *obj, char *out, npy_string_allocator *allocator)
{
    // 如果输入的 Python 对象为空，返回 -1
    if (obj == NULL) {
        return -1;
    }
    // 将 Python 对象转换为字符串对象
    PyObject *pystr_val = PyObject_Str(obj);
    // 减少对输入 Python 对象的引用计数
    Py_DECREF(obj);

    // 如果转换过程中出错，返回 -1
    if (pystr_val == NULL) {
        return -1;
    }
    // 将 Python 字符串对象转换为 UTF-8 编码的 C 字符串
    Py_ssize_t length;
    const char *cstr_val = PyUnicode_AsUTF8AndSize(pystr_val, &length);
    // 如果转换过程中出错，释放 Python 字符串对象并返回 -1
    if (cstr_val == NULL) {
        Py_DECREF(pystr_val);
        return -1;
    }
    // 将 UTF-8 编码的 C 字符串打包到输出的字符串结构体中
    npy_packed_static_string *out_ss = (npy_packed_static_string *)out;
    if (NpyString_pack(allocator, out_ss, cstr_val, length) < 0) {
        // 如果打包过程中出错，报告内存错误，并释放 Python 字符串对象
        npy_gil_error(PyExc_MemoryError,
                      "Failed to pack string while converting from python "
                      "string");
        Py_DECREF(pystr_val);
        return -1;
    }
    // 释放 Python 字符串对象并返回 0
    Py_DECREF(pystr_val);
    return 0;
}
    // 递减 Python 对象的引用计数，当引用计数为零时自动释放对象
    Py_DECREF(pystr_val);
    // 返回整数 0，表示函数执行成功
    return 0;
}
// 结束静态函数定义

static int
int_to_stringbuf(long long in, char *out, npy_string_allocator *allocator)
{
    // 将 long long 类型的输入转换为 Python 的长整型对象
    PyObject *pylong_val = PyLong_FromLongLong(in);
    // 调用通用函数将 Python 对象转换为字符串，并使用分配器进行内存管理
    return pyobj_to_string(pylong_val, out, allocator);
}

static int
uint_to_stringbuf(unsigned long long in, char *out,
                  npy_string_allocator *allocator)
{
    // 将 unsigned long long 类型的输入转换为 Python 的无符号长整型对象
    PyObject *pylong_val = PyLong_FromUnsignedLongLong(in);
    // 调用通用函数将 Python 对象转换为字符串，并使用分配器进行内存管理
    return pyobj_to_string(pylong_val, out, allocator);
}

#define DTYPES_AND_CAST_SPEC(shortname, typename)                              \
        // 获取 string 到 typename 和 typename 到 string 的数据类型元信息
        PyArray_DTypeMeta **s2##shortname##_dtypes = get_dtypes(               \
                &PyArray_StringDType,                                          \
                &PyArray_##typename##DType);                                   \
                                                                               \
        // 获取 string 到 typename 的类型转换规范
        PyArrayMethod_Spec *StringTo##typename##CastSpec =                     \
                get_cast_spec(                                                 \
                        s2##shortname##_name, NPY_UNSAFE_CASTING,              \
                        NPY_METH_REQUIRES_PYAPI, s2##shortname##_dtypes,       \
                        s2##shortname##_slots);                                \
                                                                               \
        // 获取 typename 到 string 的类型转换规范
        PyArray_DTypeMeta **shortname##2s_dtypes = get_dtypes(                 \
                &PyArray_##typename##DType,                                    \
                &PyArray_StringDType);                                         \
                                                                               \
        // 获取 typename 到 string 的类型转换规范
        PyArrayMethod_Spec *typename##ToStringCastSpec = get_cast_spec(        \
                shortname##2s_name, NPY_SAFE_CASTING,                          \
                NPY_METH_REQUIRES_PYAPI, shortname##2s_dtypes,                 \
                shortname##2s_slots);

// 定义宏 DTYPES_AND_CAST_SPEC，用于生成不同类型之间的转换规范

STRING_INT_CASTS(int8, int, i8, NPY_INT8, lli, npy_longlong, long long)
// 宏 STRING_INT_CASTS 的实例化：定义 int8 类型转换规范

STRING_INT_CASTS(int16, int, i16, NPY_INT16, lli, npy_longlong, long long)
// 宏 STRING_INT_CASTS 的实例化：定义 int16 类型转换规范

STRING_INT_CASTS(int32, int, i32, NPY_INT32, lli, npy_longlong, long long)
// 宏 STRING_INT_CASTS 的实例化：定义 int32 类型转换规范

STRING_INT_CASTS(int64, int, i64, NPY_INT64, lli, npy_longlong, long long)
// 宏 STRING_INT_CASTS 的实例化：定义 int64 类型转换规范

STRING_INT_CASTS(uint8, uint, u8, NPY_UINT8, llu, npy_ulonglong,
                 unsigned long long)
// 宏 STRING_INT_CASTS 的实例化：定义 uint8 类型转换规范

STRING_INT_CASTS(uint16, uint, u16, NPY_UINT16, llu, npy_ulonglong,
                 unsigned long long)
// 宏 STRING_INT_CASTS 的实例化：定义 uint16 类型转换规范

STRING_INT_CASTS(uint32, uint, u32, NPY_UINT32, llu, npy_ulonglong,
                 unsigned long long)
// 宏 STRING_INT_CASTS 的实例化：定义 uint32 类型转换规范

STRING_INT_CASTS(uint64, uint, u64, NPY_UINT64, llu, npy_ulonglong,
                 unsigned long long)
// 宏 STRING_INT_CASTS 的实例化：定义 uint64 类型转换规范

#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
// 如果 byte 的大小等于 short 的大小

// 宏 STRING_INT_CASTS 的实例化：定义 byte 类型转换规范
STRING_INT_CASTS(byte, int, byte, NPY_BYTE, lli, npy_longlong, long long)

// 宏 STRING_INT_CASTS 的实例化：定义 ubyte 类型转换规范
STRING_INT_CASTS(ubyte, uint, ubyte, NPY_UBYTE, llu, npy_ulonglong,
                 unsigned long long)
#endif

#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
// 如果 short 的大小等于 int 的大小
// 定义宏，用于生成字符串到整数类型的转换函数和相关信息的结构体定义
#define STRING_INT_CASTS(typename, shortname, isinf_name, npy_typename,    \
                         lli, npy_longlong, long long)                     \
    // 定义函数 string_to_##typename，用于将字符串转换为整数类型
    static int string_to_##typename(PyArrayMethod_Context * context,       \
                                    char *const data[],                    \
                                    npy_intp const dimensions[],           \
                                    npy_intp const strides[],              \
                                    NpyAuxData *NPY_UNUSED(auxdata))      \
    // 定义失败处理标签 fail，用于在出错时释放分配的字符串内存并返回错误
    fail:                                                                 \
        NpyString_release_allocator(allocator);                           \
        return -1;                                                        \
    }                                                                     \
                                                                          \
    // 定义 PyType_Slot 结构体数组 s2##shortname##_slots，包含函数指针和描述符解析方法
    static PyType_Slot s2##shortname##_slots[] = {                        \
            // 解析描述符方法，关联到函数 string_to_##typename##_resolve_descriptors
            {NPY_METH_resolve_descriptors,                                \
             &string_to_##typename##_resolve_descriptors},                \
            // 循环方法，关联到函数 string_to_##typename
            {NPY_METH_strided_loop, &string_to_##typename},               \
            // 结束标记
            {0, NULL}};                                                   \
                                                                          \
    // 定义字符串常量 s2##shortname##_name，描述类型转换的名称
    static char *s2##shortname##_name = "cast_StringDType_to_" #typename;

// 定义宏，用于生成字符串到浮点数类型的转换函数和相关信息的结构体定义
#define STRING_TO_FLOAT_CAST(typename, shortname, isinf_name,             \
                             double_to_float)                             \
    // 定义函数 string_to_##typename，用于将字符串转换为浮点数类型
    static int string_to_##typename(PyArrayMethod_Context * context,       \
                                    char *const data[],                    \
                                    npy_intp const dimensions[],           \
                                    npy_intp const strides[],              \
                                    NpyAuxData *NPY_UNUSED(auxdata))      \
    // 定义失败处理标签 fail，用于在出错时释放分配的字符串内存并返回错误
    fail:                                                                 \
        NpyString_release_allocator(allocator);                           \
        return -1;                                                        \
    }                                                                     \
                                                                          \
    // 定义 PyType_Slot 结构体数组 s2##shortname##_slots，包含函数指针和描述符解析方法
    static PyType_Slot s2##shortname##_slots[] = {                        \
            // 解析描述符方法，关联到函数 string_to_##typename##_resolve_descriptors
            {NPY_METH_resolve_descriptors,                                \
             &string_to_##typename##_resolve_descriptors},                \
            // 循环方法，关联到函数 string_to_##typename
            {NPY_METH_strided_loop, &string_to_##typename},               \
            // 结束标记
            {0, NULL}};                                                   \
                                                                          \
    // 定义字符串常量 s2##shortname##_name，描述类型转换的名称
    static char *s2##shortname##_name = "cast_StringDType_to_" #typename;

// 空的宏定义，用于未实现的宏参数
#define STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(typename, npy_typename)
    // 定义一个静态函数 string_to_##typename##_resolve_descriptors，
    // 该函数用于解析描述符并进行类型转换
    static NPY_CASTING string_to_##typename##_resolve_descriptors(         \
            // 不使用 self 参数
            PyObject *NPY_UNUSED(self),                                    \
            // 不使用 dtypes 数组
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                      \
            // 输入的描述符数组和循环使用的描述符数组
            PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2], \
            // 视图偏移量，不使用此参数
            npy_intp *NPY_UNUSED(view_offset))                             \
    {                                                                      \
        // 如果第二个给定的描述符为空
        if (given_descrs[1] == NULL) {                                     \
            // 根据给定的 numpy 类型创建新的描述符并存储在循环描述符数组中
            loop_descrs[1] = PyArray_DescrNewFromType(NPY_##npy_typename); \
        }                                                                  \
        else {                                                             \
            // 增加给定描述符的引用计数并复制到循环描述符数组中
            Py_INCREF(given_descrs[1]);                                    \
            loop_descrs[1] = given_descrs[1];                              \
        }                                                                  \
                                                                           \
        // 增加给定描述符的引用计数并复制到循环描述符数组中
        Py_INCREF(given_descrs[0]);                                        \
        loop_descrs[0] = given_descrs[0];                                  \
                                                                           \
        // 返回非安全转换标志
        return NPY_UNSAFE_CASTING;                                         \
    }
// 定义宏 FLOAT_TO_STRING_CAST，用于生成将特定类型的数组转换为字符串的函数
#define FLOAT_TO_STRING_CAST(typename, shortname, float_to_double)            \
    // 定义函数 typename##_to_string，用于将特定类型的数组转换为字符串
    static int typename##_to_string(                                          \
            PyArrayMethod_Context *context, char *const data[],               \
            npy_intp const dimensions[], npy_intp const strides[],            \
            NpyAuxData *NPY_UNUSED(auxdata))                                  \
    {                                                                         \
        // 获取数组的第一个维度大小
        npy_intp N = dimensions[0];                                           \
        // 将输入数据解释为指定类型的数组
        npy_##typename *in = (npy_##typename *)data[0];                       \
        // 输出字符串的起始位置
        char *out = data[1];                                                  \
        // 获取浮点数的描述器
        PyArray_Descr *float_descr = context->descriptors[0];                 \
                                                                              \
        // 计算输入数组的步长（每个元素占据的字节数除以数据类型的大小）
        npy_intp in_stride = strides[0] / sizeof(npy_##typename);             \
        // 输出数组的步长
        npy_intp out_stride = strides[1];                                     \
                                                                              \
        // 获取输出字符串的描述器对象
        PyArray_StringDTypeObject *descr =                                    \
                (PyArray_StringDTypeObject *)context->descriptors[1];         \
        // 获取字符串分配器
        npy_string_allocator *allocator = NpyString_acquire_allocator(descr); \
                                                                              \
        // 循环处理每个元素
        while (N--) {                                                         \
            // 将输入数组中的当前值转换为 Python 对象
            PyObject *scalar_val = PyArray_Scalar(in, float_descr, NULL);     \
            // 将 Python 对象转换为字符串并存储到输出位置，使用指定的分配器
            if (pyobj_to_string(scalar_val, out, allocator) == -1) {          \
                // 转换失败时跳转到失败处理标签
                goto fail;                                                    \
            }                                                                 \
                                                                              \
            // 更新输入和输出指针到下一个元素
            in += in_stride;                                                  \
            out += out_stride;                                                \
        }                                                                     \
                                                                              \
        // 释放字符串分配器
        NpyString_release_allocator(allocator);                               \
        // 返回成功状态
        return 0;                                                             \
    fail:                                                                     \
        // 失败时释放字符串分配器并返回失败状态
        NpyString_release_allocator(allocator);                               \
        return -1;                                                            \
    }
    // 定义 PyType_Slot 结构体数组，用于实现类型转换
    static PyType_Slot shortname##2s_slots [] = {                             \
            // 设置解析描述符方法，指向 any_to_string_SAFE_resolve_descriptors 函数
            {NPY_METH_resolve_descriptors,                                    \
             &any_to_string_SAFE_resolve_descriptors},                        \
            // 设置跨步循环方法，指向 typename##_to_string 函数
            {NPY_METH_strided_loop, &typename##_to_string},                   \
            // 终止标记，空槽和空指针
            {0, NULL}};                                                       \
                                                                              \
        // 定义 shortname##2s_name，表示类型转换的名称字符串
        static char *shortname##2s_name = "cast_" #typename "_to_StringDType";
# 解析指定的宏并生成相应的函数描述符（例如 float64 对应的 DOUBLE）
STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float64, DOUBLE)

# 定义将字符串转换为 float64 的函数
static int
string_to_float64(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    # 获取描述符对象，用于描述字符串的数据类型
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    # 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    # 检查是否存在 null 值标记
    int has_null = descr->na_object != NULL;
    # 获取默认字符串
    const npy_static_string *default_string = &descr->default_string;
    # 获取处理的数据点数
    npy_intp N = dimensions[0];
    # 输入数据的起始位置
    char *in = data[0];
    # 输出数据的起始位置（转换为 float64 类型）
    npy_float64 *out = (npy_float64 *)data[1];

    # 输入数据和输出数据的步长
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_float64);

    # 迭代处理每个数据点
    while (N--) {
        # 将输入字符串转换为 Python 的 float 对象
        PyObject *pyfloat_value =
                string_to_pyfloat(in, has_null, default_string, allocator);
        # 如果转换失败，则跳转到错误处理标签
        if (pyfloat_value == NULL) {
            goto fail;
        }
        # 将 Python float 对象的值转换为 npy_float64 类型并存入输出数组
        *out = (npy_float64)PyFloat_AS_DOUBLE(pyfloat_value);
        # 减少 Python float 对象的引用计数
        Py_DECREF(pyfloat_value);

        # 更新输入数据指针位置和输出数据指针位置
        in += in_stride;
        out += out_stride;
    }

    # 释放字符串分配器
    NpyString_release_allocator(allocator);
    return 0;

fail:
    # 失败时释放字符串分配器并返回错误码
    NpyString_release_allocator(allocator);
    return -1;
}

# 定义 float64 类型的方法槽
static PyType_Slot s2f64_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_float64_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_float64},
        {0, NULL}};

# 定义 float64 类型的名称
static char *s2f64_name = "cast_StringDType_to_float64";

# 定义将 float64 转换为字符串的宏和函数
FLOAT_TO_STRING_CAST(float64, f64, double)

# 解析指定的宏并生成相应的函数描述符（例如 float32 对应的 FLOAT）
STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float32, FLOAT)

# 定义将字符串转换为 float32 的宏和函数
STRING_TO_FLOAT_CAST(float32, f32, npy_isinf, npy_float32)
# 定义将 float32 转换为字符串的宏和函数
FLOAT_TO_STRING_CAST(float32, f32, double)

# 解析指定的宏并生成相应的函数描述符（例如 float16 对应的 HALF）
STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float16, HALF)

# 定义将字符串转换为 float16 的宏和函数
STRING_TO_FLOAT_CAST(float16, f16, npy_half_isinf, npy_double_to_half)
# 定义将 float16 转换为字符串的宏和函数
FLOAT_TO_STRING_CAST(float16, f16, npy_half_to_double)

# 解析指定的宏并生成相应的函数描述符（例如 longdouble 对应的 LONGDOUBLE）
STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(longdouble, LONGDOUBLE);

# 定义将字符串转换为 longdouble 的函数
static int
string_to_longdouble(PyArrayMethod_Context *context, char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     NpyAuxData *NPY_UNUSED(auxdata))
{
    # 获取描述符对象，用于描述字符串的数据类型
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    # 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    # 检查是否存在 null 值标记
    int has_null = descr->na_object != NULL;
    # 获取默认字符串
    const npy_static_string *default_string = &descr->default_string;
    # 获取处理的数据点数
    npy_intp N = dimensions[0];
    # 输入数据的起始位置
    char *in = data[0];
    # 输出数据的起始位置（转换为 longdouble 类型）
    npy_longdouble *out = (npy_longdouble *)data[1];

    # 输入数据和输出数据的步长
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_longdouble);
    // 循环，逐个处理直到 N 减少为零
    while (N--) {
        // 定义静态字符串结构体并初始化为空
        npy_static_string s = {0, NULL};
        // 尝试从输入流中加载非空字符串，如果失败则跳转到失败标签
        if (load_non_nullable_string(in, has_null, default_string, &s, allocator, 0) == -1) {
            goto fail;
        }

        // 分配临时的以 null 结尾的字符串拷贝
        char *buf = PyMem_RawMalloc(s.size + 1);
        memcpy(buf, s.buf, s.size);
        buf[s.size] = '\0';

        // 解析 buf 中的字符串为长双精度浮点数
        char *end = NULL;
        errno = 0;
        npy_longdouble longdouble_value = NumPyOS_ascii_strtold(buf, &end);

        // 检查是否出现了范围错误
        if (errno == ERANGE) {
            /* strtold 返回正确符号的无穷大。如果警告触发失败则释放内存并跳转到失败标签 */
            if (PyErr_Warn(PyExc_RuntimeWarning,
                           "overflow encountered in conversion from string") < 0) {
                PyMem_RawFree(buf);
                goto fail;
            }
        }
        // 检查是否出现了其他错误或者未成功解析完整个字符串
        else if (errno || end == buf || *end) {
            PyErr_Format(PyExc_ValueError,
                         "invalid literal for long double: %s (%s)",
                         buf,
                         strerror(errno));
            PyMem_RawFree(buf);
            goto fail;
        }
        PyMem_RawFree(buf);  // 释放 buf 的内存
        *out = longdouble_value;  // 将解析得到的长双精度浮点数存入 out 指向的位置

        // 更新输入和输出指针位置
        in += in_stride;
        out += out_stride;
    }

    // 释放分配器所使用的字符串分配器
    NpyString_release_allocator(allocator);
    return 0;
fail:
    // 释放字符串分配器，并返回-1表示失败
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot s2ld_slots[] = {
    {NPY_METH_resolve_descriptors, &string_to_longdouble_resolve_descriptors},
    {NPY_METH_strided_loop, &string_to_longdouble},
    {0, NULL}
};

static char *s2ld_name = "cast_StringDType_to_longdouble";

// 将 longdouble 转换为字符串

// TODO: 这里是不正确的。longdouble 到 unicode 的转换也有同样的问题。要修复这个问题，我们需要在 NumPy 中实现 ldtoa 函数。它并不在标准库中。另一个选项是使用 `snprintf`，但我们需要预先计算结果字符串的大小。

FLOAT_TO_STRING_CAST(longdouble, ld, npy_longdouble)

// 将字符串转换为 cfloat

static PyObject*
string_to_pycomplex(char *in, int has_null,
                    const npy_static_string *default_string,
                    npy_string_allocator *allocator)
{
    PyObject *val_obj = non_nullable_string_to_pystring(
            in, has_null, default_string, allocator);
    if (val_obj == NULL) {
        return NULL;
    }
    PyObject *args = PyTuple_Pack(1, val_obj);
    Py_DECREF(val_obj);
    if (args == NULL) {
        return NULL;
    }
    PyObject *pycomplex_value = PyComplex_Type.tp_new(&PyComplex_Type, args, NULL);
    Py_DECREF(args);
    return pycomplex_value;
}

fail:                                                                            \
    // 释放字符串分配器，并返回-1表示失败
    NpyString_release_allocator(allocator);                                  \
    return -1;                                                               \
    }                                                                        \
                                                                                 \
    static PyType_Slot s2##ctype##_slots[] = {                               \
        {NPY_METH_resolve_descriptors,                                   \
         &string_to_##ctype##_resolve_descriptors},                      \
        {NPY_METH_strided_loop, &string_to_##ctype},                     \
        {0, NULL}};                                                      \
                                                                                 \
    static char *s2##ctype##_name = "cast_StringDType_to_" #ctype;

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(cfloat, CFLOAT)
STRING_TO_CFLOAT_CAST(cfloat, f, float)

// 将 cfloat 转换为字符串

FLOAT_TO_STRING_CAST(cfloat, cfloat, npy_cfloat)

// 将字符串转换为 cdouble

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(cdouble, CDOUBLE)
STRING_TO_CFLOAT_CAST(cdouble, , double)

// 将 cdouble 转换为字符串

FLOAT_TO_STRING_CAST(cdouble, cdouble, npy_cdouble)

// 将字符串转换为 clongdouble

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(clongdouble, CLONGDOUBLE)
STRING_TO_CFLOAT_CAST(clongdouble, l, longdouble)

// 将 longdouble 转换为字符串

FLOAT_TO_STRING_CAST(clongdouble, clongdouble, npy_clongdouble)

// 将字符串转换为 datetime

static NPY_CASTING
// 如果给定描述符的第二个元素为NULL，表示没有指定时间单位，抛出类型错误异常并返回-1
string_to_datetime_timedelta_resolve_descriptors(
        PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "Casting from StringDType to datetimes without a unit "
                        "is not currently supported");
        return (NPY_CASTING)-1;
    }
    else {
        // 增加给定描述符的引用计数，并将其赋值给循环描述符的第二个元素
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    // 增加给定描述符的引用计数，并将其赋值给循环描述符的第一个元素
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // 返回安全类型转换标志
    return NPY_UNSAFE_CASTING;
}

// numpy将空字符串和字符串'nat'的任何大小写组合视为字符串转换中NaT的等价形式
static int
is_nat_string(const npy_static_string *s) {
    // 如果字符串长度为0或者长度为3且内容为'nat'（不区分大小写），则返回真
    return s->size == 0 || (s->size == 3 &&
             NumPyOS_ascii_tolower(s->buf[0]) == 'n' &&
             NumPyOS_ascii_tolower(s->buf[1]) == 'a' &&
             NumPyOS_ascii_tolower(s->buf[2]) == 't');
}

// 将字符串转换为日期时间
static int
string_to_datetime(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取描述符对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 检查是否存在空值对象
    int has_null = descr->na_object != NULL;
    // 检查是否有字符串NA
    int has_string_na = descr->has_string_na;
    // 默认字符串
    const npy_static_string *default_string = &descr->default_string;

    // 获取第一维度大小
    npy_intp N = dimensions[0];
    // 输入数据指针
    char *in = data[0];
    // 输出数据指针（日期时间类型）
    npy_datetime *out = (npy_datetime *)data[1];

    // 输入数据步长
    npy_intp in_stride = strides[0];
    // 输出数据步长（转换为日期时间类型）
    npy_intp out_stride = strides[1] / sizeof(npy_datetime);

    // 日期时间结构体
    npy_datetimestruct dts;
    // 输入单位
    NPY_DATETIMEUNIT in_unit = -1;
    // 输入元数据
    PyArray_DatetimeMetaData in_meta = {0, 1};
    // 输出是否特殊值
    npy_bool out_special;

    // 获取日期时间描述符对象
    _PyArray_LegacyDescr *dt_descr = (_PyArray_LegacyDescr *)context->descriptors[1];
    // 获取日期时间元数据
    PyArray_DatetimeMetaData *dt_meta =
            &(((PyArray_DatetimeDTypeMetaData *)dt_descr->c_metadata)->meta);
    # 使用 while 循环逐步处理每个输入的字符串
    while (N--) {
        # 将输入的字节序列解释为静态字符串结构
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        # 初始化静态字符串 s
        npy_static_string s = {0, NULL};
        # 调用 NpyString_load 函数加载静态字符串，检查是否为空
        int is_null = NpyString_load(allocator, ps, &s);
        # 如果加载过程中出现错误，设置内存错误并跳转到失败处理代码块
        if (is_null == -1) {
            PyErr_SetString(
                    PyExc_MemoryError,
                    "Failed to load string in string to datetime cast");
            goto fail;
        }
        # 如果字符串为空
        if (is_null) {
            # 如果存在空值并且未指定字符串的 NA 值，设置输出为 NPY_DATETIME_NAT，并跳转到下一步
            if (has_null && !has_string_na) {
                *out = NPY_DATETIME_NAT;
                goto next_step;
            }
            # 否则使用默认字符串作为 s 的值
            s = *default_string;
        }
        # 如果字符串表示为 'NaT'，设置输出为 NPY_DATETIME_NAT，并跳转到下一步
        if (is_nat_string(&s)) {
            *out = NPY_DATETIME_NAT;
            goto next_step;
        }

        // 实际解析日期时间字符串
        // 调用 NpyDatetime_ParseISO8601Datetime 函数解析 ISO 8601 格式的日期时间字符串
        if (NpyDatetime_ParseISO8601Datetime(
                    (const char *)s.buf, s.size, in_unit, NPY_UNSAFE_CASTING,
                    &dts, &in_meta.base, &out_special) < 0) {
            goto fail;
        }
        // 将日期时间结构转换为 datetime64 类型
        if (NpyDatetime_ConvertDatetimeStructToDatetime64(dt_meta, &dts, out) <
            0) {
            goto fail;
        }

    next_step:
        # 更新输入和输出指针位置
        in += in_stride;
        out += out_stride;
    }

    # 释放字符串分配器的资源
    NpyString_release_allocator(allocator);
    # 返回操作成功的标志
    return 0;
// 释放内存分配器并返回错误代码
fail:
    NpyString_release_allocator(allocator);
    return -1;
}

// 定义 PyType_Slot 结构数组，描述 s2dt 对象的行为
static PyType_Slot s2dt_slots[] = {
        // 解析描述符方法
        {NPY_METH_resolve_descriptors,
         &string_to_datetime_timedelta_resolve_descriptors},
        // 循环处理方法
        {NPY_METH_strided_loop, &string_to_datetime},
        // 数组结束标志
        {0, NULL}};

// 字符串变换为日期时间的名称
static char *s2dt_name = "cast_StringDType_to_Datetime";

// datetime 转换为字符串

static int
datetime_to_string(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取数组维度
    npy_intp N = dimensions[0];
    // 输入数据的指针
    npy_datetime *in = (npy_datetime *)data[0];
    // 输出数据的指针
    char *out = data[1];

    // 输入数据的步幅
    npy_intp in_stride = strides[0] / sizeof(npy_datetime);
    // 输出数据的步幅
    npy_intp out_stride = strides[1];

    // 获取日期时间描述符
    _PyArray_LegacyDescr *dt_descr = (_PyArray_LegacyDescr *)context->descriptors[0];
    // 获取日期时间元数据
    PyArray_DatetimeMetaData *dt_meta =
            &(((PyArray_DatetimeDTypeMetaData *)dt_descr->c_metadata)->meta);
    // 用于构建日期时间字符串的缓冲区
    char datetime_buf[NPY_DATETIME_MAX_ISO8601_STRLEN];

    // 获取字符串描述符
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)context->descriptors[1];
    // 检查是否有空值对象
    int has_null = sdescr->na_object != NULL;
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);
    while (N--) {
        npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
        // 如果输入值为 NaT (Not a Time)，处理特殊情况
        if (*in == NPY_DATETIME_NAT)
        {
            // 如果没有空值，将字符串 "NaT" 打包到输出结构中
            if (!has_null) {
                npy_static_string os = {3, "NaT"};
                // 调用 NpyString_pack 函数尝试将 "NaT" 打包到输出结构
                if (NpyString_pack(allocator, out_pss, os.buf, os.size) < 0) {
                    // 若失败，抛出内存错误并跳转到失败处理标签
                    npy_gil_error(
                            PyExc_MemoryError,
                            "Failed to pack string in datetime to string "
                            "cast");
                    goto fail;
                }
            }
            // 如果有空值，调用 NpyString_pack_null 函数将空值打包到输出结构
            else if (NpyString_pack_null(allocator, out_pss) < 0) {
                // 若失败，抛出内存错误并跳转到失败处理标签
                npy_gil_error(
                        PyExc_MemoryError,
                        "Failed to pack string in datetime to string cast");
                goto fail;
            }
        }
        // 对于正常日期时间值，进行格式转换并打包到输出结构
        else {
            npy_datetimestruct dts;
            // 将输入日期时间值转换为日期时间结构
            if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(
                        dt_meta, *in, &dts) < 0) {
                // 若转换失败，跳转到失败处理标签
                goto fail;
            }

            // 将日期时间缓冲区清零
            memset(datetime_buf, 0, NPY_DATETIME_MAX_ISO8601_STRLEN);

            // 将日期时间结构转换为 ISO8601 格式的字符串
            if (NpyDatetime_MakeISO8601Datetime(
                        &dts, datetime_buf, NPY_DATETIME_MAX_ISO8601_STRLEN, 0,
                        0, dt_meta->base, -1, NPY_UNSAFE_CASTING) < 0) {
                // 若转换失败，跳转到失败处理标签
                goto fail;
            }

            // 将转换后的 ISO8601 格式字符串打包到输出结构
            if (NpyString_pack(allocator, out_pss, datetime_buf,
                               strlen(datetime_buf)) < 0) {
                // 若打包失败，设置异常并跳转到失败处理标签
                PyErr_SetString(PyExc_MemoryError,
                                "Failed to pack string while converting "
                                "from a datetime.");
                goto fail;
            }
        }

        // 更新输入和输出指针位置
        in += in_stride;
        out += out_stride;
    }

    // 释放分配器资源
    NpyString_release_allocator(allocator);
    // 返回成功状态
    return 0;
fail:
    // 释放字符串分配器的资源
    NpyString_release_allocator(allocator);
    // 返回-1，表示函数执行失败
    return -1;
}

static PyType_Slot dt2s_slots[] = {
        // 方法插槽：解析描述符，使用安全版本的任意类型到字符串转换方法
        {NPY_METH_resolve_descriptors,
         &any_to_string_SAFE_resolve_descriptors},
        // 方法插槽：使用日期时间对象进行跨步循环的字符串转换
        {NPY_METH_strided_loop, &datetime_to_string},
        {0, NULL}};

static char *dt2s_name = "cast_Datetime_to_StringDType";

// string to timedelta

static int
string_to_timedelta(PyArrayMethod_Context *context, char *const data[],
                    npy_intp const dimensions[], npy_intp const strides[],
                    NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取字符串数据类型的描述符对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 判断是否有空值对象
    int has_null = descr->na_object != NULL;
    // 判断是否包含字符串 NA
    int has_string_na = descr->has_string_na;
    // 默认字符串
    const npy_static_string *default_string = &descr->default_string;

    // 数据维度
    npy_intp N = dimensions[0];
    // 输入数据指针
    char *in = data[0];
    // 输出数据指针
    npy_timedelta *out = (npy_timedelta *)data[1];

    // 输入和输出的步长
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_timedelta);

    // 循环处理每个元素
    while (N--) {
        // 转换为紧凑静态字符串对象
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        // 静态字符串对象
        npy_static_string s = {0, NULL};
        // 加载字符串并检查是否为空
        int is_null = NpyString_load(allocator, ps, &s);
        // 如果加载失败，设置内存错误并跳转到错误处理标签
        if (is_null == -1) {
            PyErr_SetString(
                    PyExc_MemoryError,
                    "Failed to load string in string to datetime cast");
            goto fail;
        }
        // 如果是空字符串
        if (is_null) {
            // 如果允许空值且不包含字符串 NA，设置为默认字符串
            if (has_null && !has_string_na) {
                *out = NPY_DATETIME_NAT;
                goto next_step;
            }
            s = *default_string;
        }
        // 如果是 NA 字符串
        if (is_nat_string(&s)) {
            *out = NPY_DATETIME_NAT;
            goto next_step;
        }

        // 从静态字符串创建 Python Unicode 对象
        PyObject *pystr = PyUnicode_FromStringAndSize(s.buf, s.size);
        if (pystr == NULL) {
            goto fail;
        }

        // 解释为十进制整数
        PyObject *pylong_value = PyLong_FromUnicodeObject(pystr, 10);
        Py_DECREF(pystr);
        if (pylong_value == NULL) {
            goto fail;
        }

        // 将 Python 长整型对象转换为 C 长整型
        npy_longlong value = PyLong_AsLongLong(pylong_value);
        Py_DECREF(pylong_value);
        if (value == -1 && PyErr_Occurred()) {
            goto fail;
        }

        // 将结果赋给输出的 timedelta 数组
        *out = (npy_timedelta)value;

    next_step:
        // 移动输入和输出指针到下一个元素
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器的资源
    NpyString_release_allocator(allocator);
    // 返回成功
    return 0;

fail:
    // 释放字符串分配器的资源
    NpyString_release_allocator(allocator);
    // 返回失败
    return -1;
}

static PyType_Slot s2td_slots[] = {
        // 方法插槽：解析描述符，使用字符串到日期时间类型转换的描述符
        {NPY_METH_resolve_descriptors,
         &string_to_datetime_timedelta_resolve_descriptors},
        // 方法插槽：使用字符串到 timedelta 类型的转换方法
        {NPY_METH_strided_loop, &string_to_timedelta},
        {0, NULL}};

static char *s2td_name = "cast_StringDType_to_Timedelta";

// timedelta to string

static int
// 将 timedelta 转换为字符串
timedelta_to_string(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取第一个维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数据指针，并转换为 npy_timedelta 类型
    npy_timedelta *in = (npy_timedelta *)data[0];
    // 获取输出数据指针
    char *out = data[1];

    // 计算输入数据的步长和输出数据的步长
    npy_intp in_stride = strides[0] / sizeof(npy_timedelta);
    npy_intp out_stride = strides[1];

    // 获取字符串类型描述符
    PyArray_StringDTypeObject *sdescr = (PyArray_StringDTypeObject *)context->descriptors[1];
    // 检查是否有空值对象
    int has_null = sdescr->na_object != NULL;
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(sdescr);

    // 循环处理每个元素
    while (N--) {
        // 输出数据的静态字符串指针
        npy_packed_static_string *out_pss = (npy_packed_static_string *)out;
        // 如果输入是 NaT（Not a Time），处理特殊情况
        if (*in == NPY_DATETIME_NAT)
        {
            // 如果没有空值对象，将 "NaT" 打包到输出静态字符串中
            if (!has_null) {
                npy_static_string os = {3, "NaT"};
                // 使用字符串分配器打包字符串到输出静态字符串指针中
                if (NpyString_pack(allocator, out_pss, os.buf, os.size) < 0) {
                    // 如果打包失败，抛出内存错误并跳转到失败处理标签
                    npy_gil_error(
                            PyExc_MemoryError,
                            "Failed to pack string in timedelta to string "
                            "cast");
                    goto fail;
                }
            }
            // 如果有空值对象，使用 NpyString_pack_null 函数打包空值对象
            else if (NpyString_pack_null(allocator, out_pss) < 0) {
                npy_gil_error(
                        PyExc_MemoryError,
                        "Failed to pack string in timedelta to string cast");
                goto fail;
            }
        }
        // 如果输入不是 NaT，则将其转换为字符串并存储到输出中
        else if (int_to_stringbuf((long long)*in, out, allocator) < 0) {
            // 如果转换失败，跳转到失败处理标签
            goto fail;
        }

        // 更新输入和输出指针
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocator(allocator);
    // 返回成功标志
    return 0;

fail:
    // 失败时释放字符串分配器并返回错误标志
    NpyString_release_allocator(allocator);
    return -1;
}

// timedelta 转换为字符串的方法槽
static PyType_Slot td2s_slots[] = {
        {NPY_METH_resolve_descriptors,
         &any_to_string_SAFE_resolve_descriptors},
        {NPY_METH_strided_loop, &timedelta_to_string},
        {0, NULL}};

// timedelta 转换为字符串的名称
static char *td2s_name = "cast_Timedelta_to_StringDType";

// 字符串到空值的方法：解析描述符
static NPY_CASTING
string_to_void_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *NPY_UNUSED(view_offset))
{
    // 如果给定的描述符为空，抛出类型错误并返回错误标志
    if (given_descrs[1] == NULL) {
        PyErr_SetString(
                PyExc_TypeError,
                "Casting from StringDType to a fixed-width dtype with an "
                "unspecified size is not currently supported, specify "
                "an explicit size for the output dtype instead.");
        return (NPY_CASTING)-1;
    }


这些注释解释了每行代码的作用和功能，按照要求将其包含在代码块中。
    else {
        // 如果给定描述符的类型为结构化空类型，则拒绝
        if (PyDataType_NAMES(given_descrs[1]) != NULL || PyDataType_SUBARRAY(given_descrs[1]) != NULL) {
            // 设置类型错误异常，说明从 StringDType 转换到结构化 dtype 是不支持的
            PyErr_SetString(
                    PyExc_TypeError,
                    "Casting from StringDType to a structured dtype is not "
                    "supported.");
            // 返回错误码表示转换失败
            return (NPY_CASTING)-1;
        }
        // 增加给定描述符的引用计数，确保其在返回前不被销毁
        Py_INCREF(given_descrs[1]);
        // 将给定描述符复制到循环描述符数组的相应位置
        loop_descrs[1] = given_descrs[1];
    }

    // 增加给定描述符的引用计数，确保其在返回前不被销毁
    Py_INCREF(given_descrs[0]);
    // 将给定描述符复制到循环描述符数组的相应位置
    loop_descrs[0] = given_descrs[0];

    // 返回不安全转换的标志，表示允许不安全的类型转换
    return NPY_UNSAFE_CASTING;
}

static int
string_to_void(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入描述符的字符串数据类型对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 检查是否有空值对象
    int has_null = descr->na_object != NULL;
    // 检查是否有字符串NA值
    int has_string_na = descr->has_string_na;
    // 获取默认字符串
    const npy_static_string *default_string = &descr->default_string;
    // 获取NA名称字符串
    const npy_static_string *na_name = &descr->na_name;
    // 获取第一维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数据的指针
    char *in = data[0];
    // 获取输出数据的指针
    char *out = data[1];
    // 获取输入和输出数据的步幅
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];
    // 获取输出数据的最大大小
    size_t max_out_size = context->descriptors[1]->elsize;

    // 遍历输入数据
    while (N--) {
        // 将输入数据解析为可空字符串
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        npy_static_string s = {0, NULL};
        // 加载可空字符串，处理可能的截断或错误情况
        if (load_nullable_string(ps, &s, has_null, has_string_na,
                                 default_string, na_name, allocator,
                                 "in string to void cast") == -1) {
            // 加载失败，跳转到错误处理部分
            goto fail;
        }

        // 将字符串数据复制到输出位置，可能会截断UTF-8字符
        memcpy(out, s.buf, s.size > max_out_size ? max_out_size : s.size);
        // 如果实际字符串大小小于输出大小，用零填充剩余部分
        if (s.size < max_out_size) {
            memset(out + s.size, 0, (max_out_size - s.size));
        }

        // 更新输入和输出指针位置
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocator(allocator);

    // 成功完成，返回0
    return 0;

fail:
    // 处理失败，释放字符串分配器并返回-1
    NpyString_release_allocator(allocator);
    return -1;
}

static PyType_Slot s2v_slots[] = {
    // 解析描述符的方法和函数指针
    {NPY_METH_resolve_descriptors, &string_to_void_resolve_descriptors},
    // 字符串到空值的函数指针
    {NPY_METH_strided_loop, &string_to_void},
    // 结束标记
    {0, NULL}
};

static char *s2v_name = "cast_StringDType_to_Void";

// 空值到字符串

static int
void_to_string(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取描述符数组
    PyArray_Descr *const *descrs = context->descriptors;
    // 获取输出描述符的字符串数据类型对象
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)descrs[1];

    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    // 获取输入描述符的最大大小
    long max_in_size = descrs[0]->elsize;

    // 获取第一维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数据的指针（无符号字符指针）
    unsigned char *in = (unsigned char *)data[0];
    // 获取输出数据的指针
    char *out = data[1];

    // 获取输入和输出数据的步幅
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];
    // 循环，执行 N 次，N 是循环控制变量
    while(N--) {
        // 计算输入数据的 UTF-8 编码后的字节数
        size_t out_num_bytes = utf8_buffer_size(in, max_in_size);
        // 如果计算出的字节数小于 0，表示发现无效的 UTF-8 字节序列，抛出错误并跳转到失败标签
        if (out_num_bytes < 0) {
            npy_gil_error(PyExc_TypeError,
                          "Invalid UTF-8 bytes found, cannot convert to UTF-8");
            goto fail;
        }
        // 定义一个静态字符串结构体 out_ss，并初始化为零
        npy_static_string out_ss = {0, NULL};
        // 调用 load_new_string 函数，将 out 转换为字符串，存储在 out_ss 中，使用 allocator 分配内存，描述为 "void to string cast"
        if (load_new_string((npy_packed_static_string *)out,
                            &out_ss, out_num_bytes, allocator,
                            "void to string cast") == -1) {
            // 如果 load_new_string 返回 -1，表示转换失败，跳转到失败标签
            goto fail;
        }
        // 将输入数据 in 复制到 out_buf 中，长度为 out_num_bytes
        char *out_buf = (char *)out_ss.buf;
        memcpy(out_buf, in, out_num_bytes);

        // 更新输入指针和输出指针的位置
        in += in_stride;
        out += out_stride;
    }

    // 释放 allocator 分配的内存资源
    NpyString_release_allocator(allocator);

    // 返回 0 表示成功
    return 0;
// 释放内存分配器
NpyString_release_allocator(allocator);

// 返回-1，表示函数执行失败
return -1;
}

// 定义 PyType_Slot 结构体数组 v2s_slots，用于描述一个 Python 类型的方法和数据
static PyType_Slot v2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAME_KIND_resolve_descriptors},
                                  {NPY_METH_strided_loop, &void_to_string},
                                  {0, NULL}};

// 字符串变为字节码

// static int 修饰符表明这是一个静态函数，只在当前文件可见
static int
string_to_bytes(PyArrayMethod_Context *context, char *const data[],
                npy_intp const dimensions[], npy_intp const strides[],
                NpyAuxData *NPY_UNUSED(auxdata))
{
    // 从上下文中获取字符串类型描述符
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)context->descriptors[0];
    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);
    // 检查是否有空值对象
    int has_null = descr->na_object != NULL;
    // 检查是否有字符串 NA
    int has_string_na = descr->has_string_na;
    // 获取默认字符串
    const npy_static_string *default_string = &descr->default_string;
    // 获取 NA 名称
    const npy_static_string *na_name = &descr->na_name;
    // 获取第一个维度的大小
    npy_intp N = dimensions[0];
    // 获取输入数据的指针
    char *in = data[0];
    // 获取输出数据的指针
    char *out = data[1];
    // 获取输入数据的步幅
    npy_intp in_stride = strides[0];
    // 获取输出数据的步幅
    npy_intp out_stride = strides[1];
    // 获取输出数据的最大大小
    size_t max_out_size = context->descriptors[1]->elsize;

    // 迭代处理每一个输入字符串
    while (N--) {
        // 将输入数据解析为紧凑字符串结构体
        const npy_packed_static_string *ps = (npy_packed_static_string *)in;
        // 创建静态字符串结构体 s
        npy_static_string s = {0, NULL};
        // 加载可空字符串，如果加载失败则跳转到失败标签
        if (load_nullable_string(ps, &s, has_null, has_string_na,
                                 default_string, na_name, allocator,
                                 "in string to bytes cast") == -1) {
            goto fail;
        }

        // 检查字符串中是否有超过 127 的 ASCII 字符
        for (size_t i=0; i<s.size; i++) {
            if (((unsigned char *)s.buf)[i] > 127) {
                // 异常处理：ASCII 转换错误
                NPY_ALLOW_C_API_DEF;
                NPY_ALLOW_C_API;
                // 创建并设置 UnicodeEncodeError 异常对象
                PyObject *exc = PyObject_CallFunction(
                        PyExc_UnicodeEncodeError, "ss#nns", "ascii", s.buf,
                        (Py_ssize_t)s.size, (Py_ssize_t)i, (Py_ssize_t)(i+1), "ordinal not in range(128)");
                PyErr_SetObject(PyExceptionInstance_Class(exc), exc);
                Py_DECREF(exc);
                NPY_DISABLE_C_API;
                // 转换失败，跳转到失败标签
                goto fail;
            }
        }

        // 复制字符串数据到输出缓冲区，不超过最大输出大小
        memcpy(out, s.buf, s.size > max_out_size ? max_out_size : s.size);
        // 如果字符串大小小于最大输出大小，填充剩余空间
        if (s.size < max_out_size) {
            memset(out + s.size, 0, (max_out_size - s.size));
        }

        // 更新输入和输出指针
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocator(allocator);

    // 返回成功
    return 0;

// 失败处理标签
fail:

    // 释放字符串分配器
    NpyString_release_allocator(allocator);

    // 返回 -1，表示函数执行失败
    return -1;
}

// 定义 PyType_Slot 结构体数组 s2bytes_slots，用于描述一个 Python 类型的方法和数据
static PyType_Slot s2bytes_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_fixed_width_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_bytes},
        {0, NULL}};

// 字节码变为字符串

// static int 修饰符表明这是一个静态函数，只在当前文件可见
static int
// 将字节转换为字符串的函数，使用给定的上下文、数据、维度、步长和辅助数据
static void bytes_to_string(PyArrayMethod_Context *context, char *const data[],
                            npy_intp const dimensions[], npy_intp const strides[],
                            NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取描述符数组
    PyArray_Descr *const *descrs = context->descriptors;
    // 获取第二个描述符作为字符串描述符
    PyArray_StringDTypeObject *descr = (PyArray_StringDTypeObject *)descrs[1];

    // 获取字符串分配器
    npy_string_allocator *allocator = NpyString_acquire_allocator(descr);

    // 获取最大输入大小
    size_t max_in_size = descrs[0]->elsize;

    // 获取第一个维度
    npy_intp N = dimensions[0];
    // 输入数据的无符号字符指针
    unsigned char *in = (unsigned char *)data[0];
    // 输出数据的字符指针
    char *out = data[1];

    // 输入和输出的步长
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    // 循环处理每一个元素
    while(N--) {
        // 初始化输出字节数
        size_t out_num_bytes = max_in_size;

        // 忽略末尾的空字符
        while (out_num_bytes > 0 && in[out_num_bytes - 1] == 0) {
            out_num_bytes--;
        }

        // 定义静态字符串结构体
        npy_static_string out_ss = {0, NULL};
        // 调用加载新字符串的函数，将输入转换为输出静态字符串
        if (load_new_string((npy_packed_static_string *)out,
                            &out_ss, out_num_bytes, allocator,
                            "void to string cast") == -1) {
            // 失败时跳转到标签fail
            goto fail;
        }

        // 将输入复制到输出缓冲区
        char *out_buf = (char *)out_ss.buf;
        memcpy(out_buf, in, out_num_bytes);

        // 更新输入和输出指针
        in += in_stride;
        out += out_stride;
    }

    // 释放字符串分配器
    NpyString_release_allocator(allocator);

    return;

fail:
    // 失败时同样释放字符串分配器，并返回-1
    NpyString_release_allocator(allocator);

    return;
}

// 定义静态类型槽数组，用于描述类型转换的方法
static PyType_Slot bytes2s_slots[] = {
    // 解析描述符的方法和解析相同类型的字符串描述符的方法
    {NPY_METH_resolve_descriptors, &any_to_string_SAME_KIND_resolve_descriptors},
    // 使用字节转换为字符串的方法
    {NPY_METH_strided_loop, &bytes_to_string},
    // 空槽
    {0, NULL}
};

// 定义将字节转换为字符串的名称
static char *bytes2s_name = "cast_Bytes_to_StringDType";

// 返回描述类型转换的方法规范的函数
PyArrayMethod_Spec *
get_cast_spec(const char *name, NPY_CASTING casting,
              NPY_ARRAYMETHOD_FLAGS flags, PyArray_DTypeMeta **dtypes,
              PyType_Slot *slots)
{
    // 分配内存以存储方法规范
    PyArrayMethod_Spec *ret = PyMem_Malloc(sizeof(PyArrayMethod_Spec));

    // 设置方法规范的属性
    ret->name = name;
    ret->nin = 1;
    ret->nout = 1;
    ret->casting = casting;
    ret->flags = flags;
    ret->dtypes = dtypes;
    ret->slots = slots;

    return ret;
}

// 返回两个数据类型元信息的数组
PyArray_DTypeMeta **
get_dtypes(PyArray_DTypeMeta *dt1, PyArray_DTypeMeta *dt2)
{
    // 分配内存以存储两个数据类型元信息的数组
    PyArray_DTypeMeta **ret = PyMem_Malloc(2 * sizeof(PyArray_DTypeMeta *));

    // 设置数组的元素为输入的数据类型元信息
    ret[0] = dt1;
    ret[1] = dt2;

    return ret;
}

// 返回类型转换的方法规范的数组的函数
PyArrayMethod_Spec **
get_casts()
{
    // 使用字符串到字符串转换的名称
    char *t2t_name = s2s_name;

    // 获取字符串到字符串的数据类型元信息数组
    PyArray_DTypeMeta **t2t_dtypes =
            get_dtypes(&PyArray_StringDType,
                       &PyArray_StringDType);

    // 获取字符串到字符串的方法规范
    PyArrayMethod_Spec *ThisToThisCastSpec =
            get_cast_spec(t2t_name, NPY_UNSAFE_CASTING,
                          NPY_METH_SUPPORTS_UNALIGNED, t2t_dtypes, s2s_slots);

    // 初始类型转换的数量
    int num_casts = 43;

    // 根据字节大小添加类型转换数量
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    num_casts += 4;
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    num_casts += 4;
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    num_casts += 4;
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    num_casts += 4;
#endif
#endif

    // 获取 Unicode 到 String 的数据类型元信息数组
    PyArray_DTypeMeta **u2s_dtypes = get_dtypes(
            &PyArray_UnicodeDType, &PyArray_StringDType);

    // 获取 Unicode 到 String 类型转换方法的规范
    PyArrayMethod_Spec *UnicodeToStringCastSpec = get_cast_spec(
            u2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            u2s_dtypes, u2s_slots);

    // 获取 String 到 Unicode 的数据类型元信息数组
    PyArray_DTypeMeta **s2u_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_UnicodeDType);

    // 获取 String 到 Unicode 类型转换方法的规范
    PyArrayMethod_Spec *StringToUnicodeCastSpec = get_cast_spec(
            s2u_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2u_dtypes, s2u_slots);

    // 获取 String 到 Bool 的数据类型元信息数组
    PyArray_DTypeMeta **s2b_dtypes =
            get_dtypes(&PyArray_StringDType, &PyArray_BoolDType);

    // 获取 String 到 Bool 类型转换方法的规范
    PyArrayMethod_Spec *StringToBoolCastSpec = get_cast_spec(
            s2b_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2b_dtypes, s2b_slots);

    // 获取 Bool 到 String 的数据类型元信息数组
    PyArray_DTypeMeta **b2s_dtypes =
            get_dtypes(&PyArray_BoolDType, &PyArray_StringDType);

    // 获取 Bool 到 String 类型转换方法的规范
    PyArrayMethod_Spec *BoolToStringCastSpec = get_cast_spec(
            b2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            b2s_dtypes, b2s_slots);

    // 定义各个整数类型到相应的类型转换规范
    DTYPES_AND_CAST_SPEC(i8, Int8)
    DTYPES_AND_CAST_SPEC(i16, Int16)
    DTYPES_AND_CAST_SPEC(i32, Int32)
    DTYPES_AND_CAST_SPEC(i64, Int64)
    DTYPES_AND_CAST_SPEC(u8, UInt8)
    DTYPES_AND_CAST_SPEC(u16, UInt16)
    DTYPES_AND_CAST_SPEC(u32, UInt32)
    DTYPES_AND_CAST_SPEC(u64, UInt64)
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    DTYPES_AND_CAST_SPEC(byte, Byte)
    DTYPES_AND_CAST_SPEC(ubyte, UByte)
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    DTYPES_AND_CAST_SPEC(short, Short)
    DTYPES_AND_CAST_SPEC(ushort, UShort)
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    DTYPES_AND_CAST_SPEC(int, Int)
    DTYPES_AND_CAST_SPEC(uint, UInt)
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    DTYPES_AND_CAST_SPEC(longlong, LongLong)
    DTYPES_AND_CAST_SPEC(ulonglong, ULongLong)
#endif

    // 定义各个浮点数类型到相应的类型转换规范
    DTYPES_AND_CAST_SPEC(f64, Double)
    DTYPES_AND_CAST_SPEC(f32, Float)
    DTYPES_AND_CAST_SPEC(f16, Half)

    // 获取 String 到 Datetime 的数据类型元信息数组
    PyArray_DTypeMeta **s2dt_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_DatetimeDType);

    // 获取 String 到 Datetime 类型转换方法的规范
    PyArrayMethod_Spec *StringToDatetimeCastSpec = get_cast_spec(
            s2dt_name, NPY_UNSAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            s2dt_dtypes, s2dt_slots);

    // 获取 Datetime 到 String 的数据类型元信息数组
    PyArray_DTypeMeta **dt2s_dtypes = get_dtypes(
            &PyArray_DatetimeDType, &PyArray_StringDType);

    // 获取 Datetime 到 String 类型转换方法的规范
    PyArrayMethod_Spec *DatetimeToStringCastSpec = get_cast_spec(
            dt2s_name, NPY_SAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            dt2s_dtypes, dt2s_slots);

    // 获取 String 到 Timedelta 的数据类型元信息数组
    PyArray_DTypeMeta **s2td_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_TimedeltaDType);
    // 创建 StringToTimedeltaCastSpec 变量，并使用 get_cast_spec 函数获取字符串到时间增量的转换规格
    PyArrayMethod_Spec *StringToTimedeltaCastSpec = get_cast_spec(
            s2td_name, NPY_UNSAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            s2td_dtypes, s2td_slots);

    // 使用 get_dtypes 函数获取 PyArray_TimedeltaDType 和 PyArray_StringDType 的数据类型元信息，存储在 td2s_dtypes 中
    PyArray_DTypeMeta **td2s_dtypes = get_dtypes(
            &PyArray_TimedeltaDType, &PyArray_StringDType);

    // 创建 TimedeltaToStringCastSpec 变量，并使用 get_cast_spec 函数获取时间增量到字符串的转换规格
    PyArrayMethod_Spec *TimedeltaToStringCastSpec = get_cast_spec(
            td2s_name, NPY_SAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            td2s_dtypes, td2s_slots);

    // 使用 get_dtypes 函数获取 PyArray_StringDType 和 PyArray_LongDoubleDType 的数据类型元信息，存储在 s2ld_dtypes 中
    PyArray_DTypeMeta **s2ld_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_LongDoubleDType);

    // 创建 StringToLongDoubleCastSpec 变量，并使用 get_cast_spec 函数获取字符串到长双精度浮点数的转换规格
    PyArrayMethod_Spec *StringToLongDoubleCastSpec = get_cast_spec(
            s2ld_name, NPY_UNSAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2ld_dtypes, s2ld_slots);

    // 使用 get_dtypes 函数获取 PyArray_LongDoubleDType 和 PyArray_StringDType 的数据类型元信息，存储在 ld2s_dtypes 中
    PyArray_DTypeMeta **ld2s_dtypes = get_dtypes(
            &PyArray_LongDoubleDType, &PyArray_StringDType);

    // 创建 LongDoubleToStringCastSpec 变量，并使用 get_cast_spec 函数获取长双精度浮点数到字符串的转换规格
    PyArrayMethod_Spec *LongDoubleToStringCastSpec = get_cast_spec(
            ld2s_name, NPY_SAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            ld2s_dtypes, ld2s_slots);

    // 调用宏 DTYPES_AND_CAST_SPEC 生成有关复数浮点数到浮点数的转换规格代码
    DTYPES_AND_CAST_SPEC(cfloat, CFloat)
    DTYPES_AND_CAST_SPEC(cdouble, CDouble)
    DTYPES_AND_CAST_SPEC(clongdouble, CLongDouble)

    // 使用 get_dtypes 函数获取 PyArray_StringDType 和 PyArray_VoidDType 的数据类型元信息，存储在 s2v_dtypes 中
    PyArray_DTypeMeta **s2v_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_VoidDType);

    // 创建 StringToVoidCastSpec 变量，并使用 get_cast_spec 函数获取字符串到 void 类型的转换规格
    PyArrayMethod_Spec *StringToVoidCastSpec = get_cast_spec(
            s2v_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2v_dtypes, s2v_slots);

    // 使用 get_dtypes 函数获取 PyArray_VoidDType 和 PyArray_StringDType 的数据类型元信息，存储在 v2s_dtypes 中
    PyArray_DTypeMeta **v2s_dtypes = get_dtypes(
            &PyArray_VoidDType, &PyArray_StringDType);

    // 创建 VoidToStringCastSpec 变量，并使用 get_cast_spec 函数获取 void 类型到字符串的转换规格
    PyArrayMethod_Spec *VoidToStringCastSpec = get_cast_spec(
            v2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            v2s_dtypes, v2s_slots);

    // 使用 get_dtypes 函数获取 PyArray_StringDType 和 PyArray_BytesDType 的数据类型元信息，存储在 s2bytes_dtypes 中
    PyArray_DTypeMeta **s2bytes_dtypes = get_dtypes(
            &PyArray_StringDType, &PyArray_BytesDType);

    // 创建 StringToBytesCastSpec 变量，并使用 get_cast_spec 函数获取字符串到字节流的转换规格
    PyArrayMethod_Spec *StringToBytesCastSpec = get_cast_spec(
            s2bytes_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2bytes_dtypes, s2bytes_slots);

    // 使用 get_dtypes 函数获取 PyArray_BytesDType 和 PyArray_StringDType 的数据类型元信息，存储在 bytes2s_dtypes 中
    PyArray_DTypeMeta **bytes2s_dtypes = get_dtypes(
            &PyArray_BytesDType, &PyArray_StringDType);

    // 创建 BytesToStringCastSpec 变量，并使用 get_cast_spec 函数获取字节流到字符串的转换规格
    PyArrayMethod_Spec *BytesToStringCastSpec = get_cast_spec(
            bytes2s_name, NPY_SAME_KIND_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            bytes2s_dtypes, bytes2s_slots);

    // 分配内存以存储所有转换规格的数组，并将指针存储在 casts 中
    PyArrayMethod_Spec **casts =
            PyMem_Malloc((num_casts + 1) * sizeof(PyArrayMethod_Spec *));

    // 初始化计数器 cast_i
    int cast_i = 0;

    // 将各种转换规格添加到 casts 数组中
    casts[cast_i++] = ThisToThisCastSpec;
    casts[cast_i++] = UnicodeToStringCastSpec;
    casts[cast_i++] = StringToUnicodeCastSpec;
    casts[cast_i++] = StringToBoolCastSpec;
    casts[cast_i++] = BoolToStringCastSpec;
    casts[cast_i++] = StringToInt8CastSpec;
    casts[cast_i++] = Int8ToStringCastSpec;
    casts[cast_i++] = StringToInt16CastSpec;
    # 将 Int16ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = Int16ToStringCastSpec;
    # 将 StringToInt32CastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = StringToInt32CastSpec;
    # 将 Int32ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = Int32ToStringCastSpec;
    # 将 StringToInt64CastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = StringToInt64CastSpec;
    # 将 Int64ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = Int64ToStringCastSpec;
    # 将 StringToUInt8CastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = StringToUInt8CastSpec;
    # 将 UInt8ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = UInt8ToStringCastSpec;
    # 将 StringToUInt16CastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = StringToUInt16CastSpec;
    # 将 UInt16ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = UInt16ToStringCastSpec;
    # 将 StringToUInt32CastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = StringToUInt32CastSpec;
    # 将 UInt32ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = UInt32ToStringCastSpec;
    # 将 StringToUInt64CastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = StringToUInt64CastSpec;
    # 将 UInt64ToStringCastSpec 转换函数添加到 casts 数组中，cast_i 自增
    casts[cast_i++] = UInt64ToStringCastSpec;
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    // 如果字节大小等于短整型大小，添加字节到短整型的类型转换函数到 casts 数组中
    casts[cast_i++] = StringToByteCastSpec;
    casts[cast_i++] = ByteToStringCastSpec;
    casts[cast_i++] = StringToUByteCastSpec;
    casts[cast_i++] = UByteToStringCastSpec;
#endif

#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    // 如果短整型大小等于整型大小，添加短整型到整型的类型转换函数到 casts 数组中
    casts[cast_i++] = StringToShortCastSpec;
    casts[cast_i++] = ShortToStringCastSpec;
    casts[cast_i++] = StringToUShortCastSpec;
    casts[cast_i++] = UShortToStringCastSpec;
#endif

#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    // 如果整型大小等于长整型大小，添加整型到长整型的类型转换函数到 casts 数组中
    casts[cast_i++] = StringToIntCastSpec;
    casts[cast_i++] = IntToStringCastSpec;
    casts[cast_i++] = StringToUIntCastSpec;
    casts[cast_i++] = UIntToStringCastSpec;
#endif

#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    // 如果长长整型大小等于长整型大小，添加长长整型到长整型的类型转换函数到 casts 数组中
    casts[cast_i++] = StringToLongLongCastSpec;
    casts[cast_i++] = LongLongToStringCastSpec;
    casts[cast_i++] = StringToULongLongCastSpec;
    casts[cast_i++] = ULongLongToStringCastSpec;
#endif

// 添加字符串到双精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToDoubleCastSpec;
casts[cast_i++] = DoubleToStringCastSpec;
// 添加字符串到单精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToFloatCastSpec;
casts[cast_i++] = FloatToStringCastSpec;
// 添加字符串到半精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToHalfCastSpec;
casts[cast_i++] = HalfToStringCastSpec;
// 添加字符串到日期时间的类型转换函数到 casts 数组中
casts[cast_i++] = StringToDatetimeCastSpec;
casts[cast_i++] = DatetimeToStringCastSpec;
// 添加字符串到时间增量的类型转换函数到 casts 数组中
casts[cast_i++] = StringToTimedeltaCastSpec;
casts[cast_i++] = TimedeltaToStringCastSpec;
// 添加字符串到长双精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToLongDoubleCastSpec;
casts[cast_i++] = LongDoubleToStringCastSpec;
// 添加字符串到复数单精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToCFloatCastSpec;
casts[cast_i++] = CFloatToStringCastSpec;
// 添加字符串到复数双精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToCDoubleCastSpec;
casts[cast_i++] = CDoubleToStringCastSpec;
// 添加字符串到复数长双精度浮点数的类型转换函数到 casts 数组中
casts[cast_i++] = StringToCLongDoubleCastSpec;
casts[cast_i++] = CLongDoubleToStringCastSpec;
// 添加字符串到无类型的类型转换函数到 casts 数组中
casts[cast_i++] = StringToVoidCastSpec;
casts[cast_i++] = VoidToStringCastSpec;
// 添加字符串到字节串的类型转换函数到 casts 数组中
casts[cast_i++] = StringToBytesCastSpec;
casts[cast_i++] = BytesToStringCastSpec;
// 将 NULL 添加到 casts 数组末尾作为结束标志
casts[cast_i++] = NULL;

// 断言最后一个元素为 NULL，确保 casts 数组以 NULL 结束
assert(casts[num_casts] == NULL);
// 断言 cast_i 的值为 num_casts + 1，确保 casts 数组中元素的数量正确
assert(cast_i == num_casts + 1);

// 返回填充完毕的 casts 数组
return casts;
}
```