# `.\numpy\numpy\_core\src\umath\string_ufuncs.cpp`

```
/*
 * 包含必要的头文件：Python.h 用于 Python C API，以及与 NumPy 相关的头文件。
 */
#include <Python.h>
#include <string.h>

/*
 * 定义 NPY_NO_DEPRECATED_API 以及 _MULTIARRAYMODULE 和 _UMATHMODULE 宏。
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE
#define _UMATHMODULE

/*
 * 包含 NumPy 相关的头文件，用于数组对象、数据类型、数学函数以及通用函数。
 */
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"
#include "numpy/ufuncobject.h"

/*
 * 包含其他辅助文件的头文件，如操作系统相关、调度、数据类型元信息、数据类型转换以及 GIL 工具等。
 */
#include "numpyos.h"
#include "dispatching.h"
#include "dtypemeta.h"
#include "convert_datatype.h"
#include "gil_utils.h"

/*
 * 包含与字符串相关的特定 NumPy 代码头文件，如字符串通用函数、快速搜索和字符串缓冲区。
 */
#include "string_ufuncs.h"
#include "string_fastsearch.h"
#include "string_buffer.h"


/*
 * Helper for templating, avoids warnings about uncovered switch paths.
 * 用于模板化的辅助函数，避免了未覆盖的 switch 分支的警告。
 */
enum class COMP {
    EQ, NE, LT, LE, GT, GE,
};

/*
 * 返回 COMP 枚举类型的字符串表示，用于比较操作的名称。
 */
static char const *
comp_name(COMP comp) {
    switch(comp) {
        case COMP::EQ: return "equal";
        case COMP::NE: return "not_equal";
        case COMP::LT: return "less";
        case COMP::LE: return "less_equal";
        case COMP::GT: return "greater";
        case COMP::GE: return "greater_equal";
        default:
            assert(0);  // 断言：如果出现未知的 COMP 枚举类型，则终止程序
            return nullptr;
    }
}


/*
 * 字符串比较的主循环模板函数，支持不同的比较类型和编码。
 */
template <bool rstrip, COMP comp, ENCODING enc>
static int
string_comparison_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    /*
     * 注意：从描述符中获取 elsize 是安全的，即使没有全局解释器锁 (GIL)，
     * 但最终可能会将此操作移到 auxdata 中，这样可能会稍微更快/更清晰（但更复杂）。
     */
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    npy_intp N = dimensions[0];

    while (N--) {
        // 使用指定的编码创建缓冲区对象 buf1 和 buf2
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        // 执行字符串比较，返回比较结果 cmp
        int cmp = buf1.strcmp(buf2, rstrip);
        npy_bool res;
        // 根据比较类型 comp 进行条件判断
        switch (comp) {
            case COMP::EQ:
                res = cmp == 0;
                break;
            case COMP::NE:
                res = cmp != 0;
                break;
            case COMP::LT:
                res = cmp < 0;
                break;
            case COMP::LE:
                res = cmp <= 0;
                break;
            case COMP::GT:
                res = cmp > 0;
                break;
            case COMP::GE:
                res = cmp >= 0;
                break;
        }
        // 将比较结果写入输出数组
        *(npy_bool *)out = res;

        // 更新输入和输出指针位置
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}


/*
 * 计算字符串长度的主循环模板函数，支持不同的编码。
 */
template <ENCODING enc>
static int
string_str_len_loop(PyArrayMethod_Context *context,
                    char *const data[], npy_intp const dimensions[],
                    npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    int elsize = context->descriptors[0]->elsize;

    char *in = data[0];
    char *out = data[1];

    npy_intp N = dimensions[0];
    # 使用循环，执行 N 次迭代
    while (N--) {
        # 从输入流 `in` 中读取 `elsize` 大小的数据到 `Buffer<enc>` 类型的缓冲区 `buf`
        Buffer<enc> buf(in, elsize);
        
        # 将 `buf` 缓冲区中的字符数（codepoints）赋值给 `out` 指向的内存地址（假设为整数类型）
        *(npy_intp *)out = buf.num_codepoints();

        # 更新输入流 `in` 的位置，移动到下一个元素的起始位置，根据 `strides[0]` 的大小
        in += strides[0];
        
        # 更新输出流 `out` 的位置，移动到下一个元素的起始位置，根据 `strides[1]` 的大小
        out += strides[1];
    }

    # 循环结束后返回值 `0`
    return 0;
// 定义模板函数，用于获取缓冲区操作方法的指针
template <ENCODING enc>
using buffer_method = bool (Buffer<enc>::*)();

// 字符串一元操作循环函数，处理输入数据并执行相应操作
template <ENCODING enc>
static int
string_unary_loop(PyArrayMethod_Context *context,
                  char *const data[], npy_intp const dimensions[],
                  npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 从上下文中获取缓冲区操作方法的指针
    buffer_method<enc> is_it = *(buffer_method<enc> *)(context->method->static_data);
    // 获取第一个输入缓冲区元素的大小
    int elsize = context->descriptors[0]->elsize;

    // 初始化输入和输出缓冲区指针
    char *in = data[0];
    char *out = data[1];

    // 获取处理的数据维度大小
    npy_intp N = dimensions[0];

    // 遍历每个数据元素进行操作
    while (N--) {
        // 创建输入缓冲区对象
        Buffer<enc> buf(in, elsize);
        // 执行指定的一元操作，并将结果存储到输出缓冲区
        *(npy_bool *)out = (buf.*is_it)();

        // 更新输入和输出缓冲区指针，以处理下一个元素
        in += strides[0];
        out += strides[1];
    }

    return 0;
}


// 字符串相加操作，将两个输入缓冲区的内容逐字节复制到输出缓冲区中
template <ENCODING enc>
static inline void
string_add(Buffer<enc> buf1, Buffer<enc> buf2, Buffer<enc> out)
{
    // 获取输入缓冲区1和2的字符数
    size_t len1 = buf1.num_codepoints();
    size_t len2 = buf2.num_codepoints();
    // 将缓冲区1的内容复制到输出缓冲区的起始位置
    buf1.buffer_memcpy(out, len1);
    // 将缓冲区2的内容复制到输出缓冲区的后续位置
    buf2.buffer_memcpy(out + len1, len2);
    // 将输出缓冲区剩余部分填充为零
    out.buffer_fill_with_zeros_after_index(len1 + len2);
}


// 字符串乘法操作，将输入缓冲区的内容重复指定次数，并存储到输出缓冲区中
template <ENCODING enc>
static inline void
string_multiply(Buffer<enc> buf1, npy_int64 reps, Buffer<enc> out)
{
    // 获取输入缓冲区的字符数
    size_t len1 = buf1.num_codepoints();
    
    // 如果重复次数小于1或输入缓冲区长度为0，则直接在输出缓冲区填充零
    if (reps < 1 || len1 == 0) {
        out.buffer_fill_with_zeros_after_index(0);
        return;
    }

    // 如果输入缓冲区长度为1，则使用单字符填充输出缓冲区，并在后续位置填充零
    if (len1 == 1) {
        out.buffer_memset(*buf1, reps);
        out.buffer_fill_with_zeros_after_index(reps);
    }
    else {
        // 否则，将输入缓冲区的内容重复指定次数复制到输出缓冲区中
        for (npy_int64 i = 0; i < reps; i++) {
            buf1.buffer_memcpy(out, len1);
            out += len1;
        }
        // 在复制结束后，将输出缓冲区剩余部分填充为零
        out.buffer_fill_with_zeros_after_index(0);
    }
}


// 字符串相加操作循环函数，对每对输入数据执行字符串相加操作
template <ENCODING enc>
static int
string_add_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取每个输入缓冲区元素的大小
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int outsize = context->descriptors[2]->elsize;

    // 初始化输入和输出缓冲区指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    // 获取处理的数据维度大小
    npy_intp N = dimensions[0];

    // 遍历每对数据元素进行操作
    while (N--) {
        // 创建输入缓冲区对象1和2，以及输出缓冲区对象
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf(out, outsize);
        // 执行字符串相加操作，并将结果存储到输出缓冲区
        string_add<enc>(buf1, buf2, outbuf);

        // 更新输入和输出缓冲区指针，以处理下一对元素
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}


// 字符串乘法操作循环函数，对每对输入数据执行字符串乘法操作
template <ENCODING enc>
static int
string_multiply_strint_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入缓冲区元素的大小和输出缓冲区元素的大小
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[2]->elsize;

    // 初始化输入和输出缓冲区指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];

    // 获取处理的数据维度大小
    npy_intp N = dimensions[0];

    // 遍历每对数据元素进行操作
    while (N--) {
        // 创建输入缓冲区对象1和2，以及输出缓冲区对象
        Buffer<enc> buf1(in1, elsize);
        Buffer<enc> buf2(in2, elsize);
        Buffer<enc> outbuf(out, outsize);
        // 执行字符串乘法操作，并将结果存储到输出缓冲区
        string_multiply<enc>(buf1, *(npy_int64 *)in2, outbuf);

        // 更新输入和输出缓冲区指针，以处理下一对元素
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    return 0;
}
    while (N--) {
        // 使用输入数据 in1 创建一个带有指定元素大小 elsize 的缓冲区 buf
        Buffer<enc> buf(in1, elsize);
        // 使用输出数据 out 创建一个带有指定输出大小 outsize 的缓冲区 outbuf
        Buffer<enc> outbuf(out, outsize);
        // 调用 string_multiply 函数，将 buf 中的数据乘以 in2 中的第一个 64 位整数，结果存入 outbuf
        string_multiply<enc>(buf, *(npy_int64 *)in2, outbuf);

        // 更新输入数据 in1 的位置，根据 strides[0] 调整
        in1 += strides[0];
        // 更新输入数据 in2 的位置，根据 strides[1] 调整
        in2 += strides[1];
        // 更新输出数据 out 的位置，根据 strides[2] 调整
        out += strides[2];
    }

    // 函数执行完毕，返回状态值 0
    return 0;
// 定义字符串乘法的循环处理函数模板，模板参数为编码类型ENCODING
static int
string_multiply_intstr_loop(PyArrayMethod_Context *context,
                char *const data[], npy_intp const dimensions[],
                npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取第二个输入数组元素大小和输出数组元素大小
    int elsize = context->descriptors[1]->elsize;
    int outsize = context->descriptors[2]->elsize;

    // 获取输入和输出数组的指针
    char *in1 = data[0];  // 第一个输入数组指针
    char *in2 = data[1];  // 第二个输入数组指针
    char *out = data[2];  // 输出数组指针

    // 获取第一维度大小
    npy_intp N = dimensions[0];

    // 循环处理每个元素
    while (N--) {
        // 创建第二个输入数组的缓冲区对象
        Buffer<enc> buf(in2, elsize);
        // 创建输出数组的缓冲区对象
        Buffer<enc> outbuf(out, outsize);
        // 调用字符串乘法函数，将结果写入输出数组
        string_multiply<enc>(buf, *(npy_int64 *)in1, outbuf);

        // 更新输入和输出数组的指针，根据步长
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    // 返回处理结果
    return 0;
}


// 定义查找类似函数的循环处理函数模板，模板参数为编码类型ENCODING
template <ENCODING enc>
static int
string_findlike_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取查找函数指针
    findlike_function<enc> function = *(findlike_function<enc>)(context->method->static_data);
    // 获取输入数组元素大小
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;

    // 获取输入和输出数组的指针
    char *in1 = data[0];  // 第一个输入数组指针
    char *in2 = data[1];  // 第二个输入数组指针
    char *in3 = data[2];  // 第三个输入数组指针
    char *in4 = data[3];  // 第四个输入数组指针
    char *out = data[4];  // 输出数组指针

    // 获取第一维度大小
    npy_intp N = dimensions[0];

    // 循环处理每个元素
    while (N--) {
        // 创建第一个和第二个输入数组的缓冲区对象
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        // 调用查找类似函数，获取返回值
        npy_intp idx = function(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4);
        // 如果返回值为-2，直接返回-1
        if (idx == -2) {
            return -1;
        }
        // 将返回值写入输出数组
        *(npy_intp *)out = idx;

        // 更新输入和输出数组的指针，根据步长
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    // 返回处理结果
    return 0;
}


// 定义字符串替换的循环处理函数模板，模板参数为编码类型ENCODING
template <ENCODING enc>
static int
string_replace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入数组元素大小和输出数组元素大小
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int elsize3 = context->descriptors[2]->elsize;
    int outsize = context->descriptors[4]->elsize;

    // 获取输入和输出数组的指针
    char *in1 = data[0];  // 第一个输入数组指针
    char *in2 = data[1];  // 第二个输入数组指针
    char *in3 = data[2];  // 第三个输入数组指针
    char *in4 = data[3];  // 第四个输入数组指针
    char *out = data[4];  // 输出数组指针

    // 获取第一维度大小
    npy_intp N = dimensions[0];

    // 循环处理每个元素
    while (N--) {
        // 创建输入数组的缓冲区对象
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> buf3(in3, elsize3);
        // 创建输出数组的缓冲区对象
        Buffer<enc> outbuf(out, outsize);
        // 调用字符串替换函数，将结果写入输出数组
        string_replace(buf1, buf2, buf3, *(npy_int64 *) in4, outbuf);

        // 更新输入和输出数组的指针，根据步长
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    // 返回处理结果
    return 0;
}
// 定义函数 string_startswith_endswith_loop，接受以下参数：上下文 context，数据指针数组 data，维度数组 dimensions，
// 步幅数组 strides，以及未使用的辅助数据 auxdata
string_startswith_endswith_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 从 context 中获取静态数据的起始位置
    STARTPOSITION startposition = *(STARTPOSITION *)(context->method->static_data);
    // 获取第一个输入数据块的元素大小
    int elsize1 = context->descriptors[0]->elsize;
    // 获取第二个输入数据块的元素大小
    int elsize2 = context->descriptors[1]->elsize;

    // 获取每个输入数据块的指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *in4 = data[3];
    // 获取输出数据块的指针
    char *out = data[4];

    // 获取第一维度的大小
    npy_intp N = dimensions[0];

    // 循环处理每个数据点
    while (N--) {
        // 使用 Buffer<enc> 类型创建输入缓冲区 buf1 和 buf2，分别对应 in1 和 in2
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        // 调用 tailmatch 函数对 buf1 和 buf2 进行比较，使用 in3 和 in4 作为额外参数，得到匹配结果
        npy_bool match = tailmatch<enc>(buf1, buf2, *(npy_int64 *)in3, *(npy_int64 *)in4,
                                        startposition);
        // 将匹配结果写入输出数据块中
        *(npy_bool *)out = match;

        // 根据步幅更新输入和输出指针
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        in4 += strides[3];
        out += strides[4];
    }
    // 返回 0 表示函数执行成功
    return 0;
}


// 定义模板函数 string_lrstrip_whitespace_loop，接受以下参数：上下文 context，数据指针数组 data，
// 维度数组 dimensions，步幅数组 strides，以及未使用的辅助数据 auxdata
template <ENCODING enc>
static int
string_lrstrip_whitespace_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 从 context 中获取静态数据的剥离类型
    STRIPTYPE striptype = *(STRIPTYPE *)(context->method->static_data);
    // 获取第一个输入数据块的元素大小
    int elsize = context->descriptors[0]->elsize;
    // 获取第二个输出数据块的元素大小
    int outsize = context->descriptors[1]->elsize;

    // 获取输入和输出数据块的指针
    char *in = data[0];
    char *out = data[1];

    // 获取第一维度的大小
    npy_intp N = dimensions[0];

    // 循环处理每个数据点
    while (N--) {
        // 使用 Buffer<enc> 类型创建输入缓冲区 buf 和输出缓冲区 outbuf
        Buffer<enc> buf(in, elsize);
        Buffer<enc> outbuf(out, outsize);
        // 调用 string_lrstrip_whitespace 函数对输入缓冲区 buf 进行剥离操作，结果存入输出缓冲区 outbuf
        string_lrstrip_whitespace(buf, outbuf, striptype);

        // 根据步幅更新输入和输出指针
        in += strides[0];
        out += strides[1];
    }

    // 返回 0 表示函数执行成功
    return 0;
}


// 定义模板函数 string_lrstrip_chars_loop，接受以下参数：上下文 context，数据指针数组 data，
// 维度数组 dimensions，步幅数组 strides，以及未使用的辅助数据 auxdata
template <ENCODING enc>
static int
string_lrstrip_chars_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 从 context 中获取静态数据的剥离类型
    STRIPTYPE striptype = *(STRIPTYPE *)(context->method->static_data);
    // 获取第一个输入数据块的元素大小
    int elsize1 = context->descriptors[0]->elsize;
    // 获取第二个输入数据块的元素大小
    int elsize2 = context->descriptors[1]->elsize;
    // 获取第三个输出数据块的元素大小
    int outsize = context->descriptors[2]->elsize;

    // 获取输入数据块的指针
    char *in1 = data[0];
    char *in2 = data[1];
    // 获取输出数据块的指针
    char *out = data[2];

    // 获取第一维度的大小
    npy_intp N = dimensions[0];

    // 循环处理每个数据点
    while (N--) {
        // 使用 Buffer<enc> 类型创建输入缓冲区 buf1 和 buf2，以及输出缓冲区 outbuf
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf(out, outsize);
        // 调用 string_lrstrip_chars 函数对输入缓冲区 buf1 和 buf2 进行剥离操作，结果存入输出缓冲区 outbuf
        string_lrstrip_chars(buf1, buf2, outbuf, striptype);

        // 根据步幅更新输入和输出指针
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    // 返回 0 表示函数执行成功
    return 0;
}


// 定义模板函数 string_expandtabs_length_loop，接受以下参数：上下文 context，数据指针数组 data，
// 维度数组 dimensions，步幅数组 strides，以及未使用的辅助数据 auxdata
template <ENCODING enc>
static int
string_expandtabs_length_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取第一个输入数据块的元素大小
    int elsize = context->descriptors[0]->elsize;

    // 获取输入数据块的指针
    char *in1 = data[0];
    char *in2 = data[1];
    // 获取输出数据块的指针
    char *out = data[2];

    // 获取第一维度的大小
    npy_intp N = dimensions[0];
    
    // 循环处理每个数据点
    while (N--) {
        // 使用 Buffer<enc> 类型创建输入缓冲区 buf1
        Buffer<enc> buf1(in1, elsize);
        // 调用 string_expandtabs_length 函数对输入缓冲区 buf1 进行处理，结果存入输出指针 out
        string_expandtabs_length(buf1, in2, out);

        // 根据步幅更新输入和输出指针
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    // 返回 0 表示函数执行成功
    return 0;
}
    # 当 N 大于等于 0 时执行循环，每次循环 N 自减
    while (N--) {
        # 从 in1 中读取 elsize 长度的数据创建 Buffer 对象 buf
        Buffer<enc> buf(in1, elsize);
        # 调用 string_expandtabs_length 函数，计算 buf 中的数据经过将制表符扩展后的长度，
        # 将结果存入 out 指向的内存地址中，输入参数为 in2 所指向的整型数据的地址
        *(npy_intp *)out = string_expandtabs_length(buf, *(npy_int64 *)in2);

        # 调整 in1, in2, out 的指针位置，分别根据 strides 数组中的步长值调整
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    # 循环结束后返回 0
    return 0;
// 定义一个静态函数，用于处理字符串的 expandtabs 操作的循环实现
template <ENCODING enc>
static int
string_expandtabs_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入数据元素大小和输出数据元素大小
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[2]->elsize;

    // 获取输入数据指针
    char *in1 = data[0];  // 输入字符串数组
    char *in2 = data[1];  // tabsize 数组
    char *out = data[2];  // 输出字符串数组

    // 获取字符串数组的维度
    npy_intp N = dimensions[0];

    // 循环处理每个字符串
    while (N--) {
        // 创建输入和输出的 Buffer 对象
        Buffer<enc> buf(in1, elsize);
        Buffer<enc> outbuf(out, outsize);

        // 调用 string_expandtabs 函数进行实际的 expandtabs 操作
        npy_intp new_len = string_expandtabs(buf, *(npy_int64 *)in2, outbuf);

        // 将输出缓冲区中索引之后的部分填充为零
        outbuf.buffer_fill_with_zeros_after_index(new_len);

        // 更新输入指针和输出指针，以处理下一个字符串
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    // 返回处理结果
    return 0;
}


// 定义一个静态函数，用于处理字符串的 center、ljust、rjust 操作的循环实现
template <ENCODING bufferenc, ENCODING fillenc>
static int
string_center_ljust_rjust_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取对齐方式
    JUSTPOSITION pos = *(JUSTPOSITION *)(context->method->static_data);
    
    // 获取输入和输出数据元素大小
    int elsize1 = context->descriptors[0]->elsize;
    int elsize3 = context->descriptors[2]->elsize;
    int outsize = context->descriptors[3]->elsize;

    // 获取输入数据指针
    char *in1 = data[0];  // 输入字符串数组
    char *in2 = data[1];  // width 数组
    char *in3 = data[2];  // fillchar 数组
    char *out = data[3];  // 输出字符串数组

    // 获取字符串数组的维度
    npy_intp N = dimensions[0];

    // 循环处理每个字符串
    while (N--) {
        // 创建输入和输出的 Buffer 对象
        Buffer<bufferenc> buf(in1, elsize1);
        Buffer<fillenc> fill(in3, elsize3);
        Buffer<bufferenc> outbuf(out, outsize);

        // 检查特定条件下的非 ASCII 填充字符错误
        if (bufferenc == ENCODING::ASCII && fillenc == ENCODING::UTF32 && *fill > 0x7F) {
            npy_gil_error(PyExc_ValueError, "non-ascii fill character is not allowed when buffer is ascii");
            return -1;
        }

        // 调用 string_pad 函数执行 center、ljust、rjust 操作
        npy_intp len = string_pad(buf, *(npy_int64 *)in2, *fill, pos, outbuf);

        // 如果操作失败，返回错误
        if (len < 0) {
            return -1;
        }

        // 将输出缓冲区中索引之后的部分填充为零
        outbuf.buffer_fill_with_zeros_after_index(len);

        // 更新输入指针和输出指针，以处理下一个字符串
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        out += strides[3];
    }

    // 返回处理结果
    return 0;
}


// 定义一个静态函数，用于处理字符串的 zfill 操作的循环实现
template <ENCODING enc>
static int
string_zfill_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 获取输入数据元素大小和输出数据元素大小
    int elsize = context->descriptors[0]->elsize;
    int outsize = context->descriptors[2]->elsize;

    // 获取输入数据指针
    char *in1 = data[0];  // 输入字符串数组
    char *in2 = data[1];  // width 数组
    char *out = data[2];  // 输出字符串数组

    // 获取字符串数组的维度
    npy_intp N = dimensions[0];

    // 循环处理每个字符串
    while (N--) {
        // 创建输入和输出的 Buffer 对象
        Buffer<enc> buf(in1, elsize);
        Buffer<enc> outbuf(out, outsize);

        // 调用 string_zfill 函数执行 zfill 操作
        npy_intp newlen = string_zfill(buf, *(npy_int64 *)in2, outbuf);

        // 如果操作失败，返回错误
        if (newlen < 0) {
            return -1;
        }

        // 将输出缓冲区中索引之后的部分填充为零
        outbuf.buffer_fill_with_zeros_after_index(newlen);

        // 更新输入指针和输出指针，以处理下一个字符串
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }

    // 返回处理结果
    return 0;
}
// 定义函数 string_partition_index_loop，接收一个 PyArrayMethod_Context 结构体指针 context，
// 以及数据和维度信息数组。其中最后一个参数 NPY_UNUSED(auxdata) 未被使用。
string_partition_index_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    // 从 context 结构体的静态数据中获取起始位置信息
    STARTPOSITION startposition = *(STARTPOSITION *)(context->method->static_data);
    
    // 提取各种元素大小信息
    int elsize1 = context->descriptors[0]->elsize;
    int elsize2 = context->descriptors[1]->elsize;
    int outsize1 = context->descriptors[3]->elsize;
    int outsize2 = context->descriptors[4]->elsize;
    int outsize3 = context->descriptors[5]->elsize;

    // 分别获取输入和输出数据的指针
    char *in1 = data[0];
    char *in2 = data[1];
    char *in3 = data[2];
    char *out1 = data[3];
    char *out2 = data[4];
    char *out3 = data[5];

    // 获取第一维度的大小
    npy_intp N = dimensions[0];

    // 循环处理每一个元素
    while (N--) {
        // 使用 Buffer<enc> 封装输入输出数据，便于处理
        Buffer<enc> buf1(in1, elsize1);
        Buffer<enc> buf2(in2, elsize2);
        Buffer<enc> outbuf1(out1, outsize1);
        Buffer<enc> outbuf2(out2, outsize2);
        Buffer<enc> outbuf3(out3, outsize3);

        // 调用 string_partition 函数对输入数据进行分割操作
        npy_intp final_len1, final_len2, final_len3;
        string_partition(buf1, buf2, *(npy_int64 *)in3, outbuf1, outbuf2, outbuf3,
                         &final_len1, &final_len2, &final_len3, startposition);
        
        // 检查分割后的长度是否有效，如果有任何一个小于 0，返回 -1
        if (final_len1 < 0 || final_len2 < 0 || final_len3 < 0) {
            return -1;
        }
        
        // 在输出缓冲区中填充 0，直到 final_len1、final_len2 和 final_len3
        outbuf1.buffer_fill_with_zeros_after_index(final_len1);
        outbuf2.buffer_fill_with_zeros_after_index(final_len2);
        outbuf3.buffer_fill_with_zeros_after_index(final_len3);

        // 更新输入和输出指针，以及步长信息
        in1 += strides[0];
        in2 += strides[1];
        in3 += strides[2];
        out1 += strides[3];
        out2 += strides[4];
        out3 += strides[5];
    }

    // 处理完所有元素后返回 0
    return 0;
}


// 解析描述符和提升函数的解析函数
static NPY_CASTING
string_addition_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 确保给定的描述符是规范化的
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第二个描述符是规范化的
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 创建一个新的描述符来合并第一个和第二个描述符的大小
    loop_descrs[2] = PyArray_DescrNew(loop_descrs[0]);
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    loop_descrs[2]->elsize += loop_descrs[1]->elsize;

    // 返回没有类型转换的标志
    return NPY_NO_CASTING;
}


static NPY_CASTING
string_multiply_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    # 如果给定描述符数组中的第三个描述符为NULL，则执行以下操作
    if (given_descrs[2] == NULL) {
        # 设置一个类型错误异常，说明 'out' 关键字是必需的。在没有它的情况下使用 numpy.strings.multiply。
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use numpy.strings.multiply without it.");
        # 返回一个表示在转换中发生错误的特定错误码
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    # 确保第一个给定描述符是规范的（canonical）
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    # 如果确保规范失败，则返回表示在转换中发生错误的特定错误码
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    # 确保第二个给定描述符是规范的
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    # 如果确保规范失败，则返回表示在转换中发生错误的特定错误码
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    # 确保第三个给定描述符是规范的
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    # 如果确保规范失败，则返回表示在转换中发生错误的特定错误码
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    # 返回没有进行类型转换的标志
    return NPY_NO_CASTING;
static NPY_CASTING
string_strip_whitespace_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[2]),
        PyArray_Descr *const given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 确保第一个给定描述符是规范的，如果不是，则返回错误状态
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;  // 返回错误标识
    }

    // 增加引用计数，使第一个和第二个描述符相同
    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = loop_descrs[0];

    // 返回无需强制转换的状态
    return NPY_NO_CASTING;
}


static NPY_CASTING
string_strip_chars_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 确保给定的前两个描述符是规范的，如果不是，则返回错误状态
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;  // 返回错误标识
    }

    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;  // 返回错误标识
    }

    // 增加引用计数，使第一个和第三个描述符相同
    Py_INCREF(loop_descrs[0]);
    loop_descrs[2] = loop_descrs[0];

    // 返回无需强制转换的状态
    return NPY_NO_CASTING;
}


static int
string_findlike_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加引用计数，使新操作描述符等于原操作描述符
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];

    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];

    // 设置新操作描述符为64位整数类型
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    new_op_dtypes[4] = PyArray_DTypeFromTypeNum(NPY_DEFAULT_INT);

    // 返回成功状态
    return 0;
}


static int
string_replace_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加引用计数，使新操作描述符等于原操作描述符
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];

    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];

    Py_INCREF(op_dtypes[2]);
    new_op_dtypes[2] = op_dtypes[2];

    // 设置新操作描述符为64位整数类型
    new_op_dtypes[3] = PyArray_DTypeFromTypeNum(NPY_INT64);

    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[4] = op_dtypes[0];

    // 返回成功状态
    return 0;
}


static NPY_CASTING
string_replace_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[5]),
        PyArray_Descr *const given_descrs[5],
        PyArray_Descr *loop_descrs[5],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 如果给定的第五个描述符为空，则设置错误信息并返回错误状态
    if (given_descrs[4] == NULL) {
        PyErr_SetString(PyExc_ValueError, "out kwarg should be given");
        return _NPY_ERROR_OCCURRED_IN_CAST;  // 返回错误标识
    }

    // 确保前两个给定描述符是规范的，如果不是，则返回错误状态
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;  // 返回错误标识
    }
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);


These annotations provide a clear explanation of each line's purpose in the provided C code snippets, ensuring clarity and understanding of their functionality.
    # 检查 loop_descrs[1] 是否为 NULL
    if (loop_descrs[1] == NULL) {
        # 如果是 NULL，返回错误代码 _NPY_ERROR_OCCURRED_IN_CAST
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    # 对给定描述符的第二个元素调用 ensure_canonical 函数，并将结果赋给 loop_descrs[2]
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    # 检查 loop_descrs[2] 是否为 NULL
    if (loop_descrs[2] == NULL) {
        # 如果是 NULL，返回错误代码 _NPY_ERROR_OCCURRED_IN_CAST
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    # 对给定描述符的第三个元素调用 ensure_canonical 函数，并将结果赋给 loop_descrs[3]
    loop_descrs[3] = NPY_DT_CALL_ensure_canonical(given_descrs[3]);
    # 检查 loop_descrs[3] 是否为 NULL
    if (loop_descrs[3] == NULL) {
        # 如果是 NULL，返回错误代码 _NPY_ERROR_OCCURRED_IN_CAST
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }
    # 对给定描述符的第四个元素调用 ensure_canonical 函数，并将结果赋给 loop_descrs[4]
    loop_descrs[4] = NPY_DT_CALL_ensure_canonical(given_descrs[4]);
    # 检查 loop_descrs[4] 是否为 NULL
    if (loop_descrs[4] == NULL) {
        # 如果是 NULL，返回错误代码 _NPY_ERROR_OCCURRED_IN_CAST
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    # 如果以上所有描述符的确都有有效的规范形式，返回没有类型转换的常量 NPY_NO_CASTING
    return NPY_NO_CASTING;
static int
string_startswith_endswith_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第一个操作数的数据类型设置为新操作数据类型的第一个元素
    new_op_dtypes[0] = op_dtypes[0];
    // 增加第二个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[1]);
    // 将第二个操作数的数据类型设置为新操作数据类型的第二个元素
    new_op_dtypes[1] = op_dtypes[1];
    // 将第三个新操作数据类型设置为指向 PyArray_Int64DType 的新引用
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);
    // 将第四个新操作数据类型设置为指向 PyArray_Int64DType 的新引用
    new_op_dtypes[3] = NPY_DT_NewRef(&PyArray_Int64DType);
    // 将第五个新操作数据类型设置为指向 PyArray_BoolDType 的新引用
    new_op_dtypes[4] = NPY_DT_NewRef(&PyArray_BoolDType);
    // 返回 0 表示操作成功完成
    return 0;
}


static int
string_expandtabs_length_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第一个操作数的数据类型设置为新操作数据类型的第一个元素
    new_op_dtypes[0] = op_dtypes[0];
    // 将第二个新操作数据类型设置为指向 PyArray_Int64DType 的新引用
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    // 将第三个新操作数据类型设置为指向 NPY_DEFAULT_INT 的新引用
    new_op_dtypes[2] = PyArray_DTypeFromTypeNum(NPY_DEFAULT_INT);
    // 返回 0 表示操作成功完成
    return 0;
}


static int
string_expandtabs_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第一个操作数的数据类型设置为新操作数据类型的第一个元素
    new_op_dtypes[0] = op_dtypes[0];
    // 将第二个新操作数据类型设置为指向 PyArray_Int64DType 的新引用
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第三个新操作数据类型设置为第一个操作数的数据类型
    new_op_dtypes[2] = op_dtypes[0];
    // 返回 0 表示操作成功完成
    return 0;
}


static NPY_CASTING
string_expandtabs_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 如果 'out' 参数为 NULL，则抛出类型错误
    if (given_descrs[2] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use numpy.strings.expandtabs without it.");
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第一个描述符是规范化的，并将其赋给循环描述符的第一个元素
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第二个描述符是规范化的，并将其赋给循环描述符的第二个元素
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第三个描述符是规范化的，并将其赋给循环描述符的第三个元素
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 返回 NPY_NO_CASTING 表示无需类型转换
    return NPY_NO_CASTING;
}


static int
string_center_ljust_rjust_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第一个操作数的数据类型设置为新操作数据类型的第一个元素
    new_op_dtypes[0] = op_dtypes[0];
    // 将第二个新操作数据类型设置为指向 PyArray_Int64DType 的新引用
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第三个新操作数据类型设置为第一个操作数的数据类型
    new_op_dtypes[2] = op_dtypes[0];
    // 增加第一个操作数的引用计数，以确保它在函数结束时不被释放
    Py_INCREF(op_dtypes[0]);
    // 将第四个新操作数据类型设置为第一个操作数的数据类型
    new_op_dtypes[3] = op_dtypes[0];
    // 返回 0 表示操作成功完成
    return 0;
}
// 确定字符串中心、左侧和右侧操作的描述符解析函数
string_center_ljust_rjust_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[5],
        PyArray_Descr *loop_descrs[5],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 检查是否缺少 'out' 关键字参数
    if (given_descrs[3] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use the version in numpy.strings without it.");
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第一个给定描述符是规范的，并将其赋给循环描述符数组的第一个位置
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第二个给定描述符是规范的，并将其赋给循环描述符数组的第二个位置
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第三个给定描述符是规范的，并将其赋给循环描述符数组的第三个位置
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第四个给定描述符是规范的，并将其赋给循环描述符数组的第四个位置
    loop_descrs[3] = NPY_DT_CALL_ensure_canonical(given_descrs[3]);
    if (loop_descrs[3] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 返回无需强制转换的标志
    return NPY_NO_CASTING;
}


// 字符串填充操作的推广函数
static int
string_zfill_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加第一个操作数据类型的引用计数，并将其赋给新的操作数据类型数组的第一个位置
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];

    // 在新的操作数据类型数组的第二个位置设置 int64 类型的数据类型
    new_op_dtypes[1] = NPY_DT_NewRef(&PyArray_Int64DType);

    // 再次增加第一个操作数据类型的引用计数，并将其赋给新的操作数据类型数组的第三个位置
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[2] = op_dtypes[0];

    // 返回成功标志
    return 0;
}


// 字符串填充操作的描述符解析函数
static NPY_CASTING
string_zfill_resolve_descriptors(
        PyArrayMethodObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *const NPY_UNUSED(dtypes[3]),
        PyArray_Descr *const given_descrs[3],
        PyArray_Descr *loop_descrs[3],
        npy_intp *NPY_UNUSED(view_offset))
{
    // 检查是否缺少 'out' 关键字参数
    if (given_descrs[2] == NULL) {
        PyErr_SetString(
            PyExc_TypeError,
            "The 'out' kwarg is necessary. Use numpy.strings.zfill without it.");
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第一个给定描述符是规范的，并将其赋给循环描述符数组的第一个位置
    loop_descrs[0] = NPY_DT_CALL_ensure_canonical(given_descrs[0]);
    if (loop_descrs[0] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第二个给定描述符是规范的，并将其赋给循环描述符数组的第二个位置
    loop_descrs[1] = NPY_DT_CALL_ensure_canonical(given_descrs[1]);
    if (loop_descrs[1] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 确保第三个给定描述符是规范的，并将其赋给循环描述符数组的第三个位置
    loop_descrs[2] = NPY_DT_CALL_ensure_canonical(given_descrs[2]);
    if (loop_descrs[2] == NULL) {
        return _NPY_ERROR_OCCURRED_IN_CAST;
    }

    // 返回无需强制转换的标志
    return NPY_NO_CASTING;
}


// 字符串分割操作的推广函数
static int
string_partition_promoter(PyObject *NPY_UNUSED(ufunc),
        PyArray_DTypeMeta *const op_dtypes[], PyArray_DTypeMeta *const signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    // 增加第一个操作数据类型的引用计数，并将其赋给新的操作数据类型数组的第一个位置
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[0] = op_dtypes[0];

    // 增加第二个操作数据类型的引用计数，并将其赋给新的操作数据类型数组的第二个位置
    Py_INCREF(op_dtypes[1]);
    new_op_dtypes[1] = op_dtypes[1];

    // 在新的操作数据类型数组的第三个位置设置 int64 类型的数据类型
    new_op_dtypes[2] = NPY_DT_NewRef(&PyArray_Int64DType);

    // 再次增加第一个操作数据类型的引用计数，并将其赋给新的操作数据类型数组的第四个位置
    Py_INCREF(op_dtypes[0]);
    new_op_dtypes[3] = op_dtypes[0];
}
    # 将索引为 3 的 new_op_dtypes 元素设置为 op_dtypes 中索引为 0 的元素，并增加其引用计数
    new_op_dtypes[3] = op_dtypes[0];
    # 增加 op_dtypes 中索引为 0 的元素的引用计数，因为 new_op_dtypes 中复制了这个引用
    Py_INCREF(op_dtypes[0]);
    # 将索引为 4 的 new_op_dtypes 元素设置为 op_dtypes 中索引为 0 的元素，并增加其引用计数
    new_op_dtypes[4] = op_dtypes[0];
    # 再次增加 op_dtypes 中索引为 0 的元素的引用计数
    Py_INCREF(op_dtypes[0]);
    # 将索引为 5 的 new_op_dtypes 元素设置为 op_dtypes 中索引为 0 的元素，并增加其引用计数
    new_op_dtypes[5] = op_dtypes[0];
    # 函数成功结束，返回 0
    return 0;
    // 初始化比较函数，设置初始返回值为-1
    int res = -1;
    // 定义字符串类型、Unicode 类型和布尔类型的元数据指针
    PyArray_DTypeMeta *String = &PyArray_BytesDType;
    PyArray_DTypeMeta *Unicode = &PyArray_UnicodeDType;
    PyArray_DTypeMeta *Bool = &PyArray_BoolDType;

    /* We start with the string loops: */
    // 开始处理字符串循环：
    PyArray_DTypeMeta *dtypes[] = {String, String, Bool};
    /*
     * 我们目前只有一个循环，即 strided 循环。默认类型解析器确保本机字节顺序和规范表示。
     */
    PyType_Slot slots[] = {
        {NPY_METH_strided_loop, nullptr},
        {0, nullptr}
    };

    PyArrayMethod_Spec spec = {};
    spec.name = "templated_string_comparison";
    spec.nin = 2;
    spec.nout = 1;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    /* 所有 String 类型的循环 */
    using string_looper = add_loops<false, ENCODING::ASCII, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    if (string_looper()(umath, &spec) < 0) {
        // 如果 String 类型的循环添加失败，跳转到 finish 标签处
        goto finish;
    }

    /* 所有 Unicode 类型的循环 */
    using ucs_looper = add_loops<false, ENCODING::UTF32, COMP::EQ, COMP::NE, COMP::LT, COMP::LE, COMP::GT, COMP::GE>;
    dtypes[0] = Unicode;
    dtypes[1] = Unicode;
    if (ucs_looper()(umath, &spec) < 0) {
        // 如果 Unicode 类型的循环添加失败，跳转到 finish 标签处
        goto finish;
    }

    res = 0;
  finish:
    // 返回执行结果 res
    return res;
/*
 * 初始化一个ufunc的推广函数，用于处理混合字符串dtypes的参数
 * umath: ufunc对象
 * name: ufunc的名称
 * nin: 输入参数的数量
 * nout: 输出参数的数量
 * typenums: 输入输出参数的数据类型数组
 * enc: 字符编码方式
 * loop: strided循环函数
 * resolve_descriptors: 解析描述符的函数
 * static_data: 静态数据指针
 */
static int
init_ufunc(PyObject *umath, const char *name, int nin, int nout,
           NPY_TYPES *typenums, ENCODING enc, PyArrayMethod_StridedLoop loop,
           PyArrayMethod_ResolveDescriptors resolve_descriptors,
           void *static_data)
{
    int res = -1;

    // 分配内存用于存储输入输出参数的数据类型元信息
    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **) PyMem_Malloc(
        (nin + nout) * sizeof(PyArray_DTypeMeta *));
    if (dtypes == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    // 遍历输入输出参数，根据给定的typenums和enc选择正确的数据类型
    for (int i = 0; i < nin + nout; i++) {
        if (typenums[i] == NPY_OBJECT && enc == ENCODING::UTF32) {
            dtypes[i] = NPY_DT_NewRef(&PyArray_UnicodeDType);
        }
        else if (typenums[i] == NPY_OBJECT && enc == ENCODING::ASCII) {
            dtypes[i] = NPY_DT_NewRef(&PyArray_BytesDType);
        }
        else {
            dtypes[i] = PyArray_DTypeFromTypeNum(typenums[i]);
        }
    }

    // 设置PyType_Slot数组，用于描述对象类型的特定槽位
    PyType_Slot slots[4];
    slots[0] = {NPY_METH_strided_loop, nullptr};
    slots[1] = {_NPY_METH_static_data, static_data};
    slots[3] = {0, nullptr};
    // 如果有解析描述符函数，则设置相应的槽位
    if (resolve_descriptors != NULL) {
        slots[2] = {NPY_METH_resolve_descriptors, (void *) resolve_descriptors};
    }
    else {
        slots[2] = {0, nullptr};
    }

    // 根据ufunc名称生成循环函数的名称
    char loop_name[256] = {0};
    snprintf(loop_name, sizeof(loop_name), "templated_string_%s", name);

    // 设置PyArrayMethod_Spec结构体，描述ufunc的具体规格
    PyArrayMethod_Spec spec = {};
    spec.name = loop_name;
    spec.nin = nin;
    spec.nout = nout;
    spec.dtypes = dtypes;
    spec.slots = slots;
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;

    // 将循环函数添加到ufunc中
    if (add_loop(umath, name, &spec, loop) < 0) {
        goto finish;
    }

    // 设置成功标志
    res = 0;

finish:
    // 释放分配的数据类型元信息内存
    for (int i = 0; i < nin + nout; i++) {
        Py_DECREF(dtypes[i]);
    }
    PyMem_Free((void *) dtypes);
    return res;
}
// 初始化混合类型通用函数
init_mixed_type_ufunc(PyObject *umath, const char *name, int nin, int nout,
           NPY_TYPES *typenums, PyArrayMethod_StridedLoop loop,
           PyArrayMethod_ResolveDescriptors resolve_descriptors,
           void *static_data)
{
    int res = -1;

    // 分配存储数据类型的数组内存空间
    PyArray_DTypeMeta **dtypes = (PyArray_DTypeMeta **) PyMem_Malloc(
        (nin + nout) * sizeof(PyArray_DTypeMeta *));
    if (dtypes == NULL) {
        PyErr_NoMemory();  // 内存分配失败报错
        return -1;
    }

    // 根据类型号获取数据类型对象
    for (int i = 0; i < nin+nout; i++) {
        dtypes[i] = PyArray_DTypeFromTypeNum(typenums[i]);
    }

    // 设置类型槽（slots）数组
    PyType_Slot slots[4];
    slots[0] = {NPY_METH_strided_loop, nullptr};  // 设置循环处理方法
    slots[1] = {_NPY_METH_static_data, static_data};  // 设置静态数据
    slots[3] = {0, nullptr};  // 空槽位

    // 如果 resolve_descriptors 不为空，则设置解析描述符方法
    if (resolve_descriptors != NULL) {
        slots[2] = {NPY_METH_resolve_descriptors, (void *) resolve_descriptors};
    }
    else {
        slots[2] = {0, nullptr};  // 否则置空
    }

    // 构建循环名称字符串
    char loop_name[256] = {0};
    snprintf(loop_name, sizeof(loop_name), "templated_string_%s", name);

    // 设置方法规范（spec）
    PyArrayMethod_Spec spec = {};
    spec.name = loop_name;  // 设置方法名称
    spec.nin = nin;  // 输入参数个数
    spec.nout = nout;  // 输出参数个数
    spec.dtypes = dtypes;  // 数据类型数组
    spec.slots = slots;  // 类型槽数组
    spec.flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;  // 设置方法标志

    // 将方法规范添加到通用数学对象中
    if (add_loop(umath, name, &spec, loop) < 0) {
        goto finish;  // 添加失败则跳转到结束处理
    }

    res = 0;  // 成功添加方法规范，返回值置为0
  finish:
    // 释放数据类型对象引用计数
    for (int i = 0; i < nin+nout; i++) {
        Py_DECREF(dtypes[i]);
    }
    PyMem_Free((void *) dtypes);  // 释放数据类型数组内存
    return res;  // 返回结果状态
}



// 初始化字符串类型通用函数
NPY_NO_EXPORT int
init_string_ufuncs(PyObject *umath)
{
    NPY_TYPES dtypes[] = {NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING, NPY_STRING};

    // 初始化比较函数，若失败则返回-1
    if (init_comparison(umath) < 0) {
        return -1;
    }

    // 使用 NPY_OBJECT 作为哨兵值，稍后会替换为相应的字符串数据类型（NPY_STRING 或 NPY_UNICODE）
    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;

    // 初始化 add 函数，使用 ASCII 编码
    if (init_ufunc(
            umath, "add", 2, 1, dtypes, ENCODING::ASCII,
            string_add_loop<ENCODING::ASCII>, string_addition_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    // 初始化 add 函数，使用 UTF32 编码
    if (init_ufunc(
            umath, "add", 2, 1, dtypes, ENCODING::UTF32,
            string_add_loop<ENCODING::UTF32>, string_addition_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    // 恢复为对象类型
    dtypes[0] = dtypes[2] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;

    // 初始化 multiply 函数，使用 ASCII 编码
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::ASCII,
            string_multiply_strint_loop<ENCODING::ASCII>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    // 初始化 multiply 函数，使用 UTF32 编码
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::UTF32,
            string_multiply_strint_loop<ENCODING::UTF32>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        return -1;
    }

    // 恢复为对象类型
    dtypes[1] = dtypes[2] = NPY_OBJECT;
    dtypes[0] = NPY_INT64;
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::ASCII,
            string_multiply_intstr_loop<ENCODING::ASCII>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        // 如果初始化名为 "multiply" 的 ufunc 失败，则返回 -1
        return -1;
    }
    if (init_ufunc(
            umath, "multiply", 2, 1, dtypes, ENCODING::UTF32,
            string_multiply_intstr_loop<ENCODING::UTF32>, string_multiply_resolve_descriptors,
            NULL) < 0) {
        // 如果初始化名为 "multiply" 的 ufunc 失败，则返回 -1
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_DEFAULT_INT;
    if (init_ufunc(
            umath, "str_len", 1, 1, dtypes, ENCODING::ASCII,
            string_str_len_loop<ENCODING::ASCII>, NULL, NULL) < 0) {
        // 如果初始化名为 "str_len" 的 ufunc 失败，则返回 -1
        return -1;
    }
    if (init_ufunc(
            umath, "str_len", 1, 1, dtypes, ENCODING::UTF32,
            string_str_len_loop<ENCODING::UTF32>, NULL, NULL) < 0) {
        // 如果初始化名为 "str_len" 的 ufunc 失败，则返回 -1
        return -1;
    }

    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_BOOL;

    const char *unary_buffer_method_names[] = {
        "isalpha", "isalnum", "isdigit", "isspace", "islower",
        "isupper", "istitle", "isdecimal", "isnumeric",
    };

    static buffer_method<ENCODING::ASCII> unary_buffer_ascii_methods[] = {
        &Buffer<ENCODING::ASCII>::isalpha,
        &Buffer<ENCODING::ASCII>::isalnum,
        &Buffer<ENCODING::ASCII>::isdigit,
        &Buffer<ENCODING::ASCII>::isspace,
        &Buffer<ENCODING::ASCII>::islower,
        &Buffer<ENCODING::ASCII>::isupper,
        &Buffer<ENCODING::ASCII>::istitle,
    };

    static buffer_method<ENCODING::UTF32> unary_buffer_utf32_methods[] = {
        &Buffer<ENCODING::UTF32>::isalpha,
        &Buffer<ENCODING::UTF32>::isalnum,
        &Buffer<ENCODING::UTF32>::isdigit,
        &Buffer<ENCODING::UTF32>::isspace,
        &Buffer<ENCODING::UTF32>::islower,
        &Buffer<ENCODING::UTF32>::isupper,
        &Buffer<ENCODING::UTF32>::istitle,
        &Buffer<ENCODING::UTF32>::isdecimal,
        &Buffer<ENCODING::UTF32>::isnumeric,
    };

    for (int i = 0; i < 9; i++) {
        if (i < 7) { // ASCII 编码不支持 isdecimal 和 isnumeric
            if (init_ufunc(
                    umath, unary_buffer_method_names[i], 1, 1, dtypes, ENCODING::ASCII,
                    string_unary_loop<ENCODING::ASCII>, NULL,
                    &unary_buffer_ascii_methods[i]) < 0) {
                // 如果初始化名为 unary_buffer_method_names[i] 的 ufunc 失败，则返回 -1
                return -1;
            }
        }

        if (init_ufunc(
                umath, unary_buffer_method_names[i], 1, 1, dtypes, ENCODING::UTF32,
                string_unary_loop<ENCODING::UTF32>, NULL,
                &unary_buffer_utf32_methods[i]) < 0) {
            // 如果初始化名为 unary_buffer_method_names[i] 的 ufunc 失败，则返回 -1
            return -1;
        }
    }

    dtypes[0] = dtypes[1] = NPY_OBJECT;
    dtypes[2] = dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_DEFAULT_INT;

    const char* findlike_names[] = {
        "find", "rfind", "index", "rindex", "count",
    };
    // 定义一系列用于 ASCII 编码的字符串查找函数指针数组
    findlike_function<ENCODING::ASCII> findlike_ascii_functions[] = {
        string_find<ENCODING::ASCII>,
        string_rfind<ENCODING::ASCII>,
        string_index<ENCODING::ASCII>,
        string_rindex<ENCODING::ASCII>,
        string_count<ENCODING::ASCII>,
    };

    // 定义一系列用于 UTF32 编码的字符串查找函数指针数组
    findlike_function<ENCODING::UTF32> findlike_utf32_functions[] = {
        string_find<ENCODING::UTF32>,
        string_rfind<ENCODING::UTF32>,
        string_index<ENCODING::UTF32>,
        string_rindex<ENCODING::UTF32>,
        string_count<ENCODING::UTF32>,
    };

    // 循环初始化字符串查找函数和相关处理函数
    for (int j = 0; j < 5; j++) {

        // 初始化 ASCII 编码的字符串查找函数
        if (init_ufunc(
                umath, findlike_names[j], 4, 1, dtypes, ENCODING::ASCII,
                string_findlike_loop<ENCODING::ASCII>, NULL,
                (void *) findlike_ascii_functions[j]) < 0) {
            return -1;
        }

        // 初始化 UTF32 编码的字符串查找函数
        if (init_ufunc(
                umath, findlike_names[j], 4, 1, dtypes, ENCODING::UTF32,
                string_findlike_loop<ENCODING::UTF32>, NULL,
                (void *) findlike_utf32_functions[j]) < 0) {
            return -1;
        }

        // 初始化字符串查找的促进器函数
        if (init_promoter(umath, findlike_names[j], 4, 1,
                string_findlike_promoter) < 0) {
            return -1;
        }
    }

    // 设置数组中的数据类型为对象和整数类型
    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;
    dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_OBJECT;

    // 初始化 ASCII 编码的字符串替换函数
    if (init_ufunc(
            umath, "_replace", 4, 1, dtypes, ENCODING::ASCII,
            string_replace_loop<ENCODING::ASCII>,
            string_replace_resolve_descriptors, NULL) < 0) {
        return -1;
    }

    // 初始化 UTF32 编码的字符串替换函数
    if (init_ufunc(
            umath, "_replace", 4, 1, dtypes, ENCODING::UTF32,
            string_replace_loop<ENCODING::UTF32>,
            string_replace_resolve_descriptors, NULL) < 0) {
        return -1;
    }

    // 初始化字符串替换的促进器函数
    if (init_promoter(umath, "_replace", 4, 1, string_replace_promoter) < 0) {
        return -1;
    }

    // 重新设置数据类型数组中的值
    dtypes[0] = dtypes[1] = NPY_OBJECT;
    dtypes[2] = dtypes[3] = NPY_INT64;
    dtypes[4] = NPY_BOOL;

    // 定义开始和结束字符串函数的名称数组
    const char *startswith_endswith_names[] = {
        "startswith", "endswith"
    };

    // 定义静态的开始位置枚举数组
    static STARTPOSITION startpositions[] = {
        STARTPOSITION::FRONT, STARTPOSITION::BACK
    };

    // 循环初始化开始和结束字符串函数及其相关处理函数
    for (int i = 0; i < 2; i++) {
        // 初始化 ASCII 编码的开始和结束字符串函数
        if (init_ufunc(
                umath, startswith_endswith_names[i], 4, 1, dtypes, ENCODING::ASCII,
                string_startswith_endswith_loop<ENCODING::ASCII>,
                NULL, &startpositions[i]) < 0) {
            return -1;
        }
        // 初始化 UTF32 编码的开始和结束字符串函数
        if (init_ufunc(
                umath, startswith_endswith_names[i], 4, 1, dtypes, ENCODING::UTF32,
                string_startswith_endswith_loop<ENCODING::UTF32>,
                NULL, &startpositions[i]) < 0) {
            return -1;
        }
        // 初始化开始和结束字符串的促进器函数
        if (init_promoter(umath, startswith_endswith_names[i], 4, 1,
                string_startswith_endswith_promoter) < 0) {
            return -1;
        }
    }

    // 重新设置数据类型数组中的值
    dtypes[0] = dtypes[1] = NPY_OBJECT;
    // 定义一个包含需要去除空白字符的函数名称的常量数组
    const char *strip_whitespace_names[] = {
        "_lstrip_whitespace", "_rstrip_whitespace", "_strip_whitespace"
    };

    // 定义一个静态的枚举类型数组，表示三种去除空白字符的方式
    static STRIPTYPE striptypes[] = {
        STRIPTYPE::LEFTSTRIP, STRIPTYPE::RIGHTSTRIP, STRIPTYPE::BOTHSTRIP
    };

    // 循环处理每种去除空白字符的方式
    for (int i = 0; i < 3; i++) {
        // 初始化一个通用函数对象，用于处理 ASCII 编码的字符串去除空白字符操作
        if (init_ufunc(
                umath, strip_whitespace_names[i], 1, 1, dtypes, ENCODING::ASCII,
                string_lrstrip_whitespace_loop<ENCODING::ASCII>,
                string_strip_whitespace_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
        // 初始化一个通用函数对象，用于处理 UTF32 编码的字符串去除空白字符操作
        if (init_ufunc(
                umath, strip_whitespace_names[i], 1, 1, dtypes, ENCODING::UTF32,
                string_lrstrip_whitespace_loop<ENCODING::UTF32>,
                string_strip_whitespace_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
    }

    // 将处理对象的数据类型设置为 NPY_OBJECT
    dtypes[0] = dtypes[1] = dtypes[2] = NPY_OBJECT;

    // 定义一个包含需要去除指定字符的函数名称的常量数组
    const char *strip_chars_names[] = {
        "_lstrip_chars", "_rstrip_chars", "_strip_chars"
    };

    // 再次循环处理每种去除指定字符的方式
    for (int i = 0; i < 3; i++) {
        // 初始化一个通用函数对象，用于处理 ASCII 编码的字符串去除指定字符操作
        if (init_ufunc(
                umath, strip_chars_names[i], 2, 1, dtypes, ENCODING::ASCII,
                string_lrstrip_chars_loop<ENCODING::ASCII>,
                string_strip_chars_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
        // 初始化一个通用函数对象，用于处理 UTF32 编码的字符串去除指定字符操作
        if (init_ufunc(
                umath, strip_chars_names[i], 2, 1, dtypes, ENCODING::UTF32,
                string_lrstrip_chars_loop<ENCODING::UTF32>,
                string_strip_chars_resolve_descriptors,
                &striptypes[i]) < 0) {
            return -1;
        }
    }

    // 将处理对象的数据类型设置为 NPY_OBJECT, NPY_INT64, NPY_DEFAULT_INT
    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    dtypes[2] = NPY_DEFAULT_INT;

    // 初始化一个通用函数对象，用于处理 ASCII 编码的字符串扩展制表符长度操作
    if (init_ufunc(
            umath, "_expandtabs_length", 2, 1, dtypes, ENCODING::ASCII,
            string_expandtabs_length_loop<ENCODING::ASCII>, NULL, NULL) < 0) {
        return -1;
    }
    // 初始化一个通用函数对象，用于处理 UTF32 编码的字符串扩展制表符长度操作
    if (init_ufunc(
            umath, "_expandtabs_length", 2, 1, dtypes, ENCODING::UTF32,
            string_expandtabs_length_loop<ENCODING::UTF32>, NULL, NULL) < 0) {
        return -1;
    }
    // 初始化字符串扩展制表符长度的促进器
    if (init_promoter(umath, "_expandtabs_length", 2, 1, string_expandtabs_length_promoter) < 0) {
        return -1;
    }

    // 将处理对象的数据类型设置为 NPY_OBJECT, NPY_INT64, NPY_OBJECT
    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    dtypes[2] = NPY_OBJECT;

    // 初始化一个通用函数对象，用于处理 ASCII 编码的字符串扩展制表符操作
    if (init_ufunc(
            umath, "_expandtabs", 2, 1, dtypes, ENCODING::ASCII,
            string_expandtabs_loop<ENCODING::ASCII>,
            string_expandtabs_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    // 初始化一个通用函数对象，用于处理 UTF32 编码的字符串扩展制表符操作
    if (init_ufunc(
            umath, "_expandtabs", 2, 1, dtypes, ENCODING::UTF32,
            string_expandtabs_loop<ENCODING::UTF32>,
            string_expandtabs_resolve_descriptors, NULL) < 0) {
        return -1;
    }
    // 初始化字符串扩展制表符的促进器
    if (init_promoter(umath, "_expandtabs", 2, 1, string_expandtabs_promoter) < 0) {
        return -1;
    }

    // 将处理对象的第二个数据类型设置为 NPY_INT64
    dtypes[1] = NPY_INT64;
    // 定义一个包含字符串"_center", "_ljust", "_rjust"的常量字符指针数组
    const char *center_ljust_rjust_names[] = {
        "_center", "_ljust", "_rjust"
    };

    // 静态数组，包含JUSTPOSITION枚举类型的元素，分别表示居中、左对齐、右对齐
    static JUSTPOSITION padpositions[] = {
        JUSTPOSITION::CENTER, JUSTPOSITION::LEFT, JUSTPOSITION::RIGHT
    };

    // 循环遍历center_ljust_rjust_names数组中的元素
    for (int i = 0; i < 3; i++) {
        // 设置dtypes数组的特定位置为NPY_STRING类型
        dtypes[0] = NPY_STRING;
        dtypes[2] = NPY_STRING;
        dtypes[3] = NPY_STRING;
        
        // 调用init_mixed_type_ufunc函数初始化混合类型的通用函数
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                // 使用ASCII编码的字符串处理函数和解析描述符函数
                string_center_ljust_rjust_loop<ENCODING::ASCII, ENCODING::ASCII>,
                string_center_ljust_rjust_resolve_descriptors,
                // 传入当前处理的对齐方式（居中、左对齐、右对齐）
                &padpositions[i]) < 0) {
            return -1;
        }

        // 设置dtypes数组的特定位置为NPY_STRING、NPY_UNICODE类型
        dtypes[0] = NPY_STRING;
        dtypes[2] = NPY_UNICODE;
        dtypes[3] = NPY_STRING;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                // 使用ASCII编码的字符串处理函数和解析描述符函数
                string_center_ljust_rjust_loop<ENCODING::ASCII, ENCODING::UTF32>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }

        // 设置dtypes数组的特定位置为NPY_UNICODE类型
        dtypes[0] = NPY_UNICODE;
        dtypes[2] = NPY_UNICODE;
        dtypes[3] = NPY_UNICODE;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                // 使用UTF32编码的字符串处理函数和解析描述符函数
                string_center_ljust_rjust_loop<ENCODING::UTF32, ENCODING::UTF32>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }

        // 设置dtypes数组的特定位置为NPY_UNICODE、NPY_STRING类型
        dtypes[0] = NPY_UNICODE;
        dtypes[2] = NPY_STRING;
        dtypes[3] = NPY_UNICODE;
        if (init_mixed_type_ufunc(
                umath, center_ljust_rjust_names[i], 3, 1, dtypes,
                // 使用UTF32编码的字符串处理函数和解析描述符函数
                string_center_ljust_rjust_loop<ENCODING::UTF32, ENCODING::ASCII>,
                string_center_ljust_rjust_resolve_descriptors,
                &padpositions[i]) < 0) {
            return -1;
        }

        // 调用init_promoter函数初始化提升器
        if (init_promoter(umath, center_ljust_rjust_names[i], 3, 1,
                string_center_ljust_rjust_promoter) < 0) {
            return -1;
        }
    }

    // 设置dtypes数组的特定位置为NPY_OBJECT、NPY_INT64、NPY_OBJECT类型
    dtypes[0] = NPY_OBJECT;
    dtypes[1] = NPY_INT64;
    dtypes[2] = NPY_OBJECT;
    
    // 初始化名为"_zfill"的通用函数，使用ASCII编码的字符串处理函数和解析描述符函数
    if (init_ufunc(
            umath, "_zfill", 2, 1, dtypes, ENCODING::ASCII,
            string_zfill_loop<ENCODING::ASCII>,
            string_zfill_resolve_descriptors, NULL) < 0) {
        return -1;
    }

    // 初始化名为"_zfill"的通用函数，使用UTF32编码的字符串处理函数和解析描述符函数
    if (init_ufunc(
            umath, "_zfill", 2, 1, dtypes, ENCODING::UTF32,
            string_zfill_loop<ENCODING::UTF32>,
            string_zfill_resolve_descriptors, NULL) < 0) {
        return -1;
    }

    // 调用init_promoter函数初始化提升器
    if (init_promoter(umath, "_zfill", 2, 1, string_zfill_promoter) < 0) {
        return -1;
    }

    // 设置dtypes数组的特定位置为NPY_OBJECT类型，其他位置为NPY_INT64类型
    dtypes[0] = dtypes[1] = dtypes[3] = dtypes[4] = dtypes[5] = NPY_OBJECT;
    dtypes[2] = NPY_INT64;

    // 定义一个包含字符串"_partition_index", "_rpartition_index"的常量字符指针数组
    const char *partition_names[] = {"_partition_index", "_rpartition_index"};

    // 静态数组，包含STARTPOSITION枚举类型的元素，分别表示从前开始、从后开始
    static STARTPOSITION partition_startpositions[] = {
        STARTPOSITION::FRONT, STARTPOSITION::BACK
    };
    // 循环两次，分别对两个分区进行初始化操作
    for (int i = 0; i < 2; i++) {
        // 初始化 umath 中的分区名称为 partition_names[i] 的功能
        // 使用 ASCII 编码进行初始化，传入相应的参数和函数指针
        if (init_ufunc(
                umath, partition_names[i], 3, 3, dtypes, ENCODING::ASCII,
                string_partition_index_loop<ENCODING::ASCII>,
                string_partition_resolve_descriptors, &partition_startpositions[i]) < 0) {
            // 如果初始化失败，返回 -1
            return -1;
        }
        // 再次对相同分区名称的功能进行初始化，使用 UTF32 编码进行初始化
        // 传入相同的参数和函数指针
        if (init_ufunc(
                umath, partition_names[i], 3, 3, dtypes, ENCODING::UTF32,
                string_partition_index_loop<ENCODING::UTF32>,
                string_partition_resolve_descriptors, &partition_startpositions[i]) < 0) {
            // 如果初始化失败，返回 -1
            return -1;
        }
        // 初始化 umath 中的分区名称为 partition_names[i] 的推广器功能
        // 使用 string_partition_promoter 函数进行初始化
        if (init_promoter(umath, partition_names[i], 3, 3,
                string_partition_promoter) < 0) {
            // 如果初始化失败，返回 -1
            return -1;
        }
    }

    // 如果所有初始化成功，返回 0 表示正常结束
    return 0;
/*
 * This function returns a pointer to a strided loop function template based on
 * the comparison type and encoding.
 */
template <bool rstrip, ENCODING enc>
static PyArrayMethod_StridedLoop *
get_strided_loop(int comp)
{
    // Switch statement to select the appropriate strided loop function
    switch (comp) {
        case Py_EQ:
            return string_comparison_loop<rstrip, COMP::EQ, enc>;
        case Py_NE:
            return string_comparison_loop<rstrip, COMP::NE, enc>;
        case Py_LT:
            return string_comparison_loop<rstrip, COMP::LT, enc>;
        case Py_LE:
            return string_comparison_loop<rstrip, COMP::LE, enc>;
        case Py_GT:
            return string_comparison_loop<rstrip, COMP::GT, enc>;
        case Py_GE:
            return string_comparison_loop<rstrip, COMP::GE, enc>;
        default:
            assert(false);  /* caller ensures this */
    }
    // Default return if none of the cases match (should not happen)
    return nullptr;
}
/*
 * This function is used for comparing arrays of strings (char arrays) and is also
 * applicable for void comparisons. The `rstrip` parameter possibly supports Fortran
 * compatibility. There are considerations to deprecate the chararray comparison in favor
 * of ufunc, and to optimize usage of `rstrip` on arrays before comparison.
 *
 * NOTE: This function handles unstructured voids assuming `npy_byte` is correctly set.
 */
NPY_NO_EXPORT PyObject *
_umath_strings_richcompare(
        PyArrayObject *self, PyArrayObject *other, int cmp_op, int rstrip)
{
    NpyIter *iter = nullptr;
    PyObject *result = nullptr;

    char **dataptr = nullptr;
    npy_intp *strides = nullptr;
    npy_intp *countptr = nullptr;
    npy_intp size = 0;

    PyArrayMethod_Context context = {};
    NpyIter_IterNextFunc *iternext = nullptr;

    npy_uint32 it_flags = (
            NPY_ITER_EXTERNAL_LOOP | NPY_ITER_ZEROSIZE_OK |
            NPY_ITER_BUFFERED | NPY_ITER_GROWINNER);
    npy_uint32 op_flags[3] = {
            NPY_ITER_READONLY | NPY_ITER_ALIGNED,
            NPY_ITER_READONLY | NPY_ITER_ALIGNED,
            NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE | NPY_ITER_ALIGNED};

    PyArrayMethod_StridedLoop *strided_loop = nullptr;
    NPY_BEGIN_THREADS_DEF;

    // Check if the types of self and other arrays match; return NotImplemented for mismatch
    if (PyArray_TYPE(self) != PyArray_TYPE(other)) {
        /*
         * Comparison between Bytes and Unicode is not defined in Py3K;
         * return NotImplemented.
         * TODO: This logic may need reevaluation for `compare_chararrays`,
         *       considering potential deprecation.
         */
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    PyArrayObject *ops[3] = {self, other, nullptr};
    PyArray_Descr *descrs[3] = {nullptr, nullptr, PyArray_DescrFromType(NPY_BOOL)};
    /* TODO: Native byte order is not essential for == and != */
    descrs[0] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(self));
    if (descrs[0] == nullptr) {
        goto finish;
    }
    # 将第二个描述符设为确保规范化后的描述符对象
    descrs[1] = NPY_DT_CALL_ensure_canonical(PyArray_DESCR(other));
    # 检查第二个描述符是否为 nullptr，如果是则跳转至 finish 标签
    if (descrs[1] == nullptr) {
        goto finish;
    }

    """
     * 创建迭代器：
     """
    # 使用高级方式创建迭代器，参数包括操作数组、迭代器标志、保持顺序、安全类型转换等
    iter = NpyIter_AdvancedNew(
            3, ops, it_flags, NPY_KEEPORDER, NPY_SAFE_CASTING, op_flags, descrs,
            -1, nullptr, nullptr, 0);
    # 如果迭代器创建失败，则跳转至 finish 标签
    if (iter == nullptr) {
        goto finish;
    }

    # 获取迭代器的大小
    size = NpyIter_GetIterSize(iter);
    # 如果迭代器大小为 0，则获取第三个操作数作为结果，并增加其引用计数，然后跳转至 finish 标签
    if (size == 0) {
        result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
        Py_INCREF(result);
        goto finish;
    }

    # 获取迭代器的迭代下一步函数
    iternext = NpyIter_GetIterNext(iter, nullptr);
    # 如果迭代下一步函数为 nullptr，则跳转至 finish 标签
    if (iternext == nullptr) {
        goto finish;
    }

    """
     * 准备内部循环并执行它（只需传递描述符）：
     """
    # 设置上下文中的描述符为当前描述符数组
    context.descriptors = descrs;

    # 获取迭代器的数据指针数组、内部步长数组和内部循环大小指针
    dataptr = NpyIter_GetDataPtrArray(iter);
    strides = NpyIter_GetInnerStrideArray(iter);
    countptr = NpyIter_GetInnerLoopSizePtr(iter);

    # 如果 rstrip 等于 0
    if (rstrip == 0) {
        # 注意：也用于 VOID 类型，因此可以是 STRING、UNICODE 或 VOID
        # 如果第一个描述符的类型号不是 NPY_UNICODE
        if (descrs[0]->type_num != NPY_UNICODE) {
            # 获取 ASCII 编码的步进循环函数
            strided_loop = get_strided_loop<false, ENCODING::ASCII>(cmp_op);
        }
        else:
            # 获取 UTF32 编码的步进循环函数
            strided_loop = get_strided_loop<false, ENCODING::UTF32>(cmp_op);
    }
    else:
        # 如果第一个描述符的类型号不是 NPY_UNICODE
        if (descrs[0]->type_num != NPY_UNICODE) {
            # 获取 ASCII 编码的步进循环函数
            strided_loop = get_strided_loop<true, ENCODING::ASCII>(cmp_op);
        }
        else:
            # 获取 UTF32 编码的步进循环函数
            strided_loop = get_strided_loop<true, ENCODING::UTF32>(cmp_op);

    # 开始多线程执行，如果大小超过阈值
    NPY_BEGIN_THREADS_THRESHOLDED(size);

    # 执行内部循环直到迭代结束
    do {
         /* 我们知道循环不会失败 */
         strided_loop(&context, dataptr, countptr, strides, nullptr);
    } while (iternext(iter) != 0);

    # 结束多线程执行
    NPY_END_THREADS;

    # 获取第三个操作数作为结果
    result = (PyObject *)NpyIter_GetOperandArray(iter)[2];
    # 增加结果对象的引用计数
    Py_INCREF(result);

 finish:
    # 释放迭代器对象，如果释放失败则清空结果对象引用
    if (NpyIter_Deallocate(iter) < 0) {
        Py_CLEAR(result);
    }
    # 释放第一个、第二个和第三个描述符的 Python 对象引用
    Py_XDECREF(descrs[0]);
    Py_XDECREF(descrs[1]);
    Py_XDECREF(descrs[2]);
    # 返回结果对象
    return result;
}


注释：


# 这行代码关闭了一个代码块，一般情况下与一对 '{' 对应，用于结束一个代码段的定义或逻辑结构。
```