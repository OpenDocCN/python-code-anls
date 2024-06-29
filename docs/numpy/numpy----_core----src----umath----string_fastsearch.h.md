# `.\numpy\numpy\_core\src\umath\string_fastsearch.h`

```
#ifndef _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_
#define _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_

/* stringlib: fastsearch implementation taken from CPython */

#include <Python.h>
#include <limits.h>
#include <string.h>
#include <wchar.h>

#include <type_traits>

#include <numpy/npy_common.h>


/* fast search/count implementation, based on a mix between boyer-
   moore and horspool, with a few more bells and whistles on the top.
   for some more background, see:
   https://web.archive.org/web/20201107074620/http://effbot.org/zone/stringlib.htm */

/* note: fastsearch may access s[n], but this is being checked for in this
   implementation, because NumPy strings are not NULL-terminated.
   also, the count mode returns -1 if there cannot possibly be a match
   in the target string, and 0 if it has actually checked for matches,
   but didn't find any. callers beware! */

/* If the strings are long enough, use Crochemore and Perrin's Two-Way
   algorithm, which has worst-case O(n) runtime and best-case O(n/k).
   Also compute a table of shifts to achieve O(n/k) in more cases,
   and often (data dependent) deduce larger shifts than pure C&P can
   deduce. See stringlib_find_two_way_notes.txt in this folder for a
   detailed explanation. */

// 定义常量，用于标识快速计数、快速搜索和反向快速搜索
#define FAST_COUNT 0
#define FAST_SEARCH 1
#define FAST_RSEARCH 2

// 根据平台的位数选择布隆过滤器的宽度
#if LONG_BIT >= 128
#define STRINGLIB_BLOOM_WIDTH 128
#elif LONG_BIT >= 64
#define STRINGLIB_BLOOM_WIDTH 64
#elif LONG_BIT >= 32
#define STRINGLIB_BLOOM_WIDTH 32
#else
#error "LONG_BIT is smaller than 32"
#endif

// 定义布隆过滤器的添加操作
#define STRINGLIB_BLOOM_ADD(mask, ch) \
    ((mask |= (1UL << ((ch) & (STRINGLIB_BLOOM_WIDTH -1)))))

// 定义布隆过滤器的查询操作
#define STRINGLIB_BLOOM(mask, ch)     \
    ((mask &  (1UL << ((ch) & (STRINGLIB_BLOOM_WIDTH -1)))))

// 定义前向和后向搜索的方向常量
#define FORWARD_DIRECTION 1
#define BACKWARD_DIRECTION -1

// 定义MEMCHR_CUT_OFF常量，限制使用memchr的最大长度
#define MEMCHR_CUT_OFF 15

// 定义CheckedIndexer结构模板，用于安全索引字符串
template <typename char_type>
struct CheckedIndexer {
    char_type *buffer;
    size_t length;

    CheckedIndexer()
    {
        buffer = NULL;
        length = 0;
    }

    CheckedIndexer(char_type *buf, size_t len)
    {
        buffer = buf;
        length = len;
    }

    char_type
    operator*()
    {
        return *(this->buffer);
    }

    char_type
    operator[](size_t index)
    {
        // 检查索引是否超出长度，是则返回空字符
        if (index >= this->length) {
            return (char_type) 0;
        }
        return this->buffer[index];
    }

    CheckedIndexer<char_type>
    operator+(size_t rhs)
    {
        // 如果加法超出长度，则截断
        if (rhs > this->length) {
            rhs = this->length;
        }
        return CheckedIndexer<char_type>(this->buffer + rhs, this->length - rhs);
    }

    CheckedIndexer<char_type>&
    operator+=(size_t rhs)
    {
        // 如果加法超出长度，则截断
        if (rhs > this->length) {
            rhs = this->length;
        }
        this->buffer += rhs;
        this->length -= rhs;
        return *this;
    }

    CheckedIndexer<char_type>
    operator++(int)
    {
        // 后缀自增运算符重载
        *this += 1;
        return *this;
    }

    CheckedIndexer<char_type>&
    operator-=(size_t rhs)
    {
        // 减去另一个索引器对象的长度，并增加自身的长度，然后返回自身引用
        this->buffer -= rhs;
        this->length += rhs;
        return *this;
    }
    
    CheckedIndexer<char_type>
    operator--(int)
    {
        // 后置自减运算符：先减去 1，然后返回修改后的对象的副本
        *this -= 1;
        return *this;
    }
    
    std::ptrdiff_t
    operator-(CheckedIndexer<char_type> rhs)
    {
        // 计算当前索引器对象与另一个索引器对象之间的缓冲区距离差
        return this->buffer - rhs.buffer;
    }
    
    CheckedIndexer<char_type>
    operator-(size_t rhs)
    {
        // 减去指定的长度并返回新的索引器对象
        return CheckedIndexer(this->buffer - rhs, this->length + rhs);
    }
    
    int
    operator>(CheckedIndexer<char_type> rhs)
    {
        // 比较当前索引器对象的缓冲区地址是否大于另一个索引器对象的缓冲区地址
        return this->buffer > rhs.buffer;
    }
    
    int
    operator>=(CheckedIndexer<char_type> rhs)
    {
        // 比较当前索引器对象的缓冲区地址是否大于或等于另一个索引器对象的缓冲区地址
        return this->buffer >= rhs.buffer;
    }
    
    int
    operator<(CheckedIndexer<char_type> rhs)
    {
        // 比较当前索引器对象的缓冲区地址是否小于另一个索引器对象的缓冲区地址
        return this->buffer < rhs.buffer;
    }
    
    int
    operator<=(CheckedIndexer<char_type> rhs)
    {
        // 比较当前索引器对象的缓冲区地址是否小于或等于另一个索引器对象的缓冲区地址
        return this->buffer <= rhs.buffer;
    }
    
    int
    operator==(CheckedIndexer<char_type> rhs)
    {
        // 比较当前索引器对象的缓冲区地址是否等于另一个索引器对象的缓冲区地址
        return this->buffer == rhs.buffer;
    }
// 定义一个以 char_type 类型为模板参数的内联函数，用于在 CheckedIndexer 区间内查找字符 ch 的位置
template <typename char_type>
inline Py_ssize_t
findchar(CheckedIndexer<char_type> s, Py_ssize_t n, char_type ch)
{
    // 初始化指针 p 指向 s.buffer，e 指向 (s + n).buffer
    char_type *p = s.buffer, *e = (s + n).buffer;

    // 如果 n 大于 MEMCHR_CUT_OFF，选择不同的内存查找函数
    if (n > MEMCHR_CUT_OFF) {
        // 如果 char_type 是 char 类型，使用 memchr 函数查找字符 ch
        if (std::is_same<char_type, char>::value) {
            p = (char_type *)memchr(s.buffer, ch, n);
        }
        // 如果 char_type 是 wchar_t 类型，使用 wmemchr 函数查找字符 ch
        else if (NPY_SIZEOF_WCHAR_T == 2) {
            // 遍历 s 中的字符，找到第一个等于 ch 的位置并返回
            for (Py_ssize_t i=0; i<n; i++) {
                if (s[i] == ch) {
                    return i;
                }
            }
            return -1;
        }
        // 其他情况，假设 char_type 是 wchar_t 类型，使用 wmemchr 函数查找字符 ch
        else {
            p = (char_type *)wmemchr((wchar_t *)(s.buffer), ch, n);
        }
        // 如果找到字符 ch，返回相对于 s.buffer 的偏移量
        if (p != NULL) {
            return (p - s.buffer);
        }
        // 否则返回 -1 表示未找到
        return -1;
    }

    // 在较小的区间内进行遍历查找字符 ch
    while (p < e) {
        if (*p == ch) {
            return (p - s.buffer);  // 返回相对于 s.buffer 的偏移量
        }
        p++;
    }
    // 没找到返回 -1
    return -1;
}

// 定义一个以 char_type 类型为模板参数的内联函数，用于在 CheckedIndexer 区间内从后向前查找字符 ch 的位置
template <typename char_type>
inline Py_ssize_t
rfindchar(CheckedIndexer<char_type> s, Py_ssize_t n, char_type ch)
{
    // 初始化指针 p 指向 s + n，即 s 区间的末尾
    CheckedIndexer<char_type> p = s + n;

    // 从后向前遍历 s 区间查找字符 ch
    while (p > s) {
        p--;  // 移动指针到前一个位置
        if (*p == ch)
            return (p - s);  // 返回相对于 s 的偏移量
    }
    // 没找到返回 -1
    return -1;
}

/* Change to a 1 to see logging comments walk through the algorithm. */
// 如果想要查看日志评论来跟踪算法，将下面的条件改为 1
#if 0 && STRINGLIB_SIZEOF_CHAR == 1
// 定义宏 LOG，用于打印日志信息
# define LOG(...) printf(__VA_ARGS__)
// 定义宏 LOG_STRING，用于以字符串形式打印 s 中长度为 n 的内容
# define LOG_STRING(s, n) printf("\"%.*s\"", (int)(n), s)
// 定义宏 LOG_LINEUP，用于打印对齐的日志信息
# define LOG_LINEUP() do {                                         \
    LOG("> "); LOG_STRING(haystack, len_haystack); LOG("\n> ");    \
    LOG("%*s",(int)(window_last - haystack + 1 - len_needle), ""); \
    LOG_STRING(needle, len_needle); LOG("\n");                     \
} while(0)
#else
// 如果不需要日志信息，则定义空的宏
# define LOG(...)
# define LOG_STRING(s, n)
# define LOG_LINEUP()
#endif

// 定义一个以 char_type 类型为模板参数的静态内联函数，用于执行词法搜索
template <typename char_type>
static inline Py_ssize_t
_lex_search(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            Py_ssize_t *return_period, int invert_alphabet)
{
    /* Do a lexicographic search. Essentially this:
           >>> max(needle[i:] for i in range(len(needle)+1))
       Also find the period of the right half.   */
    // 初始化变量 max_suffix 和 candidate 为 0 和 1，k 为 0，period 为 1
    Py_ssize_t max_suffix = 0;
    Py_ssize_t candidate = 1;
    Py_ssize_t k = 0;
    // 初始化变量 period 为 1，用于存储右半部分的周期长度
    // 当候选起始位置加上k小于需要匹配的字符串长度时循环执行以下操作
    while (candidate + k < len_needle) {
        // 每次循环会增加候选位置、k值和最大后缀长度
        char_type a = needle[candidate + k];
        char_type b = needle[max_suffix + k];
        // 检查候选位置处的后缀是否比最大后缀更好
        if (invert_alphabet ? (b < a) : (a < b)) {
            // 未达到最大后缀的要求。
            // 下一个 k + 1 个字符是从候选位置开始非递增的，
            // 因此它们不会成为一个最大后缀的起始点。
            candidate += k + 1;
            k = 0;
            // 排除掉比已扫描的最大后缀长度更小的任何周期。
            period = candidate - max_suffix;
        }
        else if (a == b) {
            if (k + 1 != period) {
                // 继续扫描相等的字符串段
                k++;
            }
            else {
                // 匹配了一个完整的周期。
                // 开始匹配下一个周期。
                candidate += period;
                k = 0;
            }
        }
        else {
            // 比最大后缀做得更好，因此替换最大后缀。
            max_suffix = candidate;
            candidate++;
            k = 0;
            period = 1;
        }
    }
    // 将计算得到的周期长度存入返回参数中
    *return_period = period;
    // 返回最大后缀的起始位置
    return max_suffix;
}

template <typename char_type>
static inline Py_ssize_t
_factorize(CheckedIndexer<char_type> needle,
           Py_ssize_t len_needle,
           Py_ssize_t *return_period)
{
    /* 执行“关键因式分解”，使得：
       >>> needle = (left := needle[:cut]) + (right := needle[cut:])
       其中“局部周期”在最大化的情况下。

       切分点的局部周期是一个字符串 w 的最小长度，满足 (left 以 w 结尾或者 w 以 left 结尾)
       且 (right 以 w 开始或者 w 以 left 开始)。

       “关键因式分解定理”表明这个最大局部周期即为字符串的全局周期。

       Crochemore 和 Perrin (1991) 表明这个切分点可以通过两个位置来计算：
       一个位置提供按字典顺序最大化的右半部分，另一个位置提供按反转字母表顺序的相同效果。

       这是我们希望发生的：
           >>> x = "GCAGAGAG"
           >>> cut, period = factorize(x)
           >>> x[:cut], (right := x[cut:])
           ('GC', 'AGAGAG')
           >>> period  # 右半部分的周期
           2
           >>> right[period:] == right[:-period]
           True

       这是局部周期如何在上述示例中排列的情况：
                GC | AGAGAG
           AGAGAGC = AGAGAGC
       这个最小重复的长度是 7，确实是原始字符串的周期。 */

    Py_ssize_t cut1, period1, cut2, period2, cut, period;
    cut1 = _lex_search<char_type>(needle, len_needle, &period1, 0);
    cut2 = _lex_search<char_type>(needle, len_needle, &period2, 1);

    // 取较晚的切分点。
    if (cut1 > cut2) {
        period = period1;
        cut = cut1;
    }
    else {
        period = period2;
        cut = cut2;
    }

    LOG("split: "); LOG_STRING(needle, cut);
    LOG(" + "); LOG_STRING(needle + cut, len_needle - cut);
    LOG("\n");

    *return_period = period;
    return cut;
}


#define SHIFT_TYPE uint8_t
#define MAX_SHIFT UINT8_MAX

#define TABLE_SIZE_BITS 6u
#define TABLE_SIZE (1U << TABLE_SIZE_BITS)
#define TABLE_MASK (TABLE_SIZE - 1U)

template <typename char_type>
struct prework {
    CheckedIndexer<char_type> needle;
    Py_ssize_t len_needle;
    Py_ssize_t cut;
    Py_ssize_t period;
    Py_ssize_t gap;
    int is_periodic;
    SHIFT_TYPE table[TABLE_SIZE];
};


template <typename char_type>
static void
_preprocess(CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
            prework<char_type> *p)
{
    p->needle = needle;
    p->len_needle = len_needle;
    // 对 needle 进行关键因式分解，得到切分点和周期
    p->cut = _factorize(needle, len_needle, &(p->period));
    // 断言切分点加上周期不超过 needle 的长度
    assert(p->period + p->cut <= len_needle);
    int cmp;
    // 根据 char_type 的类型判断是否需要特殊处理
    if (std::is_same<char_type, npy_ucs4>::value) {
        cmp = memcmp(needle.buffer, needle.buffer + (p->period * sizeof(npy_ucs4)), (size_t) p->cut);
    }
    else {
        cmp = memcmp(needle.buffer, needle.buffer + p->period, (size_t) p->cut);
    }
    // 设置 p->is_periodic 标志，检查是否为周期性模式
    p->is_periodic = (0 == cmp);
    if (p->is_periodic) {
        // 如果是周期性模式，确保 p->cut 小于针长度的一半
        assert(p->cut <= len_needle/2);
        // 确保 p->cut 小于周期长度
        assert(p->cut < p->period);
        // 对于周期性模式，p->gap 不被使用，设置为未使用状态
        p->gap = 0; // unused
    }
    else {
        // 如果不是周期性模式，计算一个周期的下限
        p->period = Py_MAX(p->cut, len_needle - p->cut) + 1;
        // 计算最后一个字符与之前等效字符（模 TABLE_SIZE）之间的间隔
        p->gap = len_needle;
        char_type last = needle[len_needle - 1] & TABLE_MASK;
        for (Py_ssize_t i = len_needle - 2; i >= 0; i--) {
            char_type x = needle[i] & TABLE_MASK;
            if (x == last) {
                p->gap = len_needle - 1 - i;
                break;
            }
        }
    }
    // 填充压缩的Boyer-Moore "Bad Character"表
    // 计算未找到的字符的最小移动量
    Py_ssize_t not_found_shift = Py_MIN(len_needle, MAX_SHIFT);
    for (Py_ssize_t i = 0; i < (Py_ssize_t)TABLE_SIZE; i++) {
        // 将未找到的字符移动量存入 p->table
        p->table[i] = Py_SAFE_DOWNCAST(not_found_shift,
                                       Py_ssize_t, SHIFT_TYPE);
    }
    // 填充与针末尾字符对应的表项
    for (Py_ssize_t i = len_needle - not_found_shift; i < len_needle; i++) {
        // 计算当前字符的移动量并存入 p->table
        SHIFT_TYPE shift = Py_SAFE_DOWNCAST(len_needle - 1 - i,
                                            Py_ssize_t, SHIFT_TYPE);
        p->table[needle[i] & TABLE_MASK] = shift;
    }
// 结束上一个代码块的大括号，表明这是一个静态成员函数的结束
}

// 定义静态函数 _two_way，接受一个 CheckedIndexer<char_type> 类型的 haystack、一个 Py_ssize_t 类型的 len_haystack 和一个 prework<char_type> 类型的指针 p 作为参数
static Py_ssize_t
_two_way(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
         prework<char_type> *p)
{
    // 使用 Crochemore 和 Perrin 的两向算法（1991年）进行字符串搜索
    // 参考网址：http://www-igm.univ-mlv.fr/~lecroq/string/node26.html#SECTION00260
    // 获取针的长度
    const Py_ssize_t len_needle = p->len_needle;
    // 获取针的分割点
    const Py_ssize_t cut = p->cut;
    // 获取周期
    Py_ssize_t period = p->period;
    // 获取针
    CheckedIndexer<char_type> needle = p->needle;
    // 计算窗口的最后一个元素
    CheckedIndexer<char_type> window_last = haystack + (len_needle - 1);
    // 计算 haystack 的结束位置
    CheckedIndexer<char_type> haystack_end = haystack + len_haystack;
    // 获取表
    SHIFT_TYPE *table = p->table;
    // 定义窗口
    CheckedIndexer<char_type> window;
    // 输出调试信息，显示当前使用的两向算法查找的针和 haystack
    LOG("===== Two-way: \"%s\" in \"%s\". =====\n", needle, haystack);
    # 如果模式是周期性的，则执行以下逻辑
    if (p->is_periodic) {
        # 记录日志，指示模式是周期性的
        LOG("Needle is periodic.\n");
        # 初始化内存变量为0
        Py_ssize_t memory = 0;
      periodicwindowloop:
        # 当窗口末尾位置小于haystack的结束位置时执行循环
        while (window_last < haystack_end) {
            # 断言内存变量为0，用于调试目的
            assert(memory == 0);
            # 无限循环，用于处理窗口内的匹配逻辑
            for (;;) {
                # 记录日志，指示当前位置
                LOG_LINEUP();
                # 根据表格查找当前窗口末尾字符的移动步长
                Py_ssize_t shift = table[window_last[0] & TABLE_MASK];
                # 更新窗口末尾位置
                window_last += shift;
                # 如果移动步长为0，跳出当前循环
                if (shift == 0) {
                    break;
                }
                # 如果窗口末尾位置超过了haystack的结束位置，返回-1
                if (window_last >= haystack_end) {
                    return -1;
                }
                # 记录日志，指示Horspool算法的跳跃
                LOG("Horspool skip\n");
            }
          no_shift:
            # 计算窗口的起始位置
            window = window_last - len_needle + 1;
            # 断言窗口末尾字符与模式末尾字符匹配
            assert((window[len_needle - 1] & TABLE_MASK) ==
                   (needle[len_needle - 1] & TABLE_MASK));
            # 初始化索引i为cut和memory的最大值
            Py_ssize_t i = Py_MAX(cut, memory);
            # 逐个比较模式与窗口中的字符
            for (; i < len_needle; i++) {
                if (needle[i] != window[i]) {
                    # 记录日志，指示右半部分不匹配
                    LOG("Right half does not match.\n");
                    # 更新窗口末尾位置，准备下一轮匹配
                    window_last += (i - cut + 1);
                    # 重置内存变量
                    memory = 0;
                    # 跳回periodicwindowloop标签，重新开始匹配
                    goto periodicwindowloop;
                }
            }
            # 检查左半部分是否匹配
            for (i = memory; i < cut; i++) {
                if (needle[i] != window[i]) {
                    # 记录日志，指示左半部分不匹配
                    LOG("Left half does not match.\n");
                    # 更新窗口末尾位置，跳过周期性的距离
                    window_last += period;
                    # 更新内存变量，准备下一轮匹配
                    memory = len_needle - period;
                    # 如果窗口末尾位置超过了haystack的结束位置，返回-1
                    if (window_last >= haystack_end) {
                        return -1;
                    }
                    # 根据表格查找当前窗口末尾字符的移动步长
                    Py_ssize_t shift = table[window_last[0] & TABLE_MASK];
                    # 如果移动步长不为0，执行内存跳转逻辑
                    if (shift) {
                        # 记录日志，指示使用内存跳转
                        LOG("Skip with Memory.\n");
                        # 重置内存变量
                        memory = 0;
                        # 更新窗口末尾位置，跳过最大的移动步长
                        window_last += Py_MAX(shift, Py_MAX(cut, memory) - cut + 1);
                        # 跳回periodicwindowloop标签，重新开始匹配
                        goto periodicwindowloop;
                    }
                    # 跳转到no_shift标签，重新开始匹配
                    goto no_shift;
                }
            }
            # 记录日志，指示找到匹配
            LOG("Found a match!\n");
            # 返回匹配位置在haystack中的偏移量
            return window - haystack;
        }
    }
    # 进入此代码块表示尝试在主循环中查找匹配的窗口
    else {
        # 获取 p 结构体中的间隙值
        Py_ssize_t gap = p->gap;
        # 将 period 设置为 gap 和之前计算的 period 的最大值
        period = Py_MAX(gap, period);
        # 输出日志表明找不到 needle 的周期性
        LOG("Needle is not periodic.\n");
        # 计算 gap_jump_end 作为 len_needle 和 cut + gap 的最小值
        Py_ssize_t gap_jump_end = Py_MIN(len_needle, cut + gap);
      windowloop:
        while (window_last < haystack_end) {
            for (;;) {
                # 输出日志表明进行 Horspool 算法的跳跃
                LOG_LINEUP();
                # 使用预先计算的表来计算窗口中第一个字符的移动量
                Py_ssize_t shift = table[window_last[0] & TABLE_MASK];
                # 将窗口向前移动 shift 个字符
                window_last += shift;
                # 如果 shift 为 0，则窗口无法移动，跳出内循环
                if (shift == 0) {
                    break;
                }
                # 如果窗口超过了 haystack_end，表示未找到匹配，返回 -1
                if (window_last >= haystack_end) {
                    return -1;
                }
                # 输出日志表明进行 Horspool 算法的跳跃
                LOG("Horspool skip\n");
            }
            # 计算窗口的起始位置
            window = window_last - len_needle + 1;
            # 断言窗口和 needle 的最后一个字符相等
            assert((window[len_needle - 1] & TABLE_MASK) ==
                   (needle[len_needle - 1] & TABLE_MASK));
            # 检查窗口中从 cut 到 gap_jump_end 的部分与 needle 是否匹配
            for (Py_ssize_t i = cut; i < gap_jump_end; i++) {
                # 如果不匹配，输出日志表明右半部分出现不匹配，使用 gap 进行跳跃
                LOG("Early right half mismatch: jump by gap.\n");
                # 断言 gap 大于等于 i - cut + 1
                assert(gap >= i - cut + 1);
                # 调整窗口位置以跳过不匹配的部分
                window_last += gap;
                # 跳转回到 windowloop 标签处，重新开始查找
                goto windowloop;
            }
            # 检查窗口中从 gap_jump_end 到 len_needle 的部分与 needle 是否匹配
            for (Py_ssize_t i = gap_jump_end; i < len_needle; i++) {
                # 如果不匹配，输出日志表明右半部分出现不匹配，根据 i 和 cut 进行跳跃
                LOG("Late right half mismatch.\n");
                # 断言 i - cut + 1 大于 gap
                assert(i - cut + 1 > gap);
                # 调整窗口位置以跳过不匹配的部分
                window_last += i - cut + 1;
                # 跳转回到 windowloop 标签处，重新开始查找
                goto windowloop;
            }
            # 检查窗口中从 0 到 cut 的部分与 needle 是否匹配
            for (Py_ssize_t i = 0; i < cut; i++) {
                # 如果不匹配，输出日志表明左半部分不匹配
                LOG("Left half does not match.\n");
                # 调整窗口位置以跳过不匹配的部分
                window_last += period;
                # 跳转回到 windowloop 标签处，重新开始查找
                goto windowloop;
            }
            # 输出日志表明找到了匹配的窗口
            LOG("Found a match!\n");
            # 返回匹配位置相对于 haystack 的偏移量
            return window - haystack;
        }
    }
    # 输出日志表明未找到匹配
    LOG("Not found. Returning -1.\n");
    # 返回 -1 表示未找到匹配
    return -1;
}

template <typename char_type>
static inline Py_ssize_t
_two_way_find(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
              CheckedIndexer<char_type> needle, Py_ssize_t len_needle)
{
    // 输出日志，记录在 haystack 中查找 needle 的过程
    LOG("###### Finding \"%s\" in \"%s\".\n", needle, haystack);
    
    // 创建用于预处理的 prework 对象
    prework<char_type> p;
    
    // 对 needle 进行预处理
    _preprocess(needle, len_needle, &p);
    
    // 调用 _two_way 函数执行双向查找并返回结果
    return _two_way(haystack, len_haystack, &p);
}


template <typename char_type>
static inline Py_ssize_t
_two_way_count(CheckedIndexer<char_type> haystack, Py_ssize_t len_haystack,
               CheckedIndexer<char_type> needle, Py_ssize_t len_needle,
               Py_ssize_t maxcount)
{
    // 输出日志，记录在 haystack 中统计 needle 的过程
    LOG("###### Counting \"%s\" in \"%s\".\n", needle, haystack);
    
    // 创建用于预处理的 prework 对象
    prework<char_type> p;
    
    // 对 needle 进行预处理
    _preprocess(needle, len_needle, &p);
    
    // 初始化索引和计数
    Py_ssize_t index = 0, count = 0;
    
    // 进行循环，直到达到 maxcount 或者找不到为止
    while (1) {
        Py_ssize_t result;
        // 在 haystack 中执行双向查找
        result = _two_way(haystack + index, len_haystack - index, &p);
        
        // 如果找不到，则返回当前的计数
        if (result == -1) {
            return count;
        }
        
        // 找到一个匹配项，增加计数
        count++;
        
        // 如果达到了最大计数要求，返回 maxcount
        if (count == maxcount) {
            return maxcount;
        }
        
        // 更新索引，跳过当前匹配项后的 needle 长度
        index += result + len_needle;
    }
    // 返回最终的计数结果
    return count;
}

#undef SHIFT_TYPE
#undef NOT_FOUND
#undef SHIFT_OVERFLOW
#undef TABLE_SIZE_BITS
#undef TABLE_SIZE
#undef TABLE_MASK

#undef LOG
#undef LOG_STRING
#undef LOG_LINEUP

template <typename char_type>
static inline Py_ssize_t
default_find(CheckedIndexer<char_type> s, Py_ssize_t n,
             CheckedIndexer<char_type> p, Py_ssize_t m,
             Py_ssize_t maxcount, int mode)
{
    // 计算 haystack 和 needle 的长度差
    const Py_ssize_t w = n - m;
    
    // 计算 needle 最后一个字符的索引
    Py_ssize_t mlast = m - 1, count = 0;
    
    // 初始化 gap 为 mlast
    Py_ssize_t gap = mlast;
    
    // 获取 needle 的最后一个字符
    const char_type last = p[mlast];
    
    // 设置 ss 指向 s 的末尾
    CheckedIndexer<char_type> ss = s + mlast;

    // 初始化 Bloom filter 的 mask
    unsigned long mask = 0;
    for (Py_ssize_t i = 0; i < mlast; i++) {
        // 将 p 中的字符添加到 Bloom filter 的 mask 中
        STRINGLIB_BLOOM_ADD(mask, p[i]);
        // 如果字符与 last 相同，更新 gap
        if (p[i] == last) {
            gap = mlast - i - 1;
        }
    }
    // 将 last 添加到 Bloom filter 的 mask 中
    STRINGLIB_BLOOM_ADD(mask, last);

    // 开始在 haystack 中查找 needle
    for (Py_ssize_t i = 0; i <= w; i++) {
        // 如果 ss 中的字符与 needle 的最后一个字符相同
        if (ss[i] == last) {
            /* candidate match */
            Py_ssize_t j;
            // 检查是否找到完全匹配的 needle
            for (j = 0; j < mlast; j++) {
                if (s[i+j] != p[j]) {
                    break;
                }
            }
            // 如果完全匹配，根据模式返回结果或者增加计数
            if (j == mlast) {
                /* got a match! */
                if (mode != FAST_COUNT) {
                    return i;
                }
                count++;
                if (count == maxcount) {
                    return maxcount;
                }
                i = i + mlast;
                continue;
            }
            // 如果不匹配，检查下一个字符是否是模式的一部分
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
            else {
                i = i + gap;
            }
        }
        else {
            // 如果不匹配，检查下一个字符是否是模式的一部分
            if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                i = i + m;
            }
        }
        // 继续下一个字符的检查
    }
    // 返回最终的计数结果
    return count;
}
    }
    // 如果 mode 等于 FAST_COUNT，则返回 count；否则返回 -1
    return mode == FAST_COUNT ? count : -1;
    /* 
    定义静态函数 adaptive_find，用于在字符串 s 中查找模式字符串 p 的出现位置
    返回找到的位置或计数，具体行为由 mode 参数决定
    */
    template <typename char_type>
    static Py_ssize_t
    adaptive_find(CheckedIndexer<char_type> s, Py_ssize_t n,
                  CheckedIndexer<char_type> p, Py_ssize_t m,
                  Py_ssize_t maxcount, int mode)
    {
        /*
        计算有效搜索范围 w，即 s 的长度 n 减去 p 的长度 m
        */
        const Py_ssize_t w = n - m;
        // 初始化模式字符串中最后一个字符的索引
        Py_ssize_t mlast = m - 1, count = 0;
        // gap 表示跳跃步长，默认为 mlast
        Py_ssize_t gap = mlast;
        // hits 记录部分匹配的字符数，res 保存搜索结果
        Py_ssize_t hits = 0, res;
        // 获取模式字符串的最后一个字符
        const char_type last = p[mlast];
        // ss 是 s 的子字符串，从索引 mlast 开始
        CheckedIndexer<char_type> ss = s + mlast;

        // 初始化布隆过滤器的 mask
        unsigned long mask = 0;
        // 遍历模式字符串 p 的前 mlast 个字符
        for (Py_ssize_t i = 0; i < mlast; i++) {
            // 将 p[i] 添加到布隆过滤器的 mask 中
            STRINGLIB_BLOOM_ADD(mask, p[i]);
            // 如果 p[i] 等于模式字符串的最后一个字符 last
            if (p[i] == last) {
                // 更新 gap 为 mlast - i - 1，即最大跳跃步长
                gap = mlast - i - 1;
            }
        }
        // 将模式字符串的最后一个字符 last 添加到布隆过滤器的 mask 中
        STRINGLIB_BLOOM_ADD(mask, last);

        // 主循环，遍历 s 中所有可能的起始位置 i
        for (Py_ssize_t i = 0; i <= w; i++) {
            // 如果 ss[i] 等于模式字符串的最后一个字符 last
            if (ss[i] == last) {
                /* candidate match */
                Py_ssize_t j;
                // 逐个比较 s[i+j] 和 p[j]，判断是否完全匹配
                for (j = 0; j < mlast; j++) {
                    if (s[i+j] != p[j]) {
                        break;
                    }
                }
                // 如果 j 等于 mlast，说明找到了完全匹配的位置
                if (j == mlast) {
                    /* got a match! */
                    // 如果不是快速计数模式，直接返回匹配位置 i
                    if (mode != FAST_COUNT) {
                        return i;
                    }
                    // 否则增加匹配计数
                    count++;
                    // 如果达到最大匹配数，返回最大匹配数
                    if (count == maxcount) {
                        return maxcount;
                    }
                    // 跳过已匹配的 mlast 个字符
                    i = i + mlast;
                    continue;
                }
                // 记录部分匹配字符数
                hits += j + 1;
                // 如果部分匹配字符数 hits 超过 m 的四分之一，并且剩余未搜索长度超过 2000
                if (hits > m / 4 && w - i > 2000) {
                    // 根据模式 mode 执行不同的搜索或计数操作
                    if (mode == FAST_SEARCH) {
                        res = _two_way_find(s + i, n - i, p, m);
                        // 返回搜索结果或 -1
                        return res == -1 ? -1 : res + i;
                    }
                    else {
                        res = _two_way_count(s + i, n - i, p, m, maxcount - count);
                        // 返回计数结果加上当前匹配数
                        return res + count;
                    }
                }
                /* miss: check if next character is part of pattern */
                // 如果下一个字符不是模式字符串的一部分，跳过 m 个字符
                if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                    i = i + m;
                }
                else {
                    // 否则跳过 gap 个字符
                    i = i + gap;
                }
            }
            else {
                /* skip: check if next character is part of pattern */
                // 如果下一个字符不是模式字符串的一部分，跳过 m 个字符
                if (!STRINGLIB_BLOOM(mask, ss[i+1])) {
                    i = i + m;
                }
            }
        }
        // 根据模式 mode 返回计数或 -1（未找到）
        return mode == FAST_COUNT ? count : -1;
    }
    # 从字符串末尾开始逐个检查
    for (i = w; i >= 0; i--) {
        # 如果当前位置字符与模式字符串的第一个字符匹配
        if (s[i] == p[0]) {
            /* 候选匹配 */
            # 从模式字符串末尾向前检查
            for (j = mlast; j > 0; j--) {
                # 如果当前位置后续字符与模式字符串对应位置的字符不匹配，中断循环
                if (s[i+j] != p[j]) {
                    break;
                }
            }
            # 如果 j 等于 0，表示找到了完全匹配的位置
            if (j == 0) {
                /* 找到了匹配位置！ */
                return i;
            }
            /* 匹配失败：检查前一个字符是否是模式字符串的一部分 */
            # 如果前一个字符不是模式字符串的一部分，跳过匹配长度 m
            if (i > 0 && !STRINGLIB_BLOOM(mask, s[i-1])) {
                i = i - m;
            }
            else {
                # 否则根据预先计算的跳跃值跳过
                i = i - skip;
            }
        }
        else {
            /* 跳过：检查前一个字符是否是模式字符串的一部分 */
            # 如果前一个字符不是模式字符串的一部分，跳过匹配长度 m
            if (i > 0 && !STRINGLIB_BLOOM(mask, s[i-1])) {
                i = i - m;
            }
        }
    }
    # 如果没有找到匹配，返回 -1
    return -1;
// 结束 C++ 的 extern "C" 块
}

// 定义一个静态内联函数，用于计算字符数组中指定字符的出现次数
template <typename char_type>
static inline Py_ssize_t
countchar(CheckedIndexer<char_type> s, Py_ssize_t n,
          const char_type p0, Py_ssize_t maxcount)
{
    Py_ssize_t i, count = 0;
    // 遍历字符数组 s，统计字符 p0 的出现次数，最多不超过 maxcount 次
    for (i = 0; i < n; i++) {
        if (s[i] == p0) {
            count++;
            // 如果达到了最大次数限制，则提前返回
            if (count == maxcount) {
                return maxcount;
            }
        }
    }
    // 返回字符 p0 的总出现次数
    return count;
}

// 定义一个内联函数，实现快速搜索子字符串在主字符串中的功能
template <typename char_type>
inline Py_ssize_t
fastsearch(char_type* s, Py_ssize_t n,
           char_type* p, Py_ssize_t m,
           Py_ssize_t maxcount, int mode)
{
    // 创建 CheckedIndexer 对象来包装字符数组 s 和 p
    CheckedIndexer<char_type> s_(s, n);
    CheckedIndexer<char_type> p_(p, m);

    // 如果主字符串长度小于子字符串长度，或者是 FAST_COUNT 模式且最大计数为 0，则返回 -1
    if (n < m || (mode == FAST_COUNT && maxcount == 0)) {
        return -1;
    }

    /* look for special cases */
    // 处理特殊情况
    if (m <= 1) {
        if (m <= 0) {
            return -1;
        }
        // 对于长度为 1 的子字符串，根据模式选择不同的查找方法
        if (mode == FAST_SEARCH)
            return findchar(s_, n, p_[0]);    // 在主字符串中查找字符 p_[0]
        else if (mode == FAST_RSEARCH)
            return rfindchar(s_, n, p_[0]);   // 在主字符串中反向查找字符 p_[0]
        else {
            return countchar(s_, n, p_[0], maxcount);  // 计算主字符串中字符 p_[0] 的出现次数
        }
    }

    // 处理一般情况
    if (mode != FAST_RSEARCH) {
        // 根据不同的条件选择默认的查找方法或者使用两路算法
        if (n < 2500 || (m < 100 && n < 30000) || m < 6) {
            return default_find(s_, n, p_, m, maxcount, mode);  // 默认的查找方法
        }
        else if ((m >> 2) * 3 < (n >> 2)) {
            /* 33% threshold, but don't overflow. */
            /* 对于较大的问题，其中子字符串并不占主字符串的大部分时，
               使用两路算法的启动成本可能会得到回报。 */
            if (mode == FAST_SEARCH) {
                return _two_way_find(s_, n, p_, m);   // 使用两路算法进行查找
            }
            else {
                return _two_way_count(s_, n, p_, m, maxcount);   // 使用两路算法进行计数
            }
        }
        else {
            /* To ensure that we have good worst-case behavior,
               here's an adaptive version of the algorithm, where if
               we match O(m) characters without any matches of the
               entire needle, then we predict that the startup cost of
               the two-way algorithm will probably be worth it. */
            // 使用自适应版本的算法进行查找
            return adaptive_find(s_, n, p_, m, maxcount, mode);
        }
    }
    else {
        /* FAST_RSEARCH */
        // 使用默认的反向查找方法
        return default_rfind(s_, n, p_, m, maxcount, mode);
    }
}

// 结束条件编译指令的区域
#endif  /* _NPY_CORE_SRC_UMATH_STRING_FASTSEARCH_H_ */
```