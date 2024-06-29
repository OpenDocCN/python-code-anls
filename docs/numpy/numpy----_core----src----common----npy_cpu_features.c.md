# `.\numpy\numpy\_core\src\common\npy_cpu_features.c`

```py
/*
 * Include necessary headers for CPU feature detection and definition.
 * These headers ensure that CPU baseline definitions are accessible.
 */
#include "npy_cpu_features.h"
#include "npy_cpu_dispatch.h" // To guarantee the CPU baseline definitions are in scope.
#include "numpy/npy_common.h"
#include "numpy/npy_cpu.h" // To guarantee the CPU definitions are in scope.

/******************** Private Definitions *********************/

// This array holds boolean values indicating whether each CPU feature is available.
// It is initialized during module initialization and remains immutable thereafter.
// It is not included in the global data struct due to shared usage across modules.
static unsigned char npy__cpu_have[NPY_CPU_FEATURE_MAX];

/******************** Private Declarations *********************/

// Function prototype for runtime CPU feature detection initialization
static void
npy__cpu_init_features(void);

/*
 * Enable or disable CPU dispatched features at runtime based on environment variables
 * `NPY_ENABLE_CPU_FEATURES` or `NPY_DISABLE_CPU_FEATURES`.
 *
 * Multiple features can be enabled or disabled, separated by space, comma, or tab.
 * Raises an error if parsing fails or if a specified feature is not valid or could not be enabled/disabled.
 */
static int
npy__cpu_check_env(int disable, const char *env);

/* Ensure that CPU baseline features required by the build are supported at runtime */
static int
npy__cpu_validate_baseline(void);

/******************** Public Definitions *********************/

// Function to check if a specific CPU feature is available
NPY_VISIBILITY_HIDDEN int
npy_cpu_have(int feature_id)
{
    // Check if the feature_id is within valid range
    if (feature_id <= NPY_CPU_FEATURE_NONE || feature_id >= NPY_CPU_FEATURE_MAX)
        return 0;
    // Return the boolean value indicating if the feature is available
    return npy__cpu_have[feature_id];
}

// Function to initialize CPU features detection at module initialization
NPY_VISIBILITY_HIDDEN int
npy_cpu_init(void)
{
    // Initialize CPU features detection
    npy__cpu_init_features();

    // Validate CPU baseline features required by the build
    if (npy__cpu_validate_baseline() < 0) {
        return -1;
    }

    // Check if both enable and disable environment variables are set, which is not allowed
    char *enable_env = getenv("NPY_ENABLE_CPU_FEATURES");
    char *disable_env = getenv("NPY_DISABLE_CPU_FEATURES");
    int is_enable = enable_env && enable_env[0];
    int is_disable = disable_env && disable_env[0];
    if (is_enable & is_disable) {
        PyErr_Format(PyExc_ImportError,
            "Both NPY_DISABLE_CPU_FEATURES and NPY_ENABLE_CPU_FEATURES "
            "environment variables cannot be set simultaneously."
        );
        return -1;
    }

    // If either enable or disable environment variable is set, process it
    if (is_enable | is_disable) {
        if (npy__cpu_check_env(is_disable, is_disable ? disable_env : enable_env) < 0) {
            return -1;
        }
    }

    // Initialization successful
    return 0;
}

// Structure definition to hold CPU features and their string representations
static struct {
  enum npy_cpu_features feature;
  char const *string;
# 创建一个静态的结构体数组 `features`，用于描述不同的CPU特性及其名称
} features[] = {{NPY_CPU_FEATURE_MMX, "MMX"},
                {NPY_CPU_FEATURE_SSE, "SSE"},
                {NPY_CPU_FEATURE_SSE2, "SSE2"},
                {NPY_CPU_FEATURE_SSE3, "SSE3"},
                {NPY_CPU_FEATURE_SSSE3, "SSSE3"},
                {NPY_CPU_FEATURE_SSE41, "SSE41"},
                {NPY_CPU_FEATURE_POPCNT, "POPCNT"},
                {NPY_CPU_FEATURE_SSE42, "SSE42"},
                {NPY_CPU_FEATURE_AVX, "AVX"},
                {NPY_CPU_FEATURE_F16C, "F16C"},
                {NPY_CPU_FEATURE_XOP, "XOP"},
                {NPY_CPU_FEATURE_FMA4, "FMA4"},
                {NPY_CPU_FEATURE_FMA3, "FMA3"},
                {NPY_CPU_FEATURE_AVX2, "AVX2"},
                {NPY_CPU_FEATURE_AVX512F, "AVX512F"},
                {NPY_CPU_FEATURE_AVX512CD, "AVX512CD"},
                {NPY_CPU_FEATURE_AVX512ER, "AVX512ER"},
                {NPY_CPU_FEATURE_AVX512PF, "AVX512PF"},
                {NPY_CPU_FEATURE_AVX5124FMAPS, "AVX5124FMAPS"},
                {NPY_CPU_FEATURE_AVX5124VNNIW, "AVX5124VNNIW"},
                {NPY_CPU_FEATURE_AVX512VPOPCNTDQ, "AVX512VPOPCNTDQ"},
                {NPY_CPU_FEATURE_AVX512VL, "AVX512VL"},
                {NPY_CPU_FEATURE_AVX512BW, "AVX512BW"},
                {NPY_CPU_FEATURE_AVX512DQ, "AVX512DQ"},
                {NPY_CPU_FEATURE_AVX512VNNI, "AVX512VNNI"},
                {NPY_CPU_FEATURE_AVX512IFMA, "AVX512IFMA"},
                {NPY_CPU_FEATURE_AVX512VBMI, "AVX512VBMI"},
                {NPY_CPU_FEATURE_AVX512VBMI2, "AVX512VBMI2"},
                {NPY_CPU_FEATURE_AVX512BITALG, "AVX512BITALG"},
                {NPY_CPU_FEATURE_AVX512FP16 , "AVX512FP16"},
                {NPY_CPU_FEATURE_AVX512_KNL, "AVX512_KNL"},
                {NPY_CPU_FEATURE_AVX512_KNM, "AVX512_KNM"},
                {NPY_CPU_FEATURE_AVX512_SKX, "AVX512_SKX"},
                {NPY_CPU_FEATURE_AVX512_CLX, "AVX512_CLX"},
                {NPY_CPU_FEATURE_AVX512_CNL, "AVX512_CNL"},
                {NPY_CPU_FEATURE_AVX512_ICL, "AVX512_ICL"},
                {NPY_CPU_FEATURE_AVX512_SPR, "AVX512_SPR"},
                {NPY_CPU_FEATURE_VSX, "VSX"},
                {NPY_CPU_FEATURE_VSX2, "VSX2"},
                {NPY_CPU_FEATURE_VSX3, "VSX3"},
                {NPY_CPU_FEATURE_VSX4, "VSX4"},
                {NPY_CPU_FEATURE_VX, "VX"},
                {NPY_CPU_FEATURE_VXE, "VXE"},
                {NPY_CPU_FEATURE_VXE2, "VXE2"},
                {NPY_CPU_FEATURE_NEON, "NEON"},
                {NPY_CPU_FEATURE_NEON_FP16, "NEON_FP16"},
                {NPY_CPU_FEATURE_NEON_VFPV4, "NEON_VFPV4"},
                {NPY_CPU_FEATURE_ASIMD, "ASIMD"},
                {NPY_CPU_FEATURE_FPHP, "FPHP"},
                {NPY_CPU_FEATURE_ASIMDHP, "ASIMDHP"},
                {NPY_CPU_FEATURE_ASIMDDP, "ASIMDDP"},
                {NPY_CPU_FEATURE_ASIMDFHM, "ASIMDFHM"},
                {NPY_CPU_FEATURE_SVE, "SVE"},
                {NPY_CPU_FEATURE_RVV, "RVV"}};

# 定义一个函数 `npy_cpu_features_dict`，返回一个新创建的空字典对象
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_features_dict(void)
{
    PyObject *dict = PyDict_New();
    # 如果传入的 dict 不为空，则执行下面的代码块
    if (dict) {
        # 遍历 features 数组，该数组的大小为 features 数组元素的个数
        for(unsigned i = 0; i < sizeof(features)/sizeof(features[0]); ++i)
            # 将 features[i] 的 string 成员作为键，根据 npy__cpu_have[features[i].feature] 的值设定相应的 Py_True 或 Py_False 作为值，并将键值对添加到 dict 中
            if (PyDict_SetItemString(dict, features[i].string,
                npy__cpu_have[features[i].feature] ? Py_True : Py_False) < 0) {
                # 如果 PyDict_SetItemString 出错，释放 dict 并返回 NULL
                Py_DECREF(dict);
                return NULL;
            }
    }
    # 返回填充了特征信息的 dict 或者空指针
    return dict;
/******************** Private Definitions *********************/

/**
 * 宏定义，用于在 PyList 对象中添加字符串项，将 FEATURE 转换为 PyUnicode 对象，
 * 若转换失败则释放 LIST，并返回 NULL
 */
#define NPY__CPU_PYLIST_APPEND_CB(FEATURE, LIST) \
    item = PyUnicode_FromString(NPY_TOSTRING(FEATURE)); \
    if (item == NULL) { \
        Py_DECREF(LIST); \
        return NULL; \
    } \
    PyList_SET_ITEM(LIST, index++, item);

/**
 * 返回包含 CPU 基线特性的 PyList 对象，
 * 若未禁用优化且 NPY_WITH_CPU_BASELINE_N 大于 0，则创建包含基线特性数目的列表
 * 否则返回一个空列表
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_baseline_list(void)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_BASELINE_N > 0
    PyObject *list = PyList_New(NPY_WITH_CPU_BASELINE_N), *item;
    int index = 0;
    if (list != NULL) {
        // 调用宏展开，将基线特性添加到列表中
        NPY_WITH_CPU_BASELINE_CALL(NPY__CPU_PYLIST_APPEND_CB, list)
    }
    return list;
#else
    return PyList_New(0);
#endif
}

/**
 * 返回包含 CPU 分发特性的 PyList 对象，
 * 若未禁用优化且 NPY_WITH_CPU_DISPATCH_N 大于 0，则创建包含分发特性数目的列表
 * 否则返回一个空列表
 */
NPY_VISIBILITY_HIDDEN PyObject *
npy_cpu_dispatch_list(void)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_DISPATCH_N > 0
    PyObject *list = PyList_New(NPY_WITH_CPU_DISPATCH_N), *item;
    int index = 0;
    if (list != NULL) {
        // 调用宏展开，将分发特性添加到列表中
        NPY_WITH_CPU_DISPATCH_CALL(NPY__CPU_PYLIST_APPEND_CB, list)
    }
    return list;
#else
    return PyList_New(0);
#endif
}

/**
 * 内联函数，返回给定 CPU 特性的 ID，
 * 如果该特性在通过 --cpu-baseline 配置的基线特性中，则返回其对应的 ID
 * 否则返回 0
 */
static inline int
npy__cpu_baseline_fid(const char *feature)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_BASELINE_N > 0
    NPY_WITH_CPU_BASELINE_CALL(NPY__CPU_FEATURE_ID_CB, feature)
#endif
    return 0;
}

/**
 * 内联函数，返回给定 CPU 特性的 ID，
 * 如果该特性在通过 --cpu-dispatch 配置的分发特性中，则返回其对应的 ID
 * 否则返回 0
 */
static inline int
npy__cpu_dispatch_fid(const char *feature)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_DISPATCH_N > 0
    NPY_WITH_CPU_DISPATCH_CALL(NPY__CPU_FEATURE_ID_CB, feature)
#endif
    return 0;
}

/**
 * 验证基线 CPU 特性的有效性，
 * 若未禁用优化且 NPY_WITH_CPU_BASELINE_N 大于 0，则检查所需的特性是否支持，
 * 若有不支持的特性，则抛出运行时异常，并返回 -1
 */
static int
npy__cpu_validate_baseline(void)
{
#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_BASELINE_N > 0
    char baseline_failure[sizeof(NPY_WITH_CPU_BASELINE) + 1];
    char *fptr = &baseline_failure[0];

    // 宏展开，检查基线特性是否都被支持
    #define NPY__CPU_VALIDATE_CB(FEATURE, DUMMY)                  \
        if (!npy__cpu_have[NPY_CAT(NPY_CPU_FEATURE_, FEATURE)]) { \
            const int size = sizeof(NPY_TOSTRING(FEATURE));       \
            memcpy(fptr, NPY_TOSTRING(FEATURE), size);            \
            fptr[size] = ' '; fptr += size + 1;                   \
        }
    NPY_WITH_CPU_BASELINE_CALL(NPY__CPU_VALIDATE_CB, DUMMY) // 针对 MSVC 额外的参数
    *fptr = '\0';

    if (baseline_failure[0] != '\0') {
        *(fptr-1) = '\0'; // 去掉最后的空格
        // 抛出运行时异常，指示不支持的 CPU 特性
        PyErr_Format(PyExc_RuntimeError,
            "NumPy was built with baseline optimizations: \n"
            "(" NPY_WITH_CPU_BASELINE ") but your machine "
            "doesn't support:\n(%s).",
            baseline_failure
        );
        return -1;

            );
        return -1;
#else
    return 0;
#endif
}
    }



    # 结束了一个代码块的定义，可能是函数、循环、条件语句或其他代码块的结尾
#endif
    return 0;
}

static int
npy__cpu_check_env(int disable, const char *env) {

    static const char *names[] = {
        "enable", "disable",
        "NPY_ENABLE_CPU_FEATURES", "NPY_DISABLE_CPU_FEATURES",
        "During parsing environment variable: 'NPY_ENABLE_CPU_FEATURES':\n",
        "During parsing environment variable: 'NPY_DISABLE_CPU_FEATURES':\n"
    };
    // 将 disable 转换为整数值 0 或 1
    disable = disable ? 1 : 0;
    // 根据 disable 的值选择相应的名字
    const char *act_name = names[disable];
    const char *env_name = names[disable + 2];
    const char *err_head = names[disable + 4];

#if !defined(NPY_DISABLE_OPTIMIZATION) && NPY_WITH_CPU_DISPATCH_N > 0
    // 定义最大环境变量长度为 1024
    #define NPY__MAX_VAR_LEN 1024 // More than enough for this era
    size_t var_len = strlen(env) + 1;
    // 检查环境变量长度是否超过最大长度
    if (var_len > NPY__MAX_VAR_LEN) {
        // 如果超过最大长度，抛出运行时错误
        PyErr_Format(PyExc_RuntimeError,
            "Length of environment variable '%s' is %zd, only %d accepted",
            env_name, var_len, NPY__MAX_VAR_LEN
        );
        return -1;
    }
    // 复制环境变量内容到 features 数组中
    char features[NPY__MAX_VAR_LEN];
    memcpy(features, env, var_len);

    // 定义两个字符串数组用于记录不存在和不支持的特性
    char nexist[NPY__MAX_VAR_LEN];
    char *nexist_cur = &nexist[0];

    char notsupp[sizeof(NPY_WITH_CPU_DISPATCH) + 1];
    char *notsupp_cur = &notsupp[0];

    // 定义分隔符字符串
    // 逗号和空格包括水平制表符、垂直制表符、回车符、换行符、换页符
    const char *delim = ", \t\v\r\n\f";
    // 使用 strtok 分割 features 字符串
    char *feature = strtok(features, delim);
    while (feature) {
        // 检查特性是否属于基线优化
        if (npy__cpu_baseline_fid(feature) > 0){
            if (disable) {
                // 如果试图禁用基线优化的特性，抛出运行时错误
                PyErr_Format(PyExc_RuntimeError,
                    "%s"
                    "You cannot disable CPU feature '%s', since it is part of "
                    "the baseline optimizations:\n"
                    "(" NPY_WITH_CPU_BASELINE ").",
                    err_head, feature
                );
                return -1;
            } 
            // 跳过这个特性继续处理下一个
            goto next;
        }
        // 检查特性是否属于已分派的特性
        int feature_id = npy__cpu_dispatch_fid(feature);
        if (feature_id == 0) {
            // 如果特性未被分派，记录到 nexist 数组中
            int flen = strlen(feature);
            memcpy(nexist_cur, feature, flen);
            nexist_cur[flen] = ' '; nexist_cur += flen + 1;
            // 跳过这个特性继续处理下一个
            goto next;
        }
        // 检查特性是否由当前机器支持
        if (!npy__cpu_have[feature_id]) {
            // 如果当前机器不支持该特性，记录到 notsupp 数组中
            int flen = strlen(feature);
            memcpy(notsupp_cur, feature, flen);
            notsupp_cur[flen] = ' '; notsupp_cur += flen + 1;
            // 跳过这个特性继续处理下一个
            goto next;
        }
        // 最后根据 disable 设置特性的状态为禁用或启用
        npy__cpu_have[feature_id] = disable ? 0 : 2;
    next:
        // 继续处理下一个特性
        feature = strtok(NULL, delim);
    }
    if (!disable){
        // 禁用所有未标记的已分派特性
        #define NPY__CPU_DISABLE_DISPATCH_CB(FEATURE, DUMMY) \
            if(npy__cpu_have[NPY_CAT(NPY_CPU_FEATURE_, FEATURE)] != 0)\
            {npy__cpu_have[NPY_CAT(NPY_CPU_FEATURE_, FEATURE)]--;}\

        // 调用宏 NPY_WITH_CPU_DISPATCH_CALL 来禁用未标记的分派特性
        NPY_WITH_CPU_DISPATCH_CALL(NPY__CPU_DISABLE_DISPATCH_CB, DUMMY) // extra arg for msvc
    }

    // 结束 nexist 数组以字符串形式
    *nexist_cur = '\0';
    # 如果 nexist 的第一个字符不是空字符
    if (nexist[0] != '\0') {
        *(nexist_cur-1) = '\0'; // 去除末尾的空格
        // 发出警告信息，指明无法使用某些 CPU 特性，因为它们不是分发优化的一部分
        if (PyErr_WarnFormat(PyExc_ImportWarning, 1,
            "%sYou cannot %s CPU features (%s), since "
            "they are not part of the dispatched optimizations\n"
            "(" NPY_WITH_CPU_DISPATCH ").",
            err_head, act_name, nexist
        ) < 0) {
            return -1; // 如果警告发生错误，返回 -1
        }
    }

    // 定义一个消息格式，指明某些 CPU 特性不受支持
    #define NOTSUPP_BODY \
                "%s" \
                "You cannot %s CPU features (%s), since " \
                "they are not supported by your machine.", \
                err_head, act_name, notsupp

    *notsupp_cur = '\0';
    // 如果 notsupp 的第一个字符不是空字符
    if (notsupp[0] != '\0') {
        *(notsupp_cur-1) = '\0'; // 去除末尾的空格
        // 如果禁用标志为假（即不禁用），则引发运行时错误，指明某些 CPU 特性不受支持
        if (!disable){
            PyErr_Format(PyExc_RuntimeError, NOTSUPP_BODY);
            return -1; // 返回 -1 表示出错
        }
    }
#else
    // 如果未定义特定条件，发出警告并返回错误码
    if (PyErr_WarnFormat(PyExc_ImportWarning, 1,
            "%s"
            "You cannot use environment variable '%s', since "
        #ifdef NPY_DISABLE_OPTIMIZATION
            "the NumPy library was compiled with optimization disabled.",
        #else
            "the NumPy library was compiled without any dispatched optimizations.",
        #endif
        err_head, env_name, act_name
    ) < 0) {
        return -1;
    }
#endif
    // 返回成功状态
    return 0;
}

/****************************************************************
 * This section is reserved to defining @npy__cpu_init_features
 * for each CPU architecture, please try to keep it clean. Ty
 ****************************************************************/

/***************** X86 ******************/

#if defined(NPY_CPU_AMD64) || defined(NPY_CPU_X86)

#ifdef _MSC_VER
    #include <intrin.h>
#elif defined(__INTEL_COMPILER)
    #include <immintrin.h>
#endif

static int
npy__cpu_getxcr0(void)
{
#if defined(_MSC_VER) || defined (__INTEL_COMPILER)
    // 调用平台特定的 _xgetbv 函数获取 XCR0 寄存器的值
    return _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    /* named form of xgetbv not supported on OSX, so must use byte form, see:
     * https://github.com/asmjit/asmjit/issues/78
    */
    unsigned int eax, edx;
    // 使用汇编指令直接获取 XCR0 寄存器的值
    __asm(".byte 0x0F, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return eax;
#else
    // 默认情况下返回 0
    return 0;
#endif
}

static void
npy__cpu_cpuid(int reg[4], int func_id)
{
#if defined(_MSC_VER)
    // Microsoft 编译器下使用 __cpuidex 函数获取 CPUID 信息
    __cpuidex(reg, func_id, 0);
#elif defined(__INTEL_COMPILER)
    // Intel 编译器下使用 __cpuid 函数获取 CPUID 信息
    __cpuid(reg, func_id);
#elif defined(__GNUC__) || defined(__clang__)
    #if defined(NPY_CPU_X86) && defined(__PIC__)
        // 在 PIC 模式下，使用 xchg 指令保存和恢复 %ebx 寄存器，并调用 cpuid 指令获取 CPUID 信息
        __asm__("xchg{l}\t{%%}ebx, %1\n\t"
                "cpuid\n\t"
                "xchg{l}\t{%%}ebx, %1\n\t"
                : "=a" (reg[0]), "=r" (reg[1]), "=c" (reg[2]),
                  "=d" (reg[3])
                : "a" (func_id), "c" (0)
        );
    #else
        // 直接调用 cpuid 指令获取 CPUID 信息
        __asm__("cpuid\n\t"
                : "=a" (reg[0]), "=b" (reg[1]), "=c" (reg[2]),
                  "=d" (reg[3])
                : "a" (func_id), "c" (0)
        );
    #endif
#else
    // 默认情况下将寄存器数组清零
    reg[0] = 0;
#endif
}

static void
npy__cpu_init_features(void)
{
    // 将 CPU 特性标记数组清零
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);

    // 获取 CPUID 信息，判断平台支持情况
    int reg[] = {0, 0, 0, 0};
    npy__cpu_cpuid(reg, 0);
    if (reg[0] == 0) {
       // 对于不支持 CPUID 的平台，假设基本的 MMX、SSE、SSE2 特性支持
       npy__cpu_have[NPY_CPU_FEATURE_MMX]  = 1;
       npy__cpu_have[NPY_CPU_FEATURE_SSE]  = 1;
       npy__cpu_have[NPY_CPU_FEATURE_SSE2] = 1;
       #ifdef NPY_CPU_AMD64
           npy__cpu_have[NPY_CPU_FEATURE_SSE3] = 1;
       #endif
       return;
    }

    // 查询并记录支持的 CPU 特性
    npy__cpu_cpuid(reg, 1);
    npy__cpu_have[NPY_CPU_FEATURE_MMX]    = (reg[3] & (1 << 23)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE]    = (reg[3] & (1 << 25)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE2]   = (reg[3] & (1 << 26)) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_SSE3]   = (reg[2] & (1 << 0))  != 0;
    // 检查CPU是否支持SSSE3指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_SSSE3]  = (reg[2] & (1 << 9))  != 0;
    // 检查CPU是否支持SSE4.1指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_SSE41]  = (reg[2] & (1 << 19)) != 0;
    // 检查CPU是否支持POPCNT指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_POPCNT] = (reg[2] & (1 << 23)) != 0;
    // 检查CPU是否支持SSE4.2指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_SSE42]  = (reg[2] & (1 << 20)) != 0;
    // 检查CPU是否支持F16C指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_F16C]   = (reg[2] & (1 << 29)) != 0;

    // 检查OSXSAVE位是否为0，如果是则返回，要求支持XSAVE指令集
    if ((reg[2] & (1 << 27)) == 0)
        return;
    // 获取XCR0寄存器的值，判断是否支持AVX指令集
    int xcr = npy__cpu_getxcr0();
    if ((xcr & 6) != 6)
        return;
    // 检查CPU是否支持AVX指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_AVX]    = (reg[2] & (1 << 28)) != 0;
    // 如果CPU不支持AVX指令集则返回
    if (!npy__cpu_have[NPY_CPU_FEATURE_AVX])
        return;
    // 检查CPU是否支持FMA3指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_FMA3]   = (reg[2] & (1 << 12)) != 0;

    // 第二次调用cpuid以获取扩展的AMD特性位
    npy__cpu_cpuid(reg, 0x80000001);
    // 检查CPU是否支持XOP指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_XOP]    = (reg[2] & (1 << 11)) != 0;
    // 检查CPU是否支持FMA4指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_FMA4]   = (reg[2] & (1 << 16)) != 0;

    // 第三次调用cpuid以获取扩展的AVX2和AVX512特性位
    npy__cpu_cpuid(reg, 7);
    // 检查CPU是否支持AVX2指令集，设置相应的标志位
    npy__cpu_have[NPY_CPU_FEATURE_AVX2]   = (reg[1] & (1 << 5))  != 0;
    // 如果CPU不支持AVX2指令集则返回
    if (!npy__cpu_have[NPY_CPU_FEATURE_AVX2])
        return;
    // 检查AVX512 OS支持，设置相应的标志位
    int avx512_os = (xcr & 0xe6) == 0xe6;
#if defined(__APPLE__) && defined(__x86_64__)
/**
 * 在 darwin 上，支持 AVX512 的机器默认情况下，线程被创建时 AVX512 被 XCR0 掩码屏蔽，
 * 并且使用 AVX 大小的保存区域。但是，AVX512 的能力通过 commpage 和 sysctl 公布。
 * 更多信息请参考：
 *  - https://github.com/apple/darwin-xnu/blob/0a798f6738bc1db01281fc08ae024145e84df927/osfmk/i386/fpu.c#L175-L201
 *  - https://github.com/golang/go/issues/43089
 *  - https://github.com/numpy/numpy/issues/19319
 */
if (!avx512_os) {
    npy_uintp commpage64_addr = 0x00007fffffe00000ULL;
    npy_uint16 commpage64_ver = *((npy_uint16*)(commpage64_addr + 0x01E));
    // 在版本大于 12 的情况下，读取 commpage64 的能力位
    if (commpage64_ver > 12) {
        npy_uint64 commpage64_cap = *((npy_uint64*)(commpage64_addr + 0x010));
        avx512_os = (commpage64_cap & 0x0000004000000000ULL) != 0;
    }
}
#endif

if (!avx512_os) {
    return; // 如果没有检测到 AVX512 支持，则直接返回
}

npy__cpu_have[NPY_CPU_FEATURE_AVX512F]  = (reg[1] & (1 << 16)) != 0;
npy__cpu_have[NPY_CPU_FEATURE_AVX512CD] = (reg[1] & (1 << 28)) != 0;
}

/***************** POWER ******************/

#elif defined(NPY_CPU_PPC64) || defined(NPY_CPU_PPC64LE)

#if defined(__linux__) || defined(__FreeBSD__)
#ifdef __FreeBSD__
    #include <machine/cpu.h> // 定义 PPC_FEATURE_HAS_VSX
#endif
#include <sys/auxv.h>
#ifndef AT_HWCAP2
    #define AT_HWCAP2 26
#endif
#ifndef PPC_FEATURE2_ARCH_2_07
    #define PPC_FEATURE2_ARCH_2_07 0x80000000
#endif
#ifndef PPC_FEATURE2_ARCH_3_00
    #define PPC_FEATURE2_ARCH_3_00 0x00800000
#endif
#ifndef PPC_FEATURE2_ARCH_3_1
    #define PPC_FEATURE2_ARCH_3_1  0x00040000
#endif
#endif

static void
npy__cpu_init_features(void)
{
    // 将 npy__cpu_have 数组初始化为 0
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);

#if defined(__linux__) || defined(__FreeBSD__)
#ifdef __linux__
    unsigned int hwcap = getauxval(AT_HWCAP);
    // 如果硬件没有 VSX 功能，则直接返回
    if ((hwcap & PPC_FEATURE_HAS_VSX) == 0)
        return;

    hwcap = getauxval(AT_HWCAP2);
#else
    unsigned long hwcap;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    // 如果硬件没有 VSX 功能，则直接返回
    if ((hwcap & PPC_FEATURE_HAS_VSX) == 0)
        return;

    elf_aux_info(AT_HWCAP2, &hwcap, sizeof(hwcap));
#endif // __linux__

    // 如果硬件支持 PPC 3.1 架构，则设置相应的 VSX 功能位
    if (hwcap & PPC_FEATURE2_ARCH_3_1)
    {
        npy__cpu_have[NPY_CPU_FEATURE_VSX]  =
        npy__cpu_have[NPY_CPU_FEATURE_VSX2] =
        npy__cpu_have[NPY_CPU_FEATURE_VSX3] =
        npy__cpu_have[NPY_CPU_FEATURE_VSX4] = 1;
        return;
    }

    // 设置基本的 VSX 功能位
    npy__cpu_have[NPY_CPU_FEATURE_VSX]  = 1;
    npy__cpu_have[NPY_CPU_FEATURE_VSX2] = (hwcap & PPC_FEATURE2_ARCH_2_07) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_VSX3] = (hwcap & PPC_FEATURE2_ARCH_3_00) != 0;
    npy__cpu_have[NPY_CPU_FEATURE_VSX4] = (hwcap & PPC_FEATURE2_ARCH_3_1) != 0;
// TODO: AIX, OpenBSD
#else
    // 如果不是在 Linux 或 FreeBSD 系统上，仅设置基本的 VSX 功能位
    npy__cpu_have[NPY_CPU_FEATURE_VSX]  = 1;
    #if defined(NPY_CPU_PPC64LE) || defined(NPY_HAVE_VSX2)
    #ifdef 指令检查是否定义了 NPY_CPU_PPC64LE 或 NPY_HAVE_VSX2 宏
    npy__cpu_have[NPY_CPU_FEATURE_VSX2] = 1;
    #endif
    #ifdef 指令检查是否定义了 NPY_HAVE_VSX3 宏
    npy__cpu_have[NPY_CPU_FEATURE_VSX3] = 1;
    #endif
    #ifdef 指令检查是否定义了 NPY_HAVE_VSX4 宏
    npy__cpu_have[NPY_CPU_FEATURE_VSX4] = 1;
    #endif
#endif
}

/***************** ZARCH ******************/

#elif defined(__s390x__)

#include <sys/auxv.h>
#ifndef HWCAP_S390_VXE
    #define HWCAP_S390_VXE 8192
#endif

#ifndef HWCAP_S390_VXRS_EXT2
    #define HWCAP_S390_VXRS_EXT2 32768
#endif

// 定义静态函数，初始化 CPU 特性检测数组
static void
npy__cpu_init_features(void)
{
    // 将 npy__cpu_have 数组初始化为零
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
    
    // 获取当前进程的硬件特性信息
    unsigned int hwcap = getauxval(AT_HWCAP);
    // 如果未检测到 S390 Vector Extension，则直接返回
    if ((hwcap & HWCAP_S390_VX) == 0) {
        return;
    }

    // 如果支持 S390 Vector Extension 2，则设置相关特性标志位
    if (hwcap & HWCAP_S390_VXRS_EXT2) {
       npy__cpu_have[NPY_CPU_FEATURE_VX]  =
       npy__cpu_have[NPY_CPU_FEATURE_VXE] =
       npy__cpu_have[NPY_CPU_FEATURE_VXE2] = 1;
       return;
    }
    
    // 否则，仅设置 VX 和 VXE 的特性标志位
    npy__cpu_have[NPY_CPU_FEATURE_VXE] = (hwcap & HWCAP_S390_VXE) != 0;

    npy__cpu_have[NPY_CPU_FEATURE_VX]  = 1;
}


/***************** ARM ******************/

#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)

// 定义内联函数，初始化 ARMv8 的 CPU 特性检测数组
static inline void
npy__cpu_init_features_arm8(void)
{
    // 设置 NEON 和 ASIMD 相关特性的标志位
    npy__cpu_have[NPY_CPU_FEATURE_NEON]       =
    npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16]  =
    npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] =
    npy__cpu_have[NPY_CPU_FEATURE_ASIMD]      = 1;
}

#if defined(__linux__) || defined(__FreeBSD__)
/*
 * we aren't sure of what kind kernel or clib we deal with
 * so we play it safe
*/
#include <stdio.h>
#include "npy_cpuinfo_parser.h"

#if defined(__linux__)
// 声明 getauxval 函数的弱符号，用于动态链接
__attribute__((weak)) unsigned long getauxval(unsigned long); // linker should handle it
#endif
#ifdef __FreeBSD__
// 声明 elf_aux_info 函数的弱符号，用于动态链接
__attribute__((weak)) int elf_aux_info(int, void *, int); // linker should handle it

// 定义 getauxval 函数的替代版本，用于 FreeBSD 平台
static unsigned long getauxval(unsigned long k)
{
    unsigned long val = 0ul;
    // 如果 elf_aux_info 未定义或调用失败，则返回默认值 0
    if (elf_aux_info == 0 || elf_aux_info((int)k, (void *)&val, (int)sizeof(val)) != 0) {
        return 0ul;
    }
    return val;
}
#endif

// 定义函数，用于初始化 Linux 平台下的 CPU 特性检测
static int
npy__cpu_init_features_linux(void)
{
    unsigned long hwcap = 0, hwcap2 = 0;
    #ifdef __linux__
    // 如果 getauxval 函数存在，则使用其获取硬件特性信息
    if (getauxval != 0) {
        hwcap = getauxval(NPY__HWCAP);
    #ifdef __arm__
        hwcap2 = getauxval(NPY__HWCAP2);
    #endif
    } else {
        // 否则，打开 /proc/self/auxv 文件逐行读取获取硬件特性信息
        unsigned long auxv[2];
        int fd = open("/proc/self/auxv", O_RDONLY);
        if (fd >= 0) {
            while (read(fd, &auxv, sizeof(auxv)) == sizeof(auxv)) {
                if (auxv[0] == NPY__HWCAP) {
                    hwcap = auxv[1];
                }
            #ifdef __arm__
                else if (auxv[0] == NPY__HWCAP2) {
                    hwcap2 = auxv[1];
                }
            #endif
                // 检测到末尾标志，退出循环
                else if (auxv[0] == 0 && auxv[1] == 0) {
                    break;
                }
            }
            close(fd);
        }
    }
    #else
    // 对于非 Linux 平台，直接使用 getauxval 获取硬件特性信息
    hwcap = getauxval(NPY__HWCAP);
    #ifdef __arm__
    hwcap2 = getauxval(NPY__HWCAP2);
    #endif
    #endif
    // 如果未获取到有效的硬件特性信息，则返回失败
    if (hwcap == 0 && hwcap2 == 0) {
    #ifdef __linux__
        /*
         * 如果在 Linux 平台下编译：
         * 尝试使用 /proc/cpuinfo 解析硬件特性，用于沙盒环境
         * 如果失败，则使用编译器定义的默认值
         */
        if (!get_feature_from_proc_cpuinfo(&hwcap, &hwcap2)) {
            // 如果解析失败，返回 0
            return 0;
        }
    #else
        // 如果不在 Linux 平台下编译，直接返回 0
        return 0;
    #endif
#ifdef __arm__
    // 如果编译目标是 ARM 架构

    // 检测是否为 Arm8 (aarch32 状态)，通过检查硬件特性标志位 hwcap2 来判断是否支持 AES、SHA1、SHA2、PMULL 和 CRC32
    if ((hwcap2 & NPY__HWCAP2_AES)  || (hwcap2 & NPY__HWCAP2_SHA1)  ||
        (hwcap2 & NPY__HWCAP2_SHA2) || (hwcap2 & NPY__HWCAP2_PMULL) ||
        (hwcap2 & NPY__HWCAP2_CRC32))
    {
        hwcap = hwcap2;
#else
    // 如果编译目标不是 ARM 架构

    // 始终进入此分支，用于非 ARM 架构的情况
    if (1)
    {
        // 如果硬件特性标志位 hwcap 不包含 NPY__HWCAP_FP 或 NPY__HWCAP_ASIMD，则返回 1
        if (!(hwcap & (NPY__HWCAP_FP | NPY__HWCAP_ASIMD))) {
            // 这种情况可能发生吗？也许被内核禁用了
            // 顺便说一句，这会破坏 AARCH64 的基线
            return 1;
        }
#endif
        // 根据硬件特性设置相应的标志位
        npy__cpu_have[NPY_CPU_FEATURE_FPHP]       = (hwcap & NPY__HWCAP_FPHP)     != 0;
        npy__cpu_have[NPY_CPU_FEATURE_ASIMDHP]    = (hwcap & NPY__HWCAP_ASIMDHP)  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_ASIMDDP]    = (hwcap & NPY__HWCAP_ASIMDDP)  != 0;
        npy__cpu_have[NPY_CPU_FEATURE_ASIMDFHM]   = (hwcap & NPY__HWCAP_ASIMDFHM) != 0;
        npy__cpu_have[NPY_CPU_FEATURE_SVE]        = (hwcap & NPY__HWCAP_SVE)      != 0;
        // 初始化 ARM8 架构的 CPU 特性
        npy__cpu_init_features_arm8();
    } else {
        // 如果有 NEON 指令集支持，设置 NEON 相关的特性标志位
        npy__cpu_have[NPY_CPU_FEATURE_NEON]       = (hwcap & NPY__HWCAP_NEON)   != 0;
        if (npy__cpu_have[NPY_CPU_FEATURE_NEON]) {
            // 如果 NEON 可用，则设置 NEON_FP16 和 NEON_VFPV4 的标志位
            npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16]  = (hwcap & NPY__HWCAP_HALF) != 0;
            npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] = (hwcap & NPY__HWCAP_VFPv4) != 0;
        }
    }
    // 返回 1，表示初始化成功
    return 1;
}
#endif

static void
npy__cpu_init_features(void)
{
    // 初始化 npy__cpu_have 数组，全部置为 0
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
#ifdef __linux__
    // 如果是在 Linux 平台，调用相应的初始化函数并返回
    if (npy__cpu_init_features_linux())
        return;
#endif
    // 如果是在其他平台，没有其他需要执行的任务
    // 之后的代码块处理 ARM64 或特定硬件特性的初始化
#if defined(NPY_HAVE_ASIMD) || defined(__aarch64__) || (defined(__ARM_ARCH) && __ARM_ARCH >= 8) || defined(_M_ARM64)
    #if defined(NPY_HAVE_FPHP) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // 如果支持 FPHP，设置相应标志位
    npy__cpu_have[NPY_CPU_FEATURE_FPHP] = 1;
    #endif
    #if defined(NPY_HAVE_ASIMDHP) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // 如果支持 ASIMDHP，设置相应标志位
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDHP] = 1;
    #endif
    #if defined(NPY_HAVE_ASIMDDP) || defined(__ARM_FEATURE_DOTPROD)
    // 如果支持 ASIMDDP，设置相应标志位
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDDP] = 1;
    #endif
    #if defined(NPY_HAVE_ASIMDFHM) || defined(__ARM_FEATURE_FP16FML)
    // 如果支持 ASIMDFHM，设置相应标志位
    npy__cpu_have[NPY_CPU_FEATURE_ASIMDFHM] = 1;
    #endif
    #if defined(NPY_HAVE_SVE) || defined(__ARM_FEATURE_SVE)
    // 如果支持 SVE，设置相应标志位
    npy__cpu_have[NPY_CPU_FEATURE_SVE] = 1;
    #endif
    // 初始化 ARM8 架构的 CPU 特性
    npy__cpu_init_features_arm8();
#else
    #if defined(NPY_HAVE_NEON) || defined(__ARM_NEON__)
        // 如果支持 NEON，设置 NEON 标志位
        npy__cpu_have[NPY_CPU_FEATURE_NEON] = 1;
    #endif
    #if defined(NPY_HAVE_NEON_FP16) || defined(__ARM_FP16_FORMAT_IEEE) || (defined(__ARM_FP) && (__ARM_FP & 2))
        // 如果支持 NEON_FP16，根据 NEON 的可用性设置 NEON_FP16 标志位
        npy__cpu_have[NPY_CPU_FEATURE_NEON_FP16] = npy__cpu_have[NPY_CPU_FEATURE_NEON];
    #endif
    #if defined(NPY_HAVE_NEON_VFPV4) || defined(__ARM_FEATURE_FMA)
        // 如果支持 NEON_VFPV4，根据 NEON 的可用性设置 NEON_VFPV4 标志位
        npy__cpu_have[NPY_CPU_FEATURE_NEON_VFPV4] = npy__cpu_have[NPY_CPU_FEATURE_NEON];
    #endif
#endif
}
#ifdef HWCAP_RVV
// 如果 HWCAP_RVV 已定义，则直接使用系统定义的硬件特性
#include <sys/auxv.h>

#ifndef HWCAP_RVV
    // 如果未定义 HWCAP_RVV，则定义 COMPAT_HWCAP_ISA_V 为 'V' ISA 的位掩码
    // 参考：https://github.com/torvalds/linux/blob/v6.8/arch/riscv/include/uapi/asm/hwcap.h#L24
    #define COMPAT_HWCAP_ISA_V    (1 << ('V' - 'A'))
#endif

static void
npy__cpu_init_features(void)
{
    // 清空 npy__cpu_have 数组，准备记录 CPU 特性
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);

    // 从系统获取硬件特性值
    unsigned int hwcap = getauxval(AT_HWCAP);
    // 检查是否支持 'V' ISA，如果是则设置 RVV 特性
    if (hwcap & COMPAT_HWCAP_ISA_V) {
        npy__cpu_have[NPY_CPU_FEATURE_RVV]  = 1;
    }
}

/*********** Unsupported ARCH ***********/
#else
static void
npy__cpu_init_features(void)
{
    /*
     * 如果不支持当前架构，则清空 npy__cpu_have 数组以禁用所有 CPU 特性
     * 这是为了确保在多次调用 npy__cpu_init_features 时，已禁用的特性不会受到影响
     * 通过环境变量或其他方法禁用的特性，在此处被清除
     * 可能在未来支持其他方法，如全局变量，详细了解请回到 npy__cpu_try_disable_env
     */
    memset(npy__cpu_have, 0, sizeof(npy__cpu_have[0]) * NPY_CPU_FEATURE_MAX);
}
#endif
```