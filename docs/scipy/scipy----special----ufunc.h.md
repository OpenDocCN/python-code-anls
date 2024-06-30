# `D:\src\scipysrc\scipy\scipy\special\ufunc.h`

```
// 防止头文件重复包含，只有在第一次包含时才会编译内容
#pragma once

// 定义宏以清除 PY_SSIZE_T 的旧定义，以便与最新版本的 Python 兼容
#define PY_SSIZE_T_CLEAN
// 包含 Python.h 头文件，用于与 Python 解释器交互
#include <Python.h>

// 包含 C 语言标准库的断言功能
#include <cassert>
// 包含 C 语言标准库的字符串操作功能
#include <cstring>
// 包含 C++ 标准库中的内存管理、类型特性等功能
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

// 包含 NumPy 数组对象的头文件
#include <numpy/arrayobject.h>
// NumPy 3k 兼容性支持头文件
#include <numpy/npy_3kcompat.h>
// NumPy 通用函数对象头文件
#include <numpy/ufuncobject.h>

// 包含自定义错误处理头文件 sf_error.h
#include "sf_error.h"
// 包含特殊第三方库 Kokkos 的多维跨度数组头文件
#include "special/third_party/kokkos/mdspan.hpp"


// 这是 std::accumulate 的 constexpr 版本，但在 C++20 之前不支持 constexpr
template <typename InputIt, typename T>
constexpr T initializer_accumulate(InputIt first, InputIt last, T init) {
    // 使用迭代器遍历范围 [first, last)，累加初始化值和元素值
    for (InputIt it = first; it != last; ++it) {
        init = std::move(init) + *it;
    }
    // 返回累加后的结果
    return init;
}

// 推断可调用对象 F 的参数个数
template <typename Func>
struct arity_of;

template <typename Res, typename... Args>
struct arity_of<Res (*)(Args...)> {
    static constexpr size_t value = sizeof...(Args);
};

// 获取可调用对象 F 的参数个数作为常量表达式
template <typename Func>
constexpr size_t arity_of_v = arity_of<Func>::value;

// 检查可调用对象是否有返回值
template <typename Func>
struct has_return;

template <typename Res, typename... Args>
struct has_return<Res (*)(Args...)> {
    static constexpr bool value = true;
};

template <typename... Args>
struct has_return<void (*)(Args...)> {
    static constexpr bool value = false;
};

// 获取可调用对象是否有返回值的常量表达式
template <typename Func>
constexpr size_t has_return_v = has_return<Func>::value;

// 获取类型 T 的维度
template <typename T>
struct rank_of {
    static constexpr size_t value = 0;
};

// 获取 std::mdspan 类型的维度
template <typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
struct rank_of<std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>> {
    static constexpr size_t value = Extents::rank();
};

// 获取类型 T 的维度作为常量表达式
template <typename T>
inline constexpr size_t rank_of_v = rank_of<T>::value;

// 将 C++ 类型映射到 NumPy 类型
template <typename T>
struct npy_type;

template <>
struct npy_type<bool> {
    using type = npy_bool;
};

template <>
struct npy_type<char> {
    using type = npy_byte;
};

template <>
struct npy_type<short> {
    using type = npy_short;
};

template <>
struct npy_type<int> {
    using type = npy_int;
};

template <>
struct npy_type<long> {
    using type = npy_long;
};

template <>
struct npy_type<long long> {
    using type = npy_longlong;
};

template <>
struct npy_type<unsigned char> {
    using type = npy_ubyte;
};

template <>
struct npy_type<unsigned short> {
    using type = npy_ushort;
};

template <>
struct npy_type<unsigned int> {
    using type = npy_uint;
};

template <>
struct npy_type<unsigned long> {
    using type = npy_ulong;
};

template <>
struct npy_type<unsigned long long> {
    using type = npy_ulonglong;
};

template <>
struct npy_type<float> {
    using type = npy_float;
};

template <>
struct npy_type<double> {
    using type = npy_double;
};

template <>
struct npy_type<long double> {
    using type = npy_longdouble;
};

template <>
struct npy_type<std::complex<float>> {
    using type = npy_cfloat;
};

template <>
struct npy_type<std::complex<double>> {
    using type = npy_cdouble;
};

// 以下部分代码未完整，需要继续注释，根据实际代码补充
// 定义一个结构模板，将 std::complex<long double> 映射为 npy_clongdouble
struct npy_type<std::complex<long double>> {
    using type = npy_clongdouble;
};

// 使用模板别名将类型 T 映射为其对应的 NumPy 类型
template <typename T>
using npy_type_t = typename npy_type<T>::type;

// 将 C++ 类型 T 映射为 NumPy 类型编号的结构模板
template <typename T>
struct npy_typenum {
    // 使用 npy_type_t<T> 获取类型别名，并递归地获取其对应的 NumPy 类型编号
    static constexpr NPY_TYPES value = npy_typenum<npy_type_t<T>>::value;
};

// 特化模板以处理 bool 类型，因为 npy_bool 被定义为 npy_ubyte
template <>
struct npy_typenum<bool> {
    static constexpr NPY_TYPES value = NPY_BOOL;
};

// 各种整数类型的特化模板，将 C++ 类型映射为其对应的 NumPy 类型编号
template <>
struct npy_typenum<npy_byte> {
    static constexpr NPY_TYPES value = NPY_BYTE;
};

template <>
struct npy_typenum<npy_short> {
    static constexpr NPY_TYPES value = NPY_SHORT;
};

template <>
struct npy_typenum<npy_int> {
    static constexpr NPY_TYPES value = NPY_INT;
};

template <>
struct npy_typenum<npy_long> {
    static constexpr NPY_TYPES value = NPY_LONG;
};

template <>
struct npy_typenum<npy_longlong> {
    static constexpr NPY_TYPES value = NPY_LONGLONG;
};

template <>
struct npy_typenum<npy_ubyte> {
    static constexpr NPY_TYPES value = NPY_UBYTE;
};

template <>
struct npy_typenum<npy_ushort> {
    static constexpr NPY_TYPES value = NPY_USHORT;
};

template <>
struct npy_typenum<npy_uint> {
    static constexpr NPY_TYPES value = NPY_UINT;
};

template <>
struct npy_typenum<npy_ulong> {
    static constexpr NPY_TYPES value = NPY_ULONG;
};

template <>
struct npy_typenum<npy_ulonglong> {
    static constexpr NPY_TYPES value = NPY_ULONGLONG;
};

template <>
struct npy_typenum<npy_float> {
    static constexpr NPY_TYPES value = NPY_FLOAT;
};

template <>
struct npy_typenum<npy_double> {
    static constexpr NPY_TYPES value = NPY_DOUBLE;
};

// 当 NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE 时，特化处理 npy_longdouble 类型
// 参见 https://github.com/numpy/numpy/blob/main/numpy/_core/include/numpy/npy_common.h
#if (NPY_SIZEOF_LONGDOUBLE != NPY_SIZEOF_DOUBLE)
template <>
struct npy_typenum<npy_longdouble> {
    static constexpr NPY_TYPES value = NPY_LONGDOUBLE;
};
#endif

// 复数类型的特化模板，映射为对应的 NumPy 类型编号
template <>
struct npy_typenum<npy_cfloat> {
    static constexpr NPY_TYPES value = NPY_CFLOAT;
};

template <>
struct npy_typenum<npy_cdouble> {
    static constexpr NPY_TYPES value = NPY_CDOUBLE;
};

template <>
struct npy_typenum<npy_clongdouble> {
    static constexpr NPY_TYPES value = NPY_CLONGDOUBLE;
};

// 指针类型的模板特化，映射为其指向类型的 NumPy 类型编号
template <typename T>
struct npy_typenum<T *> {
    static constexpr NPY_TYPES value = npy_typenum<T>::value;
};

// 引用类型的模板特化，映射为其引用对象类型的 NumPy 类型编号
template <typename T>
struct npy_typenum<T &> {
    static constexpr NPY_TYPES value = npy_typenum<T>::value;
};

// mdspan 类型的模板特化，映射为其模板参数类型 T 的 NumPy 类型编号
template <typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
struct npy_typenum<std::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>> {
    static constexpr NPY_TYPES value = npy_typenum<T>::value;
};

// 内联函数模板，返回类型 T 的 NumPy 类型编号
template <typename T>
inline constexpr NPY_TYPES npy_typenum_v = npy_typenum<T>::value;

// npy_traits 结构模板的起始部分，未完整给出
template <typename T>
struct npy_traits {
    // 返回从字符指针 `src` 解释为类型 `T` 的数据
    static T get(char *src, const npy_intp *dimensions, const npy_intp *steps) { return *reinterpret_cast<T *>(src); }
    
    // 将类型 `T` 的数据 `src` 设置到目标字符指针 `dst` 所指向的位置
    static void set(char *dst, const T &src) { *reinterpret_cast<npy_type_t<T> *>(dst) = src; }
};

// 模板特化：处理 std::complex<T> 类型的数据转换
template <typename T>
struct npy_traits<std::complex<T>> {
    // 从字符指针 src 中获取 std::complex<T> 类型的数据
    static std::complex<T> get(char *src, const npy_intp *dimensions, const npy_intp *steps) {
        return *reinterpret_cast<std::complex<T> *>(src);
    }

    // 将 std::complex<T> 类型的数据 src 设置到字符指针 dst 中
    static void set(char *dst, const std::complex<T> &src) {
        *reinterpret_cast<npy_type_t<T> *>(dst) = std::real(src);
        *reinterpret_cast<npy_type_t<T> *>(dst + sizeof(T)) = std::imag(src);
    }
};

// 模板特化：处理 T* 类型的数据转换
template <typename T>
struct npy_traits<T *> {
    // 从字符指针 src 中获取 T* 类型的数据
    static T *get(char *src, const npy_intp *dimensions, const npy_intp *steps) {
        static_assert(sizeof(T) == sizeof(npy_type_t<T>), "NumPy type has different size than argument type");

        return reinterpret_cast<T *>(src);
    }
};

// 模板特化：处理 T& 类型的数据转换
template <typename T>
struct npy_traits<T &> {
    // 从字符指针 src 中获取 T& 类型的数据
    static T &get(char *src, const npy_intp *dimensions, const npy_intp *steps) {
        static_assert(sizeof(T) == sizeof(npy_type_t<T>), "NumPy type has different size than argument type");

        return *reinterpret_cast<T *>(src);
    }
};

// 模板特化：处理 std::mdspan<T, Extents, std::layout_stride, AccessorPolicy> 类型的数据转换
template <typename T, typename Extents, typename AccessorPolicy>
struct npy_traits<std::mdspan<T, Extents, std::layout_stride, AccessorPolicy>> {
    // 从字符指针 src 中获取 std::mdspan<T, Extents, std::layout_stride, AccessorPolicy> 类型的数据
    static std::mdspan<T, Extents, std::layout_stride, AccessorPolicy>
    get(char *src, const npy_intp *dimensions, const npy_intp *steps) {
        static_assert(sizeof(T) == sizeof(npy_type_t<T>), "NumPy type has different size than argument type");

        // 计算步长数组 strides
        std::array<ptrdiff_t, Extents::rank()> strides;
        for (npy_uintp i = 0; i < strides.size(); ++i) {
            strides[i] = steps[i] / sizeof(T);
        }

        // 构造 extents 数组 exts
        std::array<ptrdiff_t, Extents::rank()> exts;
        for (npy_uintp i = 0; i < exts.size(); ++i) {
            exts[i] = dimensions[i];
        }

        // 返回 std::mdspan 对象，包括指向数据的指针和 extents、strides
        return {reinterpret_cast<T *>(src), {exts, strides}};
    }
};

// 结构体 base_ufunc_data，存储函数名指针
struct base_ufunc_data {
    const char *name;
};

// 模板结构体 ufunc_data，继承 base_ufunc_data，存储函数指针
template <typename Func>
struct ufunc_data : base_ufunc_data {
    Func func;
};

// 模板结构体 ufunc_traits，根据函数指针类型获取函数参数、返回值的 NumPy 类型信息
template <typename Func, typename Indices = std::make_index_sequence<arity_of_v<Func>>>
struct ufunc_traits;

// 特化模板结构体 ufunc_traits，处理函数指针为 Res (*)(Args...) 类型的情况
template <typename Res, typename... Args, size_t... I>
struct ufunc_traits<Res (*)(Args...), std::index_sequence<I...>> {
    // 定义类型数组 types，存储函数参数及返回值的 NumPy 类型编码
    static constexpr char types[sizeof...(Args) + 1] = {npy_typenum_v<Args>..., npy_typenum_v<Res>};

    // 定义数组 ranks，存储函数参数及返回值的 NumPy 维度信息
    static constexpr size_t ranks[sizeof...(Args) + 1] = {rank_of_v<Args>..., rank_of_v<Res>};

    // 定义数组 steps_offsets，存储函数参数及返回值的步长偏移量
    static constexpr size_t steps_offsets[sizeof...(Args) + 1] = {
        initializer_accumulate(ranks, ranks + I, sizeof...(Args) + 1)...,
        initializer_accumulate(ranks, ranks + sizeof...(Args) + 1, sizeof...(Args) + 1)
    };
    // 定义一个静态函数，用于迭代处理输入参数并调用函数指针执行计算
    static void loop(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
        // 从传入的数据指针中获取函数指针并进行静态转换为指定类型
        Res (*func)(Args...) = static_cast<ufunc_data<Res (*)(Args...)> *>(data)->func;
        // 迭代处理第一维度的所有元素
        for (npy_intp i = 0; i < dimensions[0]; ++i) {
            // 调用函数指针func，传入参数列表并获取结果
            Res res = func(npy_traits<Args>::get(args[I], dimensions + 1, steps + steps_offsets[I])...);
            // 将计算结果res赋值给输出指针所指向的位置
            npy_traits<Res>::set(args[sizeof...(Args)], res); // assign to the output pointer

            // 更新所有输入参数的指针位置，以便下一次迭代
            for (npy_uintp j = 0; j <= sizeof...(Args); ++j) {
                args[j] += steps[j];
            }
        }

        // 获取函数指针对应的名称，并检查浮点异常
        const char *name = static_cast<ufunc_data<Res (*)(Args...)> *>(data)->name;
        sf_error_check_fpe(name);
    }
    // This class provides traits for a function pointer with variadic arguments, indexed by std::index_sequence
    template <typename... Args, size_t... I>
    struct ufunc_traits<void (*)(Args...), std::index_sequence<I...>> {
        // Stores the NumPy type numbers for each argument in a constexpr char array
        static constexpr char types[sizeof...(Args)] = {npy_typenum_v<Args>...};

        // Stores the rank (number of dimensions) of each argument in a constexpr size_t array
        static constexpr size_t ranks[sizeof...(Args)] = {rank_of_v<Args>...};

        // Stores the cumulative offsets for indexing steps_offsets array based on ranks of arguments
        static constexpr size_t steps_offsets[sizeof...(Args)] = {
            initializer_accumulate(ranks, ranks + I, sizeof...(Args))...
        };

        // Executes a loop over the arguments, invoking the function pointer with argument indexing
        static void loop(char **args, const npy_intp *dimensions, const npy_intp *steps, void *data) {
            // Casts the void pointer data back to ufunc_data struct to retrieve the function pointer
            void (*func)(Args...) = static_cast<ufunc_data<void (*)(Args...)> *>(data)->func;

            // Iterates over the first dimension of the input array
            for (npy_intp i = 0; i < dimensions[0]; ++i) {
                // Invokes the function pointer with arguments unpacked from the array args
                func(npy_traits<Args>::get(args[I], dimensions + 1, steps + steps_offsets[I])...);

                // Adjusts the pointers in args based on step sizes
                for (npy_uintp j = 0; j < sizeof...(Args); ++j) {
                    args[j] += steps[j];
                }
            }

            // Retrieves the name from ufunc_data struct and performs error checking
            const char *name = static_cast<ufunc_data<void (*)(Args...)> *>(data)->name;
            sf_error_check_fpe(name);
        }
    };

    // This class manages special functions as universal functions (ufuncs)
    class SpecFun_UFunc {
      public:
        using data_handle_type = void *;           // Type of data handle for managing internal data
        using data_deleter_type = void (*)(void *);  // Type of function pointer for data deletion

      private:
        // Internal structure representing a special function with associated metadata
        struct SpecFun_Func {
            bool has_return;                      // Flag indicating if the function returns a value
            int nin_and_nout;                     // Total number of input and output arguments
            PyUFuncGenericFunction func;          // Function pointer to the ufunc loop
            data_handle_type data;                // Handle to internal data associated with the function
            data_deleter_type data_deleter;       // Function pointer to delete the internal data
            const char *types;                    // Character array representing NumPy types of arguments

            // Constructor template for initializing SpecFun_Func from a function pointer
            template <typename Func>
            SpecFun_Func(Func func)
                : has_return(has_return_v<Func>),    // Sets has_return based on the return type of Func
                  nin_and_nout(arity_of_v<Func> + has_return),  // Computes total arguments based on Func's arity
                  func(ufunc_traits<Func>::loop),    // Assigns ufunc loop function from ufunc_traits
                  data(new ufunc_data<Func>{{nullptr}, func}),  // Creates new ufunc_data for internal storage
                  data_deleter([](void *ptr) { delete static_cast<ufunc_data<Func> *>(ptr); }),  // Deletes ufunc_data
                  types(ufunc_traits<Func>::types) {}  // Assigns NumPy types from ufunc_traits
        };

        int m_ntypes;                               // Number of unique type signatures for the ufunc
        bool m_has_return;                          // Flag indicating if any function returns a value
        int m_nin_and_nout;                         // Total number of input and output arguments
        std::unique_ptr<PyUFuncGenericFunction[]> m_func;            // Array of ufunc function pointers
        std::unique_ptr<data_handle_type[]> m_data;                   // Array of handles to internal data
        std::unique_ptr<data_deleter_type[]> m_data_deleters;         // Array of data deletion function pointers
        std::unique_ptr<char[]> m_types;            // Character array storing NumPy types for each function

      public:
    # 根据给定的函数列表初始化 SpecFun_UFunc 对象
    SpecFun_UFunc(std::initializer_list<SpecFun_Func> func)
        : m_ntypes(func.size()),  # 初始化成员变量 m_ntypes 为函数列表的大小
          m_has_return(func.begin()->has_return),  # 初始化成员变量 m_has_return 为第一个函数的返回类型
          m_nin_and_nout(func.begin()->nin_and_nout),  # 初始化成员变量 m_nin_and_nout 为第一个函数的参数数量
          m_func(new PyUFuncGenericFunction[m_ntypes]),  # 动态分配大小为 m_ntypes 的 PyUFuncGenericFunction 数组
          m_data(new data_handle_type[m_ntypes]),  # 动态分配大小为 m_ntypes 的 data_handle_type 数组
          m_data_deleters(new data_deleter_type[m_ntypes]),  # 动态分配大小为 m_ntypes 的 data_deleter_type 数组
          m_types(new char[m_ntypes * m_nin_and_nout]) {  # 动态分配大小为 m_ntypes * m_nin_and_nout 的 char 数组
        for (auto it = func.begin(); it != func.end(); ++it) {  # 遍历函数列表
            if (it->nin_and_nout != m_nin_and_nout) {  # 检查每个函数的参数数量是否一致
                PyErr_SetString(PyExc_RuntimeError, "all functions must have the same number of arguments");
            }
            if (it->has_return != m_has_return) {  # 检查每个函数的返回类型是否一致
                PyErr_SetString(PyExc_RuntimeError, "all functions must be void if any function is");
            }

            size_t i = it - func.begin();  # 计算当前函数在列表中的索引
            m_func[i] = it->func;  # 将当前函数的函数指针存入 m_func 数组中
            m_data[i] = it->data;  # 将当前函数的数据指针存入 m_data 数组中
            m_data_deleters[i] = it->data_deleter;  # 将当前函数的数据删除器存入 m_data_deleters 数组中
            std::memcpy(m_types.get() + i * m_nin_and_nout, it->types, m_nin_and_nout);  # 将当前函数的参数类型复制到 m_types 数组中
        }
    }

    # 移动构造函数，默认实现
    SpecFun_UFunc(SpecFun_UFunc &&other) = default;

    # 析构函数
    ~SpecFun_UFunc() {
        if (m_data) {  # 检查是否有有效的数据成员
            for (int i = 0; i < m_ntypes; ++i) {  # 遍历所有类型
                data_deleter_type data_deleter = m_data_deleters[i];  # 获取当前索引对应的数据删除器
                data_deleter(m_data[i]);  # 调用数据删除器删除对应数据
            }
        }
    }

    # 返回成员变量 m_ntypes 的值
    int ntypes() const { return m_ntypes; }

    # 返回成员变量 m_has_return 的值
    bool has_return() const { return m_has_return; }

    # 返回成员变量 m_nin_and_nout 的值
    int nin_and_nout() const { return m_nin_and_nout; }

    # 返回成员变量 m_func 的值
    PyUFuncGenericFunction *func() const { return m_func.get(); }

    # 返回成员变量 m_data 的值
    data_handle_type *data() const { return m_data.get(); }

    # 返回成员变量 m_types 的值
    char *types() const { return m_types.get(); }

    # 设置成员变量 m_data 数组中每个元素的 name 属性为给定的 name
    void set_name(const char *name) {
        for (int i = 0; i < m_ntypes; ++i) {  # 遍历所有类型
            static_cast<base_ufunc_data *>(m_data[i])->name = name;  # 将 name 赋值给当前数据的 name 属性
        }
    }
};

// 定义一个名为 SpecFun_NewUFunc 的函数，用于创建一个新的 NumPy 通用函数对象
PyObject *SpecFun_NewUFunc(SpecFun_UFunc func, int nout, const char *name, const char *doc) {
    // 静态变量，存储 SpecFun_UFunc 对象的向量 ufuncs
    static std::vector<SpecFun_UFunc> ufuncs;

    // 如果 Python 中发生了异常，返回空指针
    if (PyErr_Occurred()) {
        return nullptr;
    }

    // 将 func 移动到 ufuncs 向量中，并取其引用
    SpecFun_UFunc &ufunc = ufuncs.emplace_back(std::move(func));

    // 设置 ufunc 对象的名称为传入的 name 参数
    ufunc.set_name(name);

    // 使用 PyUFunc_FromFuncAndData 函数创建一个 NumPy 通用函数对象，并返回该对象
    return PyUFunc_FromFuncAndData(
        ufunc.func(), ufunc.data(), ufunc.types(), ufunc.ntypes(), ufunc.nin_and_nout() - nout, nout, PyUFunc_None,
        name, doc, 0
    );
}

// 定义一个名为 SpecFun_NewUFunc 的函数重载，简化 nout 参数的处理
PyObject *SpecFun_NewUFunc(SpecFun_UFunc func, const char *name, const char *doc) {
    // 检测 func 是否有返回值
    int nout = func.has_return();

    // 调用前一个重载的 SpecFun_NewUFunc 函数，创建 NumPy 通用函数对象并返回
    return SpecFun_NewUFunc(std::move(func), nout, name, doc);
}

// 定义一个名为 SpecFun_NewGUFunc 的函数，用于创建一个带有签名的 NumPy 通用函数对象
PyObject *SpecFun_NewGUFunc(SpecFun_UFunc func, int nout, const char *name, const char *doc, const char *signature) {
    // 静态变量，存储 SpecFun_UFunc 对象的向量 ufuncs
    static std::vector<SpecFun_UFunc> ufuncs;

    // 如果 Python 中发生了异常，返回空指针
    if (PyErr_Occurred()) {
        return nullptr;
    }

    // 将 func 移动到 ufuncs 向量中，并取其引用
    SpecFun_UFunc &ufunc = ufuncs.emplace_back(std::move(func));

    // 设置 ufunc 对象的名称为传入的 name 参数
    ufunc.set_name(name);

    // 使用 PyUFunc_FromFuncAndDataAndSignature 函数创建一个带有签名的 NumPy 通用函数对象，并返回该对象
    return PyUFunc_FromFuncAndDataAndSignature(
        ufunc.func(), ufunc.data(), ufunc.types(), ufunc.ntypes(), ufunc.nin_and_nout() - nout, nout, PyUFunc_None,
        name, doc, 0, signature
    );
}

// 定义一个名为 SpecFun_NewGUFunc 的函数重载，简化 nout 参数的处理
PyObject *SpecFun_NewGUFunc(SpecFun_UFunc func, const char *name, const char *doc, const char *signature) {
    // 检测 func 是否有返回值
    int nout = func.has_return();

    // 调用前一个重载的 SpecFun_NewGUFunc 函数，创建带有签名的 NumPy 通用函数对象并返回
    return SpecFun_NewGUFunc(std::move(func), nout, name, doc, signature);
}
```