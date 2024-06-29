# `D:\src\scipysrc\matplotlib\src\py_exceptions.h`

```py
#ifndef MPL_PY_EXCEPTIONS_H
#define MPL_PY_EXCEPTIONS_H

#include <exception>
#include <stdexcept>

namespace mpl {

// 自定义异常类，继承自 std::exception
class exception : public std::exception
{
  public:
    // 返回异常信息 "python error has been set"
    const char *what() const throw()
    {
        return "python error has been set";
    }
};
}

// 定义一个宏，用于调用 C++ 函数并处理异常
#define CALL_CPP_FULL(name, a, cleanup, errorcode)                           \
    try                                                                      \
    {                                                                        \
        a;                                                                   \
    }                                                                        \
    // 捕获 mpl::exception 类型的异常
    catch (const mpl::exception &)                                           \
    {                                                                        \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    // 捕获 std::bad_alloc 类型的异常
    catch (const std::bad_alloc &)                                           \
    {                                                                        \
        // 格式化异常信息为内存错误，并返回
        PyErr_Format(PyExc_MemoryError, "In %s: Out of memory", (name));     \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    // 捕获 std::overflow_error 类型的异常
    catch (const std::overflow_error &e)                                     \
    {                                                                        \
        // 格式化异常信息为溢出错误，并返回
        PyErr_Format(PyExc_OverflowError, "In %s: %s", (name), e.what());    \
        {                                                                    \
            cleanup;                                                         \
        }                                                                    \
        return (errorcode);                                                  \
    }                                                                        \
    // 捕获 std::runtime_error 类型的异常
    catch (const std::runtime_error &e)                                      \
    {
        PyErr_Format(PyExc_RuntimeError, "In %s: %s", (name), e.what());
        // 抛出运行时错误异常，并使用格式化字符串指定错误消息，包括name和e.what()的内容
    
        {
            cleanup;
            // 执行清理操作，这部分代码在异常处理后被调用
        }
        return (errorcode);
        // 返回指定的错误代码
    
        }                                                                        \
        catch (...)                                                              \
        {                                                                        \
            PyErr_Format(PyExc_RuntimeError, "Unknown exception in %s", (name));
            // 捕获未知类型的异常，并生成相应的运行时错误异常消息，包括name的内容
    
            {
                cleanup;
                // 执行清理操作，这部分代码在捕获到未知异常后被调用
            }
            return (errorcode);
            // 返回指定的错误代码
        }
# 定义宏CALL_CPP_CLEANUP，简化调用C++函数时的清理操作，参数包括name（函数名）、a（参数列表）、cleanup（清理操作）
#define CALL_CPP_CLEANUP(name, a, cleanup) CALL_CPP_FULL(name, a, cleanup, 0)

# 定义宏CALL_CPP，用于调用C++函数，参数包括name（函数名）、a（参数列表）
#define CALL_CPP(name, a) CALL_CPP_FULL(name, a, , 0)

# 定义宏CALL_CPP_INIT，用于初始化调用C++函数时的操作，参数包括name（函数名）、a（参数列表）
#define CALL_CPP_INIT(name, a) CALL_CPP_FULL(name, a, , -1)

# 结束宏定义区域
#endif
```