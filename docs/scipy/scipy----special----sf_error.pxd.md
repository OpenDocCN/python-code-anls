# `D:\src\scipysrc\scipy\scipy\special\sf_error.pxd`

```
# -*-cython-*-

# 从 sf_error.h 文件中导入所需的外部定义
cdef extern from "sf_error.h":
    # 定义错误类型枚举 sf_error_t
    ctypedef enum sf_error_t:
        OK "SF_ERROR_OK"
        SINGULAR "SF_ERROR_SINGULAR"
        UNDERFLOW "SF_ERROR_UNDERFLOW"
        OVERFLOW "SF_ERROR_OVERFLOW"
        SLOW "SF_ERROR_SLOW"
        LOSS "SF_ERROR_LOSS"
        NO_RESULT "SF_ERROR_NO_RESULT"
        DOMAIN "SF_ERROR_DOMAIN"
        ARG "SF_ERROR_ARG"
        OTHER "SF_ERROR_OTHER"

    # 定义错误处理动作枚举 sf_action_t
    ctypedef enum sf_action_t:
        IGNORE "SF_ERROR_IGNORE"
        WARN "SF_ERROR_WARN"
        RAISE "SF_ERROR_RAISE"

    # 外部变量，存储错误消息字符串的数组
    char **sf_error_messages
    # 错误处理函数原型，用于报告错误
    void error "sf_error" (char *func_name, sf_error_t code, char *fmt, ...) nogil
    # 检查浮点异常的函数原型
    void check_fpe "sf_error_check_fpe" (char *func_name) nogil
    # 设置特定错误代码的处理动作函数原型
    void set_action "scipy_sf_error_set_action" (sf_error_t code, sf_action_t action) nogil
    # 获取特定错误代码的处理动作函数原型
    sf_action_t get_action "scipy_sf_error_get_action" (sf_error_t code) nogil


# 内联函数，用于测试目的，可以触发各种 sf_error 类别的错误
cdef inline int _sf_error_test_function(int code) noexcept nogil:
    """Function that can raise every sf_error category for testing
    purposes.

    """
    cdef sf_error_t sf_error_code
    
    # 根据传入的 code 值确定 sf_error_t 类型
    if code < 0 or code >= 10:
        sf_error_code = OTHER
    else:
        sf_error_code = <sf_error_t>code
    # 调用错误处理函数报告错误
    error('_err_test_function', sf_error_code, NULL)
    # 返回值，此处始终返回 0
    return 0
```