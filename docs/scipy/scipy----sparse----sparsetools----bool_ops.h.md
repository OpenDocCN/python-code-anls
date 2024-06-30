# `D:\src\scipysrc\scipy\scipy\sparse\sparsetools\bool_ops.h`

```
#ifndef BOOL_OPS_H
#define BOOL_OPS_H
/*
 * Functions to handle arithmetic operations on NumPy Bool values.
 */
#include <numpy/arrayobject.h>

/*
 * A compiler time (ct) assert macro from 
 * http://www.pixelbeat.org/programming/gcc/static_assert.html
 * This is used to assure that npy_bool_wrapper is the right size.
 */
// 定义编译时断言宏，用于确保 npy_bool_wrapper 的大小正确
#define ct_assert(e) extern char (*ct_assert(void)) [sizeof(char[1 - 2*!(e)])]

class npy_bool_wrapper {
    private:
        // 内部私有成员变量，存储布尔值
        char value;

    public:
        /* operators */
        // 类型转换操作符，将对象转换为 char 类型
        operator char() const {
            return value;
        }
        // 赋值操作符重载，从另一个 npy_bool_wrapper 对象赋值
        npy_bool_wrapper& operator=(const npy_bool_wrapper& x) {
            value = x.value;
            return (*this);
        }
        // 加法操作符重载，返回布尔或运算的结果
        npy_bool_wrapper operator+(const npy_bool_wrapper& x) {
            return value || x.value;
        }
        /* inplace operators */
        // 复合赋值加法操作符重载，进行布尔或运算并赋值给当前对象
        npy_bool_wrapper operator+=(const npy_bool_wrapper& x) {
            value = (value || x.value);
            return (*this);
        }
        // 复合赋值乘法操作符重载，进行布尔与运算并赋值给当前对象
        npy_bool_wrapper operator*=(const npy_bool_wrapper& x) {
            value = (value && x.value);
            return (*this);
        }
        /* constructors */
        // 默认构造函数，将 value 初始化为 0
        npy_bool_wrapper() { 
            value = 0; 
        }
        // 模板构造函数，根据输入值 x 的非零性将 value 初始化为 0 或 1
        template <class T>
        npy_bool_wrapper(T x) {
            value = (x != 0);
        }
};

// 使用编译时断言宏确保 char 类型的大小等于 npy_bool_wrapper 的大小
ct_assert(sizeof(char) == sizeof(npy_bool_wrapper));

#endif
```