# `.\numpy\numpy\_core\check_longdouble.c`

```py
/* 
 * "before" 是一个长度为 16 字节的数组，用于确保它和 "x" 之间没有填充。
 * 我们假设没有比 16 字节更大或者对齐要求更严格的 "long double" 类型。
 */
typedef long double test_type;  // 定义一个名为 test_type 的 long double 类型

struct {
        char         before[16];  // 结构体中的数组，长度为 16 字节
        test_type    x;           // 结构体中的 long double 类型变量
        char         after[8];    // 结构体中的数组，长度为 8 字节
} foo = {
        // 结构体 foo 的初始化：
        { '\0', '\0', '\0', '\0', '\0', '\0', '\0', '\0',  
          '\001', '\043', '\105', '\147', '\211', '\253', '\315', '\357' },  // "before" 数组的初始值
        -123456789.0,   // "x" 的初始值为 -123456789.0
        { '\376', '\334', '\272', '\230', '\166', '\124', '\062', '\020' }   // "after" 数组的初始值
};
```