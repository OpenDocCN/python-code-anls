# `.\kernels\yoso\common.h`

```py
# 定义宏函数，返回两个数中较小的一个
#define min(a, b) ((a)<(b)?(a):(b))

# 定义宏函数，返回两个数中较大的一个
#define max(a, b) ((a)>(b)?(a):(b))

# 定义宏函数，对两个数进行向上取整的除法运算
#define ceil_divide(a, b) ((a)/(b)+((a)%(b)!=0))

# 定义宏函数，根据条件选择返回其中一个值
#define select(cond, a, b) ((cond)?(a):(b))

# 定义常数 PI，表示圆周率
#define PI 3.141592

# 定义常数 EPSILON，表示一个小的正数，通常用于浮点数比较的容差
#define EPSILON 1e-8

# 定义常数 MAX_VAL，表示一个较大的数值上限
#define MAX_VAL 1e12

# 定义常数 MIN_VAL，表示一个较小的数值下限
#define MIN_VAL -1e12

# 定义常数 EMPTY_VALUE，表示一个特定的空值或未初始化值
#define EMPTY_VALUE -1
```