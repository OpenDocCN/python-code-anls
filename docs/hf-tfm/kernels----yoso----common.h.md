# `.\transformers\kernels\yoso\common.h`

```
# 定义一个取两个值中较小值的宏
#define min(a, b) ((a)<(b)?(a):(b))
# 定义一个取两个值中较大值的宏
#define max(a, b) ((a)>(b)?(a):(b))
# 定义一个将a除以b并向上取整的宏
#define ceil_divide(a, b) ((a)/(b)+((a)%(b)!=0))
# 定义一个根据条件选择返回a或b的宏
#define select(cond, a, b) ((cond)?(a):(b))
# 定义圆周率π的常量
#define PI 3.141592
# 定义一个极小值常量，用于比较浮点数时的精度
#define EPSILON 1e-8
# 定义一个最大值常量
#define MAX_VAL 1e12
# 定义一个最小值常量
#define MIN_VAL -1e12
# 定义一个空值常量，表示某些情况下的无效值
#define EMPTY_VALUE -1
```