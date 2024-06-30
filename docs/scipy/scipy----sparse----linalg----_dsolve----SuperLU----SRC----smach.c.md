# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\smach.c`

```
/*
    著作权声明和许可证信息
    这段代码是 SuperLU 辅助例程（版本 5.0）
    使用了 C99 标准常量，并且是线程安全的。
    必须使用 "-std=c99" 标志进行编译。

    函数 smach 返回单精度机器参数。

    参数
    =====
    CMACH   (输入) CHARACTER*1
            指定 SMACH 返回的值：
            = 'E' 或 'e'，  SMACH := eps
            = 'S' 或 's'，  SMACH := sfmin
            = 'B' 或 'b'，  SMACH := base
            = 'P' 或 'p'，  SMACH := eps*base
            = 'N' 或 'n'，  SMACH := t
            = 'R' 或 'r'，  SMACH := rnd
            = 'M' 或 'm'，  SMACH := emin
            = 'U' 或 'u'，  SMACH := rmin
            = 'L' 或 'l'，  SMACH := emax
            = 'O' 或 'o'，  SMACH := rmax

            其中

            eps   = 相对机器精度
            sfmin = 安全最小值，使得 1/sfmin 不会溢出
            base  = 机器的基数
            prec  = eps*base
            t     = 小数部分中的 (base) 数字的数量
            rnd   = 1.0 表示加法中发生了舍入，0.0 表示没有
            emin  = (逐渐的) 下溢之前的最小指数
            rmin  = 下溢阈值 - base**(emin-1)
            emax  = 溢出之前的最大指数
            rmax  = 溢出阈值  - (base**emax)*(1-eps)
*/

float sfmin, small, rmach;

if (strncmp(cmach, "E", 1)==0) {
    rmach = FLT_EPSILON * 0.5;
} else if (strncmp(cmach, "S", 1)==0) {
    sfmin = FLT_MIN;
    small = 1. / FLT_MAX;
    if (small >= sfmin) {
        /* 使用 SMALL 加上一点，以避免在计算 1/sfmin 时可能由于舍入而导致溢出 */
        sfmin = small * (FLT_EPSILON*0.5 + 1.);
    }
    rmach = sfmin;
} else if (strncmp(cmach, "B", 1)==0) {
    rmach = FLT_RADIX;
} else if (strncmp(cmach, "P", 1)==0) {
    rmach = FLT_EPSILON * 0.5 * FLT_RADIX;
} else if (strncmp(cmach, "N", 1)==0) {
    rmach = FLT_MANT_DIG;
} else if (strncmp(cmach, "R", 1)==0) {
    rmach = FLT_ROUNDS;
} else if (strncmp(cmach, "M", 1)==0) {
    rmach = FLT_MIN_EXP;
} else if (strncmp(cmach, "U", 1)==0) {
    rmach = FLT_MIN;
} else if (strncmp(cmach, "L", 1)==0) {
    rmach = FLT_MAX_EXP;
} else if (strncmp(cmach, "O", 1)==0) {
    rmach = FLT_MAX;
    } else {
        // 如果条件不满足，则执行以下操作
        int argument = 0;  // 定义并初始化 argument 变量为 0
        input_error("smach", &argument);  // 调用 input_error 函数，传递参数 "smach" 和 argument 的地址
        rmach = 0;  // 将 rmach 变量赋值为 0
    }

    // 返回 rmach 变量的值作为函数结果
    return rmach;
} /* end smach */
```