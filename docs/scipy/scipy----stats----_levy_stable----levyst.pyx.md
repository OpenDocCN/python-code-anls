# `D:\src\scipysrc\scipy\scipy\stats\_levy_stable\levyst.pyx`

```
# 从libc.stdlib模块中导入free函数，用于释放内存

# 从指定路径下的头文件levyst.h中导入结构体nolan_precanned的定义
cdef extern from "./c_src/levyst.h":
    struct nolan_precanned:
        # 函数指针g，接受nolan_precanned结构体指针和double类型参数，返回double类型值
        double (*g)(nolan_precanned *, double)
        # double类型成员变量alpha
        double alpha
        # double类型成员变量zeta
        double zeta
        # double类型成员变量xi
        double xi
        # double类型成员变量zeta_prefactor
        double zeta_prefactor
        # double类型成员变量alpha_exp
        double alpha_exp
        # double类型成员变量alpha_xi
        double alpha_xi
        # double类型成员变量zeta_offset
        double zeta_offset
        # double类型成员变量two_beta_div_pi
        double two_beta_div_pi
        # double类型成员变量pi_div_two_beta
        double pi_div_two_beta
        # double类型成员变量x0_div_term
        double x0_div_term
        # double类型成员变量c1
        double c1
        # double类型成员变量c2
        double c2
        # double类型成员变量c3
        double c3

# 定义一个Cython类Nolan
cdef class Nolan:
    # 使用Cython的类型声明，声明一个nolan_precanned指针p
    cdef nolan_precanned * p

    # 构造函数，接受alpha、beta、x0三个参数
    def __init__(self, alpha, beta, x0):
        # 调用外部C函数nolan_precan，返回一个nolan_precanned指针，将其赋值给self.p
        self.p = nolan_precan(alpha, beta, x0)

    # 方法g，接受参数theta，调用self.p指针所指向的nolan_precanned结构体的g函数
    def g(self, theta):
       return self.p.g(self.p, theta)

    # 属性方法zeta，返回self.p指针所指向的nolan_precanned结构体的zeta成员变量
    @property
    def zeta(self):
        return self.p.zeta

    # 属性方法xi，返回self.p指针所指向的nolan_precanned结构体的xi成员变量
    @property
    def xi(self):
        return self.p.xi

    # 属性方法c1，返回self.p指针所指向的nolan_precanned结构体的c1成员变量
    @property
    def c1(self):
        return self.p.c1

    # 属性方法c2，返回self.p指针所指向的nolan_precanned结构体的c2成员变量
    @property
    def c2(self):
        return self.p.c2

    # 属性方法c3，返回self.p指针所指向的nolan_precanned结构体的c3成员变量
    @property
    def c3(self):
        return self.p.c3

    # 析构函数，用于释放self.p指针所指向的内存空间
    def __dealloc__(self):
        free(self.p)
```