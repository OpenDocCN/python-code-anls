# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\_trlib.pyx`

```
# 导入必要的模块和类
from scipy.optimize._trustregion import BaseQuadraticSubproblem  # 导入基本二次子问题类
import numpy as np  # 导入NumPy库
from . cimport ctrlib  # 导入C扩展模块ctrlib

# 导入Cython定义的NumPy模块
cimport numpy as np  

# 导入消息流类
from scipy._lib.messagestream cimport MessageStream  

# 导入NumPy数组接口
np.import_array()

# 定义TRLIBQuadraticSubproblem类，继承自BaseQuadraticSubproblem
class TRLIBQuadraticSubproblem(BaseQuadraticSubproblem):

    # 初始化方法
    def __init__(self, x, fun, jac, hess, hessp, tol_rel_i=-2.0, tol_rel_b=-3.0,
                 disp=False):
        super().__init__(x, fun, jac, hess, hessp)  # 调用父类的初始化方法
        self.tol_rel_i = tol_rel_i  # 设置相对容差指数
        self.tol_rel_b = tol_rel_b  # 设置相对容差基数
        self.disp = disp  # 是否显示信息
        self.itmax = int(min(1e9/self.jac.shape[0], 2*self.jac.shape[0]))  # 计算最大迭代次数
        cdef long itmax, iwork_size, fwork_size, h_pointer  # 定义Cython类型的变量
        itmax = self.itmax  # 将Python中的迭代次数值传递给Cython变量itmax
        # 调用Cython函数获取内存需求和指针
        ctrlib.trlib_krylov_memory_size(itmax, &iwork_size, &fwork_size,
                                        &h_pointer)
        self.h_pointer = h_pointer  # 设置h_pointer属性
        self.fwork = np.empty([fwork_size])  # 初始化fwork数组
        cdef double [:] fwork_view = self.fwork  # 创建Cython内存视图fwork_view
        cdef double *fwork_ptr = NULL  # 初始化Cython指针fwork_ptr为NULL
        if fwork_view.shape[0] > 0:  # 检查fwork_view的长度是否大于0
            fwork_ptr = &fwork_view[0]  # 设置fwork_ptr指向fwork_view的第一个元素
        # 调用Cython函数准备Krylov迭代所需的内存
        ctrlib.trlib_krylov_prepare_memory(itmax, fwork_ptr)
        self.iwork = np.zeros([iwork_size], dtype=np.dtype("long"))  # 初始化iwork数组
        self.s  = np.empty(self.jac.shape)  # 初始化s数组
        self.g  = np.empty(self.jac.shape)  # 初始化g数组
        self.v  = np.empty(self.jac.shape)  # 初始化v数组
        self.gm = np.empty(self.jac.shape)  # 初始化gm数组
        self.p  = np.empty(self.jac.shape)  # 初始化p数组
        self.Hp = np.empty(self.jac.shape)  # 初始化Hp数组
        self.Q  = np.empty([self.itmax+1, self.jac.shape[0]])  # 初始化Q数组
        # 初始化计时数组
        self.timing = np.zeros([ctrlib.trlib_krylov_timing_size()],
                               dtype=np.dtype("long"))
        self.init = ctrlib._TRLIB_CLS_INIT  # 设置init属性为Cython中的初始化常量
```