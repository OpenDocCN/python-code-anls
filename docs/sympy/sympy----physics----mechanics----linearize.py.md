# `D:\src\scipysrc\sympy\sympy\physics\mechanics\linearize.py`

```
# 定义一个全局变量 __all__，指定了模块中可以导出的公共接口，这里只有 'Linearizer' 一个类
__all__ = ['Linearizer']

# 从 sympy 库导入需要的模块和函数
from sympy import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs, _parse_linear_solver

# 导入标准库中的 namedtuple 和 ABC 中的 Iterable 类
from collections import namedtuple
from collections.abc import Iterable

# 定义一个名为 Linearizer 的类
class Linearizer:
    """This object holds the general model form for a dynamic system. This
    model is used for computing the linearized form of the system, while
    properly dealing with constraints leading to  dependent coordinates and
    speeds. The notation and method is described in [1]_.

    Attributes
    ==========

    f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : Matrix
        Matrices holding the general system form.
    q, u, r : Matrix
        Matrices holding the generalized coordinates, speeds, and
        input vectors.
    q_i, u_i : Matrix
        Matrices of the independent generalized coordinates and speeds.
    q_d, u_d : Matrix
        Matrices of the dependent generalized coordinates and speeds.
    perm_mat : Matrix
        Permutation matrix such that [q_ind, u_ind]^T = perm_mat*[q, u]^T

    References
    ==========

    .. [1] D. L. Peterson, G. Gede, and M. Hubbard, "Symbolic linearization of
           equations of motion of constrained multibody systems," Multibody
           Syst Dyn, vol. 33, no. 2, pp. 143-161, Feb. 2015, doi:
           10.1007/s11044-014-9436-5.

    """

    def _setup(self):
        # 计算这些变量只需要在对象创建时运行一次，为了提高 Linearizer 实例创建的速度，将其从 __init__ 方法中移到这里。
        self._form_permutation_matrices()  # 调用方法 _form_permutation_matrices，形成置换矩阵
        self._form_block_matrices()  # 调用方法 _form_block_matrices，形成块状矩阵
        self._form_coefficient_matrices()  # 调用方法 _form_coefficient_matrices，形成系数矩阵
        self._setup_done = True  # 设置标志位 _setup_done 表示初始化设置已完成
    def _form_permutation_matrices(self):
        """Form the permutation matrices Pq and Pu."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Compute permutation matrices
        if n != 0:
            # Form permutation matrix Pq using q_i and q_d
            self._Pq = permutation_matrix(self.q, Matrix([self.q_i, self.q_d]))
            if l > 0:
                # Extract parts of Pq for configuration constraints
                self._Pqi = self._Pq[:, :-l]
                self._Pqd = self._Pq[:, -l:]
            else:
                self._Pqi = self._Pq
                self._Pqd = Matrix()
        if o != 0:
            # Form permutation matrix Pu using u_i and u_d
            self._Pu = permutation_matrix(self.u, Matrix([self.u_i, self.u_d]))
            if m > 0:
                # Extract parts of Pu for motion constraints
                self._Pui = self._Pu[:, :-m]
                self._Pud = self._Pu[:, -m:]
            else:
                self._Pui = self._Pu
                self._Pud = Matrix()
        # Compute combination permutation matrix for computing A and B
        P_col1 = Matrix([self._Pqi, zeros(o + k, n - l)])
        P_col2 = Matrix([zeros(n, o - m), self._Pui, zeros(k, o - m)])
        if P_col1:
            if P_col2:
                # Combine P_col1 and P_col2 into permutation matrix perm_mat
                self.perm_mat = P_col1.row_join(P_col2)
            else:
                self.perm_mat = P_col1
        else:
            self.perm_mat = P_col2

    def _form_coefficient_matrices(self):
        """Form the coefficient matrices C_0, C_1, and C_2."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Build up the coefficient matrices C_0, C_1, and C_2
        # If there are configuration constraints (l > 0), form C_0 as normal.
        # If not, C_0 is I_(nxn). Note that this works even if n=0
        if l > 0:
            # Compute C_0 using Jacobian and linear solver
            f_c_jac_q = self.f_c.jacobian(self.q)
            self._C_0 = (eye(n) - self._Pqd *
                         self.linear_solver(f_c_jac_q*self._Pqd,
                                            f_c_jac_q))*self._Pqi
        else:
            self._C_0 = eye(n)
        # If there are motion constraints (m > 0), form C_1 and C_2 as normal.
        # If not, C_1 is 0, and C_2 is I_(oxo). Note that this works even if
        # o = 0.
        if m > 0:
            # Compute C_1 and C_2 using Jacobians and linear solvers
            f_v_jac_u = self.f_v.jacobian(self.u)
            temp = f_v_jac_u * self._Pud
            if n != 0:
                f_v_jac_q = self.f_v.jacobian(self.q)
                self._C_1 = -self._Pud * self.linear_solver(temp, f_v_jac_q)
            else:
                self._C_1 = zeros(o, n)
            self._C_2 = (eye(o) - self._Pud *
                         self.linear_solver(temp, f_v_jac_u))*self._Pui
        else:
            self._C_1 = zeros(o, n)
            self._C_2 = eye(o)
    def _form_block_matrices(self):
        """Form the block matrices for composing M, A, and B."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        
        # Block Matrix Definitions. These are only defined if under certain
        # conditions. If undefined, an empty matrix is used instead
        
        # Define _M_qq and _A_qq if n is not zero
        if n != 0:
            self._M_qq = self.f_0.jacobian(self._qd)
            self._A_qq = -(self.f_0 + self.f_1).jacobian(self.q)
        else:
            self._M_qq = Matrix()
            self._A_qq = Matrix()
        
        # Define _M_uqc and _A_uqc if both n and m are not zero
        if n != 0 and m != 0:
            self._M_uqc = self.f_a.jacobian(self._qd_dup)
            self._A_uqc = -self.f_a.jacobian(self.q)
        else:
            self._M_uqc = Matrix()
            self._A_uqc = Matrix()
        
        # Define _M_uqd and _A_uqd if n and (o - m + k) are not zero
        if n != 0 and o - m + k != 0:
            self._M_uqd = self.f_3.jacobian(self._qd_dup)
            self._A_uqd = -(self.f_2 + self.f_3 + self.f_4).jacobian(self.q)
        else:
            self._M_uqd = Matrix()
            self._A_uqd = Matrix()
        
        # Define _M_uuc and _A_uuc if o and m are not zero
        if o != 0 and m != 0:
            self._M_uuc = self.f_a.jacobian(self._ud)
            self._A_uuc = -self.f_a.jacobian(self.u)
        else:
            self._M_uuc = Matrix()
            self._A_uuc = Matrix()
        
        # Define _M_uud and _A_uud if o and (o - m + k) are not zero
        if o != 0 and o - m + k != 0:
            self._M_uud = self.f_2.jacobian(self._ud)
            self._A_uud = -(self.f_2 + self.f_3).jacobian(self.u)
        else:
            self._M_uud = Matrix()
            self._A_uud = Matrix()
        
        # Define _A_qu if o and n are not zero
        if o != 0 and n != 0:
            self._A_qu = -self.f_1.jacobian(self.u)
        else:
            self._A_qu = Matrix()
        
        # Define _M_uld if k and (o - m + k) are not zero
        if k != 0 and o - m + k != 0:
            self._M_uld = self.f_4.jacobian(self.lams)
        else:
            self._M_uld = Matrix()
        
        # Define _B_u if s and (o - m + k) are not zero
        if s != 0 and o - m + k != 0:
            self._B_u = -self.f_3.jacobian(self.r)
        else:
            self._B_u = Matrix()
# 计算置换矩阵，将 orig_vec 的顺序置换为 per_vec 的顺序

def permutation_matrix(orig_vec, per_vec):
    """Compute the permutation matrix to change order of
    orig_vec into order of per_vec.

    Parameters
    ==========

    orig_vec : array_like
        Symbols in original ordering.
    per_vec : array_like
        Symbols in new ordering.

    Returns
    =======

    p_matrix : Matrix
        Permutation matrix such that orig_vec == (p_matrix * per_vec).
    """

    # 如果 orig_vec 不是列表或元组，则将其展平
    if not isinstance(orig_vec, (list, tuple)):
        orig_vec = flatten(orig_vec)
    
    # 如果 per_vec 不是列表或元组，则将其展平
    if not isinstance(per_vec, (list, tuple)):
        per_vec = flatten(per_vec)
    
    # 检查 orig_vec 和 per_vec 是否含有相同的符号集合
    if set(orig_vec) != set(per_vec):
        raise ValueError("orig_vec and per_vec must be the same length, "
                         "and contain the same symbols.")
    
    # 根据 per_vec 中元素在 orig_vec 中的索引，创建索引列表
    ind_list = [orig_vec.index(i) for i in per_vec]
    
    # 初始化全零置换矩阵，长度为 orig_vec 的长度
    p_matrix = zeros(len(orig_vec))
    
    # 根据索引列表设置置换矩阵的值
    for i, j in enumerate(ind_list):
        p_matrix[i, j] = 1
    
    # 返回计算得到的置换矩阵
    return p_matrix
```