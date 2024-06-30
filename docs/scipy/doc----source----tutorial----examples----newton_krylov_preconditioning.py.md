# `D:\src\scipysrc\scipy\doc\source\tutorial\examples\newton_krylov_preconditioning.py`

```
from scipy.optimize import root
from scipy.sparse import spdiags, kron
from scipy.sparse.linalg import spilu, LinearOperator
from numpy import cosh, zeros_like, mgrid, zeros, eye

# 定义网格的大小
nx, ny = 75, 75
# 计算步长
hx, hy = 1./(nx-1), 1./(ny-1)

# 边界条件
P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

def get_preconditioner():
    """计算预处理器 M"""
    # 创建用于 x 方向的对角线数组
    diags_x = zeros((3, nx))
    diags_x[0,:] = 1/hx/hx
    diags_x[1,:] = -2/hx/hx
    diags_x[2,:] = 1/hx/hx
    Lx = spdiags(diags_x, [-1,0,1], nx, nx)

    # 创建用于 y 方向的对角线数组
    diags_y = zeros((3, ny))
    diags_y[0,:] = 1/hy/hy
    diags_y[1,:] = -2/hy/hy
    diags_y[2,:] = 1/hy/hy
    Ly = spdiags(diags_y, [-1,0,1], ny, ny)

    # 构建 J1 矩阵
    J1 = kron(Lx, eye(ny)) + kron(eye(nx), Ly)

    # 现在我们有了矩阵 `J_1`，需要找到其逆 `M` --
    # 由于一个近似逆已经足够了，我们可以使用不完全 LU 分解（ILU）
    J1_ilu = spilu(J1)

    # 返回一个带有 .solve() 方法的对象，用于求解对应的矩阵-向量乘积
    # 在传递给 Krylov 方法之前，需要将其封装成 LinearOperator：
    M = LinearOperator(shape=(nx*ny, nx*ny), matvec=J1_ilu.solve)
    return M

def solve(preconditioning=True):
    """计算解"""
    count = [0]

    def residual(P):
        count[0] += 1

        d2x = zeros_like(P)
        d2y = zeros_like(P)

        # x 方向的二阶导数
        d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2])/hx/hx
        d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
        d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

        # y 方向的二阶导数
        d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
        d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
        d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

        # 残差函数的计算
        return d2x + d2y + 5*cosh(P).mean()**2

    # 是否使用预处理器
    if preconditioning:
        M = get_preconditioner()
    else:
        M = None

    # 初始化猜测值
    guess = zeros((nx, ny), float)

    # 使用 Krylov 方法求解非线性方程组
    sol = root(residual, guess, method='krylov',
               options={'disp': True,
                        'jac_options': {'inner_M': M}})
    print('Residual', abs(residual(sol.x)).max())
    print('Evaluations', count[0])

    return sol.x

def main():
    sol = solve(preconditioning=True)

    # 可视化结果
    import matplotlib.pyplot as plt
    x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
    plt.clf()
    plt.pcolor(x, y, sol)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
```