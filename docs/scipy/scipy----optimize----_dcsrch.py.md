# `D:\src\scipysrc\scipy\scipy\optimize\_dcsrch.py`

```
import numpy as np

"""
# 2023 - 从 minpack2.dcsrch, dcstep (Fortran) 转换到 Python
c     MINPACK-1 项目。1983 年 6 月。
c     阿贡国家实验室。
c     Jorge J. More' 和 David J. Thuente.
c
c     MINPACK-2 项目。1993 年 11 月。
c     阿贡国家实验室和明尼苏达大学。
c     Brett M. Averick, Richard G. Carter 和 Jorge J. More'.
"""

# 注意：此文件在第一次提交时由 black 进行了代码格式化，后续保持不变。

class DCSRCH:
    """
    Parameters
    ----------
    phi : callable phi(alpha)
        点 `alpha` 处的函数
    derphi : callable phi'(alpha)
        目标函数的导数。返回一个标量。
    ftol : float
        足够减少条件的非负容差。
    gtol : float
        曲率条件的非负容差。
    xtol : float
        可接受步长的非负相对容差。如果 `sty` 和 `stx` 之间的相对差小于 `xtol`，则子程序将发出警告。
    stpmin : float
        步长的非负下界。
    stpmax :
        步长的非负上界。

    Notes
    -----

    此子程序找到满足足够减少条件和曲率条件的步长。

    每次调用子程序都会更新区间，端点为 `stx` 和 `sty`。初始选择的区间
    被设计为包含修改后的函数的极小值点

           psi(stp) = f(stp) - f(0) - ftol*stp*f'(0).

    如果对于某些步长 `psi(stp) <= 0` 且 `f'(stp) >= 0`，则选择的区间
    会包含 `f` 的极小值。

    此算法旨在找到满足足够减少条件

           f(stp) <= f(0) + ftol*stp*f'(0),

    和曲率条件

           abs(f'(stp)) <= gtol*abs(f'(0)).

    如果 `ftol` 小于 `gtol` 并且例如函数在下方有界，则总是存在一个步长
    满足这两个条件。

    如果找不到满足这两个条件的步长，则算法会停止并发出警告。在这种情况下，`stp`
    只满足足够减少条件。

    调用 `dcsrch` 的典型方法如下：

    在 `stp = 0.0` 处评估函数；存储在 `f` 中。
    在 `stp = 0.0` 处评估梯度；存储在 `g` 中。
    选择一个起始步长 `stp`。

    task = 'START'
    10 continue
        调用 `dcsrch(stp,f,g,ftol,gtol,xtol,task,stpmin,stpmax,
                   isave,dsave)`
        如果 (`task .eq. 'FG'`) then
           在 `stp` 处评估函数和梯度
           转到 10
        end if

    注意：用户在调用之间不得更改工作数组。

    子程序语句是

        subroutine dcsrch(f,g,stp,ftol,gtol,xtol,stpmin,stpmax,
                         task,isave,dsave)
        where
    stp is a double precision variable.
        On entry stp is the current estimate of a satisfactory
            step. On initial entry, a positive initial estimate
            must be provided.
        On exit stp is the current estimate of a satisfactory step
            if task = 'FG'. If task = 'CONV' then stp satisfies
            the sufficient decrease and curvature condition.

    f is a double precision variable.
        On initial entry f is the value of the function at 0.
        On subsequent entries f is the value of the
            function at stp.
        On exit f is the value of the function at stp.

    g is a double precision variable.
        On initial entry g is the derivative of the function at 0.
        On subsequent entries g is the derivative of the
           function at stp.
        On exit g is the derivative of the function at stp.

    ftol is a double precision variable.
        On entry ftol specifies a nonnegative tolerance for the
           sufficient decrease condition.
        On exit ftol is unchanged.

    gtol is a double precision variable.
        On entry gtol specifies a nonnegative tolerance for the
           curvature condition.
        On exit gtol is unchanged.

    xtol is a double precision variable.
        On entry xtol specifies a nonnegative relative tolerance
          for an acceptable step. The subroutine exits with a
          warning if the relative difference between sty and stx
          is less than xtol.
        On exit xtol is unchanged.

    task is a character variable of length at least 60.
        On initial entry task must be set to 'START'.
        On exit task indicates the required action:

           If task(1:2) = 'FG' then evaluate the function and
           derivative at stp and call dcsrch again.

           If task(1:4) = 'CONV' then the search is successful.

           If task(1:4) = 'WARN' then the subroutine is not able
           to satisfy the convergence conditions. The exit value of
           stp contains the best point found during the search.

           If task(1:5) = 'ERROR' then there is an error in the
           input arguments.

        On exit with convergence, a warning or an error, the
           variable task contains additional information.

    stpmin is a double precision variable.
        On entry stpmin is a nonnegative lower bound for the step.
        On exit stpmin is unchanged.

    stpmax is a double precision variable.
        On entry stpmax is a nonnegative upper bound for the step.
        On exit stpmax is unchanged.

    isave is an integer work array of dimension 2.

    dsave is a double precision work array of dimension 13.

    Subprograms called

      MINPACK-2 ... dcstep
    MINPACK-1 Project. June 1983.
    Argonne National Laboratory.
    Jorge J. More' and David J. Thuente.

    MINPACK-2 Project. November 1993.
    Argonne National Laboratory and University of Minnesota.
    Brett M. Averick, Richard G. Carter, and Jorge J. More'.
    """

    def __init__(self, phi, derphi, ftol, gtol, xtol, stpmin, stpmax):
        # 初始化函数，设置各类属性为 None
        self.stage = None
        self.ginit = None
        self.gtest = None
        self.gx = None
        self.gy = None
        self.finit = None
        self.fx = None
        self.fy = None
        self.stx = None
        self.sty = None
        self.stmin = None
        self.stmax = None
        self.width = None
        self.width1 = None

        # 设置收敛容限和极限值，首次调用对象时计算
        self.ftol = ftol
        self.gtol = gtol
        self.xtol = xtol
        self.stpmin = stpmin
        self.stpmax = stpmax

        # 设置 phi 和 derphi 函数用于计算
        self.phi = phi
        self.derphi = derphi

    def __call__(self, alpha1, phi0=None, derphi0=None, maxiter=100):
        """
        Parameters
        ----------
        alpha1 : float
            当前满意步骤的估计值。必须提供一个正的初始估计。
        phi0 : float
            在 0 处的 `phi` 的值（如果已知）。
        derphi0 : float
            在 0 处 `derphi` 的导数（如果已知）。
        maxiter : int

        Returns
        -------
        alpha : float
            步长，如果未找到合适的步骤则返回 None。
        phi : float
            在新点 `alpha` 处的 `phi` 的值。
        phi0 : float
            在 `alpha=0` 处的 `phi` 的值。
        task : bytes
            退出时任务状态信息。

           如果 task[:4] == b'CONV'，则搜索成功。

           如果 task[:4] == b'WARN'，则子程序无法满足收敛条件。搜索结束时，stp 包含搜索期间找到的最佳点。

           如果 task[:5] == b'ERROR'，则输入参数存在错误。
        """
        # 如果 phi0 未提供，则在 0 处计算 phi 的值
        if phi0 is None:
            phi0 = self.phi(0.0)
        # 如果 derphi0 未提供，则在 0 处计算 derphi 的导数
        if derphi0 is None:
            derphi0 = self.derphi(0.0)

        phi1 = phi0
        derphi1 = derphi0

        task = b"START"
        for i in range(maxiter):
            # 调用 _iterate 方法进行迭代，更新 stp、phi1、derphi1 和 task
            stp, phi1, derphi1, task = self._iterate(
                alpha1, phi1, derphi1, task
            )

            # 如果 stp 不是有限值，则标记为警告
            if not np.isfinite(stp):
                task = b"WARN"
                stp = None
                break

            # 根据 task 的前缀更新 alpha1、phi1 和 derphi1
            if task[:2] == b"FG":
                alpha1 = stp
                phi1 = self.phi(stp)
                derphi1 = self.derphi(stp)
            else:
                break
        else:
            # 达到最大迭代次数，线搜索未收敛
            stp = None
            task = b"WARNING: dcsrch did not converge within max iterations"

        # 如果 task 的前缀为 ERROR 或 WARN，则标记为失败
        if task[:5] == b"ERROR" or task[:4] == b"WARN":
            stp = None  # 失败

        return stp, phi1, phi0, task
# Subroutine dcstep
# 计算安全步长以及更新包含满足足够减少和曲率条件的步长区间。

# 参数说明：
# stx：当前得到的具有最小函数值的步长，是包含最小值的区间的一个端点。
# fx：stx 处的函数值，在输入时为 stx 处的函数值，在输出时也是。
# dx：stx 处的函数导数值。导数必须在步长方向上为负，即 dx 和 stp - stx 必须有相反的符号。在输出时，dx 仍然表示 stx 处的函数导数值。
# sty：包含最小值的区间的第二个端点，在输入时为 sty 的值，在输出时更新为新的 sty 的值。
# fy：sty 处的函数值，在输入时为 sty 处的函数值，在输出时也是。
# dy：sty 处的函数导数值，在输入时为 sty 处的函数导数值，在输出时仍然表示 sty 处的函数导数值。
# stp：当前的步长，在输入时为当前步长的值，在输出时为新的试验步长。
# fp：stp 处的函数值，在输入时为 stp 处的函数值，在输出时保持不变。
# dp：stp 处的函数导数值，在输入时为 stp 处的函数导数值，在输出时保持不变。
# brackt：逻辑变量，在输入时指示是否已经找到了极小值点的区间。初始时必须设置为 .false.。在输出时指示是否已经找到了极小值点的区间。当找到极小值点的区间时，设置为 .true.。
# stpmin：步长的下界，在输入时为步长的下界，在输出时保持不变。

def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """
    Subroutine dcstep

    This subroutine computes a safeguarded step for a search
    procedure and updates an interval that contains a step that
    satisfies a sufficient decrease and a curvature condition.

    The parameter stx contains the step with the least function
    value. If brackt is set to .true. then a minimizer has
    been bracketed in an interval with endpoints stx and sty.
    The parameter stp contains the current step.
    The subroutine assumes that if brackt is set to .true. then

        min(stx,sty) < stp < max(stx,sty),

    and that the derivative at stx is negative in the direction
    of the step.

    The subroutine statement is

      subroutine dcstep(stx,fx,dx,sty,fy,dy,stp,fp,dp,brackt,
                        stpmin,stpmax)

    where

    stx is a double precision variable.
        On entry stx is the best step obtained so far and is an
          endpoint of the interval that contains the minimizer.
        On exit stx is the updated best step.

    fx is a double precision variable.
        On entry fx is the function at stx.
        On exit fx is the function at stx.

    dx is a double precision variable.
        On entry dx is the derivative of the function at
          stx. The derivative must be negative in the direction of
          the step, that is, dx and stp - stx must have opposite
          signs.
        On exit dx is the derivative of the function at stx.

    sty is a double precision variable.
        On entry sty is the second endpoint of the interval that
          contains the minimizer.
        On exit sty is the updated endpoint of the interval that
          contains the minimizer.

    fy is a double precision variable.
        On entry fy is the function at sty.
        On exit fy is the function at sty.

    dy is a double precision variable.
        On entry dy is the derivative of the function at sty.
        On exit dy is the derivative of the function at the exit sty.

    stp is a double precision variable.
        On entry stp is the current step. If brackt is set to .true.
          then on input stp must be between stx and sty.
        On exit stp is a new trial step.

    fp is a double precision variable.
        On entry fp is the function at stp
        On exit fp is unchanged.

    dp is a double precision variable.
        On entry dp is the derivative of the function at stp.
        On exit dp is unchanged.

    brackt is an logical variable.
        On entry brackt specifies if a minimizer has been bracketed.
            Initially brackt must be set to .false.
        On exit brackt specifies if a minimizer has been bracketed.
            When a minimizer is bracketed brackt is set to .true.

    stpmin is a double precision variable.
        On entry stpmin is a lower bound for the step.
        On exit stpmin is unchanged.
    """
    stpmax is a double precision variable.
        On entry stpmax is an upper bound for the step.
        On exit stpmax is unchanged.

    MINPACK-1 Project. June 1983
    Argonne National Laboratory.
    Jorge J. More' and David J. Thuente.

    MINPACK-2 Project. November 1993.
    Argonne National Laboratory and University of Minnesota.
    Brett M. Averick and Jorge J. More'.

    """
    # 计算 dp 和 dx 的符号函数值
    sgn_dp = np.sign(dp)
    sgn_dx = np.sign(dx)

    # 计算 dp * (dx / abs(dx))，用于确定最终的符号
    sgnd = sgn_dp * sgn_dx

    # 第一种情况：函数值更高。最小值被夹在中间。
    # 如果立方步长比二次步长更接近 stx，则采用立方步长；否则采用立方步长和二次步长的平均值。
    if fp > fx:
        # 计算用于决策的 theta
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        # 计算 gamma，用于决定最终步长 stpf
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma *= -1
        # 计算 p 和 q，用于求解最终的步长 r
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        # 根据距离选择最终的步长 stpf
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        # 第二种情况：函数值较低且导数异号。最小值被夹在中间。
        # 如果立方步长比弦长步长更远离 stp，则采用立方步长；否则采用弦长步长。
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma *= -1
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        # 根据距离选择最终的步长 stpf
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        # 第三种情况：函数值更低，导数符号相同，并且导数的绝对值减小。

        # 仅在立方函数朝步长方向趋于无穷大或者立方函数的最小值超出stp时，计算立方步长。
        # 否则，采用割线步长。
        theta = 3 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        # 当 gamma = 0 时，只有在立方函数在步长方向上不趋于无穷大时才会出现。
        gamma = s * np.sqrt(max(0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)

        if brackt:
            # 已找到极小值点。如果立方步长比割线步长更接近stp，则采用立方步长，否则采用割线步长。
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq

            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            # 未找到极小值点。如果立方步长比割线步长更远离stp，则采用立方步长，否则采用割线步长。
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = np.clip(stpf, stpmin, stpmax)

    else:
        # 第四种情况：函数值更低，导数符号相同，并且导数的绝对值不减小。如果极小值点未被找到，
        # 步长将是stpmin或stpmax，否则采用立方步长。
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    # 更新包含极小值点的区间。
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    # 计算新的步长。
    stp = stpf

    # 返回更新后的参数 stx, fx, dx, sty, fy, dy, stp, brackt
    return stx, fx, dx, sty, fy, dy, stp, brackt
```