# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\trlib_krylov.c`

```
/*
 * MIT License
 *
 * Copyright (c) 2016--2017 Felix Lenders
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#include "trlib_private.h"  // 包含私有头文件 trlib_private.h
#include "trlib.h"          // 包含公共头文件 trlib.h

// 定义 trlib_krylov_min_internal 函数，返回 trlib_int_t 类型值
trlib_int_t trlib_krylov_min_internal(
    trlib_int_t init, trlib_flt_t radius, trlib_int_t equality, trlib_int_t itmax, trlib_int_t itmax_lanczos,
    trlib_flt_t tol_rel_i, trlib_flt_t tol_abs_i,
    trlib_flt_t tol_rel_b, trlib_flt_t tol_abs_b, trlib_flt_t zero, trlib_flt_t obj_lo,
    trlib_int_t ctl_invariant, trlib_int_t convexify, trlib_int_t earlyterm,
    trlib_flt_t g_dot_g, trlib_flt_t v_dot_g, trlib_flt_t p_dot_Hp,
    trlib_int_t *iwork, trlib_flt_t *fwork, trlib_int_t refine,
    trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout, trlib_int_t *timing,
    trlib_int_t *action, trlib_int_t *iter, trlib_int_t *ityp,
    trlib_flt_t *flt1, trlib_flt_t *flt2, trlib_flt_t *flt3) {
    /* The algorithm runs by solving the trust region subproblem restricted to a Krylov subspace K(ii)
       The Krylov space K(ii) can be either described by the pCG iterates: (notation iM = M^-1)
         K(ii) = span(p_0, ..., p_ii)
       and in an equivalent way by the Lanczos iterates
         K(ii) = span(q_0, ..., q_ii)

       In one iteration the algorithms performs the following steps
       (a) expand K(ii-1) to K(ii):
           if done via pCG:
             alpha = (g, v)/(p, H p); g+ = g + alpha H p; v+ = iM g; beta = (g+, v+)/(g, v); p+ = -v+ + beta p
           if done via Lanczos:
             y = iM t; gamma = sq (t, y); w = t/gamma; q = y/gamma; delta = (q, H q); t+ = Hq - delta w - gamma w-
           we use pCG as long as it does not break down (alpha ~ 0) and continue with Lanczos in that case,
           note the relationship q = v/sq (g, v) * +-1
       (b) compute minimizer s of problem restricted to sample Krylov space K(ii)
           check if this minimizer is expected to be interior:
             do the pCG iterates satisfy the trust region constraint?
             is H positive definite on K(ii), i.e. are all alphas >= 0?
           if the minimizer is interior, set s = p
           if the minimizer is expected on the boundary, set s = Q*h with Q = [q_0, ..., q_ii]
             and let s solve a tridiagonal trust region subproblem with hessian the tridiagonal matrix
             T_ii from the Lanczos process,
             diag(T_ii) = (delta_0, ..., delta_ii) and offdiag(T_ii) = (gamma_1, ..., gamma_ii)
       (c) test for convergence */

    // Pointer to store the timing information
    trlib_int_t *leftmost_timing = NULL;

    // Conditional compilation for time measurement
    #if TRLIB_MEASURE_TIME
        // Variables to store time measurements
        struct timespec verystart, start, end;
        // Assign timing pointer to a specific index in the timing array
        leftmost_timing = timing + 1;
        // Start measuring time
        TRLIB_TIC(verystart)
    #endif

    // Workspace variables with descriptive names
    trlib_int_t *status = iwork;
    trlib_int_t *ii = iwork+1;
    trlib_int_t *pos_def = iwork+2;
    trlib_int_t *interior = iwork+3;
    trlib_int_t *warm_leftmost = iwork+4;
    trlib_int_t *ileftmost = iwork+5;
    trlib_int_t *warm_lam0 = iwork+6;
    trlib_int_t *warm_lam = iwork+7;
    trlib_int_t *lanczos_switch = iwork+8;
    trlib_int_t *exit_tri = iwork+9;
    trlib_int_t *sub_fail_tri = iwork+10;
    trlib_int_t *iter_tri = iwork+11;
    trlib_int_t *iter_last_head = iwork+12;
    trlib_int_t *type_last_head = iwork+13;
    trlib_int_t *nirblk = iwork + 15;
    trlib_int_t *irblk = iwork+16;

    // Float workspace variables
    trlib_flt_t *stop_i = fwork;
    trlib_flt_t *stop_b = fwork+1;
    trlib_flt_t *v_g = fwork+2;
    trlib_flt_t *p_Hp = fwork+3;
    trlib_flt_t *cgl = fwork+4;
    trlib_flt_t *cglm = fwork+5;
    trlib_flt_t *lam0 = fwork+6;
    trlib_flt_t *lam = fwork+7;
    trlib_flt_t *obj = fwork+8;
    trlib_flt_t *s_Mp = fwork+9;
    trlib_flt_t *p_Mp = fwork+10;
    trlib_flt_t *s_Ms = fwork+11;
    trlib_flt_t *sigma = fwork+12;
    trlib_flt_t *raymax = fwork+13;
    trlib_flt_t *raymin = fwork+14;
    trlib_flt_t *alpha = fwork+15;
    // Initialize pointers to specific positions within fwork array
    trlib_flt_t *beta = fwork+15+itmax+1; // Pointer to beta array within fwork
    trlib_flt_t *neglin = fwork+15+2*(itmax+1); // Pointer to neglin array within fwork
    trlib_flt_t *h0 = fwork+15+3*(itmax+1); // Pointer to h0 array within fwork
    trlib_flt_t *h = fwork+15+4*(itmax+1); // Pointer to h array within fwork
    trlib_flt_t *delta =  fwork+15+5*(itmax+1); // Pointer to delta array within fwork
    trlib_flt_t *delta_fac0 = fwork+15+6*(itmax+1); // Pointer to delta_fac0 array within fwork
    trlib_flt_t *delta_fac = fwork+15+7*(itmax+1); // Pointer to delta_fac array within fwork
    trlib_flt_t *gamma = fwork+15+8*(itmax+1); // Pointer to gamma array within fwork (note: shifted by 1)
    trlib_flt_t *gamma_fac0 = fwork+15+8+9*itmax; // Pointer to gamma_fac0 array within fwork
    trlib_flt_t *gamma_fac = fwork+15+8+10*itmax; // Pointer to gamma_fac array within fwork
    trlib_flt_t *ones = fwork+15+8+11*itmax; // Pointer to ones array within fwork
    trlib_flt_t *leftmost = fwork+15+9+12*itmax; // Pointer to leftmost array within fwork
    trlib_flt_t *regdiag = fwork+15+10+13*itmax; // Pointer to regdiag array within fwork
    trlib_flt_t *convhist = fwork+15+11+14*itmax; // Pointer to convhist array within fwork
    trlib_flt_t *fwork_tr = fwork+15+12+15*itmax; // Pointer to fwork_tr array within fwork

    // Initialize local variables
    trlib_int_t returnvalue = TRLIB_CLR_CONTINUE; // Return value indicating continue
    trlib_int_t warm_fac0 = 0; // Flag indicating successful update of factorization (for delta_fac0)
    trlib_int_t warm_fac = 0; // Flag indicating successful update of factorization (for delta_fac)
    trlib_int_t inc = 1; // Increment value
    trlib_flt_t one = 1.0; // Constant value 1.0
    trlib_flt_t minus = -1.0; // Constant value -1.0
    trlib_flt_t sp_Msp = 0.0; // (s+, Ms+), initialized to 0.0
    trlib_flt_t eta_i = 0.0; // Forcing parameter eta_i, initialized to 0.0
    trlib_flt_t eta_b = 0.0; // Forcing parameter eta_b, initialized to 0.0
    trlib_int_t cit = 0;     // Loop counter for convergence history, initialized to 0

    *iter = *ii; // Set iter to the value pointed by ii

    // Set status based on the value of init
    if (init == TRLIB_CLS_INIT)       { *status = TRLIB_CLS_INIT; }
    if (init == TRLIB_CLS_HOTSTART)   { *status = TRLIB_CLS_HOTSTART; }
    if (init == TRLIB_CLS_HOTSTART_P) { *status = TRLIB_CLS_HOTSTART_P; }
    if (init == TRLIB_CLS_HOTSTART_G) { *status = TRLIB_CLS_HOTSTART_G; }
    if (init == TRLIB_CLS_HOTSTART_T) { *status = TRLIB_CLS_HOTSTART_T; }
    if (init == TRLIB_CLS_HOTSTART_R) { *status = TRLIB_CLS_HOTSTART_R; }

    // Return from the function with returnvalue
    TRLIB_RETURN(returnvalue)
    // 函数 trlib_krylov_min 开始
    trlib_int_t trlib_krylov_min(
        // 初始化标志，初始搜索半径，等式约束标志，最大迭代次数，Lanczos 迭代最大次数
        trlib_int_t init, trlib_flt_t radius, trlib_int_t equality, trlib_int_t itmax, trlib_int_t itmax_lanczos,
        // 相对容差和绝对容差（初始化和后续迭代）
        trlib_flt_t tol_rel_i, trlib_flt_t tol_abs_i,
        // 相对容差和绝对容差（约束和后续迭代）
        trlib_flt_t tol_rel_b, trlib_flt_t tol_abs_b, trlib_flt_t zero, trlib_flt_t obj_lo,
        // 控制不变性标志，凸化标志，提前终止标志
        trlib_int_t ctl_invariant, trlib_int_t convexify, trlib_int_t earlyterm,
        // g_dot_g, v_dot_g, p_dot_Hp 用于迭代过程中的一些标量
        trlib_flt_t g_dot_g, trlib_flt_t v_dot_g, trlib_flt_t p_dot_Hp,
        // 整数工作区和浮点数工作区，用于存储中间结果和迭代信息
        trlib_int_t *iwork, trlib_flt_t *fwork, trlib_int_t refine,
        // 详细输出标志，Unicode 输出标志，输出前缀字符串，输出文件指针，计时信息
        trlib_int_t verbose, trlib_int_t unicode, char *prefix, FILE *fout, trlib_int_t *timing,
        // 动作标志，迭代次数，迭代类型
        trlib_int_t *action, trlib_int_t *iter, trlib_int_t *ityp,
        // 三个额外的浮点数变量，用于特定计算或输出
        trlib_flt_t *flt1, trlib_flt_t *flt2, trlib_flt_t *flt3) {

        // 返回值，默认为 -1000
        trlib_int_t ret = -1000;

        // 外部状态指针，指向整数工作区的第 14 个元素
        trlib_int_t *outerstatus = iwork + 14;
        // 更新迭代次数
        *iter = *(iwork + 1);
        
        // 如果初始化标志为 TRLIB_CLS_INIT 或 TRLIB_CLS_HOTSTART，则将外部状态置为 0
        if (init == TRLIB_CLS_INIT || init == TRLIB_CLS_HOTSTART) { *outerstatus = 0; }

        // 如果外部状态小于 100 或等于 300，则执行以下循环
        if (*outerstatus < 100 || *outerstatus == 300) {
            while (1) {
                // 调用内部函数 trlib_krylov_min_internal 进行 Krylov 迭代最小化
                ret = trlib_krylov_min_internal(init, radius, equality, itmax, itmax_lanczos,
                                                tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b, zero, obj_lo,
                                                ctl_invariant, convexify, earlyterm, g_dot_g, v_dot_g, p_dot_Hp,
                                                iwork, fwork, refine, verbose, unicode, prefix, fout, timing,
                                                action, iter, ityp, flt1, flt2, flt3);

                // 如果初始化大于 0 或返回值小于 10 或动作标志不是 TRLIB_CLA_TRIVIAL，则跳出循环
                if (init > 0 || ret < 10 || *action != TRLIB_CLA_TRIVIAL) { break; }
            }
        }
    // 函数 trlib_krylov_min 结束
    }
    # 如果返回值大于等于0或者等于-1000
    if( ret >= 0 || ret == -1000 ) {

        # 如果外部状态小于100并且返回值小于10并且动作不是TRIVIAL，则设置外部状态为100+ret，并返回10
        if( *outerstatus < 100 && ret < 10 && *action != TRLIB_CLA_TRIVIAL ) { *outerstatus = 100 + ret; return 10; }
        
        # 如果外部状态在100到200之间
        if( *outerstatus >= 100 && *outerstatus < 200 ) { ret = *outerstatus - 100; *outerstatus = 0; *action = TRLIB_CLA_TRIVIAL; }

        # 如果返回值小于10并且外部状态小于100并且convexify为真
        if( ret < 10 && *outerstatus < 100 && convexify ) {
            // 退出，检查是否需要进行凸化操作
            # 从fwork数组中取出lam值
            trlib_flt_t lam = fwork[7];
            # 如果lam大于1e-2乘以fwork[13]的最大值，并且fwork[14]小于0并且fabs(fwork[14])小于1e-8乘以fwork[13]
            # 只有在基于特征值估计有意义时才执行
            if( lam > 1e-2*fmax(1.0, fwork[13]) && fwork[14] < 0.0 && fabs(fwork[14]) < 1e-8 * fwork[13]) {
                // 请求调用者计算目标函数值
                *outerstatus = 200 + ret;
                *action = TRLIB_CLA_OBJVAL;
                return 10;
            }
        }

        # 如果外部状态大于200并且小于300
        if( *outerstatus > 200 && *outerstatus < 300 ) {
            # 从fwork数组中取出obj和g_dot_g值
            trlib_flt_t obj = fwork[8];
            trlib_flt_t realobj = g_dot_g;
            # 如果obj与realobj的绝对值大于fmax(1e-6, 1e-1乘以fabs(realobj))或者realobj大于0
            if( fabs(obj - realobj) > fmax(1e-6, 1e-1*fabs(realobj)) || realobj > 0.0) {
                # 打印左侧最值、lam、射线最大值和射线最小值的信息
                TRLIB_PRINTLN_2("leftmost: %e lam: %e raymax: %e raymin: %e\n", fwork[24+12*itmax], fwork[7], fwork[13], fwork[14])
                # 打印从三对角解中预测的目标值与实际计算值之间的不匹配
                TRLIB_PRINTLN_2("mismatch between objective value as predicted from tridiagonal solution and actually computed: tridiag: %e, actual: %e\n", obj, realobj)
                # 打印使用正则化Hessian重新计算的信息
                TRLIB_PRINTLN_2("recomputing with regularized hessian\n");
                # 初始化为热启动
                init = TRLIB_CLS_HOTSTART_P;
                # 调用内部的Krylov最小化函数
                ret = trlib_krylov_min_internal(init, radius, equality, itmax, itmax_lanczos,
                        tol_rel_i, tol_abs_i, tol_rel_b, tol_abs_b, zero, obj_lo,
                        ctl_invariant, convexify, earlyterm, g_dot_g, v_dot_g, p_dot_Hp,
                        iwork, fwork, refine, verbose, unicode, prefix, fout, timing,
                        action, iter, ityp, flt1, flt2, flt3);
                *outerstatus = 300;
                return ret;
            }
            else {
                ret = *outerstatus - 200;
                *outerstatus = 0;
                return ret;
            }

        }

        # 如果外部状态等于300并且返回值小于10
        if( *outerstatus == 300 && ret < 10 ) { *outerstatus = 0; return ret; }

    }

    # 返回原始的返回值ret
    return ret;
}

// 准备 Krylov 方法所需的内存空间
trlib_int_t trlib_krylov_prepare_memory(trlib_int_t itmax, trlib_flt_t *fwork) {
    trlib_int_t jj = 0;
    // 将 fwork 中的部分元素设为 1.0
    for(jj = 23+11*itmax; jj<24+12*itmax; ++jj) { *(fwork+jj) = 1.0; } // everything to 1.0 in ones
    // 将 fwork 中的一部分元素设为 0，用于负线性组合
    memset(fwork+17+2*itmax, 0, itmax*sizeof(trlib_flt_t)); // neglin = - gamma_0 e1, thus set neglin[1:] = 0
    // 返回操作成功的标志
    return 0;
}

// 计算 Krylov 方法所需的内存空间大小
trlib_int_t trlib_krylov_memory_size(trlib_int_t itmax, trlib_int_t *iwork_size, trlib_int_t *fwork_size, trlib_int_t *h_pointer) {
    // 计算整型工作区大小
    *iwork_size = 17+itmax;
    // 计算浮点数工作区大小
    *fwork_size = 27+15*itmax+trlib_tri_factor_memory_size(itmax+1);
    // 计算 h_pointer 的值
    *h_pointer = 19+4*itmax;
    // 返回操作成功的标志
    return 0;
}

// 计算 Krylov 方法的时间测量大小
trlib_int_t trlib_krylov_timing_size() {
#if TRLIB_MEASURE_TIME
    // 如果启用时间测量，返回 trlib_tri_timing_size() 的大小加一
    return 1 + trlib_tri_timing_size();
#endif
    // 否则返回 0
    return 0;
}

// 获取 Krylov 方法中的某个指针位置
trlib_int_t trlib_krylov_gt(trlib_int_t itmax, trlib_int_t *gt_pointer) {
    // 设置 gt_pointer 的值
    *gt_pointer = 17 + 2*itmax;
    // 返回操作成功的标志
    return 0;
}
```