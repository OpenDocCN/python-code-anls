# `D:\src\scipysrc\scikit-learn\sklearn\svm\src\liblinear\tron.cpp`

```
    // 定义默认参数 eta0, eta1, eta2 用于更新迭代步长
    double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

    // 定义默认参数 sigma1, sigma2, sigma3 用于更新信任区域大小 delta
    double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

    // 获取特征个数 n，即优化问题中的变量个数
    int n = fun_obj->get_nr_variable();

    // 初始化迭代次数和共轭梯度法迭代次数
    int iter = 1, cg_iter;

    // 初始化步长 delta 和梯度范数 snorm
    double delta, snorm;

    // 初始化 alpha, f, fnew, prered, actred 和 gs，其中 f 为目标函数值
    double alpha, f, fnew, prered, actred, gs;

    // 初始化搜索标志，迭代次数增量和 w_new 数组
    int search = 1, inc = 1;
    double *s = new double[n];
    double *r = new double[n];
    double *w_new = new double[n];
    double *g = new double[n];

    // 将 w 数组的所有元素初始化为 0
    for (i=0; i<n; i++)
        w[i] = 0;

    // 计算初始目标函数值 f 和其梯度 g
    f = fun_obj->fun(w);
    fun_obj->grad(w, g);

    // 计算初始梯度范数 delta
    delta = blas->nrm2(n, g, inc);
    double gnorm1 = delta;
    double gnorm = gnorm1;

    // 如果初始梯度范数小于等于 eps 乘以初始梯度范数 gnorm1，则停止搜索
    if (gnorm <= eps*gnorm1)
        search = 0;

    // 初始化迭代次数为 1
    iter = 1;

    // 进入迭代过程，迭代次数不超过 max_iter 并且仍在搜索中
    while (iter <= max_iter && search)
    {
        // 调用trcg函数计算共轭梯度迭代器
        cg_iter = trcg(delta, g, s, r);
    
        // 复制w数组到w_new数组，w_new是w的副本
        memcpy(w_new, w, sizeof(double)*n);
    
        // 将s向量与w_new向量相加，结果存入w_new中
        blas->axpy(n, 1.0, s, inc, w_new, inc);
    
        // 计算向量g和向量s的内积
        gs = blas->dot(n, g, inc, s, inc);
    
        // 计算预测值prered，基于共轭梯度和r的内积
        prered = -0.5*(gs - blas->dot(n, s, inc, r, inc));
    
        // 计算使用w_new求得的新的目标函数值
        fnew = fun_obj->fun(w_new);
    
        // 计算实际减少量
        actred = f - fnew;
    
        // 若是第一次迭代，调整初始步长限制delta
        snorm = blas->nrm2(n, s, inc);
        if (iter == 1)
            delta = min(delta, snorm);
    
        // 根据函数减少量计算步长预测值alpha*snorm
        if (fnew - f - gs <= 0)
            alpha = sigma3;
        else
            alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));
    
        // 根据实际减少量和预测减少量比率更新信任域限制delta
        if (actred < eta0*prered)
            delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
        else if (actred < eta1*prered)
            delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
        else if (actred < eta2*prered)
            delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
        else
            delta = max(delta, min(alpha*snorm, sigma3*delta));
    
        // 输出迭代信息，包括迭代次数、实际减少量、预测减少量、当前步长delta、目标函数值f、梯度范数gnorm和共轭梯度迭代次数cg_iter
        info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);
    
        // 如果实际减少量大于预测减少量的某个阈值，则更新w、f、g并进行下一次迭代
        if (actred > eta0*prered)
        {
            iter++;
            memcpy(w, w_new, sizeof(double)*n);
            fun_obj->grad(w, g);
    
            gnorm = blas->nrm2(n, g, inc);
            if (gnorm <= eps*gnorm1)
                break;
        }
    
        // 如果目标函数值f小于-1.0e+32，则输出警告信息并终止迭代
        if (f < -1.0e+32)
        {
            info("WARNING: f < -1.0e+32\n");
            break;
        }
    
        // 如果实际减少量和预测减少量均小于等于0，则输出警告信息并终止迭代
        if (fabs(actred) <= 0 && prered <= 0)
        {
            info("WARNING: actred and prered <= 0\n");
            break;
        }
    
        // 如果实际减少量和预测减少量的绝对值均小于目标函数值f的一定比例，则输出警告信息并终止迭代
        if (fabs(actred) <= 1.0e-12*fabs(f) &&
            fabs(prered) <= 1.0e-12*fabs(f))
        {
            info("WARNING: actred and prered too small\n");
            break;
        }
    }
    
    // 释放动态分配的内存：g、r、w_new和s
    delete[] g;
    delete[] r;
    delete[] w_new;
    delete[] s;
    
    // 返回迭代次数减一（因为最后一次迭代未完成）
    return --iter;
}

// TRON 类的 trcg 方法实现，用于执行共轭梯度法优化步骤
int TRON::trcg(double delta, double *g, double *s, double *r)
{
    int i, inc = 1;
    int n = fun_obj->get_nr_variable();
    double *d = new double[n];  // 定义一个长度为 n 的双精度数组 d
    double *Hd = new double[n];  // 定义一个长度为 n 的双精度数组 Hd
    double rTr, rnewTrnew, alpha, beta, cgtol;

    for (i=0; i<n; i++)
    {
        s[i] = 0;  // 初始化数组 s 的每个元素为 0
        r[i] = -g[i];  // 初始化数组 r 为负梯度 g
        d[i] = r[i];  // 初始化数组 d 为数组 r 的拷贝
    }
    cgtol = 0.1 * blas->nrm2(n, g, inc);  // 计算共轭梯度法的收敛容差

    int cg_iter = 0;  // 初始化共轭梯度法的迭代次数为 0
    rTr = blas->dot(n, r, inc, r, inc);  // 计算 r 的二范数的平方
    while (1)
    {
        if (blas->nrm2(n, r, inc) <= cgtol)  // 如果 r 的二范数小于等于收敛容差，则退出循环
            break;
        cg_iter++;
        fun_obj->Hv(d, Hd);  // 计算 Hessian 矩阵对向量 d 的乘积 Hd

        alpha = rTr / blas->dot(n, d, inc, Hd, inc);  // 计算 alpha，用于共轭梯度法的线性搜索
        blas->axpy(n, alpha, d, inc, s, inc);  // s = s + alpha * d
        if (blas->nrm2(n, s, inc) > delta)  // 如果 s 的二范数大于 trust region 半径 delta
        {
            info("cg reaches trust region boundary\n");  // 输出信息，表示共轭梯度法达到了 trust region 的边界
            alpha = -alpha;  // 反转 alpha
            blas->axpy(n, alpha, d, inc, s, inc);  // s = s + alpha * d

            double std = blas->dot(n, s, inc, d, inc);  // 计算 s 和 d 的内积
            double sts = blas->dot(n, s, inc, s, inc);  // 计算 s 的二范数的平方
            double dtd = blas->dot(n, d, inc, d, inc);  // 计算 d 的二范数的平方
            double dsq = delta*delta;  // 计算 delta 的平方
            double rad = sqrt(std*std + dtd*(dsq-sts));  // 计算根号内的值
            if (std >= 0)
                alpha = (dsq - sts)/(std + rad);  // 计算新的 alpha
            else
                alpha = (rad - std)/dtd;  // 计算新的 alpha
            blas->axpy(n, alpha, d, inc, s, inc);  // s = s + alpha * d
            alpha = -alpha;  // 反转 alpha
            blas->axpy(n, alpha, Hd, inc, r, inc);  // r = r + alpha * Hd
            break;  // 跳出循环
        }
        alpha = -alpha;  // 反转 alpha
        blas->axpy(n, alpha, Hd, inc, r, inc);  // r = r + alpha * Hd
        rnewTrnew = blas->dot(n, r, inc, r, inc);  // 计算更新后的 r 的二范数的平方
        beta = rnewTrnew/rTr;  // 计算 beta
        blas->scal(n, beta, d, inc);  // d = beta * d
        blas->axpy(n, 1.0, r, inc, d, inc);  // d = d + r
        rTr = rnewTrnew;  // 更新 r 的二范数的平方
    }

    delete[] d;  // 释放数组 d 的内存空间
    delete[] Hd;  // 释放数组 Hd 的内存空间

    return(cg_iter);  // 返回共轭梯度法的迭代次数
}

// TRON 类的 norm_inf 方法，用于计算向量 x 的无穷范数
double TRON::norm_inf(int n, double *x)
{
    double dmax = fabs(x[0]);  // 初始化 dmax 为向量 x 的第一个元素的绝对值
    for (int i=1; i<n; i++)
        if (fabs(x[i]) >= dmax)
            dmax = fabs(x[i]);  // 更新 dmax 的值
    return(dmax);  // 返回向量 x 的无穷范数
}

// TRON 类的 set_print_string 方法，用于设置打印字符串的函数指针
void TRON::set_print_string(void (*print_string) (const char *buf))
{
    tron_print_string = print_string;  // 设置 tron_print_string 为指定的打印字符串函数指针
}
```