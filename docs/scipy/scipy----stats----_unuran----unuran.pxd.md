# `D:\src\scipysrc\scipy\scipy\stats\_unuran\unuran.pxd`

```
# File automatically generated using autopxd2

# 导入需要的头文件
from libc.stdio cimport FILE

# 定义 UNUR 库中的结构体
cdef extern from "unuran.h" nogil:

    cdef struct unur_distr
    ctypedef unur_distr UNUR_DISTR

    cdef struct unur_par
    ctypedef unur_par UNUR_PAR

    cdef struct unur_gen
    ctypedef unur_gen UNUR_GEN

    cdef struct unur_urng
    ctypedef unur_urng UNUR_URNG

    # 定义连续分布函数指针类型
    ctypedef double UNUR_FUNCT_CONT(double x, unur_distr* distr)

    # 定义离散分布函数指针类型
    ctypedef double UNUR_FUNCT_DISCR(int x, unur_distr* distr)

    # 定义离散分布反函数指针类型
    ctypedef int UNUR_IFUNCT_DISCR(double x, unur_distr* distr)

    # 定义连续向量函数指针类型
    ctypedef double UNUR_FUNCT_CVEC(double* x, unur_distr* distr)

    # 定义连续向量函数指针类型（返回值为整型）
    ctypedef int UNUR_VFUNCT_CVEC(double* result, double* x, unur_distr* distr)

    # 定义连续向量函数指针类型（带坐标）
    ctypedef double UNUR_FUNCTD_CVEC(double* x, int coord, unur_distr* distr)

    cdef struct unur_slist

    # 定义错误处理函数指针类型
    ctypedef void UNUR_ERROR_HANDLER(char* objid, char* file, int line, char* errortype, int unur_errno, char* reason)

    # 获取默认的随机数生成器
    UNUR_URNG* unur_get_default_urng()

    # 设置默认的随机数生成器
    UNUR_URNG* unur_set_default_urng(UNUR_URNG* urng_new)

    # 设置辅助的默认随机数生成器
    UNUR_URNG* unur_set_default_urng_aux(UNUR_URNG* urng_new)

    # 获取辅助的默认随机数生成器
    UNUR_URNG* unur_get_default_urng_aux()

    # 设置随机数生成器
    int unur_set_urng(UNUR_PAR* parameters, UNUR_URNG* urng)

    # 更改随机数生成器
    UNUR_URNG* unur_chg_urng(UNUR_GEN* generator, UNUR_URNG* urng)

    # 获取随机数生成器
    UNUR_URNG* unur_get_urng(UNUR_GEN* generator)

    # 设置辅助随机数生成器
    int unur_set_urng_aux(UNUR_PAR* parameters, UNUR_URNG* urng_aux)

    # 使用默认的辅助随机数生成器
    int unur_use_urng_aux_default(UNUR_PAR* parameters)

    # 更改为默认的辅助随机数生成器
    int unur_chgto_urng_aux_default(UNUR_GEN* generator)

    # 更改辅助随机数生成器
    UNUR_URNG* unur_chg_urng_aux(UNUR_GEN* generator, UNUR_URNG* urng_aux)

    # 获取辅助随机数生成器
    UNUR_URNG* unur_get_urng_aux(UNUR_GEN* generator)

    # 生成随机数
    double unur_urng_sample(UNUR_URNG* urng)

    # 生成随机数
    double unur_sample_urng(UNUR_GEN* gen)

    # 生成数组随机数
    int unur_urng_sample_array(UNUR_URNG* urng, double* X, int dim)

    # 重置随机数
    int unur_urng_reset(UNUR_URNG* urng)

    # 同步随机数
    int unur_urng_sync(UNUR_URNG* urng)

    # 种子随机数
    int unur_urng_seed(UNUR_URNG* urng, unsigned long seed)

    # 抗变随机数
    int unur_urng_anti(UNUR_URNG* urng, int anti)

    # 下一随机数
    int unur_urng_nextsub(UNUR_URNG* urng)

    # 重置下一随机数
    int unur_urng_resetsub(UNUR_URNG* urng)

    # 同步创建
    int unur_gen_sync(UNUR_GEN* generator)

    # 生成种子
    int unur_gen_seed(UNUR_GEN* generator, unsigned long seed)

    # 抗变数据
    int unur_gen_anti(UNUR_GEN* generator, int anti)

    # 重置数
    int unur_gen_reset(UNUR_GEN* generator)

    # 下一输出
    int unur_gen_nextsub(UNUR_GEN* generator)

    # 重置输出
    int unur_gen_resetsub(UNUR_GEN* generator)

    # 新的应用数据
    ctypedef double (*_unur_urng_new_sampleunif_ft)(void* state)

    # 新的样本
    UNUR_URNG* unur_urng_new(_unur_urng_new_sampleunif_ft sampleunif, void* state)

    # 释放应用
    void unur_urng_free(UNUR_URNG* urng)

    # 创建新的列表
    ctypedef unsigned int (*_unur_urng_set_sample_array_samplearray_ft)(void* state, double* X, int dim)

    # 创建新的数量
    int unur_urng_set_sample_array(UNUR_URNG* urng, _unur_urng_set_sample_array_samplearray_ft samplearray)

    # 同步创建
    ctypedef void (*_unur_urng_set_sync_sync_ft)(void* state)

    # 生成新的输出
    int unur_urng_set_sync(UNUR_URNG* urng, _unur_urng_set_sync_sync_ft sync)
    # 定义一个函数指针类型 _unur_urng_set_seed_setseed_ft，该函数接受一个状态指针和一个无符号长整型种子参数，无返回值
    ctypedef void (*_unur_urng_set_seed_setseed_ft)(void* state, unsigned long seed)

    # 设置随机数生成器 urng 的种子设置函数为 setseed，返回整型值
    int unur_urng_set_seed(UNUR_URNG* urng, _unur_urng_set_seed_setseed_ft setseed)

    # 定义一个函数指针类型 _unur_urng_set_anti_setanti_ft，该函数接受一个状态指针和一个整型抗变量参数，无返回值
    ctypedef void (*_unur_urng_set_anti_setanti_ft)(void* state, int anti)

    # 设置随机数生成器 urng 的抗变量设置函数为 setanti，返回整型值
    int unur_urng_set_anti(UNUR_URNG* urng, _unur_urng_set_anti_setanti_ft setanti)

    # 定义一个函数指针类型 _unur_urng_set_reset_reset_ft，该函数接受一个状态指针，无返回值
    ctypedef void (*_unur_urng_set_reset_reset_ft)(void* state)

    # 设置随机数生成器 urng 的重置函数为 reset，返回整型值
    int unur_urng_set_reset(UNUR_URNG* urng, _unur_urng_set_reset_reset_ft reset)

    # 定义一个函数指针类型 _unur_urng_set_nextsub_nextsub_ft，该函数接受一个状态指针，无返回值
    ctypedef void (*_unur_urng_set_nextsub_nextsub_ft)(void* state)

    # 设置随机数生成器 urng 的下一子序列函数为 nextsub，返回整型值
    int unur_urng_set_nextsub(UNUR_URNG* urng, _unur_urng_set_nextsub_nextsub_ft nextsub)

    # 定义一个函数指针类型 _unur_urng_set_resetsub_resetsub_ft，该函数接受一个状态指针，无返回值
    ctypedef void (*_unur_urng_set_resetsub_resetsub_ft)(void* state)

    # 设置随机数生成器 urng 的重置子序列函数为 resetsub，返回整型值
    int unur_urng_set_resetsub(UNUR_URNG* urng, _unur_urng_set_resetsub_resetsub_ft resetsub)

    # 定义一个函数指针类型 _unur_urng_set_delete_fpdelete_ft，该函数接受一个状态指针，无返回值
    ctypedef void (*_unur_urng_set_delete_fpdelete_ft)(void* state)

    # 设置随机数生成器 urng 的删除函数为 fpdelete，返回整型值
    int unur_urng_set_delete(UNUR_URNG* urng, _unur_urng_set_delete_fpdelete_ft fpdelete)

    # 定义一个枚举类型，包含连续分布、离散分布等各种分布类型
    cdef enum:
        UNUR_DISTR_CONT    # 连续分布
        UNUR_DISTR_CEMP    # 自定义经验分布
        UNUR_DISTR_CVEC    # 自定义向量分布
        UNUR_DISTR_CVEMP   # 自定义向量经验分布
        UNUR_DISTR_MATR    # 矩阵分布
        UNUR_DISTR_DISCR   # 离散分布

    # 释放分布对象 distribution 的内存
    void unur_distr_free(UNUR_DISTR* distribution)

    # 设置分布对象 distribution 的名称为 name，返回整型值
    int unur_distr_set_name(UNUR_DISTR* distribution, char* name)

    # 获取分布对象 distribution 的名称，返回字符串指针
    char* unur_distr_get_name(UNUR_DISTR* distribution)

    # 获取分布对象 distribution 的维度，返回整型值
    int unur_distr_get_dim(UNUR_DISTR* distribution)

    # 获取分布对象 distribution 的类型，返回无符号整型值
    unsigned int unur_distr_get_type(UNUR_DISTR* distribution)

    # 检查分布对象 distribution 是否为连续分布，返回整型值
    int unur_distr_is_cont(UNUR_DISTR* distribution)

    # 检查分布对象 distribution 是否为向量分布，返回整型值
    int unur_distr_is_cvec(UNUR_DISTR* distribution)

    # 检查分布对象 distribution 是否为经验分布，返回整型值
    int unur_distr_is_cemp(UNUR_DISTR* distribution)

    # 检查分布对象 distribution 是否为向量经验分布，返回整型值
    int unur_distr_is_cvemp(UNUR_DISTR* distribution)

    # 检查分布对象 distribution 是否为离散分布，返回整型值
    int unur_distr_is_discr(UNUR_DISTR* distribution)

    # 检查分布对象 distribution 是否为矩阵分布，返回整型值
    int unur_distr_is_matr(UNUR_DISTR* distribution)

    # 设置分布对象 distribution 的外部对象为 extobj，返回整型值
    int unur_distr_set_extobj(UNUR_DISTR* distribution, void* extobj)

    # 获取分布对象 distribution 的外部对象，返回指向该对象的指针
    void* unur_distr_get_extobj(UNUR_DISTR* distribution)

    # 克隆分布对象 distr，返回克隆后的分布对象指针
    UNUR_DISTR* unur_distr_clone(UNUR_DISTR* distr)

    # 创建一个新的自定义经验分布对象，返回分布对象指针
    UNUR_DISTR* unur_distr_cemp_new()

    # 设置自定义经验分布对象 distribution 的数据，返回整型值
    int unur_distr_cemp_set_data(UNUR_DISTR* distribution, double* sample, int n_sample)

    # 从文件 filename 中读取数据到自定义经验分布对象 distribution，返回整型值
    int unur_distr_cemp_read_data(UNUR_DISTR* distribution, char* filename)

    # 获取自定义经验分布对象 distribution 的数据，返回数据的双重指针
    int unur_distr_cemp_get_data(UNUR_DISTR* distribution, double** sample)

    # 设置自定义经验分布对象 distribution 的直方图，返回整型值
    int unur_distr_cemp_set_hist(UNUR_DISTR* distribution, double* prob, int n_prob, double xmin, double xmax)

    # 设置自定义经验分布对象 distribution 的直方图概率，返回整型值
    int unur_distr_cemp_set_hist_prob(UNUR_DISTR* distribution, double* prob, int n_prob)

    # 设置自定义经验分布对象 distribution 的直方图定义域，返回整型值
    int unur_distr_cemp_set_hist_domain(UNUR_DISTR* distribution, double xmin, double xmax)

    # 设置自定义经验分布对象 distribution 的直方图区间，返回整型值
    int unur_distr_cemp_set_hist_bins(UNUR_DISTR* distribution, double* bins, int n_bins)

    # 创建一个新的连续分布对象，返回分布对象指针
    UNUR_DISTR* unur_distr_cont_new()

    # 设置连续分布对象 distribution 的概率密度函数，返回整型值
    int unur_distr_cont_set_pdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* pdf)

    # 设置连续分布对象 distribution 的概率密度函数导数，返回整型值
    int unur_distr_cont_set_dpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* dpdf)

    # 设置连续分布对象 distribution 的累积分布函数，返回整型值
    int unur_distr_cont_set_cdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* cdf)

    # 设置连续分布对象 distribution 的反向累积分布函数，返回整型值
    int unur_distr_cont_set_invcdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* invcdf)
    // 获取指定分布的概率密度函数 (PDF) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_pdf(UNUR_DISTR* distribution)
    
    // 获取指定分布的概率密度函数导数 (PDF 的导数) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_dpdf(UNUR_DISTR* distribution)
    
    // 获取指定分布的累积分布函数 (CDF) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_cdf(UNUR_DISTR* distribution)
    
    // 获取指定分布的反向累积分布函数 (CDF 的反函数) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_invcdf(UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的概率密度函数值
    double unur_distr_cont_eval_pdf(double x, UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的概率密度函数导数值
    double unur_distr_cont_eval_dpdf(double x, UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的累积分布函数值
    double unur_distr_cont_eval_cdf(double x, UNUR_DISTR* distribution)
    
    // 计算指定分布在给定概率 u 处的反向累积分布函数值
    double unur_distr_cont_eval_invcdf(double u, UNUR_DISTR* distribution)
    
    // 设置指定分布的对数概率密度函数 (log PDF) 连续函数对象
    int unur_distr_cont_set_logpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* logpdf)
    
    // 设置指定分布的对数概率密度函数导数 (log PDF 的导数) 连续函数对象
    int unur_distr_cont_set_dlogpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* dlogpdf)
    
    // 设置指定分布的对数累积分布函数 (log CDF) 连续函数对象
    int unur_distr_cont_set_logcdf(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* logcdf)
    
    // 获取指定分布的对数概率密度函数 (log PDF) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_logpdf(UNUR_DISTR* distribution)
    
    // 获取指定分布的对数概率密度函数导数 (log PDF 的导数) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_dlogpdf(UNUR_DISTR* distribution)
    
    // 获取指定分布的对数累积分布函数 (log CDF) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_logcdf(UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的对数概率密度函数值
    double unur_distr_cont_eval_logpdf(double x, UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的对数概率密度函数导数值
    double unur_distr_cont_eval_dlogpdf(double x, UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的对数累积分布函数值
    double unur_distr_cont_eval_logcdf(double x, UNUR_DISTR* distribution)
    
    // 设置指定分布的概率密度函数字符串表示
    int unur_distr_cont_set_pdfstr(UNUR_DISTR* distribution, char* pdfstr)
    
    // 设置指定分布的累积分布函数字符串表示
    int unur_distr_cont_set_cdfstr(UNUR_DISTR* distribution, char* cdfstr)
    
    // 获取指定分布的概率密度函数字符串表示
    char* unur_distr_cont_get_pdfstr(UNUR_DISTR* distribution)
    
    // 获取指定分布的概率密度函数导数字符串表示
    char* unur_distr_cont_get_dpdfstr(UNUR_DISTR* distribution)
    
    // 获取指定分布的累积分布函数字符串表示
    char* unur_distr_cont_get_cdfstr(UNUR_DISTR* distribution)
    
    // 设置指定分布的概率密度函数参数
    int unur_distr_cont_set_pdfparams(UNUR_DISTR* distribution, double* params, int n_params)
    
    // 获取指定分布的概率密度函数参数
    int unur_distr_cont_get_pdfparams(UNUR_DISTR* distribution, double** params)
    
    // 设置指定分布的概率密度函数向量参数
    int unur_distr_cont_set_pdfparams_vec(UNUR_DISTR* distribution, int par, double* param_vec, int n_param_vec)
    
    // 获取指定分布的概率密度函数向量参数
    int unur_distr_cont_get_pdfparams_vec(UNUR_DISTR* distribution, int par, double** param_vecs)
    
    // 设置指定分布的对数概率密度函数字符串表示
    int unur_distr_cont_set_logpdfstr(UNUR_DISTR* distribution, char* logpdfstr)
    
    // 获取指定分布的对数概率密度函数字符串表示
    char* unur_distr_cont_get_logpdfstr(UNUR_DISTR* distribution)
    
    // 获取指定分布的对数概率密度函数导数字符串表示
    char* unur_distr_cont_get_dlogpdfstr(UNUR_DISTR* distribution)
    
    // 设置指定分布的对数累积分布函数字符串表示
    int unur_distr_cont_set_logcdfstr(UNUR_DISTR* distribution, char* logcdfstr)
    
    // 获取指定分布的对数累积分布函数字符串表示
    char* unur_distr_cont_get_logcdfstr(UNUR_DISTR* distribution)
    
    // 设置指定分布的定义域范围
    int unur_distr_cont_set_domain(UNUR_DISTR* distribution, double left, double right)
    
    // 获取指定分布的定义域范围
    int unur_distr_cont_get_domain(UNUR_DISTR* distribution, double* left, double* right)
    
    // 获取指定分布的截断范围
    int unur_distr_cont_get_truncated(UNUR_DISTR* distribution, double* left, double* right)
    
    // 设置指定分布的危险率 (Hazard rate) 连续函数对象
    int unur_distr_cont_set_hr(UNUR_DISTR* distribution, UNUR_FUNCT_CONT* hazard)
    
    // 获取指定分布的危险率 (Hazard rate) 连续函数对象
    UNUR_FUNCT_CONT* unur_distr_cont_get_hr(UNUR_DISTR* distribution)
    
    // 计算指定分布在给定点 x 处的危险率值
    double unur_distr_cont_eval_hr(double x, UNUR_DISTR* distribution)
    
    // 设置指定分布的危险率字符串表示
    int unur_distr_cont_set_hrstr(UNUR_DISTR* distribution, char* hrstr)
    // 获取指定连续分布的人类可读字符串表示
    char* unur_distr_cont_get_hrstr(UNUR_DISTR* distribution)
    
    // 设置指定连续分布的众数（最高频率的值）
    int unur_distr_cont_set_mode(UNUR_DISTR* distribution, double mode)
    
    // 更新指定连续分布的众数
    int unur_distr_cont_upd_mode(UNUR_DISTR* distribution)
    
    // 获取指定连续分布的众数
    double unur_distr_cont_get_mode(UNUR_DISTR* distribution)
    
    // 设置指定连续分布的中心位置
    int unur_distr_cont_set_center(UNUR_DISTR* distribution, double center)
    
    // 获取指定连续分布的中心位置
    double unur_distr_cont_get_center(UNUR_DISTR* distribution)
    
    // 设置指定连续分布的概率密度函数区域面积
    int unur_distr_cont_set_pdfarea(UNUR_DISTR* distribution, double area)
    
    // 更新指定连续分布的概率密度函数区域面积
    int unur_distr_cont_upd_pdfarea(UNUR_DISTR* distribution)
    
    // 获取指定连续分布的概率密度函数区域面积
    double unur_distr_cont_get_pdfarea(UNUR_DISTR* distribution)
    
    // 创建一个基于给定分布的复合变换分布
    UNUR_DISTR* unur_distr_cxtrans_new(UNUR_DISTR* distribution)
    
    // 获取复合变换分布中的基础分布
    UNUR_DISTR* unur_distr_cxtrans_get_distribution(UNUR_DISTR* distribution)
    
    // 设置复合变换分布的alpha参数
    int unur_distr_cxtrans_set_alpha(UNUR_DISTR* distribution, double alpha)
    
    // 设置复合变换分布的重新缩放参数mu和sigma
    int unur_distr_cxtrans_set_rescale(UNUR_DISTR* distribution, double mu, double sigma)
    
    // 获取复合变换分布的alpha参数
    double unur_distr_cxtrans_get_alpha(UNUR_DISTR* distribution)
    
    // 获取复合变换分布的mu参数
    double unur_distr_cxtrans_get_mu(UNUR_DISTR* distribution)
    
    // 获取复合变换分布的sigma参数
    double unur_distr_cxtrans_get_sigma(UNUR_DISTR* distribution)
    
    // 设置复合变换分布的概率密度函数对数奇点及其导数
    int unur_distr_cxtrans_set_logpdfpole(UNUR_DISTR* distribution, double logpdfpole, double dlogpdfpole)
    
    // 设置复合变换分布的定义域
    int unur_distr_cxtrans_set_domain(UNUR_DISTR* distribution, double left, double right)
    
    // 创建一个有序复合分布
    UNUR_DISTR* unur_distr_corder_new(UNUR_DISTR* distribution, int n, int k)
    
    // 获取有序复合分布中的基础分布
    UNUR_DISTR* unur_distr_corder_get_distribution(UNUR_DISTR* distribution)
    
    // 设置有序复合分布的秩（排名）
    int unur_distr_corder_set_rank(UNUR_DISTR* distribution, int n, int k)
    
    // 获取有序复合分布的秩（排名）
    int unur_distr_corder_get_rank(UNUR_DISTR* distribution, int* n, int* k)
    
    // 创建一个多维复合分布
    UNUR_DISTR* unur_distr_cvec_new(int dim)
    
    // 设置多维复合分布的概率密度函数
    int unur_distr_cvec_set_pdf(UNUR_DISTR* distribution, UNUR_FUNCT_CVEC* pdf)
    
    // 设置多维复合分布的概率密度函数导数
    int unur_distr_cvec_set_dpdf(UNUR_DISTR* distribution, UNUR_VFUNCT_CVEC* dpdf)
    
    // 设置多维复合分布的概率密度函数二阶导数
    int unur_distr_cvec_set_pdpdf(UNUR_DISTR* distribution, UNUR_FUNCTD_CVEC* pdpdf)
    
    // 获取多维复合分布的概率密度函数
    UNUR_FUNCT_CVEC* unur_distr_cvec_get_pdf(UNUR_DISTR* distribution)
    
    // 获取多维复合分布的概率密度函数导数
    UNUR_VFUNCT_CVEC* unur_distr_cvec_get_dpdf(UNUR_DISTR* distribution)
    
    // 获取多维复合分布的概率密度函数二阶导数
    UNUR_FUNCTD_CVEC* unur_distr_cvec_get_pdpdf(UNUR_DISTR* distribution)
    
    // 计算多维复合分布在给定点x处的概率密度函数值
    double unur_distr_cvec_eval_pdf(double* x, UNUR_DISTR* distribution)
    
    // 计算多维复合分布在给定点x处的概率密度函数导数
    int unur_distr_cvec_eval_dpdf(double* result, double* x, UNUR_DISTR* distribution)
    
    // 计算多维复合分布在给定点x及其坐标轴上的坐标处的概率密度函数二阶导数值
    double unur_distr_cvec_eval_pdpdf(double* x, int coord, UNUR_DISTR* distribution)
    
    // 设置多维复合分布的对数概率密度函数
    int unur_distr_cvec_set_logpdf(UNUR_DISTR* distribution, UNUR_FUNCT_CVEC* logpdf)
    
    // 设置多维复合分布的对数概率密度函数导数
    int unur_distr_cvec_set_dlogpdf(UNUR_DISTR* distribution, UNUR_VFUNCT_CVEC* dlogpdf)
    
    // 设置多维复合分布的对数概率密度函数二阶导数
    int unur_distr_cvec_set_pdlogpdf(UNUR_DISTR* distribution, UNUR_FUNCTD_CVEC* pdlogpdf)
    
    // 获取多维复合分布的对数概率密度函数
    UNUR_FUNCT_CVEC* unur_distr_cvec_get_logpdf(UNUR_DISTR* distribution)
    
    // 获取多维复合分布的对数概率密度函数导数
    UNUR_VFUNCT_CVEC* unur_distr_cvec_get_dlogpdf(UNUR_DISTR* distribution)
    
    // 获取多维复合分布的对数概率密度函数二阶导数
    UNUR_FUNCTD_CVEC* unur_distr_cvec_get_pdlogpdf(UNUR_DISTR* distribution)
    
    // 计算多维复合分布在给定点x处的对数概率密度函数值
    double unur_distr_cvec_eval_logpdf(double* x, UNUR_DISTR* distribution)
    // Evaluate the logarithm of the probability density function (logpdf) of a continuous vector distribution at point x
    int unur_distr_cvec_eval_dlogpdf(double* result, double* x, UNUR_DISTR* distribution)

    // Evaluate the partial derivative of the logpdf of a continuous vector distribution at coordinate 'coord'
    double unur_distr_cvec_eval_pdlogpdf(double* x, int coord, UNUR_DISTR* distribution)

    // Set the mean vector for a continuous vector distribution
    int unur_distr_cvec_set_mean(UNUR_DISTR* distribution, double* mean)

    // Get the mean vector of a continuous vector distribution
    double* unur_distr_cvec_get_mean(UNUR_DISTR* distribution)

    // Set the covariance matrix for a continuous vector distribution
    int unur_distr_cvec_set_covar(UNUR_DISTR* distribution, double* covar)

    // Set the inverse of the covariance matrix for a continuous vector distribution
    int unur_distr_cvec_set_covar_inv(UNUR_DISTR* distribution, double* covar_inv)

    // Get the covariance matrix of a continuous vector distribution
    double* unur_distr_cvec_get_covar(UNUR_DISTR* distribution)

    // Get the Cholesky decomposition of the covariance matrix of a continuous vector distribution
    double* unur_distr_cvec_get_cholesky(UNUR_DISTR* distribution)

    // Get the inverse of the covariance matrix of a continuous vector distribution
    double* unur_distr_cvec_get_covar_inv(UNUR_DISTR* distribution)

    // Set the rank correlation matrix for a continuous vector distribution
    int unur_distr_cvec_set_rankcorr(UNUR_DISTR* distribution, double* rankcorr)

    // Get the rank correlation matrix of a continuous vector distribution
    double* unur_distr_cvec_get_rankcorr(UNUR_DISTR* distribution)

    // Get the Cholesky decomposition of the rank correlation matrix of a continuous vector distribution
    double* unur_distr_cvec_get_rk_cholesky(UNUR_DISTR* distribution)

    // Set the marginal distributions for each dimension of a continuous vector distribution
    int unur_distr_cvec_set_marginals(UNUR_DISTR* distribution, UNUR_DISTR* marginal)

    // Set an array of marginal distributions for a continuous vector distribution
    int unur_distr_cvec_set_marginal_array(UNUR_DISTR* distribution, UNUR_DISTR** marginals)

    // Set a list of marginal distributions for a continuous vector distribution
    int unur_distr_cvec_set_marginal_list(UNUR_DISTR* distribution)

    // Get the nth marginal distribution from a continuous vector distribution
    UNUR_DISTR* unur_distr_cvec_get_marginal(UNUR_DISTR* distribution, int n)

    // Set additional parameters for the pdf of a continuous vector distribution
    int unur_distr_cvec_set_pdfparams(UNUR_DISTR* distribution, double* params, int n_params)

    // Get additional parameters for the pdf of a continuous vector distribution
    int unur_distr_cvec_get_pdfparams(UNUR_DISTR* distribution, double** params)

    // Set an array of parameters for the pdf of a continuous vector distribution
    int unur_distr_cvec_set_pdfparams_vec(UNUR_DISTR* distribution, int par, double* param_vec, int n_params)

    // Get an array of parameters for the pdf of a continuous vector distribution
    int unur_distr_cvec_get_pdfparams_vec(UNUR_DISTR* distribution, int par, double** param_vecs)

    // Set the rectangular domain for a continuous vector distribution
    int unur_distr_cvec_set_domain_rect(UNUR_DISTR* distribution, double* lowerleft, double* upperright)

    // Check if a point x is within the domain of a continuous vector distribution
    int unur_distr_cvec_is_indomain(double* x, UNUR_DISTR* distribution)

    // Set the mode vector for a continuous vector distribution
    int unur_distr_cvec_set_mode(UNUR_DISTR* distribution, double* mode)

    // Update the mode vector of a continuous vector distribution
    int unur_distr_cvec_upd_mode(UNUR_DISTR* distribution)

    // Get the mode vector of a continuous vector distribution
    double* unur_distr_cvec_get_mode(UNUR_DISTR* distribution)

    // Set the center vector for a continuous vector distribution
    int unur_distr_cvec_set_center(UNUR_DISTR* distribution, double* center)

    // Get the center vector of a continuous vector distribution
    double* unur_distr_cvec_get_center(UNUR_DISTR* distribution)

    // Set the volume for the pdf of a continuous vector distribution
    int unur_distr_cvec_set_pdfvol(UNUR_DISTR* distribution, double volume)

    // Update the volume of the pdf for a continuous vector distribution
    int unur_distr_cvec_upd_pdfvol(UNUR_DISTR* distribution)

    // Get the volume of the pdf for a continuous vector distribution
    double unur_distr_cvec_get_pdfvol(UNUR_DISTR* distribution)

    // Create a new conditional distribution from an existing continuous vector distribution
    UNUR_DISTR* unur_distr_condi_new(UNUR_DISTR* distribution, double* pos, double* dir, int k)

    // Set the conditioning parameters for a conditional distribution based on a continuous vector distribution
    int unur_distr_condi_set_condition(UNUR_DISTR* distribution, double* pos, double* dir, int k)

    // Get the conditioning parameters of a conditional distribution based on a continuous vector distribution
    int unur_distr_condi_get_condition(UNUR_DISTR* distribution, double** pos, double** dir, int* k)

    // Get the base distribution of a conditional distribution based on a continuous vector distribution
    UNUR_DISTR* unur_distr_condi_get_distribution(UNUR_DISTR* distribution)

    // Create a new empirical distribution for a continuous vector distribution with specified dimension
    UNUR_DISTR* unur_distr_cvemp_new(int dim)

    // Set the data points for an empirical distribution of a continuous vector distribution
    int unur_distr_cvemp_set_data(UNUR_DISTR* distribution, double* sample, int n_sample)

    // Read data points from a file to set as data for an empirical distribution of a continuous vector distribution
    int unur_distr_cvemp_read_data(UNUR_DISTR* distribution, char* filename)
    // 获取连续型分布的样本数据
    int unur_distr_cvemp_get_data(UNUR_DISTR* distribution, double** sample)

    // 创建一个新的离散型分布对象
    UNUR_DISTR* unur_distr_discr_new()

    // 设置离散型分布的概率值
    int unur_distr_discr_set_pv(UNUR_DISTR* distribution, double* pv, int n_pv)

    // 根据概率值生成离散型分布的离散值
    int unur_distr_discr_make_pv(UNUR_DISTR* distribution)

    // 获取离散型分布的概率值
    int unur_distr_discr_get_pv(UNUR_DISTR* distribution, double** pv)

    // 设置离散型分布的概率质量函数
    int unur_distr_discr_set_pmf(UNUR_DISTR* distribution, UNUR_FUNCT_DISCR* pmf)

    // 设置离散型分布的累积分布函数
    int unur_distr_discr_set_cdf(UNUR_DISTR* distribution, UNUR_FUNCT_DISCR* cdf)

    // 设置离散型分布的反累积分布函数
    int unur_distr_discr_set_invcdf(UNUR_DISTR* distribution, UNUR_IFUNCT_DISCR* invcdf)

    // 获取离散型分布的概率质量函数
    UNUR_FUNCT_DISCR* unur_distr_discr_get_pmf(UNUR_DISTR* distribution)

    // 获取离散型分布的累积分布函数
    UNUR_FUNCT_DISCR* unur_distr_discr_get_cdf(UNUR_DISTR* distribution)

    // 获取离散型分布的反累积分布函数
    UNUR_IFUNCT_DISCR* unur_distr_discr_get_invcdf(UNUR_DISTR* distribution)

    // 计算离散型分布在给定离散值处的概率值
    double unur_distr_discr_eval_pv(int k, UNUR_DISTR* distribution)

    // 计算离散型分布在给定离散值处的概率质量函数值
    double unur_distr_discr_eval_pmf(int k, UNUR_DISTR* distribution)

    // 计算离散型分布在给定离散值处的累积分布函数值
    double unur_distr_discr_eval_cdf(int k, UNUR_DISTR* distribution)

    // 根据给定的概率值计算离散型分布的反累积分布函数的离散值
    int unur_distr_discr_eval_invcdf(double u, UNUR_DISTR* distribution)

    // 设置离散型分布的概率质量函数（字符串形式）
    int unur_distr_discr_set_pmfstr(UNUR_DISTR* distribution, char* pmfstr)

    // 设置离散型分布的累积分布函数（字符串形式）
    int unur_distr_discr_set_cdfstr(UNUR_DISTR* distribution, char* cdfstr)

    // 获取离散型分布的概率质量函数（字符串形式）
    char* unur_distr_discr_get_pmfstr(UNUR_DISTR* distribution)

    // 获取离散型分布的累积分布函数（字符串形式）
    char* unur_distr_discr_get_cdfstr(UNUR_DISTR* distribution)

    // 设置离散型分布的概率质量函数的参数
    int unur_distr_discr_set_pmfparams(UNUR_DISTR* distribution, double* params, int n_params)

    // 获取离散型分布的概率质量函数的参数
    int unur_distr_discr_get_pmfparams(UNUR_DISTR* distribution, double** params)

    // 设置离散型分布的定义域
    int unur_distr_discr_set_domain(UNUR_DISTR* distribution, int left, int right)

    // 获取离散型分布的定义域
    int unur_distr_discr_get_domain(UNUR_DISTR* distribution, int* left, int* right)

    // 设置离散型分布的模式
    int unur_distr_discr_set_mode(UNUR_DISTR* distribution, int mode)

    // 更新离散型分布的模式
    int unur_distr_discr_upd_mode(UNUR_DISTR* distribution)

    // 获取离散型分布的模式
    int unur_distr_discr_get_mode(UNUR_DISTR* distribution)

    // 设置离散型分布的概率质量函数的总和值
    int unur_distr_discr_set_pmfsum(UNUR_DISTR* distribution, double sum)

    // 更新离散型分布的概率质量函数的总和值
    int unur_distr_discr_upd_pmfsum(UNUR_DISTR* distribution)

    // 获取离散型分布的概率质量函数的总和值
    double unur_distr_discr_get_pmfsum(UNUR_DISTR* distribution)

    // 创建一个新的矩阵型分布对象
    UNUR_DISTR* unur_distr_matr_new(int n_rows, int n_cols)

    // 获取矩阵型分布对象的维度
    int unur_distr_matr_get_dim(UNUR_DISTR* distribution, int* n_rows, int* n_cols)

    // 创建一个自动参数配置对象
    UNUR_PAR* unur_auto_new(UNUR_DISTR* distribution)

    // 设置自动参数配置对象的对数标志
    int unur_auto_set_logss(UNUR_PAR* parameters, int logss)

    // 创建一个动态修正器参数配置对象
    UNUR_PAR* unur_dari_new(UNUR_DISTR* distribution)

    // 设置动态修正器参数配置对象的挤压标志
    int unur_dari_set_squeeze(UNUR_PAR* parameters, int squeeze)

    // 设置动态修正器参数配置对象的表大小
    int unur_dari_set_tablesize(UNUR_PAR* parameters, int size)

    // 设置动态修正器参数配置对象的控制点因子
    int unur_dari_set_cpfactor(UNUR_PAR* parameters, double cp_factor)

    // 设置动态修正器参数配置对象的验证标志
    int unur_dari_set_verify(UNUR_PAR* parameters, int verify)

    // 修改动态修正器生成器对象的验证标志
    int unur_dari_chg_verify(UNUR_GEN* generator, int verify)

    // 创建一个多元逆变换采样参数配置对象
    UNUR_PAR* unur_dau_new(UNUR_DISTR* distribution)

    // 设置多元逆变换采样参数配置对象的乌恩因子
    int unur_dau_set_urnfactor(UNUR_PAR* parameters, double factor)

    // 创建一个生成器参数配置对象
    UNUR_PAR* unur_dgt_new(UNUR_DISTR* distribution)
    // 设置离散傅里叶逆变换生成器的引导因子
    int unur_dgt_set_guidefactor(UNUR_PAR* parameters, double factor)

    // 设置离散傅里叶逆变换生成器的变体
    int unur_dgt_set_variant(UNUR_PAR* parameters, unsigned variant)

    // 通过循环利用上一个生成值来评估逆累积分布函数
    int unur_dgt_eval_invcdf_recycle(UNUR_GEN* generator, double u, double* recycle)

    // 评估逆累积分布函数的一般接口
    int unur_dgt_eval_invcdf(UNUR_GEN* generator, double u)

    // 创建一个新的分布随机数生成器参数对象
    UNUR_PAR* unur_dsrou_new(UNUR_DISTR* distribution)

    // 设置随机数生成器参数对象的累积分布函数在模式值处的值
    int unur_dsrou_set_cdfatmode(UNUR_PAR* parameters, double Fmode)

    // 设置随机数生成器参数对象的验证标志
    int unur_dsrou_set_verify(UNUR_PAR* parameters, int verify)

    // 修改随机数生成器对象的验证标志
    int unur_dsrou_chg_verify(UNUR_GEN* generator, int verify)

    // 修改随机数生成器对象的累积分布函数在模式值处的值
    int unur_dsrou_chg_cdfatmode(UNUR_GEN* generator, double Fmode)

    // 创建一个新的分布随机数生成器参数对象
    UNUR_PAR* unur_dss_new(UNUR_DISTR* distribution)

    // 创建一个新的分布随机数生成器参数对象
    UNUR_PAR* unur_arou_new(UNUR_DISTR* distribution)

    // 设置自适应拒绝采样算法的使用 DARS 标志
    int unur_arou_set_usedars(UNUR_PAR* parameters, int usedars)

    // 设置自适应拒绝采样算法的 DARS 因子
    int unur_arou_set_darsfactor(UNUR_PAR* parameters, double factor)

    // 设置自适应拒绝采样算法的最大平方比率
    int unur_arou_set_max_sqhratio(UNUR_PAR* parameters, double max_ratio)

    // 获取自适应拒绝采样算法生成器的当前平方比率
    double unur_arou_get_sqhratio(UNUR_GEN* generator)

    // 获取自适应拒绝采样算法生成器的当前帽区域面积
    double unur_arou_get_hatarea(UNUR_GEN* generator)

    // 获取自适应拒绝采样算法生成器的当前压缩区域面积
    double unur_arou_get_squeezearea(UNUR_GEN* generator)

    // 设置自适应拒绝采样算法的最大分段数
    int unur_arou_set_max_segments(UNUR_PAR* parameters, int max_segs)

    // 设置自适应拒绝采样算法的控制点
    int unur_arou_set_cpoints(UNUR_PAR* parameters, int n_stp, double* stp)

    // 设置自适应拒绝采样算法的中心使用标志
    int unur_arou_set_usecenter(UNUR_PAR* parameters, int usecenter)

    // 设置自适应拒绝采样算法的引导因子
    int unur_arou_set_guidefactor(UNUR_PAR* parameters, double factor)

    // 设置自适应拒绝采样算法的验证标志
    int unur_arou_set_verify(UNUR_PAR* parameters, int verify)

    // 修改自适应拒绝采样算法生成器的验证标志
    int unur_arou_chg_verify(UNUR_GEN* generator, int verify)

    // 设置自适应拒绝采样算法的严格模式标志
    int unur_arou_set_pedantic(UNUR_PAR* parameters, int pedantic)

    // 创建一个新的接受拒绝采样算法参数对象
    UNUR_PAR* unur_ars_new(UNUR_DISTR* distribution)

    // 设置接受拒绝采样算法的最大间隔数
    int unur_ars_set_max_intervals(UNUR_PAR* parameters, int max_ivs)

    // 设置接受拒绝采样算法的控制点
    int unur_ars_set_cpoints(UNUR_PAR* parameters, int n_cpoints, double* cpoints)

    // 设置接受拒绝采样算法的重新初始化百分位数
    int unur_ars_set_reinit_percentiles(UNUR_PAR* parameters, int n_percentiles, double* percentiles)

    // 修改接受拒绝采样算法生成器的重新初始化百分位数
    int unur_ars_chg_reinit_percentiles(UNUR_GEN* generator, int n_percentiles, double* percentiles)

    // 设置接受拒绝采样算法的重新初始化控制点数
    int unur_ars_set_reinit_ncpoints(UNUR_PAR* parameters, int ncpoints)

    // 修改接受拒绝采样算法生成器的重新初始化控制点数
    int unur_ars_chg_reinit_ncpoints(UNUR_GEN* generator, int ncpoints)

    // 设置接受拒绝采样算法的最大迭代次数
    int unur_ars_set_max_iter(UNUR_PAR* parameters, int max_iter)

    // 设置接受拒绝采样算法的验证标志
    int unur_ars_set_verify(UNUR_PAR* parameters, int verify)

    // 修改接受拒绝采样算法生成器的验证标志
    int unur_ars_chg_verify(UNUR_GEN* generator, int verify)

    // 设置接受拒绝采样算法的严格模式标志
    int unur_ars_set_pedantic(UNUR_PAR* parameters, int pedantic)

    // 获取接受拒绝采样算法生成器的当前帽区域对数面积
    double unur_ars_get_loghatarea(UNUR_GEN* generator)

    // 评估接受拒绝采样算法生成器的逆累积分布函数的估计值
    double unur_ars_eval_invcdfhat(UNUR_GEN* generator, double u)

    // 创建一个新的半隐式逆变换算法参数对象
    UNUR_PAR* unur_hinv_new(UNUR_DISTR* distribution)

    // 设置半隐式逆变换算法的阶数
    int unur_hinv_set_order(UNUR_PAR* parameters, int order)

    // 设置半隐式逆变换算法的 U 分辨率
    int unur_hinv_set_u_resolution(UNUR_PAR* parameters, double u_resolution)

    // 设置半隐式逆变换算法的控制点
    int unur_hinv_set_cpoints(UNUR_PAR* parameters, double* stp, int n_stp)

    // 设置半隐式逆变换算法的边界
    int unur_hinv_set_boundary(UNUR_PAR* parameters, double left, double right)

    // 设置半隐式逆变换算法的引导因子
    int unur_hinv_set_guidefactor(UNUR_PAR* parameters, double factor)
    int unur_hinv_set_max_intervals(UNUR_PAR* parameters, int max_ivs)
        # 设置逆变换方法的最大区间数目
        # 用于设置给定参数的最大区间数目，用于逆变换方法
    
    int unur_hinv_get_n_intervals(UNUR_GEN* generator)
        # 获取逆变换方法生成器中的区间数目
        # 返回当前生成器中的区间数目，用于逆变换方法
    
    double unur_hinv_eval_approxinvcdf(UNUR_GEN* generator, double u)
        # 评估逆变换方法的近似累积分布函数的逆
        # 根据给定的生成器和累积分布函数值 u，返回近似的累积分布函数的逆值
    
    int unur_hinv_chg_truncated(UNUR_GEN* generator, double left, double right)
        # 更改逆变换方法生成器的截断范围
        # 调整给定生成器的截断范围，用于逆变换方法
    
    int unur_hinv_estimate_error(UNUR_GEN* generator, int samplesize, double* max_error, double* MAE)
        # 估计逆变换方法的误差
        # 使用给定生成器和样本大小来估计逆变换方法的最大误差和平均绝对误差（MAE）
    
    UNUR_PAR* unur_hrb_new(UNUR_DISTR* distribution)
        # 创建逆比例采样生成器的参数对象
        # 根据给定的分布对象创建一个新的逆比例采样生成器的参数对象
    
    int unur_hrb_set_upperbound(UNUR_PAR* parameters, double upperbound)
        # 设置逆比例采样方法的上界
        # 为给定参数对象设置逆比例采样方法的上界
    
    int unur_hrb_set_verify(UNUR_PAR* parameters, int verify)
        # 设置逆比例采样方法的验证标志
        # 为给定参数对象设置逆比例采样方法的验证标志
    
    int unur_hrb_chg_verify(UNUR_GEN* generator, int verify)
        # 更改逆比例采样方法生成器的验证标志
        # 调整给定生成器的逆比例采样方法的验证标志
    
    UNUR_PAR* unur_hrd_new(UNUR_DISTR* distribution)
        # 创建逆正态分布生成器的参数对象
        # 根据给定的分布对象创建一个新的逆正态分布生成器的参数对象
    
    int unur_hrd_set_verify(UNUR_PAR* parameters, int verify)
        # 设置逆正态分布方法的验证标志
        # 为给定参数对象设置逆正态分布方法的验证标志
    
    int unur_hrd_chg_verify(UNUR_GEN* generator, int verify)
        # 更改逆正态分布方法生成器的验证标志
        # 调整给定生成器的逆正态分布方法的验证标志
    
    UNUR_PAR* unur_hri_new(UNUR_DISTR* distribution)
        # 创建逆重要性抽样生成器的参数对象
        # 根据给定的分布对象创建一个新的逆重要性抽样生成器的参数对象
    
    int unur_hri_set_p0(UNUR_PAR* parameters, double p0)
        # 设置逆重要性抽样方法的初始概率
        # 为给定参数对象设置逆重要性抽样方法的初始概率
    
    int unur_hri_set_verify(UNUR_PAR* parameters, int verify)
        # 设置逆重要性抽样方法的验证标志
        # 为给定参数对象设置逆重要性抽样方法的验证标志
    
    int unur_hri_chg_verify(UNUR_GEN* generator, int verify)
        # 更改逆重要性抽样方法生成器的验证标志
        # 调整给定生成器的逆重要性抽样方法的验证标志
    
    UNUR_PAR* unur_itdr_new(UNUR_DISTR* distribution)
        # 创建变换-拒绝-重采样生成器的参数对象
        # 根据给定的分布对象创建一个新的变换-拒绝-重采样生成器的参数对象
    
    int unur_itdr_set_xi(UNUR_PAR* parameters, double xi)
        # 设置变换-拒绝-重采样方法的 xi 参数
        # 为给定参数对象设置变换-拒绝-重采样方法的 xi 参数
    
    int unur_itdr_set_cp(UNUR_PAR* parameters, double cp)
        # 设置变换-拒绝-重采样方法的 cp 参数
        # 为给定参数对象设置变换-拒绝-重采样方法的 cp 参数
    
    int unur_itdr_set_ct(UNUR_PAR* parameters, double ct)
        # 设置变换-拒绝-重采样方法的 ct 参数
        # 为给定参数对象设置变换-拒绝-重采样方法的 ct 参数
    
    double unur_itdr_get_xi(UNUR_GEN* generator)
        # 获取变换-拒绝-重采样方法生成器的 xi 参数
        # 返回给定生成器当前使用的变换-拒绝-重采样方法的 xi 参数值
    
    double unur_itdr_get_cp(UNUR_GEN* generator)
        # 获取变换-拒绝-重采样方法生成器的 cp 参数
        # 返回给定生成器当前使用的变换-拒绝-重采样方法的 cp 参数值
    
    double unur_itdr_get_ct(UNUR_GEN* generator)
        # 获取变换-拒绝-重采样方法生成器的 ct 参数
        # 返回给定生成器当前使用的变换-拒绝-重采样方法的 ct 参数值
    
    double unur_itdr_get_area(UNUR_GEN* generator)
        # 获取变换-拒绝-重采样方法生成器的区域面积
        # 返回给定生成器当前使用的变换-拒绝-重采样方法的区域面积
    
    int unur_itdr_set_verify(UNUR_PAR* parameters, int verify)
        # 设置变换-拒绝-重采样方法的验证标志
        # 为给定参数对象设置变换-拒绝-重采样方法的验证标志
    
    int unur_itdr_chg_verify(UNUR_GEN* generator, int verify)
        # 更改变换-拒绝-重采样方法生成器的验证标志
        # 调整给定生成器的变换-拒绝-重采样方法的验证标志
    
    UNUR_PAR* unur_mcorr_new(UNUR_DISTR* distribution)
        # 创建多变量相关抽样生成器的参数对象
        # 根据给定的分布对象创建一个新的多变量相关抽样生成器的参数对象
    
    int unur_mcorr_set_eigenvalues(UNUR_PAR* par, double* eigenvalues)
        # 设置多变量相关抽样方法的特征值
        # 为给定参数对象设置多变量相关抽样方法的特征值
    
    int unur_mcorr_chg_eigenvalues(UNUR_GEN* gen, double* eigenvalues)
        # 更改多变量相关抽样方法生成器的特征值
        # 调整给定生成器的多变量相关抽样方法的特征值
    
    UNUR_PAR* unur_ninv_new(UNUR_DISTR* distribution)
        # 创建正态分布逆变换生成器的参数对象
        # 根据给定的分布对象创建一个新的正态分布逆变换生成器的参数对象
    
    int unur_ninv_set_useregula(UNUR_PAR* parameters)
        # 设置正态分布逆变换方法使用正则逼近
        # 为给定参数对象设置正态分布逆变换方法使用正则逼近
    
    int unur_ninv_set_usenewton(UNUR_PAR* parameters)
        # 设置正态分布逆变换方法使用牛顿法
        # 为给定参数对象设置正态分布逆变换方法使用牛顿法
    
    int unur_ninv_set_usebisect(UNUR_PAR* parameters)
    // 设置 UNUR_PAR 结构体的参数 umin 和 umax
    int unur_nrou_set_u(UNUR_PAR* parameters, double umin, double umax)
    
    // 设置 UNUR_PAR 结构体的参数 vmax
    int unur_nrou_set_v(UNUR_PAR* parameters, double vmax)
    
    // 设置 UNUR_PAR 结构体的参数 r
    int unur_nrou_set_r(UNUR_PAR* parameters, double r)
    
    // 设置 UNUR_PAR 结构体的参数 center
    int unur_nrou_set_center(UNUR_PAR* parameters, double center)
    
    // 设置 UNUR_PAR 结构体的参数 verify
    int unur_nrou_set_verify(UNUR_PAR* parameters, int verify)
    
    // 修改 UNUR_GEN 结构体的参数 verify
    int unur_nrou_chg_verify(UNUR_GEN* generator, int verify)
    
    // 创建一个新的 UNUR_PAR 结构体，基于给定的分布 distribution
    UNUR_PAR* unur_pinv_new(UNUR_DISTR* distribution)
    
    // 设置 UNUR_PAR 结构体的参数 order
    int unur_pinv_set_order(UNUR_PAR* parameters, int order)
    
    // 设置 UNUR_PAR 结构体的参数 smoothness
    int unur_pinv_set_smoothness(UNUR_PAR* parameters, int smoothness)
    
    // 设置 UNUR_PAR 结构体的参数 u_resolution
    int unur_pinv_set_u_resolution(UNUR_PAR* parameters, double u_resolution)
    
    // 设置 UNUR_PAR 结构体的参数 use_upoints
    int unur_pinv_set_use_upoints(UNUR_PAR* parameters, int use_upoints)
    
    // 设置 UNUR_PAR 结构体的参数为使用 PDF
    int unur_pinv_set_usepdf(UNUR_PAR* parameters)
    
    // 设置 UNUR_PAR 结构体的参数为使用 CDF
    int unur_pinv_set_usecdf(UNUR_PAR* parameters)
    
    // 设置 UNUR_PAR 结构体的边界参数 left 和 right
    int unur_pinv_set_boundary(UNUR_PAR* parameters, double left, double right)
    
    // 设置 UNUR_PAR 结构体的搜索边界参数 left 和 right
    int unur_pinv_set_searchboundary(UNUR_PAR* parameters, int left, int right)
    
    // 设置 UNUR_PAR 结构体的最大区间数参数 max_ivs
    int unur_pinv_set_max_intervals(UNUR_PAR* parameters, int max_ivs)
    
    // 获取 UNUR_GEN 结构体生成器的区间数
    int unur_pinv_get_n_intervals(UNUR_GEN* generator)
    
    // 设置 UNUR_PAR 结构体的参数 keepcdf
    int unur_pinv_set_keepcdf(UNUR_PAR* parameters, int keepcdf)
    
    // 计算 UNUR_GEN 结构体生成器的逆 CDF 近似值，基于输入参数 u
    double unur_pinv_eval_approxinvcdf(UNUR_GEN* generator, double u)
    
    // 计算 UNUR_GEN 结构体生成器的 CDF 近似值，基于输入参数 x
    double unur_pinv_eval_approxcdf(UNUR_GEN* generator, double x)
    
    // 评估 UNUR_GEN 结构体生成器的误差估计，使用样本大小 samplesize，返回最大误差和平均绝对误差
    int unur_pinv_estimate_error(UNUR_GEN* generator, int samplesize, double* max_error, double* MAE)
    
    // 创建一个新的 UNUR_PAR 结构体，基于给定的分布 distribution
    UNUR_PAR* unur_srou_new(UNUR_DISTR* distribution)
    
    // 设置 UNUR_PAR 结构体的参数 r
    int unur_srou_set_r(UNUR_PAR* parameters, double r)
    
    // 设置 UNUR_PAR 结构体的参数 Fmode（CDF 在模式下的值）
    int unur_srou_set_cdfatmode(UNUR_PAR* parameters, double Fmode)
    
    // 设置 UNUR_PAR 结构体的参数 fmode（PDF 在模式下的值）
    int unur_srou_set_pdfatmode(UNUR_PAR* parameters, double fmode)
    
    // 设置 UNUR_PAR 结构体的参数 usesqueeze
    int unur_srou_set_usesqueeze(UNUR_PAR* parameters, int usesqueeze)
    
    // 设置 UNUR_PAR 结构体的参数 usemirror
    int unur_srou_set_usemirror(UNUR_PAR* parameters, int usemirror)
    
    // 设置 UNUR_PAR 结构体的参数 verify
    int unur_srou_set_verify(UNUR_PAR* parameters, int verify)
    
    // 修改 UNUR_GEN 结构体的参数 verify
    int unur_srou_chg_verify(UNUR_GEN* generator, int verify)
    
    // 修改 UNUR_GEN 结构体的参数 Fmode（CDF 在模式下的值）
    int unur_srou_chg_cdfatmode(UNUR_GEN* generator, double Fmode)
    
    // 修改 UNUR_GEN 结构体的参数 fmode（PDF 在模式下的值）
    int unur_srou_chg_pdfatmode(UNUR_GEN* generator, double fmode)
    
    // 创建一个新的 UNUR_PAR 结构体，基于给定的分布 distribution
    UNUR_PAR* unur_ssr_new(UNUR_DISTR* distribution)
    
    // 设置 UNUR_PAR 结构体的参数 Fmode（CDF 在模式下的值）
    int unur_ssr_set_cdfatmode(UNUR_PAR* parameters, double Fmode)
    
    // 设置 UNUR_PAR 结构体的参数 fmode（PDF 在模式下的值）
    int unur_ssr_set_pdfatmode(UNUR_PAR* parameters, double fmode)
    
    // 设置 UNUR_PAR 结构体的参数 usesqueeze
    int unur_ssr_set_usesqueeze(UNUR_PAR* parameters, int usesqueeze)
    
    // 设置 UNUR_PAR 结构体的参数 verify
    int unur_ssr_set_verify(UNUR_PAR* parameters, int verify)
    
    // 修改 UNUR_GEN 结构体的参数 verify
    int unur_ssr_chg_verify(UNUR_GEN* generator, int verify)
    
    // 修改 UNUR_GEN 结构体的参数 Fmode（CDF 在模式下的值）
    int unur_ssr_chg_cdfatmode(UNUR_GEN* generator, double Fmode)
    
    // 修改 UNUR_GEN 结构体的参数 fmode（PDF 在模式下的值）
    int unur_ssr_chg_pdfatmode(UNUR_GEN* generator, double fmode)
    
    // 创建一个新的 UNUR_PAR 结构体，基于给定的分布 distribution
    UNUR_PAR* unur_tabl_new(UNUR_DISTR* distribution)
    
    // 设置 UNUR_PAR 结构体的参数 use_ia
    int unur_tabl_set_variant_ia(UNUR_PAR* parameters, int use_ia)
    
    // 设置 UNUR_PAR 结构体的关键点参数 n_cpoints 和 cpoints
    int unur_tabl_set_cpoints(UNUR_PAR* parameters, int n_cpoints, double* cpoints)
    
    // 设置 UNUR_PAR 结构体的步数参数 n_stp
    int unur_tabl_set_nstp(UNUR_PAR* parameters, int n_stp)
    
    // 设置 UNUR_PAR 结构体的参数 useear
    int unur_tabl_set_useear(UNUR_PAR* parameters, int useear)
    // 设置给定的参数对象的面积比例因子
    int unur_tabl_set_areafraction(UNUR_PAR* parameters, double fraction)
    
    // 设置给定的参数对象使用的ARS方法数
    int unur_tabl_set_usedars(UNUR_PAR* parameters, int usedars)
    
    // 设置给定的参数对象的ARS因子
    int unur_tabl_set_darsfactor(UNUR_PAR* parameters, double factor)
    
    // 设置给定的参数对象的分裂模式变体
    int unur_tabl_set_variant_splitmode(UNUR_PAR* parameters, unsigned splitmode)
    
    // 设置给定的参数对象的最大平方/面积比
    int unur_tabl_set_max_sqhratio(UNUR_PAR* parameters, double max_ratio)
    
    // 获取生成器对象的当前平方/面积比
    double unur_tabl_get_sqhratio(UNUR_GEN* generator)
    
    // 获取生成器对象的哈达玛面积
    double unur_tabl_get_hatarea(UNUR_GEN* generator)
    
    // 获取生成器对象的压缩区域面积
    double unur_tabl_get_squeezearea(UNUR_GEN* generator)
    
    // 设置给定的参数对象的最大区间数
    int unur_tabl_set_max_intervals(UNUR_PAR* parameters, int max_ivs)
    
    // 获取生成器对象当前使用的区间数
    int unur_tabl_get_n_intervals(UNUR_GEN* generator)
    
    // 设置给定的参数对象的斜率数组和数组长度
    int unur_tabl_set_slopes(UNUR_PAR* parameters, double* slopes, int n_slopes)
    
    // 设置给定的参数对象的指南因子
    int unur_tabl_set_guidefactor(UNUR_PAR* parameters, double factor)
    
    // 设置给定的参数对象的区间边界
    int unur_tabl_set_boundary(UNUR_PAR* parameters, double left, double right)
    
    // 修改生成器对象的截断区域
    int unur_tabl_chg_truncated(UNUR_GEN* gen, double left, double right)
    
    // 设置给定的参数对象的验证标志
    int unur_tabl_set_verify(UNUR_PAR* parameters, int verify)
    
    // 修改生成器对象的验证标志
    int unur_tabl_chg_verify(UNUR_GEN* generator, int verify)
    
    // 设置给定的参数对象的严格模式标志
    int unur_tabl_set_pedantic(UNUR_PAR* parameters, int pedantic)
    
    // 创建新的基于分布的参数对象
    UNUR_PAR* unur_tdr_new(UNUR_DISTR* distribution)
    
    // 设置给定的参数对象的常数C
    int unur_tdr_set_c(UNUR_PAR* parameters, double c)
    
    // 设置给定的参数对象的GW变体
    int unur_tdr_set_variant_gw(UNUR_PAR* parameters)
    
    // 设置给定的参数对象的PS变体
    int unur_tdr_set_variant_ps(UNUR_PAR* parameters)
    
    // 设置给定的参数对象的IA变体
    int unur_tdr_set_variant_ia(UNUR_PAR* parameters)
    
    // 设置给定的参数对象使用的ARS方法数
    int unur_tdr_set_usedars(UNUR_PAR* parameters, int usedars)
    
    // 设置给定的参数对象的ARS因子
    int unur_tdr_set_darsfactor(UNUR_PAR* parameters, double factor)
    
    // 设置给定的参数对象的控制点
    int unur_tdr_set_cpoints(UNUR_PAR* parameters, int n_stp, double* stp)
    
    // 设置给定的参数对象的重新初始化百分位数
    int unur_tdr_set_reinit_percentiles(UNUR_PAR* parameters, int n_percentiles, double* percentiles)
    
    // 修改生成器对象的重新初始化百分位数
    int unur_tdr_chg_reinit_percentiles(UNUR_GEN* generator, int n_percentiles, double* percentiles)
    
    // 设置给定的参数对象的重新初始化控制点数
    int unur_tdr_set_reinit_ncpoints(UNUR_PAR* parameters, int ncpoints)
    
    // 修改生成器对象的重新初始化控制点数
    int unur_tdr_chg_reinit_ncpoints(UNUR_GEN* generator, int ncpoints)
    
    // 修改生成器对象的截断区域
    int unur_tdr_chg_truncated(UNUR_GEN* gen, double left, double right)
    
    // 设置给定的参数对象的最大平方/面积比
    int unur_tdr_set_max_sqhratio(UNUR_PAR* parameters, double max_ratio)
    
    // 获取生成器对象的当前平方/面积比
    double unur_tdr_get_sqhratio(UNUR_GEN* generator)
    
    // 获取生成器对象的哈达玛面积
    double unur_tdr_get_hatarea(UNUR_GEN* generator)
    
    // 获取生成器对象的压缩区域面积
    double unur_tdr_get_squeezearea(UNUR_GEN* generator)
    
    // 设置给定的参数对象的最大区间数
    int unur_tdr_set_max_intervals(UNUR_PAR* parameters, int max_ivs)
    
    // 设置给定的参数对象的使用中心标志
    int unur_tdr_set_usecenter(UNUR_PAR* parameters, int usecenter)
    
    // 设置给定的参数对象的使用模式标志
    int unur_tdr_set_usemode(UNUR_PAR* parameters, int usemode)
    
    // 设置给定的参数对象的指南因子
    int unur_tdr_set_guidefactor(UNUR_PAR* parameters, double factor)
    
    // 设置给定的参数对象的验证标志
    int unur_tdr_set_verify(UNUR_PAR* parameters, int verify)
    
    // 修改生成器对象的验证标志
    int unur_tdr_chg_verify(UNUR_GEN* generator, int verify)
    
    // 设置给定的参数对象的严格模式标志
    int unur_tdr_set_pedantic(UNUR_PAR* parameters, int pedantic)
    
    // 计算生成器对象的逆CDF估计值及相关信息
    double unur_tdr_eval_invcdfhat(UNUR_GEN* generator, double u, double* hx, double* fx, double* sqx)
    
    // 检查生成器对象的ARS方法是否正在运行
    int _unur_tdr_is_ARS_running(UNUR_GEN* generator)
    # 创建一个新的 univariate discrete random variable (UTDR) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_utdr_new(UNUR_DISTR* distribution)
    
    # 设置 UTDR 参数对象的概率密度函数的模式
    int unur_utdr_set_pdfatmode(UNUR_PAR* parameters, double fmode)
    
    # 设置 UTDR 参数对象的复制因子
    int unur_utdr_set_cpfactor(UNUR_PAR* parameters, double cp_factor)
    
    # 设置 UTDR 参数对象的 delta 值
    int unur_utdr_set_deltafactor(UNUR_PAR* parameters, double delta)
    
    # 设置 UTDR 参数对象的验证模式
    int unur_utdr_set_verify(UNUR_PAR* parameters, int verify)
    
    # 修改已创建的 UTDR 生成器的验证模式
    int unur_utdr_chg_verify(UNUR_GEN* generator, int verify)
    
    # 修改已创建的 UTDR 生成器的概率密度函数的模式
    int unur_utdr_chg_pdfatmode(UNUR_GEN* generator, double fmode)
    
    # 创建一个新的 empirical kernel density (EMPK) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_empk_new(UNUR_DISTR* distribution)
    
    # 设置 EMPK 参数对象的核函数类型
    int unur_empk_set_kernel(UNUR_PAR* parameters, unsigned kernel)
    
    # 设置 EMPK 参数对象的核函数生成器、alpha 和核函数方差
    int unur_empk_set_kernelgen(UNUR_PAR* parameters, UNUR_GEN* kernelgen, double alpha, double kernelvar)
    
    # 设置 EMPK 参数对象的 beta 值
    int unur_empk_set_beta(UNUR_PAR* parameters, double beta)
    
    # 设置 EMPK 参数对象的平滑度
    int unur_empk_set_smoothing(UNUR_PAR* parameters, double smoothing)
    
    # 修改已创建的 EMPK 生成器的平滑度
    int unur_empk_chg_smoothing(UNUR_GEN* generator, double smoothing)
    
    # 设置 EMPK 参数对象的变量相关性
    int unur_empk_set_varcor(UNUR_PAR* parameters, int varcor)
    
    # 修改已创建的 EMPK 生成器的变量相关性
    int unur_empk_chg_varcor(UNUR_GEN* generator, int varcor)
    
    # 设置 EMPK 参数对象的正性约束
    int unur_empk_set_positive(UNUR_PAR* parameters, int positive)
    
    # 创建一个新的 empirical likelihood (EMPL) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_empl_new(UNUR_DISTR* distribution)
    
    # 创建一个新的 histogram (HIST) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_hist_new(UNUR_DISTR* distribution)
    
    # 创建一个新的 multivariate truncated distribution (MVTDR) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_mvtdr_new(UNUR_DISTR* distribution)
    
    # 设置 MVTDR 参数对象的最小步数
    int unur_mvtdr_set_stepsmin(UNUR_PAR* parameters, int stepsmin)
    
    # 设置 MVTDR 参数对象的边界分割因子
    int unur_mvtdr_set_boundsplitting(UNUR_PAR* parameters, double boundsplitting)
    
    # 设置 MVTDR 参数对象的最大锥数
    int unur_mvtdr_set_maxcones(UNUR_PAR* parameters, int maxcones)
    
    # 获取已创建的 MVTDR 生成器的锥数
    int unur_mvtdr_get_ncones(UNUR_GEN* generator)
    
    # 获取已创建的 MVTDR 生成器的帽子体积
    double unur_mvtdr_get_hatvol(UNUR_GEN* generator)
    
    # 设置 MVTDR 参数对象的验证模式
    int unur_mvtdr_set_verify(UNUR_PAR* parameters, int verify)
    
    # 修改已创建的 MVTDR 生成器的验证模式
    int unur_mvtdr_chg_verify(UNUR_GEN* generator, int verify)
    
    # 创建一个新的 non-central orthogonalized rank transform (NORTA) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_norta_new(UNUR_DISTR* distribution)
    
    # 创建一个新的 variable empirical kernel density (VEMPK) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_vempk_new(UNUR_DISTR* distribution)
    
    # 设置 VEMPK 参数对象的平滑度
    int unur_vempk_set_smoothing(UNUR_PAR* parameters, double smoothing)
    
    # 修改已创建的 VEMPK 生成器的平滑度
    int unur_vempk_chg_smoothing(UNUR_GEN* generator, double smoothing)
    
    # 设置 VEMPK 参数对象的变量相关性
    int unur_vempk_set_varcor(UNUR_PAR* parameters, int varcor)
    
    # 修改已创建的 VEMPK 生成器的变量相关性
    int unur_vempk_chg_varcor(UNUR_GEN* generator, int varcor)
    
    # 创建一个新的 Gibbs sampler (GIBBS) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_gibbs_new(UNUR_DISTR* distribution)
    
    # 设置 Gibbs sampler 参数对象的坐标变体
    int unur_gibbs_set_variant_coordinate(UNUR_PAR* parameters)
    
    # 设置 Gibbs sampler 参数对象的随机方向变体
    int unur_gibbs_set_variant_random_direction(UNUR_PAR* parameters)
    
    # 设置 Gibbs sampler 参数对象的常数 c
    int unur_gibbs_set_c(UNUR_PAR* parameters, double c)
    
    # 设置 Gibbs sampler 参数对象的起始点
    int unur_gibbs_set_startingpoint(UNUR_PAR* parameters, double* x0)
    
    # 设置 Gibbs sampler 参数对象的稀疏率
    int unur_gibbs_set_thinning(UNUR_PAR* parameters, int thinning)
    
    # 设置 Gibbs sampler 参数对象的燃烧期
    int unur_gibbs_set_burnin(UNUR_PAR* parameters, int burnin)
    
    # 获取已创建的 Gibbs sampler 生成器的状态
    double* unur_gibbs_get_state(UNUR_GEN* generator)
    
    # 修改已创建的 Gibbs sampler 生成器的状态
    int unur_gibbs_chg_state(UNUR_GEN* generator, double* state)
    
    # 重置已创建的 Gibbs sampler 生成器的状态
    int unur_gibbs_reset_state(UNUR_GEN* generator)
    
    # 创建一个新的 Copula standard transform (CSTD) 参数对象，并与指定的分布相关联
    UNUR_PAR* unur_cstd_new(UNUR_DISTR* distribution)
    
    # 设置 CSTD 参数对象的变体
    int unur_cstd_set_variant(UNUR_PAR* parameters, unsigned variant)
    
    # 修改已创建的 CSTD 生成器的截断值
    int unur_cstd_chg_truncated(UNUR_GEN* generator, double left, double right)
    // 声明一个函数 unur_cstd_eval_invcdf，接受 UNUR_GEN* 类型的 generator 和 double 类型的 u，返回 double 类型的值
    double unur_cstd_eval_invcdf(UNUR_GEN* generator, double u)
    
    // 创建一个新的 univariate distribution 参数对象 UNUR_PAR，参数是 UNUR_DISTR* distribution
    UNUR_PAR* unur_dstd_new(UNUR_DISTR* distribution)
    
    // 设置 univariate distribution 参数对象 UNUR_PAR 的 variant 属性，参数是 unsigned 类型的 variant
    int unur_dstd_set_variant(UNUR_PAR* parameters, unsigned variant)
    
    // 修改 generator 的截断逆累积分布函数（inverse cumulative distribution function，简称 invcdf）的左右边界，返回 int 类型的结果
    int unur_dstd_chg_truncated(UNUR_GEN* generator, int left, int right)
    
    // 计算 generator 的逆累积分布函数 invcdf 在 u 处的值，返回 int 类型的结果
    int unur_dstd_eval_invcdf(UNUR_GEN* generator, double u)
    
    // 创建一个新的 multivariate distribution 参数对象 UNUR_PAR，参数是 UNUR_DISTR* distribution
    UNUR_PAR* unur_mvstd_new(UNUR_DISTR* distribution)
    
    // 创建一个新的 mixture distribution 参数对象 UNUR_PAR，包含 n 个成分（components），每个成分的概率由 prob 数组给出，每个成分的生成器由 comp 数组给出
    UNUR_PAR* unur_mixt_new(int n, double* prob, UNUR_GEN** comp)
    
    // 设置 mixture distribution 参数对象 UNUR_PAR 的 useinversion 属性，参数是 int 类型的 useinv
    int unur_mixt_set_useinversion(UNUR_PAR* parameters, int useinv)
    
    // 计算 generator 的逆累积分布函数 invcdf 在 u 处的值，返回 double 类型的结果
    double unur_mixt_eval_invcdf(UNUR_GEN* generator, double u)
    
    // 创建一个新的 external (custom) distribution 参数对象 UNUR_PAR，参数是 UNUR_DISTR* distribution
    UNUR_PAR* unur_cext_new(UNUR_DISTR* distribution)
    
    // 定义一个函数指针类型 _unur_cext_set_init_init_ft，该函数接受 UNUR_GEN* 类型的 gen 参数，返回 int 类型的结果
    ctypedef int (*_unur_cext_set_init_init_ft)(UNUR_GEN* gen)
    
    // 设置 external distribution 参数对象 UNUR_PAR 的 init 函数指针，参数是 _unur_cext_set_init_init_ft 类型的 init 函数指针
    int unur_cext_set_init(UNUR_PAR* parameters, _unur_cext_set_init_init_ft init)
    
    // 定义一个函数指针类型 _unur_cext_set_sample_sample_ft，该函数接受 UNUR_GEN* 类型的 gen 参数，返回 double 类型的结果
    ctypedef double (*_unur_cext_set_sample_sample_ft)(UNUR_GEN* gen)
    
    // 设置 external distribution 参数对象 UNUR_PAR 的 sample 函数指针，参数是 _unur_cext_set_sample_sample_ft 类型的 sample 函数指针
    int unur_cext_set_sample(UNUR_PAR* parameters, _unur_cext_set_sample_sample_ft sample)
    
    // 获取 generator 的参数信息，返回一个指向参数数据的 void* 指针，参数 size 表示数据的大小
    void* unur_cext_get_params(UNUR_GEN* generator, size_t size)
    
    // 获取 generator 的分布参数信息，返回一个指向 double 数组的指针
    double* unur_cext_get_distrparams(UNUR_GEN* generator)
    
    // 获取 generator 的分布参数个数，返回 int 类型的结果
    int unur_cext_get_ndistrparams(UNUR_GEN* generator)
    
    // 创建一个新的 external (custom) distribution 参数对象 UNUR_PAR，参数是 UNUR_DISTR* distribution
    UNUR_PAR* unur_dext_new(UNUR_DISTR* distribution)
    
    // 定义一个函数指针类型 _unur_dext_set_init_init_ft，该函数接受 UNUR_GEN* 类型的 gen 参数，返回 int 类型的结果
    ctypedef int (*_unur_dext_set_init_init_ft)(UNUR_GEN* gen)
    
    // 设置 external distribution 参数对象 UNUR_PAR 的 init 函数指针，参数是 _unur_dext_set_init_init_ft 类型的 init 函数指针
    int unur_dext_set_init(UNUR_PAR* parameters, _unur_dext_set_init_init_ft init)
    
    // 定义一个函数指针类型 _unur_dext_set_sample_sample_ft，该函数接受 UNUR_GEN* 类型的 gen 参数，返回 int 类型的结果
    ctypedef int (*_unur_dext_set_sample_sample_ft)(UNUR_GEN* gen)
    
    // 设置 external distribution 参数对象 UNUR_PAR 的 sample 函数指针，参数是 _unur_dext_set_sample_sample_ft 类型的 sample 函数指针
    int unur_dext_set_sample(UNUR_PAR* parameters, _unur_dext_set_sample_sample_ft sample)
    
    // 获取 generator 的参数信息，返回一个指向参数数据的 void* 指针，参数 size 表示数据的大小
    void* unur_dext_get_params(UNUR_GEN* generator, size_t size)
    
    // 获取 generator 的分布参数信息，返回一个指向 double 数组的指针
    double* unur_dext_get_distrparams(UNUR_GEN* generator)
    
    // 获取 generator 的分布参数个数，返回 int 类型的结果
    int unur_dext_get_ndistrparams(UNUR_GEN* generator)
    
    // 创建一个新的 uniform distribution 参数对象 UNUR_PAR，参数是 UNUR_DISTR* dummy
    UNUR_PAR* unur_unif_new(UNUR_DISTR* dummy)
    
    // 将字符串表示的生成器类型转换为 UNUR_GEN* 类型的对象
    UNUR_GEN* unur_str2gen(char* string)
    
    // 将字符串表示的分布类型转换为 UNUR_DISTR* 类型的对象
    UNUR_DISTR* unur_str2distr(char* string)
    
    // 根据给定的字符串生成器类型、方法类型和随机数生成器 urng 创建一个新的生成器 UNUR_GEN*
    UNUR_GEN* unur_makegen_ssu(char* distrstr, char* methodstr, UNUR_URNG* urng)
    
    // 根据给定的分布类型 distribution、方法类型 methodstr 和随机数生成器 urng 创建一个新的生成器 UNUR_GEN*
    UNUR_GEN* unur_makegen_dsu(UNUR_DISTR* distribution, char* methodstr, UNUR_URNG* urng)
    
    // 根据给定的分布类型 distribution、方法类型 method 和生成器信息链表 mlist 创建一个新的参数对象 UNUR_PAR*
    UNUR_PAR* _unur_str2par(UNUR_DISTR* distribution, char* method, unur_slist** mlist)
    
    // 根据给定的参数对象 parameters 初始化生成器，返回初始化后的生成器 UNUR_GEN*
    UNUR_GEN* unur_init(UNUR_PAR* parameters)
    
    // 重新初始化给定的生成器 generator，返回 int 类型的结果
    int unur_reinit(UNUR_GEN* generator)
    
    // 从给定的离散分布生成一个样本，返回 int 类型的结果
    int unur_sample_discr(UNUR_GEN* generator)
    
    // 从给定的连续分布生成一个样本，返回 double 类型的结果
    double unur_sample_cont(UNUR_GEN* generator)
    
    // 从给定的多变量分布生成一个样本向量，vector 是一个指向 double 数组的指针，返回 int 类型的结果
    int unur_sample_vec(UNUR_GEN* generator, double* vector)
    
    // 从给定的多变量分布生成一个样本矩阵，matrix 是一个指向 double 数组的指针，返回 int 类型的结果
    int unur_sample_matr(UNUR_GEN* generator, double* matrix)
    
    // 根据给定的累积概率 U 计算逆量化（quantile）值，返回 double 类型的结果
    double unur_quantile(UNUR_GEN* generator, double U)
    
    // 释放给定的生成器 generator
    void unur_free(UNUR_GEN* generator)
    
    // 获取给定生成器 generator 的信息，help 参数表示是否打印详细帮助信息，返回一个指向信息字符串的指针
    char* unur_gen_info(UNUR_GEN* generator, int help)
    
    // 获取给定生成器 generator 的维度信息，返回 int 类型的结果
    int unur_get_dimension(UNUR_GEN* generator)
    
    // 获取给定生成器 generator 的标识符信息，返回一个指向标识符字符串的指针
    char* unur_get_genid(UNUR_GEN* generator)
    // 定义枚举类型，列出各种概率分布的名称
    cdef enum:
        UNUR_DISTR_GENERIC        // 通用概率分布
        UNUR_DISTR_CORDER         // CORDER 分布
        UNUR_DISTR_CXTRANS        // CXTRANS 分布
        UNUR_DISTR_CONDI          // CONDI 分布
        UNUR_DISTR_BETA           // Beta 分布
        UNUR_DISTR_CAUCHY         // Cauchy 分布
        UNUR_DISTR_CHI            // Chi 分布
        UNUR_DISTR_CHISQUARE      // Chi-square 分布
        UNUR_DISTR_EPANECHNIKOV   // Epanechnikov 分布
        UNUR_DISTR_EXPONENTIAL    // 指数分布
        UNUR_DISTR_EXTREME_I      // 极值分布类型 I
        UNUR_DISTR_EXTREME_II     // 极值分布类型 II
        UNUR_DISTR_F              // F 分布
        UNUR_DISTR_GAMMA          // Gamma 分布
        UNUR_DISTR_GHYP           // GHYP 分布
        UNUR_DISTR_GIG            // Generalized Inverse Gaussian (GIG) 分布
        UNUR_DISTR_GIG2           // GIG2 分布
        UNUR_DISTR_HYPERBOLIC     // 双曲线分布
        UNUR_DISTR_IG             // Inverse Gaussian (IG) 分布
        UNUR_DISTR_LAPLACE        // 拉普拉斯分布
        UNUR_DISTR_LOGISTIC       // Logistic 分布
        UNUR_DISTR_LOGNORMAL      // 对数正态分布
        UNUR_DISTR_LOMAX          // Lomax 分布
        UNUR_DISTR_NORMAL         // 正态分布
        UNUR_DISTR_GAUSSIAN       // Gaussian 分布
        UNUR_DISTR_PARETO         // Pareto 分布
        UNUR_DISTR_POWEREXPONENTIAL  // Power Exponential 分布
        UNUR_DISTR_RAYLEIGH       // Rayleigh 分布
        UNUR_DISTR_SLASH          // Slash 分布
        UNUR_DISTR_STUDENT        // Student's t 分布
        UNUR_DISTR_TRIANGULAR     // Triangular 分布
        UNUR_DISTR_UNIFORM        // 均匀分布
        UNUR_DISTR_BOXCAR         // Boxcar 分布
        UNUR_DISTR_WEIBULL        // Weibull 分布
        UNUR_DISTR_BURR_I         // Burr 分布类型 I
        UNUR_DISTR_BURR_II        // Burr 分布类型 II
        UNUR_DISTR_BURR_III       // Burr 分布类型 III
        UNUR_DISTR_BURR_IV        // Burr 分布类型 IV
        UNUR_DISTR_BURR_V         // Burr 分布类型 V
        UNUR_DISTR_BURR_VI        // Burr 分布类型 VI
        UNUR_DISTR_BURR_VII       // Burr 分布类型 VII
        UNUR_DISTR_BURR_VIII      // Burr 分布类型 VIII
        UNUR_DISTR_BURR_IX        // Burr 分布类型 IX
        UNUR_DISTR_BURR_X         // Burr 分布类型 X
        UNUR_DISTR_BURR_XI        // Burr 分布类型 XI
        UNUR_DISTR_BURR_XII       // Burr 分布类型 XII
        UNUR_DISTR_BINOMIAL       // 二项分布
        UNUR_DISTR_GEOMETRIC      // 几何分布
        UNUR_DISTR_HYPERGEOMETRIC // 超几何分布
        UNUR_DISTR_LOGARITHMIC    // 对数分布
        UNUR_DISTR_NEGATIVEBINOMIAL  // 负二项分布
        UNUR_DISTR_POISSON        // 泊松分布
        UNUR_DISTR_ZIPF           // Zipf 分布
        UNUR_DISTR_MCAUCHY        // 多变量 Cauchy 分布
        UNUR_DISTR_MNORMAL        // 多变量正态分布
        UNUR_DISTR_MSTUDENT       // 多变量 Student's t 分布
        UNUR_DISTR_MEXPONENTIAL   // 多变量指数分布
        UNUR_DISTR_COPULA         // Copula 分布
        UNUR_DISTR_MCORRELATION   // 多变量相关分布

    // 返回一个指向 Beta 分布对象的指针
    UNUR_DISTR* unur_distr_beta(double* params, int n_params)

    // 返回一个指向 Burr 分布对象的指针
    UNUR_DISTR* unur_distr_burr(double* params, int n_params)

    // 返回一个指向 Cauchy 分布对象的指针
    UNUR_DISTR* unur_distr_cauchy(double* params, int n_params)

    // 返回一个指向 Chi 分布对象的指针
    UNUR_DISTR* unur_distr_chi(double* params, int n_params)

    // 返回一个指向 Chi-square 分布对象的指针
    UNUR_DISTR* unur_distr_chisquare(double* params, int n_params)

    // 返回一个指向 Exponential 分布对象的指针
    UNUR_DISTR* unur_distr_exponential(double* params, int n_params)

    // 返回一个指向 ExtremeI 分布对象的指针
    UNUR_DISTR* unur_distr_extremeI(double* params, int n_params)

    // 返回一个指向 ExtremeII 分布对象的指针
    UNUR_DISTR* unur_distr_extremeII(double* params, int n_params)

    // 返回一个指向 F 分布对象的指针
    UNUR_DISTR* unur_distr_F(double* params, int n_params)

    // 返回一个指向 Gamma 分布对象的指针
    UNUR_DISTR* unur_distr_gamma(double* params, int n_params)

    // 返回一个指向 GHYP 分布对象的指针
    UNUR_DISTR* unur_distr_ghyp(double* params, int n_params)

    // 返回一个指向 GIG 分布对象的指针
    UNUR_DISTR* unur_distr_gig(double* params, int n_params)

    // 返回一个指向 GIG2 分布对象的指针
    UNUR_DISTR* unur_distr_gig2(double* params, int n_params)

    // 返回一个指向 Hyperbolic 分布对象的指针
    UNUR_DISTR* unur_distr_hyperbolic(double* params, int n_params)

    // 返回一个指向 IG 分布对象的指针
    UNUR_DISTR* unur_distr_ig(double* params, int n_params)

    // 返回一个指向 Laplace 分布对象的指针
    UNUR_DISTR* unur_distr_laplace(double* params, int n_params)

    // 返回一个指向 Logistic 分布对象的指针
    UNUR_DISTR* unur_distr_logistic(double* params, int n_params)

    // 返回一个指向 Lognormal 分布对象的指针
    UNUR_DISTR* unur_distr_lognormal(double* params, int n_params)

    // 返回一个指向 Lomax 分布对象的指针
    UNUR_DISTR* unur_distr_lomax(double* params, int n_params)

    // 返回一个指向 Normal 分布对象的指针
    UNUR_DISTR* unur_distr_normal(double* params, int n_params)

    // 返回一个指向 Pareto 分布对象的指针
    UNUR_DISTR* unur_distr_pareto(double* params, int n_params)
    # 创建一个双参数的幂指数分布对象
    UNUR_DISTR* unur_distr_powerexponential(double* params, int n_params)
    
    # 创建一个双参数的瑞利分布对象
    UNUR_DISTR* unur_distr_rayleigh(double* params, int n_params)
    
    # 创建一个双参数的斜对数分布对象
    UNUR_DISTR* unur_distr_slash(double* params, int n_params)
    
    # 创建一个双参数的学生 t 分布对象
    UNUR_DISTR* unur_distr_student(double* params, int n_params)
    
    # 创建一个双参数的三角分布对象
    UNUR_DISTR* unur_distr_triangular(double* params, int n_params)
    
    # 创建一个双参数的均匀分布对象
    UNUR_DISTR* unur_distr_uniform(double* params, int n_params)
    
    # 创建一个双参数的威布尔分布对象
    UNUR_DISTR* unur_distr_weibull(double* params, int n_params)
    
    # 创建一个多元正态分布对象，参数包括维度、均值向量和协方差矩阵
    UNUR_DISTR* unur_distr_multinormal(int dim, double* mean, double* covar)
    
    # 创建一个多元柯西分布对象，参数包括维度、均值向量和协方差矩阵
    UNUR_DISTR* unur_distr_multicauchy(int dim, double* mean, double* covar)
    
    # 创建一个多元学生 t 分布对象，参数包括维度、自由度、均值向量和协方差矩阵
    UNUR_DISTR* unur_distr_multistudent(int dim, double nu, double* mean, double* covar)
    
    # 创建一个多参数的多指数分布对象，参数包括维度、标准差向量和分布参数向量
    UNUR_DISTR* unur_distr_multiexponential(int dim, double* sigma, double* theta)
    
    # 创建一个多元概率分布对象，参数包括维度和秩相关系数矩阵
    UNUR_DISTR* unur_distr_copula(int dim, double* rankcorr)
    
    # 创建一个多元相关分布对象，参数为维度
    UNUR_DISTR* unur_distr_correlation(int n)
    
    # 创建一个双参数的二项分布对象
    UNUR_DISTR* unur_distr_binomial(double* params, int n_params)
    
    # 创建一个双参数的几何分布对象
    UNUR_DISTR* unur_distr_geometric(double* params, int n_params)
    
    # 创建一个双参数的超几何分布对象
    UNUR_DISTR* unur_distr_hypergeometric(double* params, int n_params)
    
    # 创建一个双参数的对数分布对象
    UNUR_DISTR* unur_distr_logarithmic(double* params, int n_params)
    
    # 创建一个双参数的负二项分布对象
    UNUR_DISTR* unur_distr_negativebinomial(double* params, int n_params)
    
    # 创建一个双参数的泊松分布对象
    UNUR_DISTR* unur_distr_poisson(double* params, int n_params)
    
    # 创建一个双参数的 Zipf 分布对象
    UNUR_DISTR* unur_distr_zipf(double* params, int n_params)
    
    # 设置一个新的文件流作为默认流，并返回新的文件流
    FILE* unur_set_stream(FILE* new_stream)
    
    # 获取当前设置的文件流
    FILE* unur_get_stream()
    
    # 设置指定参数对象的调试标志
    int unur_set_debug(UNUR_PAR* parameters, unsigned debug)
    
    # 修改生成器对象的调试标志
    int unur_chg_debug(UNUR_GEN* generator, unsigned debug)
    
    # 设置默认的调试标志
    int unur_set_default_debug(unsigned debug)
    
    # 获取当前的错误代码
    int unur_errno
    
    # 获取当前的错误代码并清零
    int unur_get_errno()
    
    # 重置当前的错误代码
    void unur_reset_errno()
    
    # 获取指定错误代码的错误信息字符串
    char* unur_get_strerror(int errnocode)
    
    # 设置新的错误处理函数，并返回之前的处理函数
    UNUR_ERROR_HANDLER* unur_set_error_handler(UNUR_ERROR_HANDLER* new_handler)
    
    # 关闭错误处理函数并返回之前的处理函数
    UNUR_ERROR_HANDLER* unur_set_error_handler_off()
    # 定义枚举类型，列出可能的返回状态码
    cdef enum:
        UNUR_SUCCESS                     # 成功返回状态
        UNUR_FAILURE                     # 失败返回状态
        UNUR_ERR_DISTR_SET               # 分布设置错误
        UNUR_ERR_DISTR_GET               # 分布获取错误
        UNUR_ERR_DISTR_NPARAMS           # 分布参数数量错误
        UNUR_ERR_DISTR_DOMAIN            # 分布定义域错误
        UNUR_ERR_DISTR_GEN               # 分布生成错误
        UNUR_ERR_DISTR_REQUIRED          # 缺少必要的分布信息错误
        UNUR_ERR_DISTR_UNKNOWN           # 未知分布错误
        UNUR_ERR_DISTR_INVALID           # 无效的分布错误
        UNUR_ERR_DISTR_DATA              # 分布数据错误
        UNUR_ERR_DISTR_PROP              # 分布属性错误
        UNUR_ERR_PAR_SET                 # 参数设置错误
        UNUR_ERR_PAR_VARIANT             # 参数变体错误
        UNUR_ERR_PAR_INVALID             # 无效的参数错误
        UNUR_ERR_GEN                     # 生成器错误
        UNUR_ERR_GEN_DATA                # 生成器数据错误
        UNUR_ERR_GEN_CONDITION           # 生成器条件错误
        UNUR_ERR_GEN_INVALID             # 无效的生成器错误
        UNUR_ERR_GEN_SAMPLING            # 生成器采样错误
        UNUR_ERR_NO_REINIT               # 不允许重新初始化错误
        UNUR_ERR_NO_QUANTILE             # 无分位数错误
        UNUR_ERR_URNG                    # 随机数生成器错误
        UNUR_ERR_URNG_MISS               # 缺少随机数生成器错误
        UNUR_ERR_STR                     # 字符串错误
        UNUR_ERR_STR_UNKNOWN             # 未知字符串错误
        UNUR_ERR_STR_SYNTAX              # 字符串语法错误
        UNUR_ERR_STR_INVALID             # 无效的字符串错误
        UNUR_ERR_FSTR_SYNTAX             # 浮点字符串语法错误
        UNUR_ERR_FSTR_DERIV              # 浮点字符串导数错误
        UNUR_ERR_DOMAIN                  # 域错误
        UNUR_ERR_ROUNDOFF                # 舍入误差错误
        UNUR_ERR_MALLOC                  # 内存分配错误
        UNUR_ERR_NULL                    # 空指针错误
        UNUR_ERR_COOKIE                  # Cookie 错误
        UNUR_ERR_GENERIC                 # 通用错误
        UNUR_ERR_SILENT                  # 静默错误
        UNUR_ERR_INF                     # 无穷大错误
        UNUR_ERR_NAN                     # 非数错误
        UNUR_ERR_COMPILE                 # 编译错误
        UNUR_ERR_SHOULD_NOT_HAPPEN       # 不应发生的错误

    # 定义正无穷大的双精度浮点数常量
    double INFINITY

    # 声明一个新的单链表结构的函数，返回一个空的单链表对象
    unur_slist* _unur_slist_new()

    # 向单链表中追加元素的函数，返回操作结果状态码
    int _unur_slist_append(unur_slist* slist, void* element)

    # 返回单链表中元素的个数
    int _unur_slist_length(unur_slist* slist)

    # 获取单链表中第 n 个元素的指针
    void* _unur_slist_get(unur_slist* slist, int n)

    # 替换单链表中第 n 个元素，并返回替换前的元素指针
    void* _unur_slist_replace(unur_slist* slist, int n, void* element)

    # 释放单链表及其元素占用的内存空间
    void _unur_slist_free(unur_slist* slist)
```