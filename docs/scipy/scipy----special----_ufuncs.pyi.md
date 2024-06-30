# `D:\src\scipysrc\scipy\scipy\special\_ufuncs.pyi`

```
# 从 typing 模块导入 Any 和 Dict 类型
from typing import Any, Dict
# 导入 numpy 库，并使用 np 别名表示
import numpy as np

# __all__ 是一个特殊的变量，定义了模块中所有公开的符号
__all__ = [
    'geterr',     # 函数 geterr 的名称在模块中是公开的
    'seterr',     # 函数 seterr 的名称在模块中是公开的
    'errstate',   # 函数 errstate 的名称在模块中是公开的
    'agm',        # 函数 agm 的名称在模块中是公开的
    'airy',       # 函数 airy 的名称在模块中是公开的
    'airye',      # 函数 airye 的名称在模块中是公开的
    'bdtr',       # 函数 bdtr 的名称在模块中是公开的
    'bdtrc',      # 函数 bdtrc 的名称在模块中是公开的
    'bdtri',      # 函数 bdtri 的名称在模块中是公开的
    'bdtrik',     # 函数 bdtrik 的名称在模块中是公开的
    'bdtrin',     # 函数 bdtrin 的名称在模块中是公开的
    'bei',        # 函数 bei 的名称在模块中是公开的
    'beip',       # 函数 beip 的名称在模块中是公开的
    'ber',        # 函数 ber 的名称在模块中是公开的
    'berp',       # 函数 berp 的名称在模块中是公开的
    'besselpoly', # 函数 besselpoly 的名称在模块中是公开的
    'beta',       # 函数 beta 的名称在模块中是公开的
    'betainc',    # 函数 betainc 的名称在模块中是公开的
    'betaincc',   # 函数 betaincc 的名称在模块中是公开的
    'betainccinv',# 函数 betainccinv 的名称在模块中是公开的
    'betaincinv', # 函数 betaincinv 的名称在模块中是公开的
    'betaln',     # 函数 betaln 的名称在模块中是公开的
    'binom',      # 函数 binom 的名称在模块中是公开的
    'boxcox',     # 函数 boxcox 的名称在模块中是公开的
    'boxcox1p',   # 函数 boxcox1p 的名称在模块中是公开的
    'btdtr',      # 函数 btdtr 的名称在模块中是公开的
    'btdtri',     # 函数 btdtri 的名称在模块中是公开的
    'btdtria',    # 函数 btdtria 的名称在模块中是公开的
    'btdtrib',    # 函数 btdtrib 的名称在模块中是公开的
    'cbrt',       # 函数 cbrt 的名称在模块中是公开的
    'chdtr',      # 函数 chdtr 的名称在模块中是公开的
    'chdtrc',     # 函数 chdtrc 的名称在模块中是公开的
    'chdtri',     # 函数 chdtri 的名称在模块中是公开的
    'chdtriv',    # 函数 chdtriv 的名称在模块中是公开的
    'chndtr',     # 函数 chndtr 的名称在模块中是公开的
    'chndtridf',  # 函数 chndtridf 的名称在模块中是公开的
    'chndtrinc',  # 函数 chndtrinc 的名称在模块中是公开的
    'chndtrix',   # 函数 chndtrix 的名称在模块中是公开的
    'cosdg',      # 函数 cosdg 的名称在模块中是公开的
    'cosm1',      # 函数 cosm1 的名称在模块中是公开的
    'cotdg',      # 函数 cotdg 的名称在模块中是公开的
    'dawsn',      # 函数 dawsn 的名称在模块中是公开的
    'ellipe',     # 函数 ellipe 的名称在模块中是公开的
    'ellipeinc',  # 函数 ellipeinc 的名称在模块中是公开的
    'ellipj',     # 函数 ellipj 的名称在模块中是公开的
    'ellipk',     # 函数 ellipk 的名称在模块中是公开的
    'ellipkinc',  # 函数 ellipkinc 的名称在模块中是公开的
    'ellipkm1',   # 函数 ellipkm1 的名称在模块中是公开的
    'elliprc',    # 函数 elliprc 的名称在模块中是公开的
    'elliprd',    # 函数 elliprd 的名称在模块中是公开的
    'elliprf',    # 函数 elliprf 的名称在模块中是公开的
    'elliprg',    # 函数 elliprg 的名称在模块中是公开的
    'elliprj',    # 函数 elliprj 的名称在模块中是公开的
    'entr',       # 函数 entr 的名称在模块中是公开的
    'erf',        # 函数 erf 的名称在模块中是公开的
    'erfc',       # 函数 erfc 的名称在模块中是公开的
    'erfcinv',    # 函数 erfcinv 的名称在模块中是公开的
    'erfcx',      # 函数 erfcx 的名称在模块中是公开的
    'erfi',       # 函数 erfi 的名称在模块中是公开的
    'erfinv',     # 函数 erfinv 的名称在模块中是公开的
    'eval_chebyc',# 函数 eval_chebyc 的名称在模块中是公开的
    'eval_chebys',# 函数 eval_chebys 的名称在模块中是公开的
    'eval_chebyt',# 函数 eval_chebyt 的名称在模块中是公开的
    'eval_chebyu',# 函数 eval_chebyu 的名称在模块中是公开的
    'eval_gegenbauer', # 函数 eval_gegenbauer 的名称在模块中是公开的
    'eval_genlaguerre',# 函数 eval_genlaguerre 的名称在模块中是公开的
    'eval_hermite',    # 函数 eval_hermite 的名称在模块中是公开的
    'eval_hermitenorm',# 函数 eval_hermitenorm 的名称在模块中是公开的
    'eval_jacobi',     # 函数 eval_jacobi 的名称在模块中是公开的
    'eval_laguerre',   # 函数 eval_laguerre 的名称在模块中是公开的
    'eval_legendre',   # 函数 eval_legendre 的名称在模块中是公开的
    'eval_sh_chebyt',  # 函数 eval_sh_chebyt 的名称在模块中是公开的
    'eval_sh_chebyu',  # 函数 eval_sh_chebyu 的名称在模块中是公开的
    'eval_sh_jacobi',  # 函数 eval_sh_jacobi 的名称在模块中是公开的
    'eval_sh_legendre',# 函数 eval_sh_legendre 的名称在模块中是公开的
    'exp1',        # 函数 exp1 的名称在模块中是公开的
    'exp10',       # 函数 exp10 的名称在模块中是公开的
    'exp2',        # 函数 exp2 的名称在模块中是公开的
    'expi',        # 函数 expi 的名称在模块中是公开的
    # 导入 scipy.special 模块，用于数学和科学计算中的特殊函数
    from scipy.special import (
        'pdtrik',            # 累积分布函数的反函数
        'poch',              # Pochhammer 符号
        'powm1',             # x^y - 1 的高效计算
        'pro_ang1',          # 相关的球谐函数
        'pro_ang1_cv',       # 相关的球谐函数的复共轭
        'pro_cv',            # 多重 Gamma 函数的一种形式
        'pro_rad1',          # 求解单参数伽马分布函数
        'pro_rad1_cv',       # 单参数伽马分布函数的复共轭
        'pro_rad2',          # 求解双参数伽马分布函数
        'pro_rad2_cv',       # 双参数伽马分布函数的复共轭
        'pseudo_huber',      # 伪 Huber 损失函数
        'psi',               # Psi (Ψ) 函数，也叫 Digamma 函数
        'radian',            # 将角度转换为弧度的乘法因子
        'rel_entr',          # 相对熵（KL 散度）
        'rgamma',            # 反 Gamma 函数
        'round',             # 四舍五入到最接近的整数
        'shichi',            # S 矢量函数的积分
        'sici',              # Sine 和 Cosine 积分
        'sindg',             # 将角度转换为弧度的乘法因子
        'smirnov',           # 小于等于两个样本的累积分布函数
        'smirnovi',          # 反两个样本的累积分布函数
        'spence',            # 斯宾斯函数
        'sph_harm',          # 球谐函数
        'stdtr',             # 累积分布函数的一种形式
        'stdtridf',          # 累积分布函数的自由度
        'stdtrit',           # 累积分布函数的三角形形式
        'struve',            # Struve 函数
        'tandg',             # 正切函数
        'tklmbda',           # 转换 KL 散度的函数
        'voigt_profile',     # Voigt 分布的复共轭
        'wofz',              # Faddeeva 函数
        'wright_bessel',     # Wright-Bessel 函数
        'wrightomega',       # Wright-Omega 函数
        'xlog1py',           # log(1+x) 的高效计算
        'xlogy',             # x * log(y) 的高效计算
        'y0',                # 贝塞尔函数的第一类的第一个种类
        'y1',                # 贝塞尔函数的第一类的第二种类
        'yn',                # 贝塞尔函数的第一类的第三种类
        'yv',                # 贝塞尔函数的第二类
        'yve',               # 贝塞尔函数的第二类的指数形式
        'zetac'              # Riemann zeta 函数的复共轭
    )
# 下面是一系列以 `_` 开头的全局变量，它们是 NumPy 的通用函数（ufunc）对象。
# 这些对象用于执行数学和科学计算中的各种函数操作。

_cosine_cdf: np.ufunc  # 余弦分布的累积分布函数
_cosine_invcdf: np.ufunc  # 余弦分布的反函数累积分布函数
_cospi: np.ufunc  # π 乘以输入数组的余弦
_ellip_harm: np.ufunc  # 椭圆函数的调和函数
_factorial: np.ufunc  # 阶乘函数
_igam_fac: np.ufunc  # γ 函数和上不完全 γ 函数的比值
_kolmogc: np.ufunc  # Kolmogorov 分布的累积分布函数
_kolmogci: np.ufunc  # Kolmogorov 分布的反函数累积分布函数
_kolmogp: np.ufunc  # Kolmogorov 分布的概率密度函数
_lambertw: np.ufunc  # Lambert W 函数的主分支
_lanczos_sum_expg_scaled: np.ufunc  # Lanczos 近似和调整指数函数的比例
_lgam1p: np.ufunc  # γ 函数的绝对误差测试函数
_log1pmx: np.ufunc  # log(1 + x) 的减 1
_riemann_zeta: np.ufunc  # 黎曼 zeta 函数
_scaled_exp1: np.ufunc  # expm1 的调整指数函数的比例
_sf_error_test_function: np.ufunc  # 科学计算库错误测试函数
_sinpi: np.ufunc  # π 乘以输入数组的正弦
_smirnovc: np.ufunc  # Smirnov 分布的累积分布函数
_smirnovci: np.ufunc  # Smirnov 分布的反函数累积分布函数
_smirnovp: np.ufunc  # Smirnov 分布的概率密度函数
_spherical_in: np.ufunc  # 球谐函数的第一类修正函数
_spherical_in_d: np.ufunc  # 球谐函数的第一类修正函数的导数
_spherical_jn: np.ufunc  # 球谐函数的第一类函数
_spherical_jn_d: np.ufunc  # 球谐函数的第一类函数的导数
_spherical_kn: np.ufunc  # 球谐函数的第二类函数
_spherical_kn_d: np.ufunc  # 球谐函数的第二类函数的导数
_spherical_yn: np.ufunc  # 球谐函数的第二类修正函数
_spherical_yn_d: np.ufunc  # 球谐函数的第二类修正函数的导数
_stirling2_inexact: np.ufunc  # Stirling 近似的第二类不精确函数
_struve_asymp_large_z: np.ufunc  # Struve 函数在大 z 下的渐近展开
_struve_bessel_series: np.ufunc  # Struve 函数的 Bessel 级数
_struve_power_series: np.ufunc  # Struve 函数的幂级数
_zeta: np.ufunc  # Riemann zeta 函数的广义形式
agm: np.ufunc  # 算术-几何平均值的计算
airy: np.ufunc  # Airy 函数
airye: np.ufunc  # Airy 函数的导数
bdtr: np.ufunc  # 贝塔分布的累积分布函数
bdtrc: np.ufunc  # 贝塔分布的补累积分布函数
bdtri: np.ufunc  # 贝塔分布的反函数累积分布函数
bdtrik: np.ufunc  # 贝塔分布的逆累积分布函数
bdtrin: np.ufunc  # 贝塔分布的不完全累积分布函数
bei: np.ufunc  # 贝塞尔函数 I 的负二类
beip: np.ufunc  # 贝塞尔函数 I 的负一类
ber: np.ufunc  # 贝塞尔函数 I 的二类
berp: np.ufunc  # 贝塞尔函数 I 的一类
besselpoly: np.ufunc  # 贝塞尔多项式
beta: np.ufunc  # 贝塔函数
betainc: np.ufunc  # 贝塔函数的不完全累积分布函数
betaincc: np.ufunc  # 贝塔函数的补不完全累积分布函数
betainccinv: np.ufunc  # 贝塔函数的反补不完全累积分布函数
betaincinv: np.ufunc  # 贝塔函数的反不完全累积分布函数
betaln: np.ufunc  # 贝塔函数的自然对数
binom: np.ufunc  # 二项式系数
boxcox1p: np.ufunc  # Box-Cox 变换的函数
boxcox: np.ufunc  # Box-Cox 变换的函数
btdtr: np.ufunc  # Beta 分布的累积分布函数
btdtri: np.ufunc  # Beta 分布的反函数累积分布函数
btdtria: np.ufunc  # Beta 分布的不完全累积分布函数
btdtrib: np.ufunc  # Beta 分布的补不完全累积分布函数
cbrt: np.ufunc  # 立方根函数
chdtr: np.ufunc  # 卡方分布的累积分布函数
chdtrc: np.ufunc  # 卡方分布的补累积分布函数
chdtri: np.ufunc  # 卡方分布的反函数累积分布函数
chdtriv: np.ufunc  # 自由度为 v 的卡方分布的累积分布函数的反函数
chndtr: np.ufunc  # 非中心卡方分布的累积分布函数
chndtridf: np.ufunc  # 非中心卡方分布的自由度
chndtrinc: np.ufunc  # 非中心卡方分布的不完全累积分布函数
chndtrix: np.ufunc  # 非中心卡方分布的反不完全累积分布函数
cosdg: np.ufunc  # 度角的余弦
cosm1: np.ufunc  # cos(x) - 1 的计算
cotdg: np.ufunc  # 度角的余切
dawsn: np.ufunc  # Dawson 函数
ellipe: np.ufunc  # 椭圆积分第二类
ellipeinc: np.ufunc  # 不完全椭圆积分第二类
ellipj: np.ufunc  # Jacobian 椭圆函数
ellipk: np.ufunc  # 椭圆积分第一类
ellipkinc: np.ufunc  # 不完全椭圆积分第一类
ellipkm1: np.ufunc  # 椭圆积分第一类减一
elliprc: np.ufunc  # 完全椭圆积分第三类
elliprd: np.ufunc  # 不完全椭圆积分第三类
elliprf: np.ufunc  # 完全椭圆积分第一类
elliprg: np.ufunc  # 完全椭圆积分第四类
elliprj: np.ufunc  # 完全椭圆积分第三类
entr: np.ufunc  # 熵函数
erf: np.ufunc  # 误差函数
erfc: np.ufunc  # 互补误差函数
erfcinv: np.ufunc  # 互补误差函数的
hyp1f1: np.ufunc
hyp2f1: np.ufunc
hyperu: np.ufunc
i0: np.ufunc
i0e: np.ufunc
i1: np.ufunc
i1e: np.ufunc
inv_boxcox1p: np.ufunc
inv_boxcox: np.ufunc
it2i0k0: np.ufunc
it2j0y0: np.ufunc
it2struve0: np.ufunc
itairy: np.ufunc
iti0k0: np.ufunc
itj0y0: np.ufunc
itmodstruve0: np.ufunc
itstruve0: np.ufunc
iv: np.ufunc
ive: np.ufunc
j0: np.ufunc
j1: np.ufunc
jn: np.ufunc
jv: np.ufunc
jve: np.ufunc
k0: np.ufunc
k0e: np.ufunc
k1: np.ufunc
k1e: np.ufunc
kei: np.ufunc
keip: np.ufunc
kelvin: np.ufunc
ker: np.ufunc
kerp: np.ufunc
kl_div: np.ufunc
kn: np.ufunc
kolmogi: np.ufunc
kolmogorov: np.ufunc
kv: np.ufunc
kve: np.ufunc
log1p: np.ufunc
log_expit: np.ufunc
log_ndtr: np.ufunc
log_wright_bessel: np.ufunc
loggamma: np.ufunc
logit: np.ufunc
lpmv: np.ufunc
mathieu_a: np.ufunc
mathieu_b: np.ufunc
mathieu_cem: np.ufunc
mathieu_modcem1: np.ufunc
mathieu_modcem2: np.ufunc
mathieu_modsem1: np.ufunc
mathieu_modsem2: np.ufunc
mathieu_sem: np.ufunc
modfresnelm: np.ufunc
modfresnelp: np.ufunc
modstruve: np.ufunc
nbdtr: np.ufunc
nbdtrc: np.ufunc
nbdtri: np.ufunc
nbdtrik: np.ufunc
nbdtrin: np.ufunc
ncfdtr: np.ufunc
ncfdtri: np.ufunc
ncfdtridfd: np.ufunc
ncfdtridfn: np.ufunc
ncfdtrinc: np.ufunc
nctdtr: np.ufunc
nctdtridf: np.ufunc
nctdtrinc: np.ufunc
nctdtrit: np.ufunc
ndtr: np.ufunc
ndtri: np.ufunc
ndtri_exp: np.ufunc
nrdtrimn: np.ufunc
nrdtrisd: np.ufunc
obl_ang1: np.ufunc
obl_ang1_cv: np.ufunc
obl_cv: np.ufunc
obl_rad1: np.ufunc
obl_rad1_cv: np.ufunc
obl_rad2: np.ufunc
obl_rad2_cv: np.ufunc
owens_t: np.ufunc
pbdv: np.ufunc
pbvv: np.ufunc
pbwa: np.ufunc
pdtr: np.ufunc
pdtrc: np.ufunc
pdtri: np.ufunc
pdtrik: np.ufunc
poch: np.ufunc
powm1: np.ufunc
pro_ang1: np.ufunc
pro_ang1_cv: np.ufunc
pro_cv: np.ufunc
pro_rad1: np.ufunc
pro_rad1_cv: np.ufunc
pro_rad2: np.ufunc
pro_rad2_cv: np.ufunc
pseudo_huber: np.ufunc
psi: np.ufunc
radian: np.ufunc
rel_entr: np.ufunc
rgamma: np.ufunc
round: np.ufunc
shichi: np.ufunc
sici: np.ufunc
sindg: np.ufunc
smirnov: np.ufunc
smirnovi: np.ufunc
spence: np.ufunc
sph_harm: np.ufunc
stdtr: np.ufunc
stdtridf: np.ufunc
stdtrit: np.ufunc
struve: np.ufunc
tandg: np.ufunc
tklmbda: np.ufunc
voigt_profile: np.ufunc
wofz: np.ufunc
wright_bessel: np.ufunc
wrightomega: np.ufunc
xlog1py: np.ufunc
xlogy: np.ufunc
y0: np.ufunc
y1: np.ufunc
yn: np.ufunc
yv: np.ufunc
yve: np.ufunc
zetac: np.ufunc
```