# `D:\src\scipysrc\scipy\scipy\special\tests\test_cython_special.py`

```
# 导入来自未来的注释特性以及类型提示 Callable
from __future__ import annotations
from typing import Callable

# 导入 pytest 测试框架，以及从 itertools 中导入 product 函数
import pytest
from itertools import product

# 从 numpy.testing 中导入 assert_allclose 和 suppress_warnings 函数
from numpy.testing import assert_allclose, suppress_warnings

# 从 scipy 库中导入 special 模块和 cython_special 模块
from scipy import special
from scipy.special import cython_special

# 定义布尔类型的测试点列表
bint_points = [True, False]
# 定义整数类型的测试点列表
int_points = [-10, -1, 1, 10]
# 定义实数类型的测试点列表
real_points = [-10.0, -1.0, 1.0, 10.0]
# 定义复数类型的测试点列表，使用 product 函数生成笛卡尔积
complex_points = [complex(*tup) for tup in product(real_points, repeat=2)]

# 定义 Cython 函数参数类型映射字典
CYTHON_SIGNATURE_MAP = {
    'b': 'bint',
    'f': 'float',
    'd': 'double',
    'g': 'long double',
    'F': 'float complex',
    'D': 'double complex',
    'G': 'long double complex',
    'i': 'int',
    'l': 'long'
}

# 定义测试点字典，每种参数类型对应相应的测试点列表
TEST_POINTS = {
    'b': bint_points,
    'f': real_points,
    'd': real_points,
    'g': real_points,
    'F': complex_points,
    'D': complex_points,
    'G': complex_points,
    'i': int_points,
    'l': int_points,
}

# 定义参数元组列表 PARAMS，每个元组包含两个 special 模块和 cython_special 模块中的函数，
# 以及函数参数签名字符串列表和描述（可选的）
PARAMS: list[tuple[Callable, Callable, tuple[str, ...], str | None]] = [
    # AGM 函数
    (special.agm, cython_special.agm, ('dd',), None),
    # Airy 函数
    (special.airy, cython_special._airy_pywrap, ('d', 'D'), None),
    # Airy 函数（修正）
    (special.airye, cython_special._airye_pywrap, ('d', 'D'), None),
    # Beta 分布函数
    (special.bdtr, cython_special.bdtr, ('dld', 'ddd'), None),
    (special.bdtrc, cython_special.bdtrc, ('dld', 'ddd'), None),
    (special.bdtri, cython_special.bdtri, ('dld', 'ddd'), None),
    (special.bdtrik, cython_special.bdtrik, ('ddd',), None),
    (special.bdtrin, cython_special.bdtrin, ('ddd',), None),
    # 贝塞尔函数（第一类）
    (special.bei, cython_special.bei, ('d',), None),
    (special.beip, cython_special.beip, ('d',), None),
    # 贝塞尔函数（第二类）
    (special.ber, cython_special.ber, ('d',), None),
    (special.berp, cython_special.berp, ('d',), None),
    # 贝塞尔多项式
    (special.besselpoly, cython_special.besselpoly, ('ddd',), None),
    # Beta 函数
    (special.beta, cython_special.beta, ('dd',), None),
    # 完全 Beta 函数
    (special.betainc, cython_special.betainc, ('ddd',), None),
    (special.betaincc, cython_special.betaincc, ('ddd',), None),
    (special.betaincinv, cython_special.betaincinv, ('ddd',), None),
    (special.betainccinv, cython_special.betainccinv, ('ddd',), None),
    # Beta 函数的自然对数
    (special.betaln, cython_special.betaln, ('dd',), None),
    # 二项式系数
    (special.binom, cython_special.binom, ('dd',), None),
    # Box-Cox 变换
    (special.boxcox, cython_special.boxcox, ('dd',), None),
    (special.boxcox1p, cython_special.boxcox1p, ('dd',), None),
    # Beta 分布累积分布函数
    (special.btdtr, cython_special.btdtr, ('ddd',), None),
    (special.btdtri, cython_special.btdtri, ('ddd',), None),
    (special.btdtria, cython_special.btdtria, ('ddd',), None),
    (special.btdtrib, cython_special.btdtrib, ('ddd',), None),
    # 立方根函数
    (special.cbrt, cython_special.cbrt, ('d',), None),
    # 卡方分布的累积分布函数
    (special.chdtr, cython_special.chdtr, ('dd',), None),
    (special.chdtrc, cython_special.chdtrc, ('dd',), None),
    (special.chdtri, cython_special.chdtri, ('dd',), None),
    (special.chdtriv, cython_special.chdtriv, ('dd',), None),
    # 非中心卡方分布的累积分布函数
    (special.chndtr, cython_special.chndtr, ('ddd',), None),
    (special.chndtridf, cython_special.chndtridf, ('ddd',), None),
    # 调用特殊函数 special.chndtrinc 和 cython_special.chndtrinc，传入参数 ('ddd',) 和 None
    (special.chndtrinc, cython_special.chndtrinc, ('ddd',), None),
    # 调用特殊函数 special.chndtrix 和 cython_special.chndtrix，传入参数 ('ddd',) 和 None
    (special.chndtrix, cython_special.chndtrix, ('ddd',), None),
    # 调用特殊函数 special.cosdg 和 cython_special.cosdg，传入参数 ('d',) 和 None
    (special.cosdg, cython_special.cosdg, ('d',), None),
    # 调用特殊函数 special.cosm1 和 cython_special.cosm1，传入参数 ('d',) 和 None
    (special.cosm1, cython_special.cosm1, ('d',), None),
    # 调用特殊函数 special.cotdg 和 cython_special.cotdg，传入参数 ('d',) 和 None
    (special.cotdg, cython_special.cotdg, ('d',), None),
    # 调用特殊函数 special.dawsn 和 cython_special.dawsn，传入参数 ('d', 'D') 和 None
    (special.dawsn, cython_special.dawsn, ('d', 'D'), None),
    # 调用特殊函数 special.ellipe 和 cython_special.ellipe，传入参数 ('d',) 和 None
    (special.ellipe, cython_special.ellipe, ('d',), None),
    # 调用特殊函数 special.ellipeinc 和 cython_special.ellipeinc，传入参数 ('dd',) 和 None
    (special.ellipeinc, cython_special.ellipeinc, ('dd',), None),
    # 调用特殊函数 special.ellipj 和 cython_special._ellipj_pywrap，传入参数 ('dd',) 和 None
    (special.ellipj, cython_special._ellipj_pywrap, ('dd',), None),
    # 调用特殊函数 special.ellipkinc 和 cython_special.ellipkinc，传入参数 ('dd',) 和 None
    (special.ellipkinc, cython_special.ellipkinc, ('dd',), None),
    # 调用特殊函数 special.ellipkm1 和 cython_special.ellipkm1，传入参数 ('d',) 和 None
    (special.ellipkm1, cython_special.ellipkm1, ('d',), None),
    # 调用特殊函数 special.ellipk 和 cython_special.ellipk，传入参数 ('d',) 和 None
    (special.ellipk, cython_special.ellipk, ('d',), None),
    # 调用特殊函数 special.elliprc 和 cython_special.elliprc，传入参数 ('dd', 'DD') 和 None
    (special.elliprc, cython_special.elliprc, ('dd', 'DD'), None),
    # 调用特殊函数 special.elliprd 和 cython_special.elliprd，传入参数 ('ddd', 'DDD') 和 None
    (special.elliprd, cython_special.elliprd, ('ddd', 'DDD'), None),
    # 调用特殊函数 special.elliprf 和 cython_special.elliprf，传入参数 ('ddd', 'DDD') 和 None
    (special.elliprf, cython_special.elliprf, ('ddd', 'DDD'), None),
    # 调用特殊函数 special.elliprg 和 cython_special.elliprg，传入参数 ('ddd', 'DDD') 和 None
    (special.elliprg, cython_special.elliprg, ('ddd', 'DDD'), None),
    # 调用特殊函数 special.elliprj 和 cython_special.elliprj，传入参数 ('dddd', 'DDDD') 和 None
    (special.elliprj, cython_special.elliprj, ('dddd', 'DDDD'), None),
    # 调用特殊函数 special.entr 和 cython_special.entr，传入参数 ('d',) 和 None
    (special.entr, cython_special.entr, ('d',), None),
    # 调用特殊函数 special.erf 和 cython_special.erf，传入参数 ('d', 'D') 和 None
    (special.erf, cython_special.erf, ('d', 'D'), None),
    # 调用特殊函数 special.erfc 和 cython_special.erfc，传入参数 ('d', 'D') 和 None
    (special.erfc, cython_special.erfc, ('d', 'D'), None),
    # 调用特殊函数 special.erfcx 和 cython_special.erfcx，传入参数 ('d', 'D') 和 None
    (special.erfcx, cython_special.erfcx, ('d', 'D'), None),
    # 调用特殊函数 special.erfi 和 cython_special.erfi，传入参数 ('d', 'D') 和 None
    (special.erfi, cython_special.erfi, ('d', 'D'), None),
    # 调用特殊函数 special.erfinv 和 cython_special.erfinv，传入参数 ('d',) 和 None
    (special.erfinv, cython_special.erfinv, ('d',), None),
    # 调用特殊函数 special.erfcinv 和 cython_special.erfcinv，传入参数 ('d',) 和 None
    (special.erfcinv, cython_special.erfcinv, ('d',), None),
    # 调用特殊函数 special.eval_chebyc 和 cython_special.eval_chebyc，传入参数 ('dd', 'dD', 'ld') 和 None
    (special.eval_chebyc, cython_special.eval_chebyc, ('dd', 'dD', 'ld'), None),
    # 调用特殊函数 special.eval_chebys 和 cython_special.eval_chebys，传入参数 ('dd', 'dD', 'ld') 和 'd and l differ for negative int'
    (special.eval_chebys, cython_special.eval_chebys, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    # 调用特殊函数 special.eval_chebyt 和 cython_special.eval_chebyt，传入参数 ('dd', 'dD', 'ld') 和 'd and l differ for negative int'
    (special.eval_chebyt, cython_special.eval_chebyt, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    # 调用特殊函数 special.eval_chebyu 和 cython_special.eval_chebyu，传入参数 ('dd', 'dD', 'ld') 和 'd and l differ for negative int'
    (special.eval_chebyu, cython_special.eval_chebyu, ('dd', 'dD', 'ld'),
     'd and l differ for negative int'),
    # 调用特殊函数 special.eval_gegenbauer 和 cython_special.eval_gegenbauer，传入参数 ('ddd', 'ddD', 'ldd') 和 'd and l differ for negative int'
    (special.eval_gegenbauer, cython_special.eval_gegenbauer, ('ddd', 'ddD', 'ldd'),
     'd and l differ for negative int'),
    # 调用特殊函数 special.eval_genlaguerre 和 cython_special.eval_genlaguerre，传入参数 ('ddd', 'ddD', 'ldd') 和 'd and l differ for negative int'
    (special.eval_genlaguerre, cython_special.eval_genlaguerre, ('ddd', 'ddD', 'ldd'),
     'd and l differ for negative int'),
    # 调用特殊函数 special.eval_hermite 和 cython_special.eval_hermite，传入参数 ('ld',) 和 None
    (special.eval_hermite, cython_special.eval_hermite, ('ld',), None),
    # 调用特殊函数 special.eval_hermitenorm 和 cython_special.eval_hermitenorm，传入参数 ('ld',) 和 None
    (special.eval_hermitenorm, cython_special.eval_hermitenorm, ('ld',), None),
    # 调用特殊函数 special.eval_jacobi 和 cython_special.eval_jacobi，传入参数 ('dddd', 'dddD', 'lddd') 和 'd and l differ for negative int'
    (special.eval_jacobi, cython_special.eval_jacobi, ('dddd', 'dddD', 'lddd'),
     'd and l differ for negative int'),
    # 调用特殊函数
    # 使用 special 模块中的 eval_sh_legendre 函数和 cython_special 模块中的 eval_sh_legendre 函数，
    # 参数类型为 ('dd', 'dD', 'ld')，无需额外的注释信息
    (special.eval_sh_legendre, cython_special.eval_sh_legendre, ('dd', 'dD', 'ld'), None),

    # 使用 special 模块中的 exp1 函数和 cython_special 模块中的 exp1 函数，
    # 参数类型为 ('d', 'D')，无需额外的注释信息
    (special.exp1, cython_special.exp1, ('d', 'D'), None),

    # 使用 special 模块中的 exp10 函数和 cython_special 模块中的 exp10 函数，
    # 参数类型为 ('d',)，无需额外的注释信息
    (special.exp10, cython_special.exp10, ('d',), None),

    # 使用 special 模块中的 exp2 函数和 cython_special 模块中的 exp2 函数，
    # 参数类型为 ('d',)，无需额外的注释信息
    (special.exp2, cython_special.exp2, ('d',), None),

    # 使用 special 模块中的 expi 函数和 cython_special 模块中的 expi 函数，
    # 参数类型为 ('d', 'D')，无需额外的注释信息
    (special.expi, cython_special.expi, ('d', 'D'), None),

    # 使用 special 模块中的 expit 函数和 cython_special 模块中的 expit 函数，
    # 参数类型为 ('f', 'd', 'g')，无需额外的注释信息
    (special.expit, cython_special.expit, ('f', 'd', 'g'), None),

    # 使用 special 模块中的 expm1 函数和 cython_special 模块中的 expm1 函数，
    # 参数类型为 ('d', 'D')，无需额外的注释信息
    (special.expm1, cython_special.expm1, ('d', 'D'), None),

    # 使用 special 模块中的 expn 函数和 cython_special 模块中的 expn 函数，
    # 参数类型为 ('ld', 'dd')，无需额外的注释信息
    (special.expn, cython_special.expn, ('ld', 'dd'), None),

    # 使用 special 模块中的 exprel 函数和 cython_special 模块中的 exprel 函数，
    # 参数类型为 ('d',)，无需额外的注释信息
    (special.exprel, cython_special.exprel, ('d',), None),

    # 使用 special 模块中的 fdtr 函数和 cython_special 模块中的 fdtr 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.fdtr, cython_special.fdtr, ('ddd',), None),

    # 使用 special 模块中的 fdtrc 函数和 cython_special 模块中的 fdtrc 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.fdtrc, cython_special.fdtrc, ('ddd',), None),

    # 使用 special 模块中的 fdtri 函数和 cython_special 模块中的 fdtri 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.fdtri, cython_special.fdtri, ('ddd',), None),

    # 使用 special 模块中的 fdtridfd 函数和 cython_special 模块中的 fdtridfd 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.fdtridfd, cython_special.fdtridfd, ('ddd',), None),

    # 使用 special 模块中的 fresnel 函数和 cython_special 模块中的 _fresnel_pywrap 函数，
    # 参数类型为 ('d', 'D')，无需额外的注释信息
    (special.fresnel, cython_special._fresnel_pywrap, ('d', 'D'), None),

    # 使用 special 模块中的 gamma 函数和 cython_special 模块中的 gamma 函数，
    # 参数类型为 ('d', 'D')，无需额外的注释信息
    (special.gamma, cython_special.gamma, ('d', 'D'), None),

    # 使用 special 模块中的 gammainc 函数和 cython_special 模块中的 gammainc 函数，
    # 参数类型为 ('dd',)，无需额外的注释信息
    (special.gammainc, cython_special.gammainc, ('dd',), None),

    # 使用 special 模块中的 gammaincc 函数和 cython_special 模块中的 gammaincc 函数，
    # 参数类型为 ('dd',)，无需额外的注释信息
    (special.gammaincc, cython_special.gammaincc, ('dd',), None),

    # 使用 special 模块中的 gammainccinv 函数和 cython_special 模块中的 gammainccinv 函数，
    # 参数类型为 ('dd',)，无需额外的注释信息
    (special.gammainccinv, cython_special.gammainccinv, ('dd',), None),

    # 使用 special 模块中的 gammaincinv 函数和 cython_special 模块中的 gammaincinv 函数，
    # 参数类型为 ('dd',)，无需额外的注释信息
    (special.gammaincinv, cython_special.gammaincinv, ('dd',), None),

    # 使用 special 模块中的 gammaln 函数和 cython_special 模块中的 gammaln 函数，
    # 参数类型为 ('d',)，无需额外的注释信息
    (special.gammaln, cython_special.gammaln, ('d',), None),

    # 使用 special 模块中的 gammasgn 函数和 cython_special 模块中的 gammasgn 函数，
    # 参数类型为 ('d',)，无需额外的注释信息
    (special.gammasgn, cython_special.gammasgn, ('d',), None),

    # 使用 special 模块中的 gdtr 函数和 cython_special 模块中的 gdtr 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.gdtr, cython_special.gdtr, ('ddd',), None),

    # 使用 special 模块中的 gdtrc 函数和 cython_special 模块中的 gdtrc 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.gdtrc, cython_special.gdtrc, ('ddd',), None),

    # 使用 special 模块中的 gdtria 函数和 cython_special 模块中的 gdtria 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.gdtria, cython_special.gdtria, ('ddd',), None),

    # 使用 special 模块中的 gdtrib 函数和 cython_special 模块中的 gdtrib 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.gdtrib, cython_special.gdtrib, ('ddd',), None),

    # 使用 special 模块中的 gdtrix 函数和 cython_special 模块中的 gdtrix 函数，
    # 参数类型为 ('ddd',)，无需额外的注释信息
    (special.gdtrix, cython_special.gdtrix, ('ddd',), None),

    # 使用 special 模块中的 hankel1 函数和 cython_special 模块中的 hankel1 函数，
    # 参数类型为 ('dD',)，无需额外的注释信息
    (special.hankel1, cython_special.hankel1, ('dD',), None),

    # 使用 special 模块中的 hankel1e 函数和 cython_special 模块中的 hankel1e 函数，
    # 参数类型为 ('dD',)，无需额外的注释信息
    (special.hankel1e, cython_special.hankel1e, ('dD',), None),

    # 使用 special 模块中的 hankel2 函数和 cython_special 模块中的 hankel2 函数，
    # 参数类型为 ('dD',)，无需额外的注释
    (special.itmodstruve0, cython_special.itmodstruve0, ('d',), None),
    # 调用特殊函数库中的 itmodstruve0 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.itstruve0, cython_special.itstruve0, ('d',), None),
    # 调用特殊函数库中的 itstruve0 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.iv, cython_special.iv, ('dd', 'dD'), None),
    # 调用特殊函数库中的 iv 函数和对应的 Cython 版本，参数为 ('dd', 'dD')，无返回值
    (special.ive, cython_special.ive, ('dd', 'dD'), None),
    # 调用特殊函数库中的 ive 函数和对应的 Cython 版本，参数为 ('dd', 'dD')，无返回值
    (special.j0, cython_special.j0, ('d',), None),
    # 调用特殊函数库中的 j0 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.j1, cython_special.j1, ('d',), None),
    # 调用特殊函数库中的 j1 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.jv, cython_special.jv, ('dd', 'dD'), None),
    # 调用特殊函数库中的 jv 函数和对应的 Cython 版本，参数为 ('dd', 'dD')，无返回值
    (special.jve, cython_special.jve, ('dd', 'dD'), None),
    # 调用特殊函数库中的 jve 函数和对应的 Cython 版本，参数为 ('dd', 'dD')，无返回值
    (special.k0, cython_special.k0, ('d',), None),
    # 调用特殊函数库中的 k0 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.k0e, cython_special.k0e, ('d',), None),
    # 调用特殊函数库中的 k0e 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.k1, cython_special.k1, ('d',), None),
    # 调用特殊函数库中的 k1 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.k1e, cython_special.k1e, ('d',), None),
    # 调用特殊函数库中的 k1e 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.kei, cython_special.kei, ('d',), None),
    # 调用特殊函数库中的 kei 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.keip, cython_special.keip, ('d',), None),
    # 调用特殊函数库中的 keip 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.kelvin, cython_special._kelvin_pywrap, ('d',), None),
    # 调用特殊函数库中的 kelvin 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.ker, cython_special.ker, ('d',), None),
    # 调用特殊函数库中的 ker 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.kerp, cython_special.kerp, ('d',), None),
    # 调用特殊函数库中的 kerp 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.kl_div, cython_special.kl_div, ('dd',), None),
    # 调用特殊函数库中的 kl_div 函数和对应的 Cython 版本，参数为 ('dd',)，无返回值
    (special.kn, cython_special.kn, ('ld', 'dd'), None),
    # 调用特殊函数库中的 kn 函数和对应的 Cython 版本，参数为 ('ld', 'dd')，无返回值
    (special.kolmogi, cython_special.kolmogi, ('d',), None),
    # 调用特殊函数库中的 kolmogi 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.kolmogorov, cython_special.kolmogorov, ('d',), None),
    # 调用特殊函数库中的 kolmogorov 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.kv, cython_special.kv, ('dd', 'dD'), None),
    # 调用特殊函数库中的 kv 函数和对应的 Cython 版本，参数为 ('dd', 'dD')，无返回值
    (special.kve, cython_special.kve, ('dd', 'dD'), None),
    # 调用特殊函数库中的 kve 函数和对应的 Cython 版本，参数为 ('dd', 'dD')，无返回值
    (special.log1p, cython_special.log1p, ('d', 'D'), None),
    # 调用特殊函数库中的 log1p 函数和对应的 Cython 版本，参数为 ('d', 'D')，无返回值
    (special.log_expit, cython_special.log_expit, ('f', 'd', 'g'), None),
    # 调用特殊函数库中的 log_expit 函数和对应的 Cython 版本，参数为 ('f', 'd', 'g')，无返回值
    (special.log_ndtr, cython_special.log_ndtr, ('d', 'D'), None),
    # 调用特殊函数库中的 log_ndtr 函数和对应的 Cython 版本，参数为 ('d', 'D')，无返回值
    (special.log_wright_bessel, cython_special.log_wright_bessel, ('ddd',), None),
    # 调用特殊函数库中的 log_wright_bessel 函数和对应的 Cython 版本，参数为 ('ddd',)，无返回值
    (special.ndtri_exp, cython_special.ndtri_exp, ('d',), None),
    # 调用特殊函数库中的 ndtri_exp 函数和对应的 Cython 版本，参数为 ('d',)，无返回值
    (special.loggamma, cython_special.loggamma, ('D',), None),
    # 调用特殊函数库中的 loggamma 函数和对应的 Cython 版本，参数为 ('D',)，无返回值
    (special.logit, cython_special.logit, ('f', 'd', 'g'), None),
    # 调用特殊函数库中的 logit 函数和对应的 Cython 版本，参数为 ('f', 'd', 'g')，无返回值
    (special.lpmv, cython_special.lpmv, ('ddd',), None),
    # 调用特殊函数库中的 lpmv 函数和对应的 Cython 版本，参数为 ('ddd',)，无返回值
    (special.mathieu_a, cython_special.mathieu_a, ('dd',), None),
    # 调用特殊函数库中的 mathieu_a 函数和对应的 Cython 版本，参数为 ('dd',)，无返回值
    (special.mathieu_b, cython_special.mathieu_b, ('dd',), None),
    # 调用特殊函数库中的 mathieu_b 函数和对应的 Cython 版本，参数为 ('dd',)，无返回值
    (special.mathieu_cem, cython_special._mathieu_cem_pywrap, ('ddd',), None),
    # 调用特殊函数库中的 mathieu_cem 函数和对应的 Cython 版本，参数为 ('ddd',)，无返回值
    (special.mathieu_modcem1, cython_special._mathieu_modcem1_pywrap,
    # 调用 special 和 cython_special 模块中的函数，并传入相应的参数
    (special.nbdtrin, cython_special.nbdtrin, ('ddd',), None),
    (special.ncfdtr, cython_special.ncfdtr, ('dddd',), None),
    (special.ncfdtri, cython_special.ncfdtri, ('dddd',), None),
    (special.ncfdtridfd, cython_special.ncfdtridfd, ('dddd',), None),
    (special.ncfdtridfn, cython_special.ncfdtridfn, ('dddd',), None),
    (special.ncfdtrinc, cython_special.ncfdtrinc, ('dddd',), None),
    (special.nctdtr, cython_special.nctdtr, ('ddd',), None),
    (special.nctdtridf, cython_special.nctdtridf, ('ddd',), None),
    (special.nctdtrinc, cython_special.nctdtrinc, ('ddd',), None),
    (special.nctdtrit, cython_special.nctdtrit, ('ddd',), None),
    (special.ndtr, cython_special.ndtr, ('d', 'D'), None),
    (special.ndtri, cython_special.ndtri, ('d',), None),
    (special.nrdtrimn, cython_special.nrdtrimn, ('ddd',), None),
    (special.nrdtrisd, cython_special.nrdtrisd, ('ddd',), None),
    (special.obl_ang1, cython_special._obl_ang1_pywrap, ('dddd',), None),
    (special.obl_ang1_cv, cython_special._obl_ang1_cv_pywrap, ('ddddd',), None),
    (special.obl_cv, cython_special.obl_cv, ('ddd',), None),
    (special.obl_rad1, cython_special._obl_rad1_pywrap, ('dddd',), "see gh-6211"),
    (special.obl_rad1_cv, cython_special._obl_rad1_cv_pywrap, ('ddddd',),
     "see gh-6211"),
    (special.obl_rad2, cython_special._obl_rad2_pywrap, ('dddd',), "see gh-6211"),
    (special.obl_rad2_cv, cython_special._obl_rad2_cv_pywrap, ('ddddd',),
     "see gh-6211"),
    (special.pbdv, cython_special._pbdv_pywrap, ('dd',), None),
    (special.pbvv, cython_special._pbvv_pywrap, ('dd',), None),
    (special.pbwa, cython_special._pbwa_pywrap, ('dd',), None),
    (special.pdtr, cython_special.pdtr, ('dd', 'dd'), None),
    (special.pdtrc, cython_special.pdtrc, ('dd', 'dd'), None),
    (special.pdtri, cython_special.pdtri, ('ld', 'dd'), None),
    (special.pdtrik, cython_special.pdtrik, ('dd',), None),
    (special.poch, cython_special.poch, ('dd',), None),
    (special.powm1, cython_special.powm1, ('dd',), None),
    (special.pro_ang1, cython_special._pro_ang1_pywrap, ('dddd',), None),
    (special.pro_ang1_cv, cython_special._pro_ang1_cv_pywrap, ('ddddd',), None),
    (special.pro_cv, cython_special.pro_cv, ('ddd',), None),
    (special.pro_rad1, cython_special._pro_rad1_pywrap, ('dddd',), "see gh-6211"),
    (special.pro_rad1_cv, cython_special._pro_rad1_cv_pywrap, ('ddddd',),
     "see gh-6211"),
    (special.pro_rad2, cython_special._pro_rad2_pywrap, ('dddd',), "see gh-6211"),
    (special.pro_rad2_cv, cython_special._pro_rad2_cv_pywrap, ('ddddd',),
     "see gh-6211"),
    (special.pseudo_huber, cython_special.pseudo_huber, ('dd',), None),
    (special.psi, cython_special.psi, ('d', 'D'), None),
    (special.radian, cython_special.radian, ('ddd',), None),
    (special.rel_entr, cython_special.rel_entr, ('dd',), None),
    (special.rgamma, cython_special.rgamma, ('d', 'D'), None),
    (special.round, cython_special.round, ('d',), None),
    (special.spherical_jn, cython_special.spherical_jn, ('ld', 'ldb', 'lD', 'lDb'),
     None),
    # Tuple containing the functions special.spherical_jn and cython_special.spherical_jn,
    # with associated specializations ('ld', 'ldb', 'lD', 'lDb'). Fourth element is None.

    (special.spherical_yn, cython_special.spherical_yn, ('ld', 'ldb', 'lD', 'lDb'),
     None),
    # Tuple containing the functions special.spherical_yn and cython_special.spherical_yn,
    # with associated specializations ('ld', 'ldb', 'lD', 'lDb'). Fourth element is None.

    (special.spherical_in, cython_special.spherical_in, ('ld', 'ldb', 'lD', 'lDb'),
     None),
    # Tuple containing the functions special.spherical_in and cython_special.spherical_in,
    # with associated specializations ('ld', 'ldb', 'lD', 'lDb'). Fourth element is None.

    (special.spherical_kn, cython_special.spherical_kn, ('ld', 'ldb', 'lD', 'lDb'),
     None),
    # Tuple containing the functions special.spherical_kn and cython_special.spherical_kn,
    # with associated specializations ('ld', 'ldb', 'lD', 'lDb'). Fourth element is None.

    (special.shichi, cython_special._shichi_pywrap, ('d', 'D'), None),
    # Tuple containing the functions special.shichi and cython_special._shichi_pywrap,
    # with associated specializations ('d', 'D'). Fourth element is None.

    (special.sici, cython_special._sici_pywrap, ('d', 'D'), None),
    # Tuple containing the functions special.sici and cython_special._sici_pywrap,
    # with associated specializations ('d', 'D'). Fourth element is None.

    (special.sindg, cython_special.sindg, ('d',), None),
    # Tuple containing the functions special.sindg and cython_special.sindg,
    # with associated specializations ('d',). Fourth element is None.

    (special.smirnov, cython_special.smirnov, ('ld', 'dd'), None),
    # Tuple containing the functions special.smirnov and cython_special.smirnov,
    # with associated specializations ('ld', 'dd'). Fourth element is None.

    (special.smirnovi, cython_special.smirnovi, ('ld', 'dd'), None),
    # Tuple containing the functions special.smirnovi and cython_special.smirnovi,
    # with associated specializations ('ld', 'dd'). Fourth element is None.

    (special.spence, cython_special.spence, ('d', 'D'), None),
    # Tuple containing the functions special.spence and cython_special.spence,
    # with associated specializations ('d', 'D'). Fourth element is None.

    (special.sph_harm, cython_special.sph_harm, ('lldd', 'dddd'), None),
    # Tuple containing the functions special.sph_harm and cython_special.sph_harm,
    # with associated specializations ('lldd', 'dddd'). Fourth element is None.

    (special.stdtr, cython_special.stdtr, ('dd',), None),
    # Tuple containing the functions special.stdtr and cython_special.stdtr,
    # with associated specializations ('dd',). Fourth element is None.

    (special.stdtridf, cython_special.stdtridf, ('dd',), None),
    # Tuple containing the functions special.stdtridf and cython_special.stdtridf,
    # with associated specializations ('dd',). Fourth element is None.

    (special.stdtrit, cython_special.stdtrit, ('dd',), None),
    # Tuple containing the functions special.stdtrit and cython_special.stdtrit,
    # with associated specializations ('dd',). Fourth element is None.

    (special.struve, cython_special.struve, ('dd',), None),
    # Tuple containing the functions special.struve and cython_special.struve,
    # with associated specializations ('dd',). Fourth element is None.

    (special.tandg, cython_special.tandg, ('d',), None),
    # Tuple containing the functions special.tandg and cython_special.tandg,
    # with associated specializations ('d',). Fourth element is None.

    (special.tklmbda, cython_special.tklmbda, ('dd',), None),
    # Tuple containing the functions special.tklmbda and cython_special.tklmbda,
    # with associated specializations ('dd',). Fourth element is None.

    (special.voigt_profile, cython_special.voigt_profile, ('ddd',), None),
    # Tuple containing the functions special.voigt_profile and cython_special.voigt_profile,
    # with associated specializations ('ddd',). Fourth element is None.

    (special.wofz, cython_special.wofz, ('D',), None),
    # Tuple containing the functions special.wofz and cython_special.wofz,
    # with associated specializations ('D',). Fourth element is None.

    (special.wright_bessel, cython_special.wright_bessel, ('ddd',), None),
    # Tuple containing the functions special.wright_bessel and cython_special.wright_bessel,
    # with associated specializations ('ddd',). Fourth element is None.

    (special.wrightomega, cython_special.wrightomega, ('D',), None),
    # Tuple containing the functions special.wrightomega and cython_special.wrightomega,
    # with associated specializations ('D',). Fourth element is None.

    (special.xlog1py, cython_special.xlog1py, ('dd', 'DD'), None),
    # Tuple containing the functions special.xlog1py and cython_special.xlog1py,
    # with associated specializations ('dd', 'DD'). Fourth element is None.

    (special.xlogy, cython_special.xlogy, ('dd', 'DD'), None),
    # Tuple containing the functions special.xlogy and cython_special.xlogy,
    # with associated specializations ('dd', 'DD'). Fourth element is None.

    (special.y0, cython_special.y0, ('d',), None),
    # Tuple containing the functions special.y0 and cython_special.y0,
    # with associated specializations ('d',). Fourth element is None.

    (special.y1, cython_special.y1, ('d',), None),
    # Tuple containing the functions special.y1 and cython_special.y1,
    # with associated specializations ('d',). Fourth element is None.

    (special.yn, cython_special.yn, ('ld', 'dd'), None),
    # Tuple containing the functions special.yn and cython_special.yn,
    # with associated specializations ('ld', 'dd'). Fourth element is None.

    (special.yv, cython_special.yv, ('dd', 'dD'), None),
    # Tuple containing the functions special.yv and cython_special.yv,
    # with associated specializations ('dd', 'dD'). Fourth element is None.

    (special.yve, cython_special.yve, ('dd', 'dD'), None),
    # Tuple containing the functions special.yve and cython_special.yve,
    # with associated specializations ('dd', 'dD'). Fourth element is None.

    (special.zetac, cython_special.zetac, ('d',), None),
    # Tuple containing the functions special.zetac and cython_special.zetac,
    # with associated specializations ('d',). Fourth element is None.

    (special.owens_t, cython_special.owens_t, ('dd',), None)
    # Tuple containing the functions special.owens_t and cython_special.owens_t,
    # with associated specializations ('dd',). Fourth element is None.
# 从 PARAMS 列表中提取每个元素的第一个元素的名称，作为 IDS 列表的内容
IDS = [x[0].__name__ for x in PARAMS]

# 定义一个函数 _generate_test_points，用于生成测试点
def _generate_test_points(typecodes):
    # 根据给定的 typecodes 列表，获取 TEST_POINTS 中对应的值，并形成一个元组 axes
    axes = tuple(TEST_POINTS[x] for x in typecodes)
    # 使用 itertools.product 生成所有可能的组合，存储在 pts 列表中并返回
    pts = list(product(*axes))
    return pts

# 定义一个名为 test_cython_api_completeness 的测试函数，用于检查所有函数都已经进行了测试
def test_cython_api_completeness():
    # 遍历 cython_special 模块中的所有属性名
    for name in dir(cython_special):
        # 获取属性名对应的对象
        func = getattr(cython_special, name)
        # 如果 func 是可调用的且不以 '_' 开头
        if callable(func) and not name.startswith('_'):
            # 遍历 PARAMS 列表，检查是否有函数与 cython_special 模块中的函数匹配
            for _, cyfun, _, _ in PARAMS:
                if cyfun is func:
                    break
            else:
                # 如果没有找到匹配的函数，则抛出 RuntimeError 异常
                raise RuntimeError(f"{name} missing from tests!")

# 使用 pytest.mark.fail_slow(20) 标记，参数 param 使用 PARAMS 列表进行参数化，使用 IDS 作为标识符
@pytest.mark.fail_slow(20)
@pytest.mark.parametrize("param", PARAMS, ids=IDS)
def test_cython_api(param):
    pyfunc, cyfunc, specializations, knownfailure = param
    # 如果 knownfailure 为真，则使用 pytest.xfail 标记这个测试为预期失败
    if knownfailure:
        pytest.xfail(reason=knownfailure)

    # 检查哪些参数预期为融合类型
    max_params = max(len(spec) for spec in specializations)
    values = [set() for _ in range(max_params)]
    for typecodes in specializations:
        for j, v in enumerate(typecodes):
            values[j].add(v)
    seen = set()
    is_fused_code = [False] * len(values)
    for j, v in enumerate(values):
        vv = tuple(sorted(v))
        if vv in seen:
            continue
        # 如果某个参数类型有多于一个的取值，则标记为融合类型
        is_fused_code[j] = (len(v) > 1)
        seen.add(vv)

    # 检查结果
    for typecodes in specializations:
        # 选择正确的特化函数
        signature = [CYTHON_SIGNATURE_MAP[code]
                     for j, code in enumerate(typecodes)
                     if is_fused_code[j]]

        if signature:
            cy_spec_func = cyfunc[tuple(signature)]
        else:
            signature = None
            cy_spec_func = cyfunc

        # 测试该函数
        pts = _generate_test_points(typecodes)
        for pt in pts:
            # 使用 suppress_warnings 上下文管理器抑制 DeprecationWarning
            with suppress_warnings() as sup:
                sup.filter(DeprecationWarning)
                # 调用 pyfunc 和 cy_spec_func 函数计算结果
                pyval = pyfunc(*pt)
                cyval = cy_spec_func(*pt)
            # 使用 assert_allclose 检查 cyval 和 pyval 的近似程度，如果不满足条件则抛出错误信息
            assert_allclose(cyval, pyval, err_msg=f"{pt} {typecodes} {signature}")
```