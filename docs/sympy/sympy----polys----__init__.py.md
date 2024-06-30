# `D:\src\scipysrc\sympy\sympy\polys\__init__.py`

```
# 多项式操作算法和代数对象的定义

__all__ = [
    'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
    'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
    'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
    'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
    'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
    'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
    'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
    'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
    'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
    'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly',

    'symmetrize', 'horner', 'interpolate', 'rational_interpolate', 'viete',

    'together',

    'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
    'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
    'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
    'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
    'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed',
    'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed',
    'UnivariatePolynomialError', 'MultivariatePolynomialError',
    'PolificationFailed', 'OptionError', 'FlagError',

    'minpoly', 'minimal_polynomial', 'primitive_element', 'field_isomorphism',
    'to_number_field', 'isolate', 'round_two', 'prime_decomp',
    'prime_valuation', 'galois_group',

    'itermonomials', 'Monomial',

    'lex', 'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex',

    'CRootOf', 'rootof', 'RootOf', 'ComplexRootOf', 'RootSum',

    'roots',

    'Domain', 'FiniteField', 'IntegerRing', 'RationalField', 'RealField',
    'ComplexField', 'PythonFiniteField', 'GMPYFiniteField',
    'PythonIntegerRing', 'GMPYIntegerRing', 'PythonRational',
    'GMPYRationalField', 'AlgebraicField', 'PolynomialRing', 'FractionField',
    'ExpressionDomain', 'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy',
    'QQ_python', 'QQ_gmpy', 'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR',
    'CC', 'EX', 'EXRAW',

    'construct_domain',

    'swinnerton_dyer_poly', 'cyclotomic_poly', 'symmetric_poly',
    'random_poly', 'interpolating_poly',

    'jacobi_poly', 'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly',
    'hermite_prob_poly', 'legendre_poly', 'laguerre_poly',

    'bernoulli_poly', 'bernoulli_c_poly', 'genocchi_poly', 'euler_poly',
    'andre_poly',

    'apart', 'apart_list', 'assemble_partfrac_list',

    'Options',

    'ring', 'xring', 'vring', 'sring',

    'field', 'xfield', 'vfield', 'sfield'
]
# 从 polytools 模块中导入多项式相关的函数和类
from .polytools import (Poly, PurePoly, poly_from_expr,
        parallel_poly_from_expr, degree, total_degree, degree_list, LC, LM,
        LT, pdiv, prem, pquo, pexquo, div, rem, quo, exquo, half_gcdex, gcdex,
        invert, subresultants, resultant, discriminant, cofactors, gcd_list,
        gcd, lcm_list, lcm, terms_gcd, trunc, monic, content, primitive,
        compose, decompose, sturm, gff_list, gff, sqf_norm, sqf_part,
        sqf_list, sqf, factor_list, factor, intervals, refine_root,
        count_roots, all_roots, real_roots, nroots, ground_roots,
        nth_power_roots_poly, cancel, reduced, groebner, is_zero_dimensional,
        GroebnerBasis, poly)

# 从 polyfuncs 模块中导入多项式函数相关的函数
from .polyfuncs import (symmetrize, horner, interpolate,
        rational_interpolate, viete)

# 从 rationaltools 模块中导入处理有理数相关的函数
from .rationaltools import together

# 从 polyerrors 模块中导入多项式操作可能引发的异常类
from .polyerrors import (BasePolynomialError, ExactQuotientFailed,
        PolynomialDivisionFailed, OperationNotSupported, HeuristicGCDFailed,
        HomomorphismFailed, IsomorphismFailed, ExtraneousFactors,
        EvaluationFailed, RefinementFailed, CoercionFailed, NotInvertible,
        NotReversible, NotAlgebraic, DomainError, PolynomialError,
        UnificationFailed, GeneratorsError, GeneratorsNeeded,
        ComputationFailed, UnivariatePolynomialError,
        MultivariatePolynomialError, PolificationFailed, OptionError,
        FlagError)

# 从 numberfields 模块中导入处理数域相关的函数
from .numberfields import (minpoly, minimal_polynomial, primitive_element,
        field_isomorphism, to_number_field, isolate, round_two, prime_decomp,
        prime_valuation, galois_group)

# 从 monomials 模块中导入处理单项式相关的函数和类
from .monomials import itermonomials, Monomial

# 从 orderings 模块中导入多项式排序相关的函数
from .orderings import lex, grlex, grevlex, ilex, igrlex, igrevlex

# 从 rootoftools 模块中导入处理根式相关的函数和类
from .rootoftools import CRootOf, rootof, RootOf, ComplexRootOf, RootSum

# 从 polyroots 模块中导入处理多项式根的函数
from .polyroots import roots

# 从 domains 模块中导入处理各种数学域的类
from .domains import (Domain, FiniteField, IntegerRing, RationalField,
        RealField, ComplexField, PythonFiniteField, GMPYFiniteField,
        PythonIntegerRing, GMPYIntegerRing, PythonRational, GMPYRationalField,
        AlgebraicField, PolynomialRing, FractionField, ExpressionDomain,
        FF_python, FF_gmpy, ZZ_python, ZZ_gmpy, QQ_python, QQ_gmpy, GF, FF,
        ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX, EXRAW)

# 从 constructor 模块中导入构造数学域的函数
from .constructor import construct_domain

# 从 specialpolys 模块中导入特殊多项式的生成函数
from .specialpolys import (swinnerton_dyer_poly, cyclotomic_poly,
        symmetric_poly, random_poly, interpolating_poly)

# 从 orthopolys 模块中导入正交多项式相关的函数
from .orthopolys import (jacobi_poly, chebyshevt_poly, chebyshevu_poly,
        hermite_poly, hermite_prob_poly, legendre_poly, laguerre_poly)

# 从 appellseqs 模块中导入 Appell 序列相关的函数
from .appellseqs import (bernoulli_poly, bernoulli_c_poly, genocchi_poly,
        euler_poly, andre_poly)

# 从 partfrac 模块中导入处理部分分式相关的函数
from .partfrac import apart, apart_list, assemble_partfrac_list

# 从 polyoptions 模块中导入多项式操作的选项类
from .polyoptions import Options

# 从 rings 模块中导入环相关的函数和类
from .rings import ring, xring, vring, sring

# 从 fields 模块中导入域相关的函数和类
from .fields import field, xfield, vfield, sfield
```