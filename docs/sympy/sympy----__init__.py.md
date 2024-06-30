# `D:\src\scipysrc\sympy\sympy\__init__.py`

```
"""
SymPy is a Python library for symbolic mathematics. It aims to become a
full-featured computer algebra system (CAS) while keeping the code as simple
as possible in order to be comprehensible and easily extensible.  SymPy is
written entirely in Python. It depends on mpmath, and other external libraries
may be optionally for things like plotting support.

See the webpage for more information and documentation:

    https://sympy.org

"""


# 导入 sys 模块
import sys
# 检查 Python 版本是否符合要求（3.8 或以上）
if sys.version_info < (3, 8):
    # 如果版本不符合，抛出 ImportError 异常
    raise ImportError("Python version 3.8 or above is required for SymPy.")
# 删除 sys 模块的引用，清理命名空间
del sys


# 尝试导入 mpmath 模块
try:
    import mpmath
# 如果导入失败，抛出 ImportError 异常
except ImportError:
    raise ImportError("SymPy now depends on mpmath as an external library. "
    "See https://docs.sympy.org/latest/install.html#mpmath for more information.")
# 删除 mpmath 模块的引用，清理命名空间
del mpmath


# 从 sympy.release 模块导入 SymPy 的版本号 __version__
from sympy.release import __version__
# 从 sympy.core.cache 模块导入 lazy_function 函数
from sympy.core.cache import lazy_function


# 如果版本号中包含 'dev'，定义 enable_warnings 函数
if 'dev' in __version__:
    def enable_warnings():
        import warnings
        # 启用所有关于 DeprecationWarning 的警告
        warnings.filterwarnings('default',   '.*',   DeprecationWarning, module='sympy.*')
        del warnings
    # 调用 enable_warnings 函数
    enable_warnings()
    # 删除 enable_warnings 函数的引用，清理命名空间
    del enable_warnings


# 定义 __sympy_debug 函数，用于从环境变量 SYMPY_DEBUG 中获取调试信息的布尔值
def __sympy_debug():
    # 导入 os 模块，获取环境变量 SYMPY_DEBUG 的值
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    # 判断 debug_str 是否是 'True' 或 'False'，返回其对应的布尔值
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        # 如果值不在预期范围内，抛出 RuntimeError 异常
        raise RuntimeError("unrecognized value for SYMPY_DEBUG: %s" %
                           debug_str)
# 调用 __sympy_debug 函数，获取 SYMPY_DEBUG 的布尔值，并赋给 SYMPY_DEBUG 变量
SYMPY_DEBUG = __sympy_debug()  # type: bool


# 从当前包中导入 sympify 等一系列函数和类
from .core import (sympify, SympifyError, cacheit, Basic, Atom,
        preorder_traversal, S, Expr, AtomicExpr, UnevaluatedExpr, Symbol,
        Wild, Dummy, symbols, var, Number, Float, Rational, Integer,
        NumberSymbol, RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo,
        AlgebraicNumber, comp, mod_inverse, Pow, integer_nthroot, integer_log,
        trailing, Mul, prod, Add, Mod, Rel, Eq, Ne, Lt, Le, Gt, Ge, Equality,
        GreaterThan, LessThan, Unequality, StrictGreaterThan, StrictLessThan,
        vectorize, Lambda, WildFunction, Derivative, diff, FunctionClass,
        Function, Subs, expand, PoleError, count_ops, expand_mul, expand_log,
        expand_func, expand_trig, expand_complex, expand_multinomial, nfloat,
        expand_power_base, expand_power_exp, arity, PrecisionExhausted, N,
        evalf, Tuple, Dict, gcd_terms, factor_terms, factor_nc, evaluate,
        Catalan, EulerGamma, GoldenRatio, TribonacciConstant, bottom_up, use,
        postorder_traversal, default_sort_key, ordered, num_digits)


# 从 .logic 模块导入逻辑相关函数和类
from .logic import (to_cnf, to_dnf, to_nnf, And, Or, Not, Xor, Nand, Nor,
        Implies, Equivalent, ITE, POSform, SOPform, simplify_logic, bool_map,
        true, false, satisfiable)


# 从 .assumptions 模块导入假设相关函数和类
from .assumptions import (AppliedPredicate, Predicate, AssumptionsContext,
        assuming, Q, ask, register_handler, remove_handler, refine)
# 从.polys模块中导入多项式相关的类和函数
from .polys import (Poly, PurePoly, poly_from_expr, parallel_poly_from_expr,
        degree, total_degree, degree_list, LC, LM, LT, pdiv, prem, pquo,
        pexquo, div, rem, quo, exquo, half_gcdex, gcdex, invert,
        subresultants, resultant, discriminant, cofactors, gcd_list, gcd,
        lcm_list, lcm, terms_gcd, trunc, monic, content, primitive, compose,
        decompose, sturm, gff_list, gff, sqf_norm, sqf_part, sqf_list, sqf,
        factor_list, factor, intervals, refine_root, count_roots, all_roots,
        real_roots, nroots, ground_roots, nth_power_roots_poly, cancel,
        reduced, groebner, is_zero_dimensional, GroebnerBasis, poly,
        symmetrize, horner, interpolate, rational_interpolate, viete, together,
        BasePolynomialError, ExactQuotientFailed, PolynomialDivisionFailed,
        OperationNotSupported, HeuristicGCDFailed, HomomorphismFailed,
        IsomorphismFailed, ExtraneousFactors, EvaluationFailed,
        RefinementFailed, CoercionFailed, NotInvertible, NotReversible,
        NotAlgebraic, DomainError, PolynomialError, UnificationFailed,
        GeneratorsError, GeneratorsNeeded, ComputationFailed,
        UnivariatePolynomialError, MultivariatePolynomialError,
        PolificationFailed, OptionError, FlagError, minpoly,
        minimal_polynomial, primitive_element, field_isomorphism,
        to_number_field, isolate, round_two, prime_decomp, prime_valuation,
        galois_group, itermonomials, Monomial, lex, grlex,
        grevlex, ilex, igrlex, igrevlex, CRootOf, rootof, RootOf,
        ComplexRootOf, RootSum, roots, Domain, FiniteField, IntegerRing,
        RationalField, RealField, ComplexField, PythonFiniteField,
        GMPYFiniteField, PythonIntegerRing, GMPYIntegerRing, PythonRational,
        GMPYRationalField, AlgebraicField, PolynomialRing, FractionField,
        ExpressionDomain, FF_python, FF_gmpy, ZZ_python, ZZ_gmpy, QQ_python,
        QQ_gmpy, GF, FF, ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX, EXRAW,
        construct_domain, swinnerton_dyer_poly, cyclotomic_poly,
        symmetric_poly, random_poly, interpolating_poly, jacobi_poly,
        chebyshevt_poly, chebyshevu_poly, hermite_poly, hermite_prob_poly,
        legendre_poly, laguerre_poly, apart, apart_list, assemble_partfrac_list,
        Options, ring, xring, vring, sring, field, xfield, vfield, sfield)

# 从.series模块中导入序列相关的类和函数
from .series import (Order, O, limit, Limit, gruntz, series, approximants,
        residue, EmptySequence, SeqPer, SeqFormula, sequence, SeqAdd, SeqMul,
        fourier_series, fps, difference_delta, limit_seq)
# 从自定义模块中导入多个函数和类
from .functions import (factorial, factorial2, rf, ff, binomial,
        RisingFactorial, FallingFactorial, subfactorial, carmichael,
        fibonacci, lucas, motzkin, tribonacci, harmonic, bernoulli, bell, euler,
        catalan, genocchi, andre, partition, divisor_sigma, legendre_symbol,
        jacobi_symbol, kronecker_symbol, mobius, primenu, primeomega,
        totient, reduced_totient, primepi, sqrt, root, Min, Max, Id,
        real_root, Rem, cbrt, re, im, sign, Abs, conjugate, arg, polar_lift,
        periodic_argument, unbranched_argument, principal_branch, transpose,
        adjoint, polarify, unpolarify, sin, cos, tan, sec, csc, cot, sinc,
        asin, acos, atan, asec, acsc, acot, atan2, exp_polar, exp, ln, log,
        LambertW, sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh,
        acoth, asech, acsch, floor, ceiling, frac, Piecewise, piecewise_fold,
        piecewise_exclusive, erf, erfc, erfi, erf2, erfinv, erfcinv, erf2inv,
        Ei, expint, E1, li, Li, Si, Ci, Shi, Chi, fresnels, fresnelc, gamma,
        lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
        multigamma, dirichlet_eta, zeta, lerchphi, polylog, stieltjes, Eijk,
        LeviCivita, KroneckerDelta, SingularityFunction, DiracDelta, Heaviside,
        bspline_basis, bspline_basis_set, interpolating_spline, besselj,
        bessely, besseli, besselk, hankel1, hankel2, jn, yn, jn_zeros, hn1,
        hn2, airyai, airybi, airyaiprime, airybiprime, marcumq, hyper,
        meijerg, appellf1, legendre, assoc_legendre, hermite, hermite_prob,
        chebyshevt, chebyshevu, chebyshevu_root, chebyshevt_root, laguerre,
        assoc_laguerre, gegenbauer, jacobi, jacobi_normalized, Ynm, Ynm_c,
        Znm, elliptic_k, elliptic_f, elliptic_e, elliptic_pi, beta, mathieus,
        mathieuc, mathieusprime, mathieucprime, riemann_xi, betainc, betainc_regularized)

# 从自定义模块中导入多个函数和类
from .ntheory import (nextprime, prevprime, prime, primerange,
        randprime, Sieve, sieve, primorial, cycle_length, composite,
        compositepi, isprime, divisors, proper_divisors, factorint,
        multiplicity, perfect_power, pollard_pm1, pollard_rho, primefactors,
        divisor_count, proper_divisor_count,
        factorrat,
        mersenne_prime_exponent, is_perfect, is_mersenne_prime, is_abundant,
        is_deficient, is_amicable, is_carmichael, abundance, npartitions, is_primitive_root,
        is_quad_residue, n_order, sqrt_mod,
        quadratic_residues, primitive_root, nthroot_mod, is_nthpow_residue,
        sqrt_mod_iter, discrete_log, quadratic_congruence,
        binomial_coefficients, binomial_coefficients_list,
        multinomial_coefficients, continued_fraction_periodic,
        continued_fraction_iterator, continued_fraction_reduce,
        continued_fraction_convergents, continued_fraction, egyptian_fraction)

# 从自定义模块中导入多个函数和类
from .concrete import product, Product, summation, Sum
# 从 .discrete 模块导入以下函数：快速傅里叶变换（fft）、逆快速傅里叶变换（ifft）、
# 数论变换（ntt）、逆数论变换（intt）、快速Walsh-Hadamard变换（fwht）、
# 逆快速Walsh-Hadamard变换（ifwht）、莫比乌斯变换（mobius_transform）、
# 逆莫比乌斯变换（inverse_mobius_transform）、卷积运算（convolution）、
# 覆盖积（covering_product）、交叉积（intersecting_product）
from .discrete import (fft, ifft, ntt, intt, fwht, ifwht, mobius_transform,
        inverse_mobius_transform, convolution, covering_product,
        intersecting_product)

# 从 .simplify 模块导入以下函数：简化（simplify）、超简化（hypersimp）、
# 超相似化（hypersimilar）、对数合并（logcombine）、变量分离（separatevars）、
# 取正（posify）、贝塞尔简化（besselsimp）、Kronecker简化（kroneckersimp）、
# 符号简化（signsimp）、数值简化（nsimplify）、CSE（cse）、表达式路径（epath）、
# 表达式路径类（EPath）、超展开（hyperexpand）、收集（collect）、
# 逆向收集（rcollect）、收集常数（collect_const）、分数化简（fraction）、
# 分子提取（numer）、分母提取（denom）、三角函数简化（trigsimp）、
# 指数三角函数简化（exptrigsimp）、幂简化（powsimp）、幂展开（powdenest）、
# 组合简化（combsimp）、Gamma函数简化（gammasimp）、有理数简化（ratsimp）、
# 在模素数下的有理数简化（ratsimpmodprime）
from .simplify import (simplify, hypersimp, hypersimilar, logcombine,
        separatevars, posify, besselsimp, kroneckersimp, signsimp,
        nsimplify, FU, fu, sqrtdenest, cse, epath, EPath, hyperexpand,
        collect, rcollect, radsimp, collect_const, fraction, numer, denom,
        trigsimp, exptrigsimp, powsimp, powdenest, combsimp, gammasimp,
        ratsimp, ratsimpmodprime)

# 从 .sets 模块导入以下类和函数：集合（Set）、区间（Interval）、并集（Union）、
# 空集（EmptySet）、有限集（FiniteSet）、乘积集（ProductSet）、交集（Intersection）、
# 不相交并集（DisjointUnion）、图像集（imageset）、补集（Complement）、
# 对称差（SymmetricDifference）、图像集（ImageSet）、范围（Range）、
# 复数区域（ComplexRegion）、复数集（Complexes）、实数集（Reals）、
# 包含（Contains）、条件集（ConditionSet）、序数（Ordinal）、
# Omega幂（OmegaPower）、零序数（ord0）、幂集（PowerSet）、
# 自然数（Naturals）、自然数包含零（Naturals0）、全集（UniversalSet）、
# 整数（Integers）、有理数（Rationals）
from .sets import (Set, Interval, Union, EmptySet, FiniteSet, ProductSet,
        Intersection, DisjointUnion, imageset, Complement, SymmetricDifference, ImageSet,
        Range, ComplexRegion, Complexes, Reals, Contains, ConditionSet, Ordinal,
        OmegaPower, ord0, PowerSet, Naturals, Naturals0, UniversalSet,
        Integers, Rationals)

# 从 .solvers 模块导入以下函数：求解（solve）、解线性系统（solve_linear_system）、
# LU分解解线性系统（solve_linear_system_LU）、求未定系数解（solve_undetermined_coeffs）、
# 数值求解（nsolve）、线性求解（solve_linear）、检查解（checksol）、
# 快速行列式计算（det_quick）、快速逆矩阵计算（inv_quick）、
# 检查假设（check_assumptions）、失败假设（failing_assumptions）、
# 整数解（diophantine）、递推求解（rsolve）、多项式递推求解（rsolve_poly）、
# 比率递推求解（rsolve_ratio）、超越递推求解（rsolve_hyper）、
# 检查ODE解（checkodesol）、分类ODE（classify_ode）、解ODE（dsolve）、
# 齐次ODE阶数（homogeneous_order）、多项式系统求解（solve_poly_system）、
# 解三角分解系统（solve_triangulated）、偏微分方程分离（pde_separate）、
# 偏微分方程分离加法（pde_separate_add）、偏微分方程分离乘法（pde_separate_mul）、
# 偏微分方程求解（pdsolve）、分类偏微分方程（classify_pde）、检查PDE解（checkpdesol）、
# ODE阶数（ode_order）、不等式简化（reduce_inequalities）、绝对值不等式简化（reduce_abs_inequality）、
# 绝对值不等式组简化（reduce_abs_inequalities）、多项式不等式求解（solve_poly_inequality）、
# 有理数不等式求解（solve_rational_inequalities）、一元不等式求解（solve_univariate_inequality）、
# 分解式生成（decompogen）、解集求解（solveset）、线性系统求解（linsolve）、
# 线性方程转矩阵形式（linear_eq_to_matrix）、非线性系统求解（nonlinsolve）、替换（substitution）
from .solvers import (solve, solve_linear_system, solve_linear_system_LU,
        solve_undetermined_coeffs, nsolve, solve_linear, checksol, det_quick,
        inv_quick, check_assumptions, failing_assumptions, diophantine,
        rsolve, rsolve_poly, rsolve_ratio, rsolve_hyper, checkodesol,
        classify_ode, dsolve, homogeneous_order, solve_poly_system,
        solve_triangulated, pde_separate, pde_separate_add, pde_separate_mul,
        pdsolve, classify_pde, checkpdesol, ode_order, reduce_inequalities,
        reduce_abs_inequality, reduce_abs_inequalities, solve_poly_inequality,
        solve_rational_inequalities, solve_univariate_inequality, decompogen,
        solveset, linsolve, linear_eq_to_matrix, nonlinsolve, substitution)

# 从 .matrices 模块导入以下类和函数：形状错误（ShapeError）、非方阵错误（NonSquareMatrixError）、
# Gram-Schmidt正交化（GramSchmidt）、Casoratian（casoratian）、对角矩阵（diag）、单位矩阵（eye）、
# 海森矩阵（hessian）、Jordan标准形（jordan_cell）、列表转numpy数组（list2numpy）、
# 矩阵转numpy数组（matrix2numpy）、逐元素矩阵乘法（matrix_multiply_elementwise）、
# 全1矩阵（ones）、随机矩阵（randMatrix）、绕轴1旋转矩阵（rot_axis1）、
# 绕轴2旋转矩阵（rot_axis2）、绕轴3旋转矩阵（rot_axis3）、符号数组（symarray）、
# Wronski行列式（wronskian）、全0矩阵（zeros）、可变密集矩阵（MutableDenseMatrix）、
# 延迟向量（DeferredVector）、矩阵基类（MatrixBase）、矩阵（Matrix）、
# 可变矩阵（MutableMatrix）、可变稀疏矩阵（MutableSparseMatrix）、
# 不可变密集矩阵（ImmutableDenseMatrix）、不可变稀疏矩阵（ImmutableSparseMatrix）、
# 不可
# 从.geometry模块中导入多个几何类和函数
from .geometry import (Point, Point2D, Point3D, Line, Ray, Segment, Line2D,
        Segment2D, Ray2D, Line3D, Segment3D, Ray3D, Plane, Ellipse, Circle,
        Polygon, RegularPolygon, Triangle, rad, deg, are_similar, centroid,
        convex_hull, idiff, intersection, closest_points, farthest_points,
        GeometryError, Curve, Parabola)

# 从.utilities模块中导入多个实用函数和工具
from .utilities import (flatten, group, take, subsets, variations,
        numbered_symbols, cartes, capture, dict_merge, prefixes, postfixes,
        sift, topological_sort, unflatten, has_dups, has_variety, reshape,
        rotations, filldedent, lambdify,
        threaded, xthreaded, public, memoize_property, timed)

# 从.integrals模块中导入多个积分相关函数和类
from .integrals import (integrate, Integral, line_integrate, mellin_transform,
        inverse_mellin_transform, MellinTransform, InverseMellinTransform,
        laplace_transform, laplace_correspondence, laplace_initial_conds,
        inverse_laplace_transform, LaplaceTransform,
        InverseLaplaceTransform, fourier_transform, inverse_fourier_transform,
        FourierTransform, InverseFourierTransform, sine_transform,
        inverse_sine_transform, SineTransform, InverseSineTransform,
        cosine_transform, inverse_cosine_transform, CosineTransform,
        InverseCosineTransform, hankel_transform, inverse_hankel_transform,
        HankelTransform, InverseHankelTransform, singularityintegrate)

# 从.tensor模块中导入多个张量相关类和函数
from .tensor import (IndexedBase, Idx, Indexed, get_contraction_structure,
        get_indices, shape, MutableDenseNDimArray, ImmutableDenseNDimArray,
        MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray,
        tensorproduct, tensorcontraction, tensordiagonal, derive_by_array,
        permutedims, Array, DenseNDimArray, SparseNDimArray)

# 从.parsing模块中导入parse_expr函数
from .parsing import parse_expr

# 从.calculus模块中导入多个微积分相关函数
from .calculus import (euler_equations, singularities, is_increasing,
        is_strictly_increasing, is_decreasing, is_strictly_decreasing,
        is_monotonic, finite_diff_weights, apply_finite_diff,
        differentiate_finite, periodicity, not_empty_in, AccumBounds,
        is_convex, stationary_points, minimum, maximum)

# 从.algebras模块中导入Quaternion类
from .algebras import Quaternion

# 从.printing模块中导入多个打印相关函数和类
from .printing import (pager_print, pretty, pretty_print, pprint,
        pprint_use_unicode, pprint_try_use_unicode, latex, print_latex,
        multiline_latex, mathml, print_mathml, python, print_python, pycode,
        ccode, print_ccode, smtlib_code, glsl_code, print_glsl, cxxcode, fcode,
        print_fcode, rcode, print_rcode, jscode, print_jscode, julia_code,
        mathematica_code, octave_code, rust_code, print_gtk, preview, srepr,
        print_tree, StrPrinter, sstr, sstrrepr, TableForm, dotprint,
        maple_code, print_maple_code)

# 导入lazy_function函数，用于延迟加载测试模块和文档测试模块
test = lazy_function('sympy.testing.runtests_pytest', 'test')
doctest = lazy_function('sympy.testing.runtests', 'doctest')

# 以下几行注释描述了一些导入操作的细节，包括模块可能引起的冲突和导入速度问题
# 此模块与其他模块可能引起冲突：
# from .stats import *
# 这会增加约0.04-0.05秒的导入时间
# from combinatorics import *
# 此模块的导入速度较慢：
# 导入模块中的特定函数和对象，使它们可以直接使用
#from physics import units

# 从自定义模块中导入绘图相关的函数和对象
from .plotting import plot, textplot, plot_backends, plot_implicit, plot_parametric

# 从交互模块中导入会话初始化、打印初始化和交互遍历相关的函数和对象
from .interactive import init_session, init_printing, interactive_traversal

# 调用 evalf 对象的 _create_evalf_table 方法，可能是初始化或创建评估函数的内部数据结构
evalf._create_evalf_table()

# 定义 __all__ 列表，用于声明模块导出的公共接口
__all__ = [
    '__version__',

    # sympy.core 模块导出的各种类和函数
    'sympify', 'SympifyError', 'cacheit', 'Basic', 'Atom',
    'preorder_traversal', 'S', 'Expr', 'AtomicExpr', 'UnevaluatedExpr',
    'Symbol', 'Wild', 'Dummy', 'symbols', 'var', 'Number', 'Float',
    'Rational', 'Integer', 'NumberSymbol', 'RealNumber', 'igcd', 'ilcm',
    'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo', 'AlgebraicNumber', 'comp',
    'mod_inverse', 'Pow', 'integer_nthroot', 'integer_log', 'trailing', 'Mul', 'prod',
    'Add', 'Mod', 'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality',
    'GreaterThan', 'LessThan', 'Unequality', 'StrictGreaterThan',
    'StrictLessThan', 'vectorize', 'Lambda', 'WildFunction', 'Derivative',
    'diff', 'FunctionClass', 'Function', 'Subs', 'expand', 'PoleError',
    'count_ops', 'expand_mul', 'expand_log', 'expand_func', 'expand_trig',
    'expand_complex', 'expand_multinomial', 'nfloat', 'expand_power_base',
    'expand_power_exp', 'arity', 'PrecisionExhausted', 'N', 'evalf', 'Tuple',
    'Dict', 'gcd_terms', 'factor_terms', 'factor_nc', 'evaluate', 'Catalan',
    'EulerGamma', 'GoldenRatio', 'TribonacciConstant', 'bottom_up', 'use',
    'postorder_traversal', 'default_sort_key', 'ordered', 'num_digits',

    # sympy.logic 模块导出的逻辑运算相关的函数和对象
    'to_cnf', 'to_dnf', 'to_nnf', 'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'Implies', 'Equivalent', 'ITE', 'POSform', 'SOPform', 'simplify_logic',
    'bool_map', 'true', 'false', 'satisfiable',

    # sympy.assumptions 模块导出的符号推理相关的函数和对象
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming', 'Q',
    'ask', 'register_handler', 'remove_handler', 'refine',

    # sympy.polys 模块导出的多项式相关的函数和对象
    'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
    'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
    'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
    'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
    'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
    'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
    'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
    'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
    'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
    'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly', 'symmetrize',
    'horner', 'interpolate', 'rational_interpolate', 'viete', 'together',
    'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
    'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
    'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
    'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
    `
    # 定义一系列字符串，这些字符串可能是函数名、类名或其他标识符
    'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed', 
    'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed', 
    'UnivariatePolynomialError', 'MultivariatePolynomialError', 
    'PolificationFailed', 'OptionError', 'FlagError', 'minpoly', 
    'minimal_polynomial', 'primitive_element', 'field_isomorphism', 
    'to_number_field', 'isolate', 'round_two', 'prime_decomp', 
    'prime_valuation', 'galois_group', 'itermonomials', 'Monomial', 'lex', 
    'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex', 'CRootOf', 'rootof', 
    'RootOf', 'ComplexRootOf', 'RootSum', 'roots', 'Domain', 'FiniteField', 
    'IntegerRing', 'RationalField', 'RealField', 'ComplexField', 
    'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing', 
    'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField', 
    'AlgebraicField', 'PolynomialRing', 'FractionField', 
    'ExpressionDomain', 'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 
    'QQ_python', 'QQ_gmpy', 'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 
    'CC', 'EX', 'EXRAW', 'construct_domain', 'swinnerton_dyer_poly', 
    'cyclotomic_poly', 'symmetric_poly', 'random_poly', 'interpolating_poly', 
    'jacobi_poly', 'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly', 
    'hermite_prob_poly', 'legendre_poly', 'laguerre_poly', 'apart', 'apart_list', 
    'assemble_partfrac_list', 'Options', 'ring', 'xring', 'vring', 'sring', 
    'field', 'xfield', 'vfield', 'sfield',
    
    # sympy.series 模块中定义的函数或变量
    'Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants', 
    'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence', 
    'SeqAdd', 'SeqMul', 'fourier_series', 'fps', 'difference_delta', 'limit_seq',
    
    # sympy.functions 模块中定义的数学函数或常量
    'factorial', 'factorial2', 'rf', 'ff', 'binomial', 'RisingFactorial', 
    'FallingFactorial', 'subfactorial', 'carmichael', 'fibonacci', 'lucas', 
    'motzkin', 'tribonacci', 'harmonic', 'bernoulli', 'bell', 'euler', 'catalan', 
    'genocchi', 'andre', 'partition', 'divisor_sigma', 'legendre_symbol', 
    'jacobi_symbol', 'kronecker_symbol', 'mobius', 'primenu', 'primeomega', 
    'totient', 'primepi', 'reduced_totient', 'sqrt', 'root', 'Min', 'Max', 'Id', 
    'real_root', 'Rem', 'cbrt', 're', 'im', 'sign', 'Abs', 'conjugate', 'arg', 
    'polar_lift', 'periodic_argument', 'unbranched_argument', 'principal_branch', 
    'transpose', 'adjoint', 'polarify', 'unpolarify', 'sin', 'cos', 'tan', 'sec', 
    'csc', 'cot', 'sinc', 'asin', 'acos', 'atan', 'asec', 'acsc', 'acot', 'atan2', 
    'exp_polar', 'exp', 'ln', 'log', 'LambertW', 'sinh', 'cosh', 'tanh', 'coth', 
    'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch', 
    'floor', 'ceiling', 'frac', 'Piecewise', 'piecewise_fold', 
    'piecewise_exclusive', 'erf', 'erfc', 'erfi', 'erf2', 'erfinv', 'erfcinv', 
    'erf2inv', 'Ei', 'expint', 'E1', 'li', 'Li', 'Si', 'Ci', 'Shi', 'Chi', 
    'fresnels', 'fresnelc', 'gamma', 'lowergamma', 'uppergamma', 'polygamma', 
    'loggamma', 'digamma', 'trigamma', 'multigamma',
    # sympy.functions.special
    'dirichlet_eta', 'zeta', 'lerchphi', 'polylog', 'stieltjes', 'Eijk', 'LeviCivita',
    'KroneckerDelta', 'SingularityFunction', 'DiracDelta', 'Heaviside',
    'bspline_basis', 'bspline_basis_set', 'interpolating_spline', 'besselj',
    'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'jn', 'yn',
    'jn_zeros', 'hn1', 'hn2', 'airyai', 'airybi', 'airyaiprime',
    'airybiprime', 'marcumq', 'hyper', 'meijerg', 'appellf1', 'legendre',
    'assoc_legendre', 'hermite', 'hermite_prob', 'chebyshevt', 'chebyshevu',
    'chebyshevu_root', 'chebyshevt_root', 'laguerre', 'assoc_laguerre',
    'gegenbauer', 'jacobi', 'jacobi_normalized', 'Ynm', 'Ynm_c', 'Znm',
    'elliptic_k', 'elliptic_f', 'elliptic_e', 'elliptic_pi', 'beta',
    'mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime', 'riemann_xi','betainc',
    'betainc_regularized',

    # sympy.ntheory
    'nextprime', 'prevprime', 'prime', 'primerange', 'randprime',
    'Sieve', 'sieve', 'primorial', 'cycle_length', 'composite', 'compositepi',
    'isprime', 'divisors', 'proper_divisors', 'factorint', 'multiplicity',
    'perfect_power', 'pollard_pm1', 'pollard_rho', 'primefactors',
    'divisor_count', 'proper_divisor_count',
    'factorrat',
    'mersenne_prime_exponent', 'is_perfect', 'is_mersenne_prime',
    'is_abundant', 'is_deficient', 'is_amicable', 'is_carmichael', 'abundance',
    'npartitions',
    'is_primitive_root', 'is_quad_residue',
    'n_order', 'sqrt_mod', 'quadratic_residues',
    'primitive_root', 'nthroot_mod', 'is_nthpow_residue', 'sqrt_mod_iter',
    'discrete_log', 'quadratic_congruence', 'binomial_coefficients',
    'binomial_coefficients_list', 'multinomial_coefficients',
    'continued_fraction_periodic', 'continued_fraction_iterator',
    'continued_fraction_reduce', 'continued_fraction_convergents',
    'continued_fraction', 'egyptian_fraction',

    # sympy.concrete
    'product', 'Product', 'summation', 'Sum',

    # sympy.discrete
    'fft', 'ifft', 'ntt', 'intt', 'fwht', 'ifwht', 'mobius_transform',
    'inverse_mobius_transform', 'convolution', 'covering_product',
    'intersecting_product',

    # sympy.simplify
    'simplify', 'hypersimp', 'hypersimilar', 'logcombine', 'separatevars',
    'posify', 'besselsimp', 'kroneckersimp', 'signsimp',
    'nsimplify', 'FU', 'fu', 'sqrtdenest', 'cse', 'epath', 'EPath',
    'hyperexpand', 'collect', 'rcollect', 'radsimp', 'collect_const',
    'fraction', 'numer', 'denom', 'trigsimp', 'exptrigsimp', 'powsimp',
    'powdenest', 'combsimp', 'gammasimp', 'ratsimp', 'ratsimpmodprime',

    # sympy.sets
    'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
    'Intersection', 'imageset', 'DisjointUnion', 'Complement', 'SymmetricDifference',
    'ImageSet', 'Range', 'ComplexRegion', 'Reals', 'Contains', 'ConditionSet',
    'Ordinal', 'OmegaPower', 'ord0', 'PowerSet', 'Naturals',
    'Naturals0', 'UniversalSet', 'Integers', 'Rationals', 'Complexes',

    # sympy.solvers
    # Import SymPy functions related to algebraic solving
    'solve', 'solve_linear_system', 'solve_linear_system_LU',
    'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
    'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',
    'diophantine', 'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',
    'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',
    'solve_poly_system', 'solve_triangulated', 'pde_separate',
    'pde_separate_add', 'pde_separate_mul', 'pdsolve', 'classify_pde',
    'checkpdesol', 'ode_order', 'reduce_inequalities',
    'reduce_abs_inequality', 'reduce_abs_inequalities',
    'solve_poly_inequality', 'solve_rational_inequalities',
    'solve_univariate_inequality', 'decompogen', 'solveset', 'linsolve',
    'linear_eq_to_matrix', 'nonlinsolve', 'substitution',
    
    # Import SymPy matrix-related classes and functions
    # sympy.matrices
    'ShapeError', 'NonSquareMatrixError', 'GramSchmidt', 'casoratian', 'diag',
    'eye', 'hessian', 'jordan_cell', 'list2numpy', 'matrix2numpy',
    'matrix_multiply_elementwise', 'ones', 'randMatrix', 'rot_axis1',
    'rot_axis2', 'rot_axis3', 'symarray', 'wronskian', 'zeros',
    'MutableDenseMatrix', 'DeferredVector', 'MatrixBase', 'Matrix',
    'MutableMatrix', 'MutableSparseMatrix', 'banded', 'ImmutableDenseMatrix',
    'ImmutableSparseMatrix', 'ImmutableMatrix', 'SparseMatrix', 'MatrixSlice',
    'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix', 'Identity', 'Inverse',
    'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr', 'MatrixSymbol', 'Trace',
    'Transpose', 'ZeroMatrix', 'OneMatrix', 'blockcut', 'block_collapse',
    'matrix_symbols', 'Adjoint', 'hadamard_product', 'HadamardProduct',
    'HadamardPower', 'Determinant', 'det', 'diagonalize_vector', 'DiagMatrix',
    'DiagonalMatrix', 'DiagonalOf', 'trace', 'DotProduct',
    'kronecker_product', 'KroneckerProduct', 'PermutationMatrix',
    'MatrixPermute', 'Permanent', 'per', 'rot_ccw_axis1', 'rot_ccw_axis2',
    'rot_ccw_axis3', 'rot_givens',
    
    # Import SymPy geometric shapes and functions
    # sympy.geometry
    'Point', 'Point2D', 'Point3D', 'Line', 'Ray', 'Segment', 'Line2D',
    'Segment2D', 'Ray2D', 'Line3D', 'Segment3D', 'Ray3D', 'Plane', 'Ellipse',
    'Circle', 'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',
    'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
    'closest_points', 'farthest_points', 'GeometryError', 'Curve', 'Parabola',
    
    # Import SymPy utility functions
    # sympy.utilities
    'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
    'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
    'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
    'rotations', 'filldedent', 'lambdify', 'threaded', 'xthreaded',
    'public', 'memoize_property', 'timed',
    
    # Import SymPy integral and transform functions
    # sympy.integrals
    'integrate', 'Integral', 'line_integrate', 'mellin_transform',
    'inverse_mellin_transform', 'MellinTransform', 'InverseMellinTransform',
    'laplace_transform', 'inverse_laplace_transform', 'LaplaceTransform',
    'laplace_correspondence', 'laplace_initial_conds',
    'InverseLaplaceTransform', 'fourier_transform',
    'inverse_fourier_transform', 'FourierTransform',
    'InverseFourierTransform', 'sine_transform', 'inverse_sine_transform',
    'SineTransform', 'InverseSineTransform', 'cosine_transform',
    'inverse_cosine_transform', 'CosineTransform', 'InverseCosineTransform',
    'hankel_transform', 'inverse_hankel_transform', 'HankelTransform',
    'InverseHankelTransform', 'singularityintegrate',

    # SymPy transforms and integrals
    # 反演拉普拉斯变换、傅里叶变换及其逆变换、正弦变换及其逆变换、余弦变换及其逆变换、汉克尔变换及其逆变换、奇点积分

    # sympy.tensor
    'IndexedBase', 'Idx', 'Indexed', 'get_contraction_structure',
    'get_indices', 'shape', 'MutableDenseNDimArray', 'ImmutableDenseNDimArray',
    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',
    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array',
    'permutedims', 'Array', 'DenseNDimArray', 'SparseNDimArray',

    # SymPy tensor algebra and arrays
    # 索引基类、索引、获取收缩结构、获取索引、形状、可变稠密多维数组、不可变稠密多维数组、可变稀疏多维数组、不可变稀疏多维数组、多维数组、张量积、张量收缩、张量对角线、数组、稠密多维数组、稀疏多维数组

    # sympy.parsing
    'parse_expr',

    # SymPy expression parsing
    # 解析表达式

    # sympy.calculus
    'euler_equations', 'singularities', 'is_increasing',
    'is_strictly_increasing', 'is_decreasing', 'is_strictly_decreasing',
    'is_monotonic', 'finite_diff_weights', 'apply_finite_diff',
    'differentiate_finite', 'periodicity', 'not_empty_in',
    'AccumBounds', 'is_convex', 'stationary_points', 'minimum', 'maximum',

    # SymPy calculus and analysis
    # 欧拉方程、奇点、增函数、严格增函数、减函数、严格减函数、单调函数、有限差分权重、应用有限差分、有限差分求导、周期性、非空区间内、累积边界、凸函数、驻点、最小值、最大值

    # sympy.algebras
    'Quaternion',

    # SymPy algebras
    # 四元数

    # sympy.printing
    'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode', 'latex', 'print_latex', 'multiline_latex',
    'mathml', 'print_mathml', 'python', 'print_python', 'pycode', 'ccode',
    'print_ccode', 'smtlib_code', 'glsl_code', 'print_glsl', 'cxxcode', 'fcode',
    'print_fcode', 'rcode', 'print_rcode', 'jscode', 'print_jscode',
    'julia_code', 'mathematica_code', 'octave_code', 'rust_code', 'print_gtk',
    'preview', 'srepr', 'print_tree', 'StrPrinter', 'sstr', 'sstrrepr',
    'TableForm', 'dotprint', 'maple_code', 'print_maple_code',

    # SymPy printing and code generation
    # 分页打印、漂亮打印、漂亮打印、Python 打印、Python 打印使用 Unicode、尝试 Python 打印使用 Unicode、LaTeX 打印、LaTeX 输出、多行 LaTeX 输出、MathML 输出、MathML 打印、Python 代码生成、C 代码生成、C 代码输出、SMT-LIB 代码生成、GLSL 代码生成、GLSL 打印、C++ 代码生成、Fortran 代码生成、Fortran 打印、R 代码生成、R 打印、JavaScript 代码生成、JavaScript 打印、Julia 代码生成、Mathematica 代码生成、Octave 代码生成、Rust 代码生成、GTK 打印、预览、表达式表示、打印树形结构、字符串打印、字符串表示、表格形式、点形式打印、Maple 代码生成、Maple 打印

    # sympy.plotting
    'plot', 'textplot', 'plot_backends', 'plot_implicit', 'plot_parametric',

    # SymPy plotting
    # 绘图、文本绘图、绘图后端、隐式绘图、参数绘图

    # sympy.interactive
    'init_session', 'init_printing', 'interactive_traversal',

    # SymPy interactive session management
    # 初始化会话、初始化打印、交互遍历

    # sympy.testing
    'test', 'doctest',

    # SymPy testing utilities
    # 测试、文档测试
# 扩展 __all__ 列表以包含以下模块名称，这些模块在 SymPy 1.6 之前可以通过 `from sympy import *` 导入
# 这些模块名是在 `__init__.py` 文件中未定义 `__all__` 时隐式可导入的。未来版本可能不再支持从 `sympy` 中通配符导入这些模块。
__all__.extend((
    'algebras',          # 代数
    'assumptions',       # 假设
    'calculus',          # 微积分
    'concrete',          # 具体
    'discrete',          # 离散
    'external',          # 外部
    'functions',         # 函数
    'geometry',          # 几何
    'interactive',       # 交互
    'multipledispatch',  # 多重分派
    'ntheory',           # 数论
    'parsing',           # 解析
    'plotting',          # 绘图
    'polys',             # 多项式
    'printing',          # 打印
    'release',           # 发布
    'strategies',        # 策略
    'tensor',            # 张量
    'utilities',         # 实用工具
))
```