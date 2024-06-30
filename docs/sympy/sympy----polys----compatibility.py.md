# `D:\src\scipysrc\sympy\sympy\polys\compatibility.py`

```
"""Compatibility interface between dense and sparse polys. """

# 导入从密集多项式到稀疏多项式的兼容接口函数

# 导入从密集多项式到稀疏多项式的加项函数
from sympy.polys.densearith import dup_add_term
from sympy.polys.densearith import dmp_add_term

# 导入从密集多项式到稀疏多项式的减项函数
from sympy.polys.densearith import dup_sub_term
from sympy.polys.densearith import dmp_sub_term

# 导入从密集多项式到稀疏多项式的乘项函数
from sympy.polys.densearith import dup_mul_term
from sympy.polys.densearith import dmp_mul_term

# 导入从密集多项式到稀疏多项式的加常数函数
from sympy.polys.densearith import dup_add_ground
from sympy.polys.densearith import dmp_add_ground

# 导入从密集多项式到稀疏多项式的减常数函数
from sympy.polys.densearith import dup_sub_ground
from sympy.polys.densearith import dmp_sub_ground

# 导入从密集多项式到稀疏多项式的乘常数函数
from sympy.polys.densearith import dup_mul_ground
from sympy.polys.densearith import dmp_mul_ground

# 导入从密集多项式到稀疏多项式的除以常数函数
from sympy.polys.densearith import dup_quo_ground
from sympy.polys.densearith import dmp_quo_ground

# 导入从密集多项式到稀疏多项式的除以常数并取余函数
from sympy.polys.densearith import dup_exquo_ground
from sympy.polys.densearith import dmp_exquo_ground

# 导入从密集多项式到密集多项式的左移位函数
from sympy.polys.densearith import dup_lshift

# 导入从密集多项式到密集多项式的右移位函数
from sympy.polys.densearith import dup_rshift

# 导入从密集多项式到密集多项式的绝对值函数
from sympy.polys.densearith import dup_abs

# 导入从稀疏多项式到稀疏多项式的绝对值函数
from sympy.polys.densearith import dmp_abs

# 导入从密集多项式到密集多项式的取反函数
from sympy.polys.densearith import dup_neg

# 导入从稀疏多项式到稀疏多项式的取反函数
from sympy.polys.densearith import dmp_neg

# 导入从密集多项式到密集多项式的加法函数
from sympy.polys.densearith import dup_add

# 导入从稀疏多项式到稀疏多项式的加法函数
from sympy.polys.densearith import dmp_add

# 导入从密集多项式到密集多项式的减法函数
from sympy.polys.densearith import dup_sub

# 导入从稀疏多项式到稀疏多项式的减法函数
from sympy.polys.densearith import dmp_sub

# 导入从密集多项式到密集多项式的加法与乘法组合函数
from sympy.polys.densearith import dup_add_mul

# 导入从稀疏多项式到稀疏多项式的加法与乘法组合函数
from sympy.polys.densearith import dmp_add_mul

# 导入从密集多项式到密集多项式的减法与乘法组合函数
from sympy.polys.densearith import dup_sub_mul

# 导入从稀疏多项式到稀疏多项式的减法与乘法组合函数
from sympy.polys.densearith import dmp_sub_mul

# 导入从密集多项式到密集多项式的乘法函数
from sympy.polys.densearith import dup_mul

# 导入从稀疏多项式到稀疏多项式的乘法函数
from sympy.polys.densearith import dmp_mul

# 导入从密集多项式到密集多项式的平方函数
from sympy.polys.densearith import dup_sqr

# 导入从稀疏多项式到稀疏多项式的平方函数
from sympy.polys.densearith import dmp_sqr

# 导入从密集多项式到密集多项式的幂函数
from sympy.polys.densearith import dup_pow

# 导入从稀疏多项式到稀疏多项式的幂函数
from sympy.polys.densearith import dmp_pow

# 导入从密集多项式到密集多项式的多项式除法函数
from sympy.polys.densearith import dup_pdiv

# 导入从密集多项式到密集多项式的多项式余数函数
from sympy.polys.densearith import dup_prem

# 导入从密集多项式到密集多项式的多项式商函数
from sympy.polys.densearith import dup_pquo

# 导入从密集多项式到密集多项式的多项式除法并取余函数
from sympy.polys.densearith import dup_pexquo

# 导入从稀疏多项式到稀疏多项式的多项式除法函数
from sympy.polys.densearith import dmp_pdiv

# 导入从稀疏多项式到稀疏多项式的多项式余数函数
from sympy.polys.densearith import dmp_prem

# 导入从稀疏多项式到稀疏多项式的多项式商函数
from sympy.polys.densearith import dmp_pquo

# 导入从稀疏多项式到稀疏多项式的多项式除法并取余函数
from sympy.polys.densearith import dmp_pexquo

# 导入从密集多项式到密集多项式的有理数域上的右除法函数
from sympy.polys.densearith import dup_rr_div

# 导入从稀疏多项式到稀疏多项式的有理数域上的右除法函数
from sympy.polys.densearith import dmp_rr_div

# 导入从密集多项式到有理数域上的有理数域上的除法函数
from sympy.polys.densearith import dup_ff_div

# 导入从稀疏多项式到有理数域上的有理数域上的除法函数
from sympy.polys.densearith import dmp_ff_div

# 导入从密集多项式到密集多项式的除法函数
from sympy.polys.densearith import dup_div

# 导入从密集多项式到密集多项式的取余函数
from sympy.polys.densearith import dup_rem

# 导入从密集多项式到密集多项式的商函数
from sympy.polys.densearith import dup_quo

# 导入从密集多项式到密集多项式的除法并取余函数
from sympy.polys.densearith import dup_exquo

# 导入从稀疏多项式到稀疏多项式的除法函数
from sympy.polys.densearith import dmp_div

# 导入从稀疏多项式到稀疏多项式的取余函数
from sympy.polys.densearith import dmp_rem

# 导入从稀疏多项式到稀疏多项式的商函数
from sympy.polys.densearith import dmp_quo

# 导入从稀疏多项式到稀疏多项式的除法并取余函数
from sympy.polys.densearith import dmp_exquo

# 导入从密集多项式到密集多项式的最大范数函数
from sympy.polys.densearith import dup_max_norm

# 导入从稀疏多项式到稀疏多项式的最大
# 导入 sympy.polys.densearith 模块中的 dmp_expand 函数
from sympy.polys.densearith import dmp_expand
# 导入 sympy.polys.densebasic 模块中的以下函数：
from sympy.polys.densebasic import dup_LC      # 导入 dup_LC 函数
from sympy.polys.densebasic import dmp_LC      # 导入 dmp_LC 函数
from sympy.polys.densebasic import dup_TC      # 导入 dup_TC 函数
from sympy.polys.densebasic import dmp_TC      # 导入 dmp_TC 函数
from sympy.polys.densebasic import dmp_ground_LC   # 导入 dmp_ground_LC 函数
from sympy.polys.densebasic import dmp_ground_TC   # 导入 dmp_ground_TC 函数
from sympy.polys.densebasic import dup_degree   # 导入 dup_degree 函数
from sympy.polys.densebasic import dmp_degree   # 导入 dmp_degree 函数
from sympy.polys.densebasic import dmp_degree_in   # 导入 dmp_degree_in 函数
from sympy.polys.densebasic import dmp_to_dict   # 导入 dmp_to_dict 函数
from sympy.polys.densetools import dup_integrate   # 导入 dup_integrate 函数
from sympy.polys.densetools import dmp_integrate   # 导入 dmp_integrate 函数
from sympy.polys.densetools import dmp_integrate_in   # 导入 dmp_integrate_in 函数
from sympy.polys.densetools import dup_diff   # 导入 dup_diff 函数
from sympy.polys.densetools import dmp_diff   # 导入 dmp_diff 函数
from sympy.polys.densetools import dmp_diff_in   # 导入 dmp_diff_in 函数
from sympy.polys.densetools import dup_eval   # 导入 dup_eval 函数
from sympy.polys.densetools import dmp_eval   # 导入 dmp_eval 函数
from sympy.polys.densetools import dmp_eval_in   # 导入 dmp_eval_in 函数
from sympy.polys.densetools import dmp_eval_tail   # 导入 dmp_eval_tail 函数
from sympy.polys.densetools import dmp_diff_eval_in   # 导入 dmp_diff_eval_in 函数
from sympy.polys.densetools import dup_trunc   # 导入 dup_trunc 函数
from sympy.polys.densetools import dmp_trunc   # 导入 dmp_trunc 函数
from sympy.polys.densetools import dmp_ground_trunc   # 导入 dmp_ground_trunc 函数
from sympy.polys.densetools import dup_monic   # 导入 dup_monic 函数
from sympy.polys.densetools import dmp_ground_monic   # 导入 dmp_ground_monic 函数
from sympy.polys.densetools import dup_content   # 导入 dup_content 函数
from sympy.polys.densetools import dmp_ground_content   # 导入 dmp_ground_content 函数
from sympy.polys.densetools import dup_primitive   # 导入 dup_primitive 函数
from sympy.polys.densetools import dmp_ground_primitive   # 导入 dmp_ground_primitive 函数
from sympy.polys.densetools import dup_extract   # 导入 dup_extract 函数
from sympy.polys.densetools import dmp_ground_extract   # 导入 dmp_ground_extract 函数
from sympy.polys.densetools import dup_real_imag   # 导入 dup_real_imag 函数
from sympy.polys.densetools import dup_mirror   # 导入 dup_mirror 函数
from sympy.polys.densetools import dup_scale   # 导入 dup_scale 函数
from sympy.polys.densetools import dup_shift   # 导入 dup_shift 函数
from sympy.polys.densetools import dmp_shift   # 导入 dmp_shift 函数
from sympy.polys.densetools import dup_transform   # 导入 dup_transform 函数
from sympy.polys.densetools import dup_compose   # 导入 dup_compose 函数
from sympy.polys.densetools import dmp_compose   # 导入 dmp_compose 函数
from sympy.polys.densetools import dup_decompose   # 导入 dup_decompose 函数
from sympy.polys.densetools import dmp_lift   # 导入 dmp_lift 函数
from sympy.polys.densetools import dup_sign_variations   # 导入 dup_sign_variations 函数
from sympy.polys.densetools import dup_clear_denoms   # 导入 dup_clear_denoms 函数
from sympy.polys.densetools import dmp_clear_denoms   # 导入 dmp_clear_denoms 函数
from sympy.polys.densetools import dup_revert   # 导入 dup_revert 函数
from sympy.polys.euclidtools import dup_half_gcdex   # 导入 dup_half_gcdex 函数
from sympy.polys.euclidtools import dmp_half_gcdex   # 导入 dmp_half_gcdex 函数
from sympy.polys.euclidtools import dup_gcdex   # 导入 dup_gcdex 函数
from sympy.polys.euclidtools import dmp_gcdex   # 导入 dmp_gcdex 函数
from sympy.polys.euclidtools import dup_invert   # 导入 dup_invert 函数
from sympy.polys.euclidtools import dmp_invert   # 导入 dmp_invert 函数
from sympy.polys.euclidtools import dup_euclidean_prs   # 导入 dup_euclidean_prs 函数
from sympy.polys.euclidtools import dmp_euclidean_prs   # 导入 dmp_euclidean_prs 函数
from sympy.polys.euclidtools import dup_primitive_prs   # 导入 dup_primitive_prs 函数
from sympy.polys.euclidtools import dmp_primitive_prs   # 导入 dmp_primitive_prs 函数
from sympy.polys.euclidtools import dup_inner_subresultants   # 导入 dup_inner_subresultants 函数
from sympy.polys.euclidtools import dup_subresultants   # 导入 dup_subresultants 函数
from sympy.polys.euclidtools import dup_prs_resultant   # 导入 dup_prs_resultant 函数
from sympy.polys.euclidtools import dup_resultant   # 导入 dup_resultant 函数
# 导入从 sympy.polys.euclidtools 模块中的函数
from sympy.polys.euclidtools import dmp_inner_subresultants  # 导入 dmp_inner_subresultants 函数
from sympy.polys.euclidtools import dmp_subresultants  # 导入 dmp_subresultants 函数
from sympy.polys.euclidtools import dmp_prs_resultant  # 导入 dmp_prs_resultant 函数
from sympy.polys.euclidtools import dmp_zz_modular_resultant  # 导入 dmp_zz_modular_resultant 函数
from sympy.polys.euclidtools import dmp_zz_collins_resultant  # 导入 dmp_zz_collins_resultant 函数
from sympy.polys.euclidtools import dmp_qq_collins_resultant  # 导入 dmp_qq_collins_resultant 函数
from sympy.polys.euclidtools import dmp_resultant  # 导入 dmp_resultant 函数
from sympy.polys.euclidtools import dup_discriminant  # 导入 dup_discriminant 函数
from sympy.polys.euclidtools import dmp_discriminant  # 导入 dmp_discriminant 函数
from sympy.polys.euclidtools import dup_rr_prs_gcd  # 导入 dup_rr_prs_gcd 函数
from sympy.polys.euclidtools import dup_ff_prs_gcd  # 导入 dup_ff_prs_gcd 函数
from sympy.polys.euclidtools import dmp_rr_prs_gcd  # 导入 dmp_rr_prs_gcd 函数
from sympy.polys.euclidtools import dmp_ff_prs_gcd  # 导入 dmp_ff_prs_gcd 函数
from sympy.polys.euclidtools import dup_zz_heu_gcd  # 导入 dup_zz_heu_gcd 函数
from sympy.polys.euclidtools import dmp_zz_heu_gcd  # 导入 dmp_zz_heu_gcd 函数
from sympy.polys.euclidtools import dup_qq_heu_gcd  # 导入 dup_qq_heu_gcd 函数
from sympy.polys.euclidtools import dmp_qq_heu_gcd  # 导入 dmp_qq_heu_gcd 函数
from sympy.polys.euclidtools import dup_inner_gcd  # 导入 dup_inner_gcd 函数
from sympy.polys.euclidtools import dmp_inner_gcd  # 导入 dmp_inner_gcd 函数
from sympy.polys.euclidtools import dup_gcd  # 导入 dup_gcd 函数
from sympy.polys.euclidtools import dmp_gcd  # 导入 dmp_gcd 函数
from sympy.polys.euclidtools import dup_rr_lcm  # 导入 dup_rr_lcm 函数
from sympy.polys.euclidtools import dup_ff_lcm  # 导入 dup_ff_lcm 函数
from sympy.polys.euclidtools import dup_lcm  # 导入 dup_lcm 函数
from sympy.polys.euclidtools import dmp_rr_lcm  # 导入 dmp_rr_lcm 函数
from sympy.polys.euclidtools import dmp_ff_lcm  # 导入 dmp_ff_lcm 函数
from sympy.polys.euclidtools import dmp_lcm  # 导入 dmp_lcm 函数
from sympy.polys.euclidtools import dmp_content  # 导入 dmp_content 函数
from sympy.polys.euclidtools import dmp_primitive  # 导入 dmp_primitive 函数
from sympy.polys.euclidtools import dup_cancel  # 导入 dup_cancel 函数
from sympy.polys.euclidtools import dmp_cancel  # 导入 dmp_cancel 函数
from sympy.polys.factortools import dup_trial_division  # 导入 dup_trial_division 函数
from sympy.polys.factortools import dmp_trial_division  # 导入 dmp_trial_division 函数
from sympy.polys.factortools import dup_zz_mignotte_bound  # 导入 dup_zz_mignotte_bound 函数
from sympy.polys.factortools import dmp_zz_mignotte_bound  # 导入 dmp_zz_mignotte_bound 函数
from sympy.polys.factortools import dup_zz_hensel_step  # 导入 dup_zz_hensel_step 函数
from sympy.polys.factortools import dup_zz_hensel_lift  # 导入 dup_zz_hensel_lift 函数
from sympy.polys.factortools import dup_zz_zassenhaus  # 导入 dup_zz_zassenhaus 函数
from sympy.polys.factortools import dup_zz_irreducible_p  # 导入 dup_zz_irreducible_p 函数
from sympy.polys.factortools import dup_cyclotomic_p  # 导入 dup_cyclotomic_p 函数
from sympy.polys.factortools import dup_zz_cyclotomic_poly  # 导入 dup_zz_cyclotomic_poly 函数
from sympy.polys.factortools import dup_zz_cyclotomic_factor  # 导入 dup_zz_cyclotomic_factor 函数
from sympy.polys.factortools import dup_zz_factor_sqf  # 导入 dup_zz_factor_sqf 函数
from sympy.polys.factortools import dup_zz_factor  # 导入 dup_zz_factor 函数
from sympy.polys.factortools import dmp_zz_wang_non_divisors  # 导入 dmp_zz_wang_non_divisors 函数
from sympy.polys.factortools import dmp_zz_wang_lead_coeffs  # 导入 dmp_zz_wang_lead_coeffs 函数
from sympy.polys.factortools import dup_zz_diophantine  # 导入 dup_zz_diophantine 函数
from sympy.polys.factortools import dmp_zz_diophantine  # 导入 dmp_zz_diophantine 函数
from sympy.polys.factortools import dmp_zz_wang_hensel_lifting  # 导入 dmp_zz_wang_hensel_lifting 函数
from sympy.polys.factortools import dmp_zz_wang  # 导入 dmp_zz_wang 函数
from sympy.polys.factortools import dmp_zz_factor  # 导入 dmp_zz_factor 函数
from sympy.polys.factortools import dup_qq_i_factor  # 导入 dup_qq_i_factor 函数
from sympy.polys.factortools import dup_zz_i_factor  # 导入 dup_zz_i_factor 函数
from sympy.polys.factortools import dmp_qq_i_factor  # 导入 dmp_qq_i_factor 函数
from sympy.polys.factortools import dmp_zz_i_factor  # 导入 dmp_zz_i_factor 函数
from sympy.polys.factortools import dup_ext_factor  # 导入 dup_ext_factor 函数
from sympy.polys.factortools import dmp_ext_factor  # 导入 dmp_ext_factor 函数
# 导入 sympy.polys.factortools 模块中的函数和类
from sympy.polys.factortools import dup_gf_factor
from sympy.polys.factortools import dmp_gf_factor
from sympy.polys.factortools import dup_factor_list
from sympy.polys.factortools import dup_factor_list_include
from sympy.polys.factortools import dmp_factor_list
from sympy.polys.factortools import dmp_factor_list_include
from sympy.polys.factortools import dup_irreducible_p
from sympy.polys.factortools import dmp_irreducible_p

# 导入 sympy.polys.rootisolation 模块中的函数
from sympy.polys.rootisolation import dup_sturm
from sympy.polys.rootisolation import dup_root_upper_bound
from sympy.polys.rootisolation import dup_root_lower_bound
from sympy.polys.rootisolation import dup_step_refine_real_root
from sympy.polys.rootisolation import dup_inner_refine_real_root
from sympy.polys.rootisolation import dup_outer_refine_real_root
from sympy.polys.rootisolation import dup_refine_real_root
from sympy.polys.rootisolation import dup_inner_isolate_real_roots
from sympy.polys.rootisolation import dup_inner_isolate_positive_roots
from sympy.polys.rootisolation import dup_inner_isolate_negative_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_sqf
from sympy.polys.rootisolation import dup_isolate_real_roots
from sympy.polys.rootisolation import dup_isolate_real_roots_list
from sympy.polys.rootisolation import dup_count_real_roots
from sympy.polys.rootisolation import dup_count_complex_roots
from sympy.polys.rootisolation import dup_isolate_complex_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots_sqf
from sympy.polys.rootisolation import dup_isolate_all_roots

# 导入 sympy.polys.sqfreetools 模块中的函数
from sympy.polys.sqfreetools import (
    dup_sqf_p, dmp_sqf_p, dmp_norm, dup_sqf_norm, dmp_sqf_norm,
    dup_gf_sqf_part, dmp_gf_sqf_part, dup_sqf_part, dmp_sqf_part,
    dup_gf_sqf_list, dmp_gf_sqf_list, dup_sqf_list, dup_sqf_list_include,
    dmp_sqf_list, dmp_sqf_list_include, dup_gff_list, dmp_gff_list)

# 导入 sympy.polys.galoistools 模块中的函数
from sympy.polys.galoistools import (
    gf_degree, gf_LC, gf_TC, gf_strip, gf_from_dict,
    gf_to_dict, gf_from_int_poly, gf_to_int_poly, gf_neg, gf_add_ground, gf_sub_ground,
    gf_mul_ground, gf_quo_ground, gf_add, gf_sub, gf_mul, gf_sqr, gf_add_mul, gf_sub_mul,
    gf_expand, gf_div, gf_rem, gf_quo, gf_exquo, gf_lshift, gf_rshift, gf_pow, gf_pow_mod,
    gf_gcd, gf_lcm, gf_cofactors, gf_gcdex, gf_monic, gf_diff, gf_eval, gf_multi_eval,
    gf_compose, gf_compose_mod, gf_trace_map, gf_random, gf_irreducible, gf_irred_p_ben_or,
    gf_irred_p_rabin, gf_irreducible_p, gf_sqf_p, gf_sqf_part, gf_Qmatrix,
    gf_berlekamp, gf_ddf_zassenhaus, gf_edf_zassenhaus, gf_ddf_shoup, gf_edf_shoup,
    gf_zassenhaus, gf_shoup, gf_factor_sqf, gf_factor)

# 导入 sympy.utilities 模块中的 public 类装饰器
from sympy.utilities import public

@public
class IPolys:
    symbols = None  # 类属性：符号列表，默认为 None
    ngens = None    # 类属性：生成元数量，默认为 None
    domain = None   # 类属性：域，默认为 None
    order = None    # 类属性：排序方式，默认为 None
    gens = None     # 类属性：生成元列表，默认为 None

    def drop(self, gen):
        pass  # 方法：从多项式中去除指定的生成元

    def clone(self, symbols=None, domain=None, order=None):
        pass  # 方法：克隆对象，可选地指定符号、域和排序方式

    def to_ground(self):
        pass  # 方法：将多项式转换为整数系数的多项式

    def ground_new(self, element):
        pass  # 方法：生成一个新的以给定元素为根的多项式
    # 定义一个名为 domain_new 的方法，接受一个参数 element
    def domain_new(self, element):
        pass

    # 定义一个名为 from_dict 的方法，接受一个参数 d
    def from_dict(self, d):
        pass

    # 定义一个名为 wrap 的方法，接受一个参数 element
    def wrap(self, element):
        # 导入 SymPy 中的 Polys 模块中的 PolyElement 类
        from sympy.polys.rings import PolyElement
        # 如果 element 是 PolyElement 类型
        if isinstance(element, PolyElement):
            # 如果 element 的环等于当前实例的环，直接返回 element
            if element.ring == self:
                return element
            else:
                # 否则抛出未实现的错误，指出不支持的域转换
                raise NotImplementedError("domain conversions")
        else:
            # 否则调用 ground_new 方法，返回处理后的 element
            return self.ground_new(element)

    # 定义一个名为 to_dense 的方法，接受一个参数 element
    def to_dense(self, element):
        # 调用 wrap 方法处理 element 后，再调用其 to_dense 方法返回结果
        return self.wrap(element).to_dense()

    # 定义一个名为 from_dense 的方法，接受一个参数 element
    def from_dense(self, element):
        # 调用 dmp_to_dict 方法将 element 转换为字典格式，返回处理结果
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))

    # 定义一个名为 dup_add_term 的方法，接受三个参数 f, c, i
    def dup_add_term(self, f, c, i):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_add_term 方法进行加法操作，最后返回处理结果
        return self.from_dense(dup_add_term(self.to_dense(f), c, i, self.domain))

    # 定义一个名为 dmp_add_term 的方法，接受四个参数 f, c, i
    def dmp_add_term(self, f, c, i):
        # 调用 to_dense 方法将 f 转换为密集表示，调用 wrap 方法处理 c 后进行加法操作，最后返回处理结果
        return self.from_dense(dmp_add_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))

    # 定义一个名为 dup_sub_term 的方法，接受三个参数 f, c, i
    def dup_sub_term(self, f, c, i):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_sub_term 方法进行减法操作，最后返回处理结果
        return self.from_dense(dup_sub_term(self.to_dense(f), c, i, self.domain))

    # 定义一个名为 dmp_sub_term 的方法，接受四个参数 f, c, i
    def dmp_sub_term(self, f, c, i):
        # 调用 to_dense 方法将 f 转换为密集表示，调用 wrap 方法处理 c 后进行减法操作，最后返回处理结果
        return self.from_dense(dmp_sub_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))

    # 定义一个名为 dup_mul_term 的方法，接受三个参数 f, c, i
    def dup_mul_term(self, f, c, i):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_mul_term 方法进行乘法操作，最后返回处理结果
        return self.from_dense(dup_mul_term(self.to_dense(f), c, i, self.domain))

    # 定义一个名为 dmp_mul_term 的方法，接受四个参数 f, c, i
    def dmp_mul_term(self, f, c, i):
        # 调用 to_dense 方法将 f 转换为密集表示，调用 wrap 方法处理 c 后进行乘法操作，最后返回处理结果
        return self.from_dense(dmp_mul_term(self.to_dense(f), self.wrap(c).drop(0).to_dense(), i, self.ngens-1, self.domain))

    # 定义一个名为 dup_add_ground 的方法，接受两个参数 f, c
    def dup_add_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_add_ground 方法进行常数加法操作，最后返回处理结果
        return self.from_dense(dup_add_ground(self.to_dense(f), c, self.domain))

    # 定义一个名为 dmp_add_ground 的方法，接受两个参数 f, c
    def dmp_add_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dmp_add_ground 方法进行常数加法操作，最后返回处理结果
        return self.from_dense(dmp_add_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    # 定义一个名为 dup_sub_ground 的方法，接受两个参数 f, c
    def dup_sub_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_sub_ground 方法进行常数减法操作，最后返回处理结果
        return self.from_dense(dup_sub_ground(self.to_dense(f), c, self.domain))

    # 定义一个名为 dmp_sub_ground 的方法，接受两个参数 f, c
    def dmp_sub_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dmp_sub_ground 方法进行常数减法操作，最后返回处理结果
        return self.from_dense(dmp_sub_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    # 定义一个名为 dup_mul_ground 的方法，接受两个参数 f, c
    def dup_mul_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_mul_ground 方法进行常数乘法操作，最后返回处理结果
        return self.from_dense(dup_mul_ground(self.to_dense(f), c, self.domain))

    # 定义一个名为 dmp_mul_ground 的方法，接受两个参数 f, c
    def dmp_mul_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dmp_mul_ground 方法进行常数乘法操作，最后返回处理结果
        return self.from_dense(dmp_mul_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    # 定义一个名为 dup_quo_ground 的方法，接受两个参数 f, c
    def dup_quo_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_quo_ground 方法进行常数除法操作，最后返回处理结果
        return self.from_dense(dup_quo_ground(self.to_dense(f), c, self.domain))

    # 定义一个名为 dmp_quo_ground 的方法，接受两个参数 f, c
    def dmp_quo_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dmp_quo_ground 方法进行常数除法操作，最后返回处理结果
        return self.from_dense(dmp_quo_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    # 定义一个名为 dup_exquo_ground 的方法，接受两个参数 f, c
    def dup_exquo_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dup_exquo_ground 方法进行常数整除操作，最后返回处理结果
        return self.from_dense(dup_exquo_ground(self.to_dense(f), c, self.domain))

    # 定义一个名为 dmp_exquo_ground 的方法，接受两个参数 f, c
    def dmp_exquo_ground(self, f, c):
        # 调用 to_dense 方法将 f 转换为密集表示，再调用 dmp_exquo_ground 方法进行常数整除操作，最后返回处理结果
        return self.from_dense(dmp_exquo_ground(self.to_dense(f), c, self.ngens-1, self.domain))

    # 定义一个名为 dup_lshift 的方法，接受两个参数 f, n
    def
    # 对给定的多项式 f 计算其绝对值（dup_abs）
    def dup_abs(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其绝对值，再将结果转回原多项式的表示
        return self.from_dense(dup_abs(self.to_dense(f), self.domain))

    # 对给定的多项式 f 计算其绝对值（dmp_abs）
    def dmp_abs(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其多项式绝对值（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_abs(self.to_dense(f), self.ngens-1, self.domain))

    # 对给定的多项式 f 计算其负数（dup_neg）
    def dup_neg(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其负数，再将结果转回原多项式的表示
        return self.from_dense(dup_neg(self.to_dense(f), self.domain))

    # 对给定的多项式 f 计算其负数（dmp_neg）
    def dmp_neg(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其多项式负数（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_neg(self.to_dense(f), self.ngens-1, self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的和（dup_add）
    def dup_add(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的和，再将结果转回原多项式的表示
        return self.from_dense(dup_add(self.to_dense(f), self.to_dense(g), self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的和（dmp_add）
    def dmp_add(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的多项式和（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_add(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的差（dup_sub）
    def dup_sub(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的差，再将结果转回原多项式的表示
        return self.from_dense(dup_sub(self.to_dense(f), self.to_dense(g), self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的差（dmp_sub）
    def dmp_sub(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的多项式差（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_sub(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 对给定的三个多项式 f, g, h 计算 f + g*h（dup_add_mul）
    def dup_add_mul(self, f, g, h):
        # 将多项式 f, g, h 转换为稠密表示，然后计算 f + g*h，再将结果转回原多项式的表示
        return self.from_dense(dup_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))

    # 对给定的三个多项式 f, g, h 计算 f + g*h（dmp_add_mul）
    def dmp_add_mul(self, f, g, h):
        # 将多项式 f, g, h 转换为稠密表示，然后计算 f + g*h 的多项式形式（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_add_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))

    # 对给定的三个多项式 f, g, h 计算 f - g*h（dup_sub_mul）
    def dup_sub_mul(self, f, g, h):
        # 将多项式 f, g, h 转换为稠密表示，然后计算 f - g*h，再将结果转回原多项式的表示
        return self.from_dense(dup_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.domain))

    # 对给定的三个多项式 f, g, h 计算 f - g*h（dmp_sub_mul）
    def dmp_sub_mul(self, f, g, h):
        # 将多项式 f, g, h 转换为稠密表示，然后计算 f - g*h 的多项式形式（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_sub_mul(self.to_dense(f), self.to_dense(g), self.to_dense(h), self.ngens-1, self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的乘积（dup_mul）
    def dup_mul(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的乘积，再将结果转回原多项式的表示
        return self.from_dense(dup_mul(self.to_dense(f), self.to_dense(g), self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的乘积（dmp_mul）
    def dmp_mul(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的多项式乘积（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_mul(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 对给定的多项式 f 计算其平方（dup_sqr）
    def dup_sqr(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其平方，再将结果转回原多项式的表示
        return self.from_dense(dup_sqr(self.to_dense(f), self.domain))

    # 对给定的多项式 f 计算其平方（dmp_sqr）
    def dmp_sqr(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其多项式平方（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_sqr(self.to_dense(f), self.ngens-1, self.domain))

    # 对给定的多项式 f 和整数 n 计算 f 的 n 次幂（dup_pow）
    def dup_pow(self, f, n):
        # 将多项式 f 转换为稠密表示，然后计算其幂次为 n 的结果，再将结果转回原多项式的表示
        return self.from_dense(dup_pow(self.to_dense(f), n, self.domain))

    # 对给定的多项式 f 和整数 n 计算 f 的 n 次幂（dmp_pow）
    def dmp_pow(self, f, n):
        # 将多项式 f 转换为稠密表示，然后计算其多项式幂次为 n 的结果（对每个系数分别计算），再将结果转回原多项式的表示
        return self.from_dense(dmp_pow(self.to_dense(f), n, self.ngens-1, self.domain))

    # 对给定的两个多项式 f 和 g 计算它们的除法商和余数（dup_pdiv）
    def dup_pdiv(self, f, g):
        # 将多项式 f 和 g 转换为稠密表示，然后计算它们的多项式除法商和余数，再将结果转回
    # 计算两个多项式的首项排除法结果，并返回转换后的稠密表示
    def dmp_prem(self, f, g):
        return self.from_dense(dmp_prem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 计算两个多项式的首项商，并返回转换后的稠密表示
    def dmp_pquo(self, f, g):
        return self.from_dense(dmp_pquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 计算两个多项式的首项扩展商，并返回转换后的稠密表示
    def dmp_pexquo(self, f, g):
        return self.from_dense(dmp_pexquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 计算两个一元多项式的重根剩余与商，并返回转换后的稠密表示
    def dup_rr_div(self, f, g):
        q, r = dup_rr_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))

    # 计算两个多元多项式的重根剩余与商，并返回转换后的稠密表示
    def dmp_rr_div(self, f, g):
        q, r = dmp_rr_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))

    # 计算两个一元有限域多项式的有限域除法结果，并返回转换后的稠密表示
    def dup_ff_div(self, f, g):
        q, r = dup_ff_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))

    # 计算两个多元有限域多项式的有限域除法结果，并返回转换后的稠密表示
    def dmp_ff_div(self, f, g):
        q, r = dmp_ff_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))

    # 计算两个一元多项式的除法结果，并返回转换后的稠密表示
    def dup_div(self, f, g):
        q, r = dup_div(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(q), self.from_dense(r))

    # 计算两个一元多项式的剩余，并返回转换后的稠密表示
    def dup_rem(self, f, g):
        return self.from_dense(dup_rem(self.to_dense(f), self.to_dense(g), self.domain))

    # 计算两个一元多项式的商，并返回转换后的稠密表示
    def dup_quo(self, f, g):
        return self.from_dense(dup_quo(self.to_dense(f), self.to_dense(g), self.domain))

    # 计算两个一元多项式的扩展商，并返回转换后的稠密表示
    def dup_exquo(self, f, g):
        return self.from_dense(dup_exquo(self.to_dense(f), self.to_dense(g), self.domain))

    # 计算两个多元多项式的除法结果，并返回转换后的稠密表示
    def dmp_div(self, f, g):
        q, r = dmp_div(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(q), self.from_dense(r))

    # 计算两个多元多项式的剩余，并返回转换后的稠密表示
    def dmp_rem(self, f, g):
        return self.from_dense(dmp_rem(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 计算两个多元多项式的商，并返回转换后的稠密表示
    def dmp_quo(self, f, g):
        return self.from_dense(dmp_quo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 计算两个多元多项式的扩展商，并返回转换后的稠密表示
    def dmp_exquo(self, f, g):
        return self.from_dense(dmp_exquo(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 计算一元多项式的最大范数
    def dup_max_norm(self, f):
        return dup_max_norm(self.to_dense(f), self.domain)

    # 计算多元多项式的最大范数
    def dmp_max_norm(self, f):
        return dmp_max_norm(self.to_dense(f), self.ngens-1, self.domain)

    # 计算一元多项式的 L1 范数
    def dup_l1_norm(self, f):
        return dup_l1_norm(self.to_dense(f), self.domain)

    # 计算多元多项式的 L1 范数
    def dmp_l1_norm(self, f):
        return dmp_l1_norm(self.to_dense(f), self.ngens-1, self.domain)

    # 计算一元多项式的 L2 范数的平方
    def dup_l2_norm_squared(self, f):
        return dup_l2_norm_squared(self.to_dense(f), self.domain)

    # 计算多元多项式的 L2 范数的平方
    def dmp_l2_norm_squared(self, f):
        return dmp_l2_norm_squared(self.to_dense(f), self.ngens-1, self.domain)

    # 将一组多项式扩展为单个多项式，并返回转换后的稠密表示
    def dup_expand(self, polys):
        return self.from_dense(dup_expand(list(map(self.to_dense, polys)), self.domain))
    # 将多项式列表 polys 映射为稠密表示后，再通过 dmp_expand 扩展，返回新多项式对象
    def dmp_expand(self, polys):
        return self.from_dense(dmp_expand(list(map(self.to_dense, polys)), self.ngens-1, self.domain))

    # 返回多项式 f 的最高次数系数，转换为稠密表示后进行计算
    def dup_LC(self, f):
        return dup_LC(self.to_dense(f), self.domain)

    # 返回多项式 f 的最高次数系数，对于多项式列表，则返回从索引 1 开始的新对象
    def dmp_LC(self, f):
        LC = dmp_LC(self.to_dense(f), self.domain)
        if isinstance(LC, list):
            return self[1:].from_dense(LC)
        else:
            return LC

    # 返回多项式 f 的首项系数，转换为稠密表示后进行计算
    def dup_TC(self, f):
        return dup_TC(self.to_dense(f), self.domain)

    # 返回多项式 f 的首项系数，对于多项式列表，则返回从索引 1 开始的新对象
    def dmp_TC(self, f):
        TC = dmp_TC(self.to_dense(f), self.domain)
        if isinstance(TC, list):
            return self[1:].from_dense(TC)
        else:
            return TC

    # 返回多项式 f 的常数项系数，转换为稠密表示后进行计算
    def dmp_ground_LC(self, f):
        return dmp_ground_LC(self.to_dense(f), self.ngens-1, self.domain)

    # 返回多项式 f 的常数项系数，转换为稠密表示后进行计算
    def dmp_ground_TC(self, f):
        return dmp_ground_TC(self.to_dense(f), self.ngens-1, self.domain)

    # 返回多项式 f 的次数，转换为稠密表示后进行计算
    def dup_degree(self, f):
        return dup_degree(self.to_dense(f))

    # 返回多项式 f 的次数，对于多项式列表，则返回最高次数的变量索引
    def dmp_degree(self, f):
        return dmp_degree(self.to_dense(f), self.ngens-1)

    # 返回多项式 f 在指定变量索引 j 上的次数，转换为稠密表示后进行计算
    def dmp_degree_in(self, f, j):
        return dmp_degree_in(self.to_dense(f), j, self.ngens-1)

    # 返回多项式 f 在变量 x 上的 m 次积分，转换为稠密表示后进行计算
    def dup_integrate(self, f, m):
        return self.from_dense(dup_integrate(self.to_dense(f), m, self.domain))

    # 返回多项式 f 在变量索引 j 上的 m 次积分，转换为稠密表示后进行计算
    def dmp_integrate(self, f, m):
        return self.from_dense(dmp_integrate(self.to_dense(f), m, self.ngens-1, self.domain))

    # 返回多项式 f 在变量 x 上的 m 次导数，转换为稠密表示后进行计算
    def dup_diff(self, f, m):
        return self.from_dense(dup_diff(self.to_dense(f), m, self.domain))

    # 返回多项式 f 在变量索引 j 上的 m 次导数，转换为稠密表示后进行计算
    def dmp_diff(self, f, m):
        return self.from_dense(dmp_diff(self.to_dense(f), m, self.ngens-1, self.domain))

    # 返回多项式 f 在变量索引 j 上的 m 次导数，不同于 dmp_diff，结果通过剔除第 j 变量返回新多项式对象
    def dmp_diff_in(self, f, m, j):
        return self.from_dense(dmp_diff_in(self.to_dense(f), m, j, self.ngens-1, self.domain))

    # 返回多项式 f 在变量索引 j 上的 m 次积分，不同于 dmp_integrate，结果通过剔除第 j 变量返回新多项式对象
    def dmp_integrate_in(self, f, m, j):
        return self.from_dense(dmp_integrate_in(self.to_dense(f), m, j, self.ngens-1, self.domain))

    # 对多项式 f 在变量 x 上进行 a 处的求值，转换为稠密表示后进行计算
    def dup_eval(self, f, a):
        return dup_eval(self.to_dense(f), a, self.domain)

    # 对多项式 f 在变量 x 上进行 a 处的求值，结果为列表时，返回从索引 1 开始的新多项式对象
    def dmp_eval(self, f, a):
        result = dmp_eval(self.to_dense(f), a, self.ngens-1, self.domain)
        return self[1:].from_dense(result)

    # 对多项式 f 在变量索引 j 上进行 a 处的求值，结果为列表时，剔除第 j 变量返回新多项式对象
    def dmp_eval_in(self, f, a, j):
        result = dmp_eval_in(self.to_dense(f), a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)

    # 对多项式 f 在变量索引 j 上进行 m 次导数后再进行 a 处的求值，剔除第 j 变量返回新多项式对象
    def dmp_diff_eval_in(self, f, m, a, j):
        result = dmp_diff_eval_in(self.to_dense(f), m, a, j, self.ngens-1, self.domain)
        return self.drop(j).from_dense(result)

    # 对多项式 f 进行尾部元素 A 的求值，转换为稠密表示后进行计算
    def dmp_eval_tail(self, f, A):
        result = dmp_eval_tail(self.to_dense(f), A, self.ngens-1, self.domain)
        if isinstance(result, list):
            return self[:-len(A)].from_dense(result)
        else:
            return result

    # 对多项式 f 进行在变量 x 上的模 p 截断，转换为稠密表示后进行计算
    def dup_trunc(self, f, p):
        return self.from_dense(dup_trunc(self.to_dense(f), p, self.domain))

    # 对多项式 f 在变量 x 上进行在变量 g 上的模截断，转换为稠密表示后进行计算
    def dmp_trunc(self, f, g):
        return self.from_dense(dmp_trunc(self.to_dense(f), self[1:].to_dense(g), self.ngens-1, self.domain))
    # 从稠密表示转换为多项式，然后截断系数列表的高次项，返回截断后的多项式
    def dmp_ground_trunc(self, f, p):
        return self.from_dense(dmp_ground_trunc(self.to_dense(f), p, self.ngens-1, self.domain))

    # 从稠密表示转换为多项式，然后将多项式转换为首项系数为1的首一多项式
    def dup_monic(self, f):
        return self.from_dense(dup_monic(self.to_dense(f), self.domain))

    # 从稠密表示转换为多项式，然后将多项式转换为首项系数为1的首一多项式
    def dmp_ground_monic(self, f):
        return self.from_dense(dmp_ground_monic(self.to_dense(f), self.ngens-1, self.domain))

    # 从稠密表示转换为多项式，提取出两个多项式的最高公因式和对应的商多项式
    def dup_extract(self, f, g):
        c, F, G = dup_extract(self.to_dense(f), self.to_dense(g), self.domain)
        return (c, self.from_dense(F), self.from_dense(G))

    # 从稠密表示转换为多项式，提取出两个多项式的最高公因式和对应的商多项式
    def dmp_ground_extract(self, f, g):
        c, F, G = dmp_ground_extract(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (c, self.from_dense(F), self.from_dense(G))

    # 从稠密表示转换为多项式，将复数表示的多项式转换为实部和虚部的多项式
    def dup_real_imag(self, f):
        p, q = dup_real_imag(self.wrap(f).drop(1).to_dense(), self.domain)
        return (self.from_dense(p), self.from_dense(q))

    # 从稠密表示转换为多项式，将多项式翻转（系数倒序）
    def dup_mirror(self, f):
        return self.from_dense(dup_mirror(self.to_dense(f), self.domain))

    # 从稠密表示转换为多项式，将多项式中的系数乘以标量 a
    def dup_scale(self, f, a):
        return self.from_dense(dup_scale(self.to_dense(f), a, self.domain))

    # 从稠密表示转换为多项式，将多项式中的每个系数加上标量 a
    def dup_shift(self, f, a):
        return self.from_dense(dup_shift(self.to_dense(f), a, self.domain))

    # 从稠密表示转换为多项式，将多项式的每个系数向上/向下移动 a 个位置
    def dmp_shift(self, f, a):
        return self.from_dense(dmp_shift(self.to_dense(f), a, self.ngens-1, self.domain))

    # 从稠密表示转换为多项式，使用给定的多项式 p 和 q 对多项式 f 进行线性变换
    def dup_transform(self, f, p, q):
        return self.from_dense(dup_transform(self.to_dense(f), self.to_dense(p), self.to_dense(q), self.domain))

    # 从稠密表示转换为多项式，将两个多项式 f 和 g 进行复合运算（f(g(x))）
    def dup_compose(self, f, g):
        return self.from_dense(dup_compose(self.to_dense(f), self.to_dense(g), self.domain))

    # 从稠密表示转换为多项式，将两个多项式 f 和 g 进行复合运算（f(g(x))）
    def dmp_compose(self, f, g):
        return self.from_dense(dmp_compose(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 从稠密表示转换为多项式，将多项式 f 进行因式分解
    def dup_decompose(self, f):
        components = dup_decompose(self.to_dense(f), self.domain)
        return list(map(self.from_dense, components))

    # 从稠密表示转换为多项式，将多项式 f 的系数提升到整数环
    def dmp_lift(self, f):
        result = dmp_lift(self.to_dense(f), self.ngens-1, self.domain)
        return self.to_ground().from_dense(result)

    # 计算多项式 f 的符号变化次数
    def dup_sign_variations(self, f):
        return dup_sign_variations(self.to_dense(f), self.domain)

    # 从稠密表示转换为多项式，将多项式 f 的分母清除，并返回清除后的结果
    def dup_clear_denoms(self, f, convert=False):
        c, F = dup_clear_denoms(self.to_dense(f), self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))

    # 从稠密表示转换为多项式，将多项式 f 的分母清除，并返回清除后的结果
    def dmp_clear_denoms(self, f, convert=False):
        c, F = dmp_clear_denoms(self.to_dense(f), self.ngens-1, self.domain, convert=convert)
        if convert:
            ring = self.clone(domain=self.domain.get_ring())
        else:
            ring = self
        return (c, ring.from_dense(F))

    # 从稠密表示转换为多项式，将多项式 f 的系数按 n 倍还原
    def dup_revert(self, f, n):
        return self.from_dense(dup_revert(self.to_dense(f), n, self.domain))
    # 使用 dup_half_gcdex 算法计算多项式 f 和 g 的半扩展欧几里德算法
    def dup_half_gcdex(self, f, g):
        s, h = dup_half_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(h))

    # 使用 dmp_half_gcdex 算法计算多项式 f 和 g 的半扩展欧几里德算法
    def dmp_half_gcdex(self, f, g):
        s, h = dmp_half_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(h))

    # 使用 dup_gcdex 算法计算多项式 f 和 g 的扩展欧几里德算法
    def dup_gcdex(self, f, g):
        s, t, h = dup_gcdex(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))

    # 使用 dmp_gcdex 算法计算多项式 f 和 g 的扩展欧几里德算法
    def dmp_gcdex(self, f, g):
        s, t, h = dmp_gcdex(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(s), self.from_dense(t), self.from_dense(h))

    # 使用 dup_invert 算法计算多项式 f 在模 g 下的逆元
    def dup_invert(self, f, g):
        return self.from_dense(dup_invert(self.to_dense(f), self.to_dense(g), self.domain))

    # 使用 dmp_invert 算法计算多项式 f 在模 g 下的逆元
    def dmp_invert(self, f, g):
        return self.from_dense(dmp_invert(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain))

    # 使用 dup_euclidean_prs 算法计算多项式 f 和 g 的欧几里德算法
    def dup_euclidean_prs(self, f, g):
        prs = dup_euclidean_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))

    # 使用 dmp_euclidean_prs 算法计算多项式 f 和 g 的欧几里德算法
    def dmp_euclidean_prs(self, f, g):
        prs = dmp_euclidean_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    # 使用 dup_primitive_prs 算法计算多项式 f 和 g 的原始部分算法
    def dup_primitive_prs(self, f, g):
        prs = dup_primitive_prs(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))

    # 使用 dmp_primitive_prs 算法计算多项式 f 和 g 的原始部分算法
    def dmp_primitive_prs(self, f, g):
        prs = dmp_primitive_prs(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    # 使用 dup_inner_subresultants 算法计算多项式 f 和 g 的内子结果算法
    def dup_inner_subresultants(self, f, g):
        prs, sres = dup_inner_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return (list(map(self.from_dense, prs)), sres)

    # 使用 dmp_inner_subresultants 算法计算多项式 f 和 g 的内子结果算法
    def dmp_inner_subresultants(self, f, g):
        prs, sres  = dmp_inner_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (list(map(self.from_dense, prs)), sres)

    # 使用 dup_subresultants 算法计算多项式 f 和 g 的子结果算法
    def dup_subresultants(self, f, g):
        prs = dup_subresultants(self.to_dense(f), self.to_dense(g), self.domain)
        return list(map(self.from_dense, prs))

    # 使用 dmp_subresultants 算法计算多项式 f 和 g 的子结果算法
    def dmp_subresultants(self, f, g):
        prs = dmp_subresultants(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return list(map(self.from_dense, prs))

    # 使用 dup_prs_resultant 算法计算多项式 f 和 g 的结果算法
    def dup_prs_resultant(self, f, g):
        res, prs = dup_prs_resultant(self.to_dense(f), self.to_dense(g), self.domain)
        return (res, list(map(self.from_dense, prs)))

    # 使用 dmp_prs_resultant 算法计算多项式 f 和 g 的结果算法
    def dmp_prs_resultant(self, f, g):
        res, prs = dmp_prs_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self[1:].from_dense(res), list(map(self.from_dense, prs)))
    # 计算两个多项式的整式结果 resultant 在整数环中
    def dmp_zz_modular_resultant(self, f, g, p):
        # 转换 f 和 g 为密集表示，使用给定的 p 和当前环境中的域计算模 p 的多项式结果 resultant
        res = dmp_zz_modular_resultant(self.to_dense(f), self.to_dense(g), self.domain_new(p), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return self[1:].from_dense(res)

    # 计算两个多项式的 Collins 结果 resultant 在整数环中
    def dmp_zz_collins_resultant(self, f, g):
        # 转换 f 和 g 为密集表示，使用 Collins 方法计算多项式结果 resultant
        res = dmp_zz_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return self[1:].from_dense(res)

    # 计算两个有理数系数的 Collins 结果 resultant
    def dmp_qq_collins_resultant(self, f, g):
        # 转换 f 和 g 为密集表示，使用 Collins 方法计算有理数系数的多项式结果 resultant
        res = dmp_qq_collins_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return self[1:].from_dense(res)

    # 计算两个多项式的 resultant，在整数环中
    def dup_resultant(self, f, g): #, includePRS=False):
        # 转换 f 和 g 为密集表示，使用给定的域计算多项式 resultant
        return dup_resultant(self.to_dense(f), self.to_dense(g), self.domain) #, includePRS=includePRS)

    # 计算两个多项式的 resultant
    def dmp_resultant(self, f, g): #, includePRS=False):
        # 转换 f 和 g 为密集表示，使用给定的域计算多项式 resultant
        res = dmp_resultant(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain) #, includePRS=includePRS)
        # 如果结果是列表，则将其转换为当前对象的表示方式
        if isinstance(res, list):
            return self[1:].from_dense(res)
        else:
            return res

    # 计算多项式的 discriminant，在整数环中
    def dup_discriminant(self, f):
        # 转换 f 为密集表示，使用给定的域计算多项式 discriminant
        return dup_discriminant(self.to_dense(f), self.domain)

    # 计算多项式的 discriminant
    def dmp_discriminant(self, f):
        # 转换 f 为密集表示，使用给定的域计算多项式 discriminant
        disc = dmp_discriminant(self.to_dense(f), self.ngens-1, self.domain)
        # 如果结果是列表，则将其转换为当前对象的表示方式
        if isinstance(disc, list):
            return self[1:].from_dense(disc)
        else:
            return disc

    # 使用具有重根分解的最大公因式算法计算两个多项式的最大公因式，系数是实数
    def dup_rr_prs_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用重根分解算法计算多项式的最大公因式
        H, F, G = dup_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用具有重根分解的最大公因式算法计算两个多项式的最大公因式，系数是有理数
    def dup_ff_prs_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用重根分解算法计算有理数系数的多项式的最大公因式
        H, F, G = dup_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用具有重根分解的最大公因式算法计算两个多项式的最大公因式，在整数环中
    def dmp_rr_prs_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用重根分解算法计算多项式的最大公因式
        H, F, G = dmp_rr_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用具有重根分解的最大公因式算法计算两个多项式的最大公因式，系数是有理数，在整数环中
    def dmp_ff_prs_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用重根分解算法计算有理数系数的多项式的最大公因式
        H, F, G = dmp_ff_prs_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用启发式算法计算两个整数系数的最大公因式
    def dup_zz_heu_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用启发式算法计算整数系数的多项式的最大公因式
        H, F, G = dup_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用启发式算法计算两个多项式的最大公因式，在整数环中
    def dmp_zz_heu_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用启发式算法计算多项式的最大公因式
        H, F, G = dmp_zz_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用启发式算法计算两个有理数系数的最大公因式
    def dup_qq_heu_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用启发式算法计算有理数系数的多项式的最大公因式
        H, F, G = dup_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用启发式算法计算两个有理数系数的最大公因式，在整数环中
    def dmp_qq_heu_gcd(self, f, g):
        # 转换 f 和 g 为密集表示，使用启发式算法计算有理数系数的多项式的最大公因式
        H, F, G = dmp_qq_heu_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        # 将结果从密集表示转换为当前对象的表示方式
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))
    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dup_inner_gcd 函数计算多项式 f 和 g 的内部最大公约数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dup_inner_gcd(self, f, g):
        H, F, G = dup_inner_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dmp_inner_gcd 函数计算多项式 f 和 g 的分母多项式的内部最大公约数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dmp_inner_gcd(self, f, g):
        H, F, G = dmp_inner_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return (self.from_dense(H), self.from_dense(F), self.from_dense(G))

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dup_gcd 函数计算多项式 f 和 g 的最大公约数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dup_gcd(self, f, g):
        H = dup_gcd(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dmp_gcd 函数计算多项式 f 和 g 的分母多项式的最大公约数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dmp_gcd(self, f, g):
        H = dmp_gcd(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dup_rr_lcm 函数计算多项式 f 和 g 的最小公倍数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dup_rr_lcm(self, f, g):
        H = dup_rr_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dup_ff_lcm 函数计算多项式 f 和 g 的有理系数的最小公倍数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dup_ff_lcm(self, f, g):
        H = dup_ff_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dup_lcm 函数计算多项式 f 和 g 的最小公倍数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dup_lcm(self, f, g):
        H = dup_lcm(self.to_dense(f), self.to_dense(g), self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dmp_rr_lcm 函数计算多项式 f 和 g 的分母多项式的最小公倍数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dmp_rr_lcm(self, f, g):
        H = dmp_rr_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dmp_ff_lcm 函数计算多项式 f 和 g 的有理系数的最小公倍数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dmp_ff_lcm(self, f, g):
        H = dmp_ff_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dmp_lcm 函数计算多项式 f 和 g 的分母多项式的最小公倍数
    # 将计算结果转换为当前对象的稠密表示后返回
    def dmp_lcm(self, f, g):
        H = dmp_lcm(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain)
        return self.from_dense(H)

    # 使用当前对象的 to_dense 方法将输入多项式 f 转换为稠密表示
    # 调用 dup_content 函数计算多项式 f 的内容
    # 返回内容的稠密表示
    def dup_content(self, f):
        cont = dup_content(self.to_dense(f), self.domain)
        return cont

    # 使用当前对象的 to_dense 方法将输入多项式 f 转换为稠密表示
    # 调用 dup_primitive 函数计算多项式 f 的本原部分和本原多项式
    # 返回内容的稠密表示和本原多项式的稠密表示
    def dup_primitive(self, f):
        cont, prim = dup_primitive(self.to_dense(f), self.domain)
        return cont, self.from_dense(prim)

    # 使用当前对象的 to_dense 方法将输入多项式 f 转换为稠密表示
    # 调用 dmp_content 函数计算多项式 f 的分母多项式的内容
    # 如果内容为列表，则将其转换为当前对象的稠密表示后返回
    # 否则直接返回内容的稠密表示
    def dmp_content(self, f):
        cont = dmp_content(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return self[1:].from_dense(cont)
        else:
            return cont

    # 使用当前对象的 to_dense 方法将输入多项式 f 转换为稠密表示
    # 调用 dmp_primitive 函数计算多项式 f 的分母多项式的本原部分和本原多项式
    # 如果内容为列表，则将其转换为当前对象的稠密表示后返回
    # 否则返回内容的稠密表示和本原多项式的稠密表示
    def dmp_primitive(self, f):
        cont, prim = dmp_primitive(self.to_dense(f), self.ngens-1, self.domain)
        if isinstance(cont, list):
            return (self[1:].from_dense(cont), self.from_dense(prim))
        else:
            return (cont, self.from_dense(prim))

    # 使用当前对象的 to_dense 方法将输入多项式 f 转换为稠密表示
    # 调用 dmp_ground_content 函数计算多项式 f 的基本多项式的内容
    # 返回内容的稠密表示
    def dmp_ground_content(self, f):
        cont = dmp_ground_content(self.to_dense(f), self.ngens-1, self.domain)
        return cont

    # 使用当前对象的 to_dense 方法将输入多项式 f 转换为稠密表示
    # 调用 dmp_ground_primitive 函数计算多项式 f 的基本多项式的本原部分和本原多项式
    # 返回内容的稠密表示和本原多项式的稠密表示
    def dmp_ground_primitive(self, f):
        cont, prim = dmp_ground_primitive(self.to_dense(f), self.ngens-1, self.domain)
        return (cont, self.from_dense(prim))

    # 使用当前对象的 to_dense 方法将输入多项式 f 和 g 转换为稠密表示
    # 调用 dup_cancel 函数计算多项式 f 和 g 的消去法结果
    # 如果不包含包括消去理想，则返回结果的稠密表示
    # 否则返回结果的
    # 使用 dmp_cancel 函数取消多项式 f 和 g 的公共因子
    def dmp_cancel(self, f, g, include=True):
        # 调用 dmp_cancel 函数，传入转换为稠密表示的多项式 f 和 g，取消它们的公共因子
        result = dmp_cancel(self.to_dense(f), self.to_dense(g), self.ngens-1, self.domain, include=include)
        # 如果 include 为 False，则返回 cf, cg, F, G
        if not include:
            cf, cg, F, G = result
            return (cf, cg, self.from_dense(F), self.from_dense(G))
        else:
            # 否则返回 F, G
            F, G = result
            return (self.from_dense(F), self.from_dense(G))

    # 使用 dup_trial_division 函数对多项式 f 进行试除法分解
    def dup_trial_division(self, f, factors):
        # 将 f 和 factors 转换为稠密表示，然后调用 dup_trial_division 进行试除法分解
        factors = dup_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.domain)
        # 返回分解结果，转换为稀疏表示
        return [ (self.from_dense(g), k) for g, k in factors ]

    # 使用 dmp_trial_division 函数对多项式 f 进行试除法分解
    def dmp_trial_division(self, f, factors):
        # 将 f 和 factors 转换为稠密表示，然后调用 dmp_trial_division 进行试除法分解
        factors = dmp_trial_division(self.to_dense(f), list(map(self.to_dense, factors)), self.ngens-1, self.domain)
        # 返回分解结果，转换为稀疏表示
        return [ (self.from_dense(g), k) for g, k in factors ]

    # 使用 dup_zz_mignotte_bound 函数计算多项式 f 的 Mignotte 边界
    def dup_zz_mignotte_bound(self, f):
        return dup_zz_mignotte_bound(self.to_dense(f), self.domain)

    # 使用 dmp_zz_mignotte_bound 函数计算多项式 f 的 Mignotte 边界
    def dmp_zz_mignotte_bound(self, f):
        return dmp_zz_mignotte_bound(self.to_dense(f), self.ngens-1, self.domain)

    # 使用 dup_zz_hensel_step 函数进行 Hensel 步骤
    def dup_zz_hensel_step(self, m, f, g, h, s, t):
        # 将 f, g, h, s, t 转换为稠密表示，然后调用 dup_zz_hensel_step 进行 Hensel 步骤
        D = self.to_dense
        G, H, S, T = dup_zz_hensel_step(m, D(f), D(g), D(h), D(s), D(t), self.domain)
        # 返回结果，转换为稀疏表示
        return (self.from_dense(G), self.from_dense(H), self.from_dense(S), self.from_dense(T))

    # 使用 dup_zz_hensel_lift 函数进行 Hensel 提升
    def dup_zz_hensel_lift(self, p, f, f_list, l):
        # 将 f 和 f_list 转换为稠密表示，然后调用 dup_zz_hensel_lift 进行 Hensel 提升
        D = self.to_dense
        polys = dup_zz_hensel_lift(p, D(f), list(map(D, f_list)), l, self.domain)
        # 返回结果，将每个多项式转换为稀疏表示
        return list(map(self.from_dense, polys))

    # 使用 dup_zz_zassenhaus 函数对多项式 f 进行 Zassenhaus 分解
    def dup_zz_zassenhaus(self, f):
        factors = dup_zz_zassenhaus(self.to_dense(f), self.domain)
        # 返回分解结果，转换为稀疏表示
        return [ (self.from_dense(g), k) for g, k in factors ]

    # 使用 dup_zz_irreducible_p 函数判断多项式 f 是否是不可约的
    def dup_zz_irreducible_p(self, f):
        return dup_zz_irreducible_p(self.to_dense(f), self.domain)

    # 使用 dup_cyclotomic_p 函数判断多项式 f 是否是分圆多项式
    def dup_cyclotomic_p(self, f, irreducible=False):
        return dup_cyclotomic_p(self.to_dense(f), self.domain, irreducible=irreducible)

    # 使用 dup_zz_cyclotomic_poly 函数生成次数为 n 的分圆多项式
    def dup_zz_cyclotomic_poly(self, n):
        F = dup_zz_cyclotomic_poly(n, self.domain)
        # 返回分圆多项式，转换为稀疏表示
        return self.from_dense(F)

    # 使用 dup_zz_cyclotomic_factor 函数对多项式 f 进行分圆因子分解
    def dup_zz_cyclotomic_factor(self, f):
        result = dup_zz_cyclotomic_factor(self.to_dense(f), self.domain)
        if result is None:
            return result
        else:
            # 返回分解结果，将每个因子转换为稀疏表示
            return list(map(self.from_dense, result))

    # 使用 dmp_zz_wang_non_divisors 函数计算 Wang 方法的非除子
    # E: List[ZZ], cs: ZZ, ct: ZZ
    def dmp_zz_wang_non_divisors(self, E, cs, ct):
        return dmp_zz_wang_non_divisors(E, cs, ct, self.domain)

    # 此函数是注释掉的，未被调用或者未实现

    # 此函数是注释掉的，未被调用或者未实现

    # 此函数是注释掉的，未被调用或者未实现
    # 获取多项式 f 中的除了首项外的所有系数
    mv = self[1:]
    # 将 T 中每个元组的第一个元素转换为稠密表示，并保留原始的 k 值
    T = [ (mv.to_dense(t), k) for t, k in T ]
    # 获取多项式 H 中每个元素的稠密表示
    H = list(map(uv.to_dense, H))
    # 调用 dmp_zz_wang_lead_coeffs 函数，将多项式 f 转换为稠密表示后处理
    f, HH, CC = dmp_zz_wang_lead_coeffs(self.to_dense(f), T, cs, E, H, A, self.ngens-1, self.domain)
    # 将 f 转换为稀疏表示，并返回 HH 和 CC 的稀疏表示
    return self.from_dense(f), list(map(uv.from_dense, HH)), list(map(mv.from_dense, CC))

    # 将 F 中的每个多项式转换为稠密表示后，调用 dup_zz_diophantine 函数
    def dup_zz_diophantine(self, F, m, p):
        result = dup_zz_diophantine(list(map(self.to_dense, F)), m, p, self.domain)
        # 将结果列表中的每个多项式转换为稀疏表示并返回
        return list(map(self.from_dense, result))

    # 将 F、c 中的每个多项式转换为稠密表示后，调用 dmp_zz_diophantine 函数
    def dmp_zz_diophantine(self, F, c, A, d, p):
        result = dmp_zz_diophantine(list(map(self.to_dense, F)), self.to_dense(c), A, d, p, self.ngens-1, self.domain)
        # 将结果列表中的每个多项式转换为稀疏表示并返回
        return list(map(self.from_dense, result))

    # 将 f、H、LC 中的每个多项式转换为稠密表示后，调用 dmp_zz_wang_hensel_lifting 函数
    def dmp_zz_wang_hensel_lifting(self, f, H, LC, A, p):
        uv = self[:1]
        mv = self[1:]
        # 将 H、LC 中的每个多项式转换为稠密表示
        H = list(map(uv.to_dense, H))
        LC = list(map(mv.to_dense, LC))
        # 将 f 转换为稠密表示后，调用 dmp_zz_wang_hensel_lifting 函数
        result = dmp_zz_wang_hensel_lifting(self.to_dense(f), H, LC, A, p, self.ngens-1, self.domain)
        # 将结果列表中的每个多项式转换为稀疏表示并返回
        return list(map(self.from_dense, result))

    # 将 f 转换为稠密表示后，调用 dmp_zz_wang 函数
    def dmp_zz_wang(self, f, mod=None, seed=None):
        # 将 f 转换为稠密表示后，调用 dmp_zz_wang 函数
        factors = dmp_zz_wang(self.to_dense(f), self.ngens-1, self.domain, mod=mod, seed=seed)
        # 将结果列表中的每个多项式转换为稀疏表示并返回
        return [ self.from_dense(g) for g in factors ]

    # 将 f 转换为稠密表示后，调用 dup_zz_factor_sqf 函数
    def dup_zz_factor_sqf(self, f):
        coeff, factors = dup_zz_factor_sqf(self.to_dense(f), self.domain)
        # 将结果列表中的每个多项式转换为稀疏表示并返回
        return (coeff, [ self.from_dense(g) for g in factors ])

    # 将 f 转换为稠密表示后，调用 dup_zz_factor 函数
    def dup_zz_factor(self, f):
        coeff, factors = dup_zz_factor(self.to_dense(f), self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    # 将 f 转换为稠密表示后，调用 dmp_zz_factor 函数
    def dmp_zz_factor(self, f):
        coeff, factors = dmp_zz_factor(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    # 将 f 转换为稠密表示后，调用 dup_qq_i_factor 函数
    def dup_qq_i_factor(self, f):
        coeff, factors = dup_qq_i_factor(self.to_dense(f), self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    # 将 f 转换为稠密表示后，调用 dmp_qq_i_factor 函数
    def dmp_qq_i_factor(self, f):
        coeff, factors = dmp_qq_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    # 将 f 转换为稠密表示后，调用 dup_zz_i_factor 函数
    def dup_zz_i_factor(self, f):
        coeff, factors = dup_zz_i_factor(self.to_dense(f), self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    # 将 f 转换为稠密表示后，调用 dmp_zz_i_factor 函数
    def dmp_zz_i_factor(self, f):
        coeff, factors = dmp_zz_i_factor(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    # 将 f 转换为稠密表示后，调用 dup_ext_factor 函数
    def dup_ext_factor(self, f):
        coeff, factors = dup_ext_factor(self.to_dense(f), self.domain)
        # 将结果列表中的每个多项式及其对应的整数转换为稀疏表示并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    def dmp_ext_factor(self, f):
        # 调用 dmp_ext_factor 函数，对多项式 f 进行因式分解
        coeff, factors = dmp_ext_factor(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果转换为 (系数, [(因子, 指数)]) 的形式，并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_gf_factor(self, f):
        # 调用 dup_gf_factor 函数，对整数系数的多项式 f 进行 Galois 域因式分解
        coeff, factors = dup_gf_factor(self.to_dense(f), self.domain)
        # 将结果转换为 (系数, [(因子, 指数)]) 的形式，并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dmp_gf_factor(self, f):
        # 调用 dmp_gf_factor 函数，对多项式 f 进行 Galois 域因式分解
        coeff, factors = dmp_gf_factor(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果转换为 (系数, [(因子, 指数)]) 的形式，并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_factor_list(self, f):
        # 调用 dup_factor_list 函数，返回整数系数的多项式 f 的因子列表
        coeff, factors = dup_factor_list(self.to_dense(f), self.domain)
        # 将结果转换为 (系数, [(因子, 指数)]) 的形式，并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dup_factor_list_include(self, f):
        # 调用 dup_factor_list_include 函数，返回整数系数的多项式 f 的包含幂因子的列表
        factors = dup_factor_list_include(self.to_dense(f), self.domain)
        # 将结果转换为 [(因子, 指数)] 的形式，并返回
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dmp_factor_list(self, f):
        # 调用 dmp_factor_list 函数，返回多项式 f 的因子列表
        coeff, factors = dmp_factor_list(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果转换为 (系数, [(因子, 指数)]) 的形式，并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])

    def dmp_factor_list_include(self, f):
        # 调用 dmp_factor_list_include 函数，返回多项式 f 的包含幂因子的列表
        factors = dmp_factor_list_include(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果转换为 [(因子, 指数)] 的形式，并返回
        return [ (self.from_dense(g), k) for g, k in factors ]

    def dup_irreducible_p(self, f):
        # 调用 dup_irreducible_p 函数，判断整数系数的多项式 f 是否为不可约多项式
        return dup_irreducible_p(self.to_dense(f), self.domain)

    def dmp_irreducible_p(self, f):
        # 调用 dmp_irreducible_p 函数，判断多项式 f 是否为不可约多项式
        return dmp_irreducible_p(self.to_dense(f), self.ngens-1, self.domain)

    def dup_sturm(self, f):
        # 调用 dup_sturm 函数，返回整数系数的多项式 f 的斯图姆序列
        seq = dup_sturm(self.to_dense(f), self.domain)
        # 将结果转换为从稠密表示到稀疏表示，并返回列表
        return list(map(self.from_dense, seq))

    def dup_sqf_p(self, f):
        # 调用 dup_sqf_p 函数，判断整数系数的多项式 f 是否为平方自由多项式
        return dup_sqf_p(self.to_dense(f), self.domain)

    def dmp_sqf_p(self, f):
        # 调用 dmp_sqf_p 函数，判断多项式 f 是否为平方自由多项式
        return dmp_sqf_p(self.to_dense(f), self.ngens-1, self.domain)

    def dmp_norm(self, f):
        # 调用 dmp_norm 函数，返回多项式 f 的范数
        n = dmp_norm(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果转换为从稠密表示到地面域表示，并返回
        return self.to_ground().from_dense(n)

    def dup_sqf_norm(self, f):
        # 调用 dup_sqf_norm 函数，返回整数系数的多项式 f 的平方自由标准形
        s, F, R = dup_sqf_norm(self.to_dense(f), self.domain)
        # 将结果转换为 (符号, 系数, 地面域的系数) 的形式，并返回
        return (s, self.from_dense(F), self.to_ground().from_dense(R))

    def dmp_sqf_norm(self, f):
        # 调用 dmp_sqf_norm 函数，返回多项式 f 的平方自由标准形
        s, F, R = dmp_sqf_norm(self.to_dense(f), self.ngens-1, self.domain)
        # 将结果转换为 (符号, 系数, 地面域的系数) 的形式，并返回
        return (s, self.from_dense(F), self.to_ground().from_dense(R))

    def dup_gf_sqf_part(self, f):
        # 调用 dup_gf_sqf_part 函数，返回整数系数的多项式 f 的 Galois 域平方自由部分
        return self.from_dense(dup_gf_sqf_part(self.to_dense(f), self.domain))

    def dmp_gf_sqf_part(self, f):
        # 调用 dmp_gf_sqf_part 函数，返回多项式 f 的 Galois 域平方自由部分
        return self.from_dense(dmp_gf_sqf_part(self.to_dense(f), self.domain))

    def dup_sqf_part(self, f):
        # 调用 dup_sqf_part 函数，返回整数系数的多项式 f 的平方自由部分
        return self.from_dense(dup_sqf_part(self.to_dense(f), self.domain))

    def dmp_sqf_part(self, f):
        # 调用 dmp_sqf_part 函数，返回多项式 f 的平方自由部分
        return self.from_dense(dmp_sqf_part(self.to_dense(f), self.ngens-1, self.domain))

    def dup_gf_sqf_list(self, f, all=False):
        # 调用 dup_gf_sqf_list 函数，返回整数系数的多项式 f 的 Galois 域平方自由因式列表
        coeff, factors = dup_gf_sqf_list(self.to_dense(f), self.domain, all=all)
        # 将结果转换为 (系数, [(因子, 指数)]) 的形式，并返回
        return (coeff, [ (self.from_dense(g), k) for g, k in factors ])
    # 返回多项式 f 在有限域上的平方自由因子列表，包括其系数和因子列表
    def dmp_gf_sqf_list(self, f, all=False):
        # 将多项式 f 转换为稠密表示，然后计算其在域上的平方自由因子列表
        coeff, factors = dmp_gf_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        # 返回结果，其中因子部分转换为稀疏表示
        return (coeff, [(self.from_dense(g), k) for g, k in factors])

    # 返回多项式 f 的平方自由因子列表，包括其系数和因子列表
    def dup_sqf_list(self, f, all=False):
        # 将多项式 f 转换为稠密表示，然后计算其在整数环上的平方自由因子列表
        coeff, factors = dup_sqf_list(self.to_dense(f), self.domain, all=all)
        # 返回结果，其中因子部分转换为稀疏表示
        return (coeff, [(self.from_dense(g), k) for g, k in factors])

    # 返回多项式 f 在整数环上的平方自由因子列表，包括其因子列表
    def dup_sqf_list_include(self, f, all=False):
        # 将多项式 f 转换为稠密表示，然后计算其在整数环上的包含平方自由因子列表
        factors = dup_sqf_list_include(self.to_dense(f), self.domain, all=all)
        # 返回结果，其中因子部分转换为稀疏表示
        return [(self.from_dense(g), k) for g, k in factors]

    # 返回多项式 f 在有限域上的平方自由因子列表，包括其系数和因子列表
    def dmp_sqf_list(self, f, all=False):
        # 将多项式 f 转换为稠密表示，然后计算其在多项式环上的平方自由因子列表
        coeff, factors = dmp_sqf_list(self.to_dense(f), self.ngens-1, self.domain, all=all)
        # 返回结果，其中因子部分转换为稀疏表示
        return (coeff, [(self.from_dense(g), k) for g, k in factors])

    # 返回多项式 f 在多项式环上的平方自由因子列表，包括其因子列表
    def dmp_sqf_list_include(self, f, all=False):
        # 将多项式 f 转换为稠密表示，然后计算其在多项式环上的包含平方自由因子列表
        factors = dmp_sqf_list_include(self.to_dense(f), self.ngens-1, self.domain, all=all)
        # 返回结果，其中因子部分转换为稀疏表示
        return [(self.from_dense(g), k) for g, k in factors]

    # 返回多项式 f 在整数环上的因子分解列表，包括其因子列表
    def dup_gff_list(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其在整数环上的因子分解列表
        factors = dup_gff_list(self.to_dense(f), self.domain)
        # 返回结果，其中因子部分转换为稀疏表示
        return [(self.from_dense(g), k) for g, k in factors]

    # 返回多项式 f 在多项式环上的因子分解列表，包括其因子列表
    def dmp_gff_list(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其在多项式环上的因子分解列表
        factors = dmp_gff_list(self.to_dense(f), self.ngens-1, self.domain)
        # 返回结果，其中因子部分转换为稀疏表示
        return [(self.from_dense(g), k) for g, k in factors]

    # 返回多项式 f 在整数环上的实根的上界
    def dup_root_upper_bound(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其在整数环上实根的上界
        return dup_root_upper_bound(self.to_dense(f), self.domain)

    # 返回多项式 f 在整数环上的实根的下界
    def dup_root_lower_bound(self, f):
        # 将多项式 f 转换为稠密表示，然后计算其在整数环上实根的下界
        return dup_root_lower_bound(self.to_dense(f), self.domain)

    # 使用牛顿法对多项式 f 在整数环上进行实根的步骤精化
    def dup_step_refine_real_root(self, f, M, fast=False):
        # 将多项式 f 转换为稠密表示，然后使用牛顿法对其实根进行步骤精化
        return dup_step_refine_real_root(self.to_dense(f), M, self.domain, fast=fast)

    # 使用内部实根分离算法对多项式 f 在整数环上进行实根的内部分离
    def dup_inner_refine_real_root(self, f, M, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
        # 将多项式 f 转换为稠密表示，然后使用内部实根分离算法对其进行实根的内部分离
        return dup_inner_refine_real_root(self.to_dense(f), M, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast, mobius=mobius)

    # 使用外部实根分离算法对多项式 f 在整数环上进行实根的外部分离
    def dup_outer_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        # 将多项式 f 转换为稠密表示，然后使用外部实根分离算法对其进行实根的外部分离
        return dup_outer_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)

    # 对多项式 f 在整数环上进行实根的分离精化
    def dup_refine_real_root(self, f, s, t, eps=None, steps=None, disjoint=None, fast=False):
        # 将多项式 f 转换为稠密表示，然后对其进行实根的分离精化
        return dup_refine_real_root(self.to_dense(f), s, t, self.domain, eps=eps, steps=steps, disjoint=disjoint, fast=fast)

    # 使用内部算法分离多项式 f 在整数环上的正实根
    def dup_inner_isolate_real_roots(self, f, eps=None, fast=False):
        # 将多项式 f 转换为稠密表示，然后使用内部算法分离其正实根
        return dup_inner_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, fast=fast)

    # 使用内部算法分离多项式 f 在整数环上的正实数正根
    def dup_inner_isolate_positive_roots(self, f, eps=None, inf=None, sup=None, fast=False, mobius=False):
        # 将多项式 f 转换为稠密表示，然后使用内部算法分离其正实数正根
        return dup_inner_isolate_positive_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, mobius=mobius)
    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_inner_isolate_negative_roots` 方法进行负实根的隔离
    def dup_inner_isolate_negative_roots(self, f, inf=None, sup=None, eps=None, fast=False, mobius=False):
        return dup_inner_isolate_negative_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, eps=eps, fast=fast, mobius=mobius)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_isolate_real_roots_sqf` 方法进行实根的隔离
    def dup_isolate_real_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_real_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_isolate_real_roots` 方法进行实根的隔离
    def dup_isolate_real_roots(self, f, eps=None, inf=None, sup=None, basis=False, fast=False):
        return dup_isolate_real_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)

    # 使用当前对象的密集表示转换给定多项式列表 `polys`，并调用 `dup_isolate_real_roots_list` 方法进行实根的隔离
    def dup_isolate_real_roots_list(self, polys, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
        return dup_isolate_real_roots_list(list(map(self.to_dense, polys)), self.domain, eps=eps, inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_count_real_roots` 方法统计实根的个数
    def dup_count_real_roots(self, f, inf=None, sup=None):
        return dup_count_real_roots(self.to_dense(f), self.domain, inf=inf, sup=sup)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_count_complex_roots` 方法统计复根的个数
    def dup_count_complex_roots(self, f, inf=None, sup=None, exclude=None):
        return dup_count_complex_roots(self.to_dense(f), self.domain, inf=inf, sup=sup, exclude=exclude)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_isolate_complex_roots_sqf` 方法进行复根的隔离
    def dup_isolate_complex_roots_sqf(self, f, eps=None, inf=None, sup=None, blackbox=False):
        return dup_isolate_complex_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, blackbox=blackbox)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_isolate_all_roots_sqf` 方法进行所有根的隔离
    def dup_isolate_all_roots_sqf(self, f, eps=None, inf=None, sup=None, fast=False, blackbox=False):
        return dup_isolate_all_roots_sqf(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox)

    # 使用当前对象的密集表示转换给定多项式 `f`，并调用 `dup_isolate_all_roots` 方法进行所有根的隔离
    def dup_isolate_all_roots(self, f, eps=None, inf=None, sup=None, fast=False):
        return dup_isolate_all_roots(self.to_dense(f), self.domain, eps=eps, inf=inf, sup=sup, fast=fast)

    # 返回特定的Fateman多项式F_1，通过从 `sympy.polys.specialpolys` 导入并在当前对象的域中创建
    def fateman_poly_F_1(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_1
        return tuple(map(self.from_dense, dmp_fateman_poly_F_1(self.ngens-1, self.domain)))

    # 返回特定的Fateman多项式F_2，通过从 `sympy.polys.specialpolys` 导入并在当前对象的域中创建
    def fateman_poly_F_2(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_2
        return tuple(map(self.from_dense, dmp_fateman_poly_F_2(self.ngens-1, self.domain)))

    # 返回特定的Fateman多项式F_3，通过从 `sympy.polys.specialpolys` 导入并在当前对象的域中创建
    def fateman_poly_F_3(self):
        from sympy.polys.specialpolys import dmp_fateman_poly_F_3
        return tuple(map(self.from_dense, dmp_fateman_poly_F_3(self.ngens-1, self.domain)))

    # 将给定元素转换为域上的密集表示，并返回结果
    def to_gf_dense(self, element):
        return gf_strip([ self.domain.dom.convert(c, self.domain) for c in self.wrap(element).to_dense() ])

    # 将给定元素转换为当前对象域的密集表示，并返回结果
    def from_gf_dense(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain.dom))

    # 返回给定多项式 `f` 的次数，通过转换为域上的密集表示进行计算
    def gf_degree(self, f):
        return gf_degree(self.to_gf_dense(f))

    # 返回给定多项式 `f` 的首项系数，通过转换为域上的密集表示进行计算
    def gf_LC(self, f):
        return gf_LC(self.to_gf_dense(f), self.domain.dom)
    def gf_TC(self, f):
        # 将多项式 f 转换为稠密表示，并进行 TC 变换
        return gf_TC(self.to_gf_dense(f), self.domain.dom)

    def gf_strip(self, f):
        # 将多项式 f 转换为稠密表示，然后从稠密表示中去除高次项的零系数
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f)))

    def gf_trunc(self, f):
        # 将多项式 f 转换为稠密表示，然后从稠密表示中截断到指定模数
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod))

    def gf_normal(self, f):
        # 将多项式 f 转换为稠密表示，然后进行标准化处理，包括截断和去除高次项的零系数
        return self.from_gf_dense(gf_strip(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_from_dict(self, f):
        # 将字典表示的多项式 f 转换为稠密表示，同时进行模数和定义域的处理
        return self.from_gf_dense(gf_from_dict(f, self.domain.mod, self.domain.dom))

    def gf_to_dict(self, f, symmetric=True):
        # 将多项式 f 转换为字典表示，同时进行模数的处理，支持对称性
        return gf_to_dict(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_from_int_poly(self, f):
        # 将整数多项式 f 转换为稠密表示，同时进行模数的处理
        return self.from_gf_dense(gf_from_int_poly(f, self.domain.mod))

    def gf_to_int_poly(self, f, symmetric=True):
        # 将多项式 f 转换为整数多项式表示，同时进行模数的处理，支持对称性
        return gf_to_int_poly(self.to_gf_dense(f), self.domain.mod, symmetric=symmetric)

    def gf_neg(self, f):
        # 计算多项式 f 在有限域上的负多项式
        return self.from_gf_dense(gf_neg(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_ground(self, f, a):
        # 将多项式 f 在有限域上与地面值 a 相加
        return self.from_gf_dense(gf_add_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_sub_ground(self, f, a):
        # 将多项式 f 在有限域上与地面值 a 相减
        return self.from_gf_dense(gf_sub_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_mul_ground(self, f, a):
        # 将多项式 f 在有限域上与地面值 a 相乘
        return self.from_gf_dense(gf_mul_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_quo_ground(self, f, a):
        # 将多项式 f 在有限域上与地面值 a 相除
        return self.from_gf_dense(gf_quo_ground(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom))

    def gf_add(self, f, g):
        # 将多项式 f 和 g 在有限域上进行加法运算
        return self.from_gf_dense(gf_add(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_sub(self, f, g):
        # 将多项式 f 和 g 在有限域上进行减法运算
        return self.from_gf_dense(gf_sub(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_mul(self, f, g):
        # 将多项式 f 和 g 在有限域上进行乘法运算
        return self.from_gf_dense(gf_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_sqr(self, f):
        # 计算多项式 f 在有限域上的平方
        return self.from_gf_dense(gf_sqr(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_add_mul(self, f, g, h):
        # 将多项式 f, g 和 h 在有限域上进行加法和乘法混合运算
        return self.from_gf_dense(gf_add_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))

    def gf_sub_mul(self, f, g, h):
        # 将多项式 f, g 和 h 在有限域上进行减法和乘法混合运算
        return self.from_gf_dense(gf_sub_mul(self.to_gf_dense(f), self.to_gf_dense(g), self.to_gf_dense(h), self.domain.mod, self.domain.dom))

    def gf_expand(self, F):
        # 将多项式列表 F 在有限域上进行展开
        return self.from_gf_dense(gf_expand(list(map(self.to_gf_dense, F)), self.domain.mod, self.domain.dom))

    def gf_div(self, f, g):
        # 将多项式 f 和 g 在有限域上进行除法运算，返回商和余数
        q, r = gf_div(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(q), self.from_gf_dense(r)

    def gf_rem(self, f, g):
        # 将多项式 f 和 g 在有限域上进行求余数运算
        return self.from_gf_dense(gf_rem(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))
    def gf_quo(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的商，返回结果
        return self.from_gf_dense(gf_quo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_exquo(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的扩展欧几里得商，返回结果
        return self.from_gf_dense(gf_exquo(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_lshift(self, f, n):
        # 将输入多项式 f 转换为稠密表示，然后进行左移操作 n 次，返回结果
        return self.from_gf_dense(gf_lshift(self.to_gf_dense(f), n, self.domain.dom))

    def gf_rshift(self, f, n):
        # 将输入多项式 f 转换为稠密表示，然后进行右移操作 n 次，返回结果
        return self.from_gf_dense(gf_rshift(self.to_gf_dense(f), n, self.domain.dom))

    def gf_pow(self, f, n):
        # 将输入多项式 f 转换为稠密表示，然后计算其幂次方 n，返回结果
        return self.from_gf_dense(gf_pow(self.to_gf_dense(f), n, self.domain.mod, self.domain.dom))

    def gf_pow_mod(self, f, n, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算 f 的 n 次方模 g，返回结果
        return self.from_gf_dense(gf_pow_mod(self.to_gf_dense(f), n, self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_cofactors(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的最大公因式以及对应的系数，返回结果
        h, cff, cfg = gf_cofactors(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom)
        return self.from_gf_dense(h), self.from_gf_dense(cff), self.from_gf_dense(cfg)

    def gf_gcd(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的最大公因式，返回结果
        return self.from_gf_dense(gf_gcd(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_lcm(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的最小公倍式，返回结果
        return self.from_gf_dense(gf_lcm(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_gcdex(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的扩展欧几里得算法结果，返回结果
        return self.from_gf_dense(gf_gcdex(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_monic(self, f):
        # 将输入多项式 f 转换为稠密表示，然后计算其首一化，返回结果
        return self.from_gf_dense(gf_monic(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_diff(self, f):
        # 将输入多项式 f 转换为稠密表示，然后计算其微分，返回结果
        return self.from_gf_dense(gf_diff(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_eval(self, f, a):
        # 将输入多项式 f 转换为稠密表示，然后计算在点 a 处的值，返回结果
        return gf_eval(self.to_gf_dense(f), a, self.domain.mod, self.domain.dom)

    def gf_multi_eval(self, f, A):
        # 将输入多项式 f 转换为稠密表示，然后计算在点集合 A 上的值，返回结果
        return gf_multi_eval(self.to_gf_dense(f), A, self.domain.mod, self.domain.dom)

    def gf_compose(self, f, g):
        # 将输入多项式 f 和 g 转换为稠密表示，然后计算它们的复合，返回结果
        return self.from_gf_dense(gf_compose(self.to_gf_dense(f), self.to_gf_dense(g), self.domain.mod, self.domain.dom))

    def gf_compose_mod(self, g, h, f):
        # 将输入多项式 g, h 和 f 转换为稠密表示，然后计算 g 和 h 模 f 的复合，返回结果
        return self.from_gf_dense(gf_compose_mod(self.to_gf_dense(g), self.to_gf_dense(h), self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    def gf_trace_map(self, a, b, c, n, f):
        # 将输入多项式 a, b, c 和 f 转换为稠密表示，然后计算迹映射的结果 U 和 V，返回结果
        a = self.to_gf_dense(a)
        b = self.to_gf_dense(b)
        c = self.to_gf_dense(c)
        f = self.to_gf_dense(f)
        U, V = gf_trace_map(a, b, c, n, f, self.domain.mod, self.domain.dom)
        return self.from_gf_dense(U), self.from_gf_dense(V)

    def gf_random(self, n):
        # 生成一个随机的次数为 n 的多项式，转换为稠密表示后返回结果
        return self.from_gf_dense(gf_random(n, self.domain.mod, self.domain.dom))

    def gf_irreducible(self, n):
        # 生成一个次数为 n 的不可约多项式，转换为稠密表示后返回结果
        return self.from_gf_dense(gf_irreducible(n, self.domain.mod, self.domain.dom))

    def gf_irred_p_ben_or(self, f):
        # 将输入多项式 f 转换为稠密表示，然后判断其是否是 Ben-Or 不可约多项式，返回布尔值
        return gf_irred_p_ben_or(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
    # 使用 Rabin 算法判断多项式 f 是否为有限域中的不可约多项式
    def gf_irred_p_rabin(self, f):
        return gf_irred_p_rabin(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    # 判断多项式 f 是否为有限域中的不可约多项式
    def gf_irreducible_p(self, f):
        return gf_irreducible_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    # 判断多项式 f 是否为平方自由的
    def gf_sqf_p(self, f):
        return gf_sqf_p(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    # 获取多项式 f 的平方自由部分
    def gf_sqf_part(self, f):
        return self.from_gf_dense(gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom))

    # 获取多项式 f 的平方自由因式分解列表
    def gf_sqf_list(self, f, all=False):
        coeff, factors = gf_sqf_part(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]

    # 构造多项式 f 的 Q 矩阵
    def gf_Qmatrix(self, f):
        return gf_Qmatrix(self.to_gf_dense(f), self.domain.mod, self.domain.dom)

    # 使用 Berlekamp 算法对多项式 f 进行因式分解
    def gf_berlekamp(self, f):
        factors = gf_berlekamp(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    # 使用 Zassenhaus 算法对多项式 f 进行分解为不可约因子
    def gf_ddf_zassenhaus(self, f):
        factors = gf_ddf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]

    # 使用 Zassenhaus 算法对多项式 f 进行提取不可约因子的扩展分解
    def gf_edf_zassenhaus(self, f, n):
        factors = gf_edf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    # 使用 Shoup 算法对多项式 f 进行分解为不可约因子
    def gf_ddf_shoup(self, f):
        factors = gf_ddf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ (self.from_gf_dense(g), k) for g, k in factors ]

    # 使用 Shoup 算法对多项式 f 进行提取不可约因子的扩展分解
    def gf_edf_shoup(self, f, n):
        factors = gf_edf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    # 使用 Zassenhaus 算法对多项式 f 进行因式分解
    def gf_zassenhaus(self, f):
        factors = gf_zassenhaus(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    # 使用 Shoup 算法对多项式 f 进行因式分解
    def gf_shoup(self, f):
        factors = gf_shoup(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return [ self.from_gf_dense(g) for g in factors ]

    # 对多项式 f 进行平方自由因式分解，可以选择使用指定的分解方法
    def gf_factor_sqf(self, f, method=None):
        coeff, factors = gf_factor_sqf(self.to_gf_dense(f), self.domain.mod, self.domain.dom, method=method)
        return coeff, [ self.from_gf_dense(g) for g in factors ]

    # 对多项式 f 进行因式分解
    def gf_factor(self, f):
        coeff, factors = gf_factor(self.to_gf_dense(f), self.domain.mod, self.domain.dom)
        return coeff, [ (self.from_gf_dense(g), k) for g, k in factors ]
```