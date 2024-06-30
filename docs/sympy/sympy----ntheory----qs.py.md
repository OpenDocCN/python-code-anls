# `D:\src\scipysrc\sympy\sympy\ntheory\qs.py`

```
# 导入所需的函数和类
from sympy.core.random import _randint
from sympy.external.gmpy import gcd, invert, sqrt as isqrt
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt

# 定义一个类，表示筛选多项式
class SievePolynomial:
    def __init__(self, modified_coeff=(), a=None, b=None):
        """This class denotes the seive polynomial.
        If ``g(x) = (a*x + b)**2 - N``. `g(x)` can be expanded
        to ``a*x**2 + 2*a*b*x + b**2 - N``, so the coefficient
        is stored in the form `[a**2, 2*a*b, b**2 - N]`. This
        ensures faster `eval` method because we dont have to
        perform `a**2, 2*a*b, b**2` every time we call the
        `eval` method. As multiplication is more expensive
        than addition, by using modified_coefficient we get
        a faster seiving process.

        Parameters
        ==========

        modified_coeff : modified_coefficient of sieve polynomial
        a : parameter of the sieve polynomial
        b : parameter of the sieve polynomial
        """
        self.modified_coeff = modified_coeff  # 存储筛选多项式的系数
        self.a = a  # 筛选多项式的参数a
        self.b = b  # 筛选多项式的参数b

    def eval(self, x):
        """
        Compute the value of the sieve polynomial at point x.

        Parameters
        ==========

        x : Integer parameter for sieve polynomial
        """
        ans = 0
        for coeff in self.modified_coeff:
            ans *= x  # 乘以x，逐步计算多项式的值
            ans += coeff  # 加上当前系数的值
        return ans


class FactorBaseElem:
    """This class stores an element of the `factor_base`.
    """
    def __init__(self, prime, tmem_p, log_p):
        """
        Initialization of factor_base_elem.

        Parameters
        ==========

        prime : prime number of the factor_base
        tmem_p : Integer square root of x**2 = n mod prime
        log_p : Compute Natural Logarithm of the prime
        """
        self.prime = prime  # 存储因子基的素数
        self.tmem_p = tmem_p  # 存储模素数的平方根
        self.log_p = log_p  # 存储素数的自然对数
        self.soln1 = None  # 初始化第一个解为None
        self.soln2 = None  # 初始化第二个解为None
        self.a_inv = None  # 初始化a的逆为None
        self.b_ainv = None  # 初始化b * a^(-1)为None


def _generate_factor_base(prime_bound, n):
    """Generate `factor_base` for Quadratic Sieve. The `factor_base`
    consists of all the points whose ``legendre_symbol(n, p) == 1``
    and ``p < num_primes``. Along with the prime `factor_base` also stores
    natural logarithm of prime and the residue n modulo p.
    It also returns the of primes numbers in the `factor_base` which are
    close to 1000 and 5000.

    Parameters
    ==========

    prime_bound : upper prime bound of the factor_base
    n : integer to be factored
    """
    from sympy.ntheory.generate import sieve  # 导入sieve函数
    factor_base = []  # 初始化因子基为空列表
    idx_1000, idx_5000 = None, None  # 初始化索引值为None
    # 使用筛法生成器对象sieve，生成小于prime_bound的素数，逐个遍历
    for prime in sieve.primerange(1, prime_bound):
        # 如果满足勒让德符号条件（即(n / prime) = 1）
        if pow(n, (prime - 1) // 2, prime) == 1:
            # 如果当前素数大于1000且idx_1000尚未赋值，则更新idx_1000
            if prime > 1000 and idx_1000 is None:
                idx_1000 = len(factor_base) - 1
            # 如果当前素数大于5000且idx_5000尚未赋值，则更新idx_5000
            if prime > 5000 and idx_5000 is None:
                idx_5000 = len(factor_base) - 1
            # 使用_sqrt_mod_prime_power函数计算模素数prime的平方根
            residue = _sqrt_mod_prime_power(n, prime, 1)[0]
            # 计算素数prime的对数，并进行舍入处理
            log_p = round(log(prime)*2**10)
            # 将素数prime及其计算得到的值加入到factor_base列表中
            factor_base.append(FactorBaseElem(prime, residue, log_p))
    # 返回索引值idx_1000、idx_5000以及已填充的factor_base列表
    return idx_1000, idx_5000, factor_base
# 初始化第一个筛选多项式的步骤。
# 在这里，选择 `a` 作为因子基数几个质数的乘积，使得 `a` 大约为 ``sqrt(2*N) / M``。
# 还初始化了因子基数元素的其他初始值，包括 a_inv、b_ainv、soln1、soln2，
# 这些在筛选多项式更改时会用到。b_ainv 用于快速多项式更改，因为我们不必每次计算 `2*b*invert(a, prime)`。
# 确保构成 `a` 的因子基数质数介于 1000 和 5000 之间。

def _initialize_first_polynomial(N, M, factor_base, idx_1000, idx_5000, seed=None):
    """This step is the initialization of the 1st sieve polynomial.
    Here `a` is selected as a product of several primes of the factor_base
    such that `a` is about to ``sqrt(2*N) / M``. Other initial values of
    factor_base elem are also initialized which includes a_inv, b_ainv, soln1,
    soln2 which are used when the sieve polynomial is changed. The b_ainv
    is required for fast polynomial change as we do not have to calculate
    `2*b*invert(a, prime)` every time.
    We also ensure that the `factor_base` primes which make `a` are between
    1000 and 5000.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    idx_1000 : index of prime number in the factor_base near 1000
    idx_5000 : index of prime number in the factor_base near to 5000
    seed : Generate pseudoprime numbers
    """
    randint = _randint(seed)
    approx_val = sqrt(2*N) / M
    # `a` is a parameter of the sieve polynomial and `q` is the prime factors of `a`
    # randomly search for a combination of primes whose multiplication is close to approx_val
    # This multiplication of primes will be `a` and the primes will be `q`
    # `best_a` denotes that `a` is close to approx_val in the random search of combination
    best_a, best_q, best_ratio = None, None, None
    start = 0 if idx_1000 is None else idx_1000
    end = len(factor_base) - 1 if idx_5000 is None else idx_5000
    for _ in range(50):
        a = 1
        q = []
        while(a < approx_val):
            rand_p = 0
            while(rand_p == 0 or rand_p in q):
                rand_p = randint(start, end)
            p = factor_base[rand_p].prime
            a *= p
            q.append(rand_p)
        ratio = a / approx_val
        if best_ratio is None or abs(ratio - 1) < abs(best_ratio - 1):
            best_q = q
            best_a = a
            best_ratio = ratio

    a = best_a
    q = best_q

    B = []
    for val in q:
        q_l = factor_base[val].prime
        gamma = factor_base[val].tmem_p * invert(a // q_l, q_l) % q_l
        if gamma > q_l / 2:
            gamma = q_l - gamma
        B.append(a//q_l*gamma)

    b = sum(B)
    g = SievePolynomial([a*a, 2*a*b, b*b - N], a, b)

    for fb in factor_base:
        if a % fb.prime == 0:
            continue
        fb.a_inv = invert(a, fb.prime)
        fb.b_ainv = [2*b_elem*fb.a_inv % fb.prime for b_elem in B]
        fb.soln1 = (fb.a_inv*(fb.tmem_p - b)) % fb.prime
        fb.soln2 = (fb.a_inv*(-fb.tmem_p - b)) % fb.prime
    return g, B
    """
    Calculate the (i + 1)th Sieve polynomial based on specific conditions related to prime factorization.
    
    Parameters
    ==========
    
    N : number to be factored
    factor_base : factor_base primes
    i : integer denoting ith polynomial
    g : (i - 1)th polynomial
    B : array that stores a//q_l*gamma
    """
    # 导入 ceil 函数用于向上取整
    from sympy.functions.elementary.integers import ceiling
    
    # 初始化变量 v 为 1，j 为 i
    v = 1
    j = i
    
    # 当 j 是偶数时，递增 v 直到 j 变为奇数
    while(j % 2 == 0):
        v += 1
        j //= 2
    
    # 根据特定条件确定 neg_pow 的值
    if ceiling(i / (2**v)) % 2 == 1:
        neg_pow = -1
    else:
        neg_pow = 1
    
    # 计算 b 的值
    b = g.b + 2*neg_pow*B[v - 1]
    a = g.a
    
    # 创建新的筛选多项式 g
    g = SievePolynomial([a*a, 2*a*b, b*b - N], a, b)
    
    # 更新 factor_base 中的解集
    for fb in factor_base:
        if a % fb.prime == 0:
            continue
        fb.soln1 = (fb.soln1 - neg_pow*fb.b_ainv[v - 1]) % fb.prime
        fb.soln2 = (fb.soln2 - neg_pow*fb.b_ainv[v - 1]) % fb.prime
    
    # 返回更新后的筛选多项式 g
    return g
    """
# 创建一个空的筛选数组，长度为 2*M+1，用于存储每个位置的累积值
sieve_array = [0]*(2*M + 1)

# 对于 factor_base 中的每个素数因子
for factor in factor_base:
    # 如果 factor 不整除系数 a 的解 soln1，则跳过当前循环
    if factor.soln1 is None:
        continue
    
    # 从 (M + factor.soln1) % factor.prime 开始，以 factor.prime 为步长，遍历筛选数组
    # 并在每个位置上添加 factor.log_p 到筛选数组中
    for idx in range((M + factor.soln1) % factor.prime, 2*M, factor.prime):
        sieve_array[idx] += factor.log_p
    
    # 当 factor.prime 等于 2 时，只使用 soln1 进行筛选
    if factor.prime == 2:
        continue
    
    # 从 (M + factor.soln2) % factor.prime 开始，以 factor.prime 为步长，再次遍历筛选数组
    # 并在每个位置上添加 factor.log_p 到筛选数组中
    for idx in range((M + factor.soln2) % factor.prime, 2*M, factor.prime):
        sieve_array[idx] += factor.log_p

# 返回填充了因子对数的筛选数组
return sieve_array
    # Calculate the integer square root of N
    sqrt_n = isqrt(N)
    # Compute accumulated_val using the sieve interval M, sqrt_n, and error term ERROR_TERM
    accumulated_val = log(M * sqrt_n) * 2**10 - ERROR_TERM
    # Initialize an empty list to store smooth relations found
    smooth_relations = []
    # Initialize a set to store proper factors found during the process
    proper_factor = set()
    # Set an upper bound for the large prime in partial relations
    partial_relation_upper_bound = 128 * factor_base[-1].prime
    
    # Iterate through sieve_array with index and value pairs
    for idx, val in enumerate(sieve_array):
        # Skip values less than accumulated_val
        if val < accumulated_val:
            continue
        # Calculate x based on idx and M
        x = idx - M
        # Evaluate the sieve polynomial at x to get v
        v = sieve_poly.eval(x)
        # Check if v is smooth with respect to the factor base
        vec, is_smooth = _check_smoothness(v, factor_base)
        
        # If v is neither smooth nor partial, continue to the next iteration
        if is_smooth is None:
            continue
        
        # Calculate u based on sieve polynomial coefficients and x
        u = sieve_poly.a * x + sieve_poly.b
        
        # Update partial relations if v is a partial relation
        if is_smooth is False:  # partial relation found
            large_prime = vec
            # Consider only large primes under the specified upper bound
            if large_prime > partial_relation_upper_bound:
                continue
            # If large_prime is not already in partial_relations, add it
            if large_prime not in partial_relations:
                partial_relations[large_prime] = (u, v)
                continue
            else:
                # If a second partial relation with the same large prime is found, generate a smooth relation
                u_prev, v_prev = partial_relations[large_prime]
                partial_relations.pop(large_prime)
                try:
                    # Compute the modular inverse of large_prime modulo N
                    large_prime_inv = invert(large_prime, N)
                except ZeroDivisionError:
                    # If large_prime divides N, add it to proper factors and continue
                    proper_factor.add(large_prime)
                    continue
                # Combine u, u_prev, and large_prime_inv to form u for smooth relation
                u = u * u_prev * large_prime_inv
                # Combine v, v_prev, and large_prime^2 to form v for smooth relation
                v = v * v_prev // (large_prime * large_prime)
                # Check again if v is smooth after combining
                vec, is_smooth = _check_smoothness(v, factor_base)
        
        # Append the tuple (u, v, vec) to smooth_relations if v is smooth
        smooth_relations.append((u, v, vec))
    
    # Return the list of smooth relations and the set of proper factors found
    return smooth_relations, proper_factor
# LINEAR ALGEBRA STAGE

# 从平滑关系中构建一个二维矩阵。
def _build_matrix(smooth_relations):
    """Build a 2D matrix from smooth relations.

    Parameters
    ==========

    smooth_relations : Stores smooth relations
    """
    matrix = []
    # 遍历平滑关系列表，将每个平滑关系的第三个元素添加到矩阵中
    for s_relation in smooth_relations:
        matrix.append(s_relation[2])
    return matrix


# 在模2下进行快速高斯消元法。
def _gauss_mod_2(A):
    """Fast gaussian reduction for modulo 2 matrix.

    Parameters
    ==========

    A : Matrix

    Examples
    ========

    >>> from sympy.ntheory.qs import _gauss_mod_2
    >>> _gauss_mod_2([[0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 1]])
    ([[[1, 0, 1], 3]],
     [True, True, True, False],
     [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]])

    Reference
    ==========

    .. [1] A fast algorithm for gaussian elimination over GF(2) and
    its implementation on the GAPP. Cetin K.Koc, Sarath N.Arachchige"""
    import copy
    matrix = copy.deepcopy(A)
    row = len(matrix)
    col = len(matrix[0])
    mark = [False]*row
    # 对每一列进行处理
    for c in range(col):
        # 找到第一个为1的元素所在的行
        for r in range(row):
            if matrix[r][c] == 1:
                break
        # 将该行标记为依赖行
        mark[r] = True
        # 对于除了当前列以外的每一列
        for c1 in range(col):
            if c1 == c:
                continue
            # 如果当前行的该列为1，则将该行的值与当前行进行模2相加
            if matrix[r][c1] == 1:
                for r2 in range(row):
                    matrix[r2][c1] = (matrix[r2][c1] + matrix[r2][c]) % 2
    dependent_row = []
    # 标记为False的行即为依赖行
    for idx, val in enumerate(mark):
        if val == False:
            dependent_row.append([matrix[idx], idx])
    return dependent_row, mark, matrix


# 找到N的适当因子。
def _find_factor(dependent_rows, mark, gauss_matrix, index, smooth_relations, N):
    """Finds proper factor of N. Here, transform the dependent rows as a
    combination of independent rows of the gauss_matrix to form the desired
    relation of the form ``X**2 = Y**2 modN``. After obtaining the desired relation
    we obtain a proper factor of N by `gcd(X - Y, N)`.

    Parameters
    ==========

    dependent_rows : denoted dependent rows in the reduced matrix form
    mark : boolean array to denoted dependent and independent rows
    gauss_matrix : Reduced form of the smooth relations matrix
    index : denoted the index of the dependent_rows
    smooth_relations : Smooth relations vectors matrix
    N : Number to be factored
    """
    # 获取依赖行的索引
    idx_in_smooth = dependent_rows[index][1]
    independent_u = [smooth_relations[idx_in_smooth][0]]
    independent_v = [smooth_relations[idx_in_smooth][1]]
    dept_row = dependent_rows[index][0]

    # 根据依赖行将独立行添加到相应的列表中
    for idx, val in enumerate(dept_row):
        if val == 1:
            for row in range(len(gauss_matrix)):
                if gauss_matrix[row][idx] == 1 and mark[row] == True:
                    independent_u.append(smooth_relations[row][0])
                    independent_v.append(smooth_relations[row][1])
                    break

    u = 1
    v = 1
    # 计算独立行的乘积
    for i in independent_u:
        u *= i
    for i in independent_v:
        v *= i
    # 确保 u**2 ≡ v mod N
    v = isqrt(v)
    return gcd(u - v, N)
    # 将 ERROR_TERM 扩大 2^10 倍，增加平滑性检查的容忍度
    ERROR_TERM *= 2**10
    
    # 使用 _generate_factor_base 函数生成因子基的索引和因子基本身
    idx_1000, idx_5000, factor_base = _generate_factor_base(prime_bound, N)
    
    # 初始化平滑关系列表
    smooth_relations = []
    
    # 初始化多项式计数器
    ith_poly = 0
    
    # 部分关系的字典
    partial_relations = {}
    
    # 用于存储找到的合适因子
    proper_factor = set()
    
    # 阈值用于额外的平滑关系，加快因子分解
    threshold = 5 * len(factor_base) // 100
    
    # 开始循环执行自举二次筛法
    while True:
        # 如果是第一个多项式，则初始化第一个多项式和 B 数组
        if ith_poly == 0:
            ith_sieve_poly, B_array = _initialize_first_polynomial(N, M, factor_base, idx_1000, idx_5000)
        else:
            # 否则，初始化第 ith 个多项式
            ith_sieve_poly = _initialize_ith_poly(N, factor_base, ith_poly, ith_sieve_poly, B_array)
        
        # 增加多项式计数器
        ith_poly += 1
        
        # 如果多项式计数器超过了 B 数组长度减 1 的 2 次方，则重新开始新的多项式
        if ith_poly >= 2**(len(B_array) - 1):
            ith_poly = 0
        
        # 生成筛选数组
        sieve_array = _gen_sieve_array(M, factor_base)
        
        # 在试除阶段执行试除分解
        s_rel, p_f = _trial_division_stage(N, M, factor_base, sieve_array, ith_sieve_poly, partial_relations, ERROR_TERM)
        
        # 将找到的平滑关系添加到平滑关系列表中
        smooth_relations += s_rel
        
        # 将找到的合适因子加入合适因子集合中
        proper_factor |= p_f
        
        # 如果平滑关系列表的长度达到因子基和阈值的和，则跳出循环
        if len(smooth_relations) >= len(factor_base) + threshold:
            break
    
    # 建立矩阵，以进行高斯消元
    matrix = _build_matrix(smooth_relations)
    
    # 对矩阵进行模 2 的高斯消元
    dependent_row, mark, gauss_matrix = _gauss_mod_2(matrix)
    
    # 备份 N 的值
    N_copy = N
    # 遍历依赖行的索引范围
    for index in range(len(dependent_row)):
        # 调用函数 _find_factor，查找因子
        factor = _find_factor(dependent_row, mark, gauss_matrix, index, smooth_relations, N)
        # 如果找到的因子大于 1 并且小于 N
        if factor > 1 and factor < N:
            # 将该因子添加到 proper_factor 集合中
            proper_factor.add(factor)
            # 将 N 的副本按该因子除尽
            while(N_copy % factor == 0):
                N_copy //= factor
            # 如果剩余的数是素数，则将其也添加到 proper_factor 中并结束循环
            if isprime(N_copy):
                proper_factor.add(N_copy)
                break
            # 如果剩余的数变为 1，则结束循环
            if(N_copy == 1):
                break
    # 返回包含所有适当因子的 proper_factor 集合
    return proper_factor
```