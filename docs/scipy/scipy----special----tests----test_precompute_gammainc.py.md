# `D:\src\scipysrc\scipy\scipy\special\tests\test_precompute_gammainc.py`

```
# 导入pytest库，用于测试框架
import pytest

# 从scipy.special._testutils模块中导入MissingModule和check_version函数
from scipy.special._testutils import MissingModule, check_version

# 从scipy.special._mptestutils模块中导入Arg, IntArg, mp_assert_allclose和assert_mpmath_equal函数
from scipy.special._mptestutils import (
    Arg, IntArg, mp_assert_allclose, assert_mpmath_equal)

# 从scipy.special._precompute.gammainc_asy模块中导入compute_g, compute_alpha和compute_d函数
from scipy.special._precompute.gammainc_asy import (
    compute_g, compute_alpha, compute_d)

# 从scipy.special._precompute.gammainc_data模块中导入gammainc和gammaincc函数
from scipy.special._precompute.gammainc_data import gammainc, gammaincc

# 尝试导入sympy库，如果失败则使用MissingModule表示缺失
try:
    import sympy
except ImportError:
    sympy = MissingModule('sympy')

# 尝试导入mpmath库，如果失败则使用MissingModule表示缺失
try:
    import mpmath as mp
except ImportError:
    mp = MissingModule('mpmath')


# 使用装饰器@check_version(mp, '0.19')，确保mpmath库版本不低于0.19
@check_version(mp, '0.19')
def test_g():
    # 测试g_k的数据，参考DLMF 5.11.4节
    with mp.workdps(30):  # 设置mpmath的工作精度为30位小数
        # g_k的系数列表
        g = [mp.mpf(1), mp.mpf(1)/12, mp.mpf(1)/288,
             -mp.mpf(139)/51840, -mp.mpf(571)/2488320,
             mp.mpf(163879)/209018880, mp.mpf(5246819)/75246796800]
        # 断言compute_g(7)计算结果与g的各项相等
        mp_assert_allclose(compute_g(7), g)


# 使用装饰器@pytest.mark.slow标记为慢速测试
@check_version(mp, '0.19')
@check_version(sympy, '0.7')
@pytest.mark.xfail_on_32bit("rtol only 2e-11, see gh-6938")
def test_alpha():
    # 测试alpha_k的数据，参考DLMF 8.12.14节
    with mp.workdps(30):  # 设置mpmath的工作精度为30位小数
        # alpha_k的系数列表
        alpha = [mp.mpf(0), mp.mpf(1), mp.mpf(1)/3, mp.mpf(1)/36,
                 -mp.mpf(1)/270, mp.mpf(1)/4320, mp.mpf(1)/17010,
                 -mp.mpf(139)/5443200, mp.mpf(1)/204120]
        # 断言compute_alpha(9)计算结果与alpha的各项相等
        mp_assert_allclose(compute_alpha(9), alpha)


# 使用装饰器@pytest.mark.xslow标记为极慢速测试
@check_version(mp, '0.19')
@check_version(sympy, '0.7')
def test_d():
    # 将d_{k, n}的计算结果与文献[1]附录F中的结果进行比较
    #
    # 参考资料
    # -------
    # [1] DiDonato and Morris, Computation of the Incomplete Gamma
    #     Function Ratios and their Inverse, ACM Transactions on
    #     Mathematical Software, 1986.
    pass  # 这里只是标记函数体为空，实际应该包含测试代码
    # 设置多精度计算的工作精度为50位
    with mp.workdps(50):
        # 定义数据集，每个元组包含三个元素：k, n, 和对应的多精度浮点数值
        dataset = [(0, 0, -mp.mpf('0.333333333333333333333333333333')),
                   (0, 12, mp.mpf('0.102618097842403080425739573227e-7')),
                   (1, 0, -mp.mpf('0.185185185185185185185185185185e-2')),
                   (1, 12, mp.mpf('0.119516285997781473243076536700e-7')),
                   (2, 0, mp.mpf('0.413359788359788359788359788360e-2')),
                   (2, 12, -mp.mpf('0.140925299108675210532930244154e-7')),
                   (3, 0, mp.mpf('0.649434156378600823045267489712e-3')),
                   (3, 12, -mp.mpf('0.191111684859736540606728140873e-7')),
                   (4, 0, -mp.mpf('0.861888290916711698604702719929e-3')),
                   (4, 12, mp.mpf('0.288658297427087836297341274604e-7')),
                   (5, 0, -mp.mpf('0.336798553366358150308767592718e-3')),
                   (5, 12, mp.mpf('0.482409670378941807563762631739e-7')),
                   (6, 0, mp.mpf('0.531307936463992223165748542978e-3')),
                   (6, 12, -mp.mpf('0.882860074633048352505085243179e-7')),
                   (7, 0, mp.mpf('0.344367606892377671254279625109e-3')),
                   (7, 12, -mp.mpf('0.175629733590604619378669693914e-6')),
                   (8, 0, -mp.mpf('0.652623918595309418922034919727e-3')),
                   (8, 12, mp.mpf('0.377358774161109793380344937299e-6')),
                   (9, 0, -mp.mpf('0.596761290192746250124390067179e-3')),
                   (9, 12, mp.mpf('0.870823417786464116761231237189e-6'))]
        
        # 调用 compute_d 函数计算结果 d，传入参数为 10 和 13
        d = compute_d(10, 13)
        
        # 从计算结果 d 中提取出指定的结果 res，对每个 (k, n, std) 元组进行处理
        res = [d[k][n] for k, n, std in dataset]
        
        # 提取 dataset 中的标准值 std，即每个元组的第三个元素
        std = [x[2] for x in dataset]
        
        # 使用多精度断言函数 mp_assert_allclose 检查 res 和 std 的接近程度
        mp_assert_allclose(res, std)
@check_version(mp, '0.19')
def test_gammainc():
    # 定义测试函数 test_gammainc，用于验证 gammainc 函数的正确性
    # 使用 assert_mpmath_equal 函数比较 gammainc 和 mpmath 中的对应函数结果
    # 参数包括：a 和 x 的范围，regularized 设置为 True，不允许返回 NaN，相对误差限制为 1e-17，迭代次数为 50，精度为 50 位小数
    assert_mpmath_equal(gammainc,
                        lambda a, x: mp.gammainc(a, b=x, regularized=True),
                        [Arg(0, 100, inclusive_a=False), Arg(0, 100)],
                        nan_ok=False, rtol=1e-17, n=50, dps=50)


@pytest.mark.xslow
@check_version(mp, '0.19')
def test_gammaincc():
    # 定义测试函数 test_gammaincc，用于验证 gammaincc 函数的正确性
    # 使用 assert_mpmath_equal 函数比较 gammaincc 和 mpmath 中的对应函数结果
    # 参数包括：a 和 x 的范围，dps 设置为 1000，不允许返回 NaN，相对误差限制为 1e-17，迭代次数为 50，精度为 1000 位小数
    assert_mpmath_equal(lambda a, x: gammaincc(a, x, dps=1000),
                        lambda a, x: mp.gammainc(a, a=x, regularized=True),
                        [Arg(20, 100), Arg(20, 100)],
                        nan_ok=False, rtol=1e-17, n=50, dps=1000)

    # 测试快速整数路径
    # 使用 assert_mpmath_equal 函数比较 gammaincc 和 mpmath 中的对应函数结果
    # 参数包括：a 的范围为 1 到 100，x 的范围为 0 到 100，不允许返回 NaN，相对误差限制为 1e-17，迭代次数为 50，精度为 50 位小数
    assert_mpmath_equal(gammaincc,
                        lambda a, x: mp.gammainc(a, a=x, regularized=True),
                        [IntArg(1, 100), Arg(0, 100)],
                        nan_ok=False, rtol=1e-17, n=50, dps=50)
```