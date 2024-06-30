# `D:\src\scipysrc\scipy\scipy\fft\tests\test_fftlog.py`

```
# 导入警告模块
import warnings
# 导入 numpy 库并简写为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 scipy 库中的 fftlog 相关函数
from scipy.fft._fftlog import fht, ifht, fhtoffset
# 导入 scipy 库中的 poch 函数
from scipy.special import poch

# 导入 scipy 库中的 array_api_compatible 函数
from scipy.conftest import array_api_compatible
# 导入 scipy 库中的 xp_assert_close 函数
from scipy._lib._array_api import xp_assert_close

# 将当前测试标记为与数组 API 兼容
pytestmark = array_api_compatible


def test_fht_agrees_with_fftlog(xp):
    # 检查 fht 函数的数值结果与 Fortran FFTLog 输出一致性
    # 这些结果是通过提供的 `fftlogtest` 程序生成的，
    # 在修复如何生成 k 数组之后（将范围除以 n-1，而不是 n）

    # 测试函数，分析汉克尔变换具有相同的形式
    def f(r, mu):
        return r**(mu+1)*np.exp(-r**2/2)

    # 在对数空间中生成均匀间隔的 r 值
    r = np.logspace(-4, 4, 16)

    # 计算 r 的对数间隔
    dln = np.log(r[1]/r[0])
    # 设置 mu 值
    mu = 0.3
    # 设置偏移量
    offset = 0.0
    # 设置偏差
    bias = 0.0

    # 将 f 函数应用于 r，并将结果转换为 xp 对象
    a = xp.asarray(f(r, mu))

    # 测试 1: 计算给定条件下的 fht 结果
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    # 预期的结果数组
    theirs = [-0.1159922613593045E-02, +0.1625822618458832E-02,
              -0.1949518286432330E-02, +0.3789220182554077E-02,
              +0.5093959119952945E-03, +0.2785387803618774E-01,
              +0.9944952700848897E-01, +0.4599202164586588E+00,
              +0.3157462160881342E+00, -0.8201236844404755E-03,
              -0.7834031308271878E-03, +0.3931444945110708E-03,
              -0.2697710625194777E-03, +0.3568398050238820E-03,
              -0.5554454827797206E-03, +0.8286331026468585E-03]
    # 将预期结果数组转换为 xp 对象
    theirs = xp.asarray(theirs, dtype=xp.float64)
    # 比较计算结果和预期结果的接近程度
    xp_assert_close(ours, theirs)

    # 测试 2: 更改为最佳偏移量
    offset = fhtoffset(dln, mu, bias=bias)
    # 重新计算 fht 结果
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    # 预期的结果数组
    theirs = [+0.4353768523152057E-04, -0.9197045663594285E-05,
              +0.3150140927838524E-03, +0.9149121960963704E-03,
              +0.5808089753959363E-02, +0.2548065256377240E-01,
              +0.1339477692089897E+00, +0.4821530509479356E+00,
              +0.2659899781579785E+00, -0.1116475278448113E-01,
              +0.1791441617592385E-02, -0.4181810476548056E-03,
              +0.1314963536765343E-03, -0.5422057743066297E-04,
              +0.3208681804170443E-04, -0.2696849476008234E-04]
    # 将预期结果数组转换为 xp 对象
    theirs = xp.asarray(theirs, dtype=xp.float64)
    # 比较计算结果和预期结果的接近程度
    xp_assert_close(ours, theirs)

    # 测试 3: 正偏差
    bias = 0.8
    # 使用给定偏差重新计算偏移量
    offset = fhtoffset(dln, mu, bias=bias)
    # 重新计算 fht 结果
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    # 预期的结果数组
    theirs = [-7.3436673558316850E+00, +0.1710271207817100E+00,
              +0.1065374386206564E+00, -0.5121739602708132E-01,
              +0.2636649319269470E-01, +0.1697209218849693E-01,
              +0.1250215614723183E+00, +0.4739583261486729E+00,
              +0.2841149874912028E+00, -0.8312764741645729E-02,
              +0.1024233505508988E-02, -0.1644902767389120E-03,
              +0.3305775476926270E-04, -0.7786993194882709E-05,
              +0.1962258449520547E-05, -0.8977895734909250E-06]
    # 将预期结果数组转换为 xp 对象
    theirs = xp.asarray(theirs, dtype=xp.float64)
    # 比较计算结果和预期结果的接近程度
    xp_assert_close(ours, theirs)

    # 测试 4: 负偏差
    bias = -0.8
    # 计算偏移量，使用给定的参数 dln, mu 和 bias
    offset = fhtoffset(dln, mu, bias=bias)
    # 执行快速哈达玛变换（FHT），使用数组 a, dln, mu 和计算得到的偏移量
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    # 给定的参考结果数组，这里使用科学计数法表示
    theirs = [+0.8985777068568745E-05, +0.4074898209936099E-04,
              +0.2123969254700955E-03, +0.1009558244834628E-02,
              +0.5131386375222176E-02, +0.2461678673516286E-01,
              +0.1235812845384476E+00, +0.4719570096404403E+00,
              +0.2893487490631317E+00, -0.1686570611318716E-01,
              +0.2231398155172505E-01, -0.1480742256379873E-01,
              +0.1692387813500801E+00, +0.3097490354365797E+00,
              +2.7593607182401860E+00, 10.5251075070045800E+00]
    # 将参考结果数组转换为与 ours 相同的数据类型，这里使用了 xp（可能是某个库的别名）
    theirs = xp.asarray(theirs, dtype=xp.float64)
    # 断言 ours 与 theirs 的值在允许的误差范围内相近
    xp_assert_close(ours, theirs)
# 使用 pytest 的 parametrize 装饰器设置多个测试参数组合
@pytest.mark.parametrize('optimal', [True, False])
@pytest.mark.parametrize('offset', [0.0, 1.0, -1.0])
@pytest.mark.parametrize('bias', [0, 0.1, -0.1])
@pytest.mark.parametrize('n', [64, 63])
def test_fht_identity(n, bias, offset, optimal, xp):
    # 创建一个随机数生成器对象，用于生成随机数
    rng = np.random.RandomState(3491349965)

    # 使用随机数生成器创建一个长度为 n 的数组 a，转换为 xp 对象
    a = xp.asarray(rng.standard_normal(n))
    
    # 从均匀分布 [-1, 1] 中生成一个随机数 dln
    dln = rng.uniform(-1, 1)
    
    # 从均匀分布 [-2, 2] 中生成一个随机数 mu
    mu = rng.uniform(-2, 2)

    # 如果 optimal 为 True，则调用 fhtoffset 函数计算新的 offset
    if optimal:
        offset = fhtoffset(dln, mu, initial=offset, bias=bias)

    # 调用 fht 函数计算 A，输入 a, dln, mu, offset 和 bias 作为参数
    A = fht(a, dln, mu, offset=offset, bias=bias)
    
    # 调用 ifht 函数计算 a_，输入 A, dln, mu, offset 和 bias 作为参数
    a_ = ifht(A, dln, mu, offset=offset, bias=bias)

    # 使用 xp_assert_close 断言函数验证 a_ 和 a 的近似程度，相对误差容忍度设为 1.5e-7
    xp_assert_close(a_, a, rtol=1.5e-7)


# 定义一个特殊情况的测试函数，测试不同的 mu 和 bias 组合
def test_fht_special_cases(xp):
    # 创建一个随机数生成器对象，用于生成随机数
    rng = np.random.RandomState(3491349965)

    # 使用随机数生成器创建一个长度为 64 的数组 a，转换为 xp 对象
    a = xp.asarray(rng.standard_normal(64))
    
    # 从均匀分布 [-1, 1] 中生成一个随机数 dln
    dln = rng.uniform(-1, 1)

    # case 1: x in M, y in M => well-defined transform
    mu, bias = -4.0, 1.0
    # 在捕获警告的上下文中，调用 fht 函数并断言没有警告被记录
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias)
        assert not record, 'fht warned about a well-defined transform'

    # case 2: x not in M, y in M => well-defined transform
    mu, bias = -2.5, 0.5
    # 在捕获警告的上下文中，调用 fht 函数并断言没有警告被记录
    with warnings.catch_warnings(record=True) as record:
        fht(a, dln, mu, bias=bias)
        assert not record, 'fht warned about a well-defined transform'

    # case 3: x in M, y not in M => singular transform
    mu, bias = -3.5, 0.5
    # 使用 pytest 的 warns 断言捕获 Warning 类型的警告，并检查是否有警告被记录
    with pytest.warns(Warning) as record:
        fht(a, dln, mu, bias=bias)
        assert record, 'fht did not warn about a singular transform'

    # case 4: x not in M, y in M => singular inverse transform
    mu, bias = -2.5, 0.5
    # 使用 pytest 的 warns 断言捕获 Warning 类型的警告，并检查是否有警告被记录
    with pytest.warns(Warning) as record:
        ifht(a, dln, mu, bias=bias)
        assert record, 'ifht did not warn about a singular transform'


# 使用 pytest 的 parametrize 装饰器设置多个测试参数组合
@pytest.mark.parametrize('n', [64, 63])
def test_fht_exact(n, xp):
    # 创建一个随机数生成器对象，用于生成随机数
    rng = np.random.RandomState(3491349965)

    # 从均匀分布 [0, 3] 中生成一个随机数 mu
    mu = rng.uniform(0, 3)

    # 从均匀分布 [-1-mu, 1/2] 中生成一个随机数 gamma
    gamma = rng.uniform(-1-mu, 1/2)

    # 生成一个长度为 n 的对数空间数组 r
    r = np.logspace(-2, 2, n)
    
    # 使用 r 的幂次生成一个数组 a，转换为 xp 对象
    a = xp.asarray(r**gamma)

    # 计算数组 r 的对数步长 dln
    dln = np.log(r[1]/r[0])

    # 调用 fhtoffset 函数计算初始偏移量 offset
    offset = fhtoffset(dln, mu, initial=0.0, bias=gamma)

    # 调用 fht 函数计算 A，输入 a, dln, mu, offset 和 gamma 作为参数
    A = fht(a, dln, mu, offset=offset, bias=gamma)

    # 计算变量 k
    k = np.exp(offset)/r[::-1]

    # 计算理论值数组 At
    At = xp.asarray((2/k)**gamma * poch((mu+1-gamma)/2, gamma))

    # 使用 xp_assert_close 断言函数验证 A 和 At 的近似程度
    xp_assert_close(A, At)
```