# `D:\src\scipysrc\scipy\scipy\special\tests\test_hyp2f1.py`

```
"""
Tests for hyp2f1 for complex values.

Author: Albert Steppi, with credit to Adam Kullberg (FormerPhycisist) for
the implementation of mp_hyp2f1 below, which modifies mpmath's hyp2f1 to
return the same branch as scipy's on the standard branch cut.
"""

# 导入系统库
import sys
# 导入 pytest 库
import pytest
# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 NamedTuple 类型
from typing import NamedTuple
# 导入 numpy.testing 中的 assert_allclose 函数
from numpy.testing import assert_allclose

# 导入 scipy.special 中的 hyp2f1 函数
from scipy.special import hyp2f1
# 导入 scipy.special._testutils 中的 check_version, MissingModule 函数
from scipy.special._testutils import check_version, MissingModule

# 尝试导入 mpmath 库，如果失败则创建一个 MissingModule 对象
try:
    import mpmath
except ImportError:
    mpmath = MissingModule("mpmath")


def mp_hyp2f1(a, b, c, z):
    """Return mpmath hyp2f1 calculated on same branch as scipy hyp2f1.

    For most values of a,b,c mpmath returns the x - 0j branch of hyp2f1 on the
    branch cut x=(1,inf) whereas scipy's hyp2f1 calculates the x + 0j branch.
    Thus, to generate the right comparison values on the branch cut, we
    evaluate mpmath.hyp2f1 at x + 1e-15*j.

    The exception to this occurs when c-a=-m in which case both mpmath and
    scipy calculate the x + 0j branch on the branch cut. When this happens
    mpmath.hyp2f1 will be evaluated at the original z point.
    """
    # 判断 z 是否在分支切割处
    on_branch_cut = z.real > 1.0 and abs(z.imag) < 1.0e-15
    cond1 = abs(c - a - round(c - a)) < 1.0e-15 and round(c - a) <= 0
    cond2 = abs(c - b - round(c - b)) < 1.0e-15 and round(c - b) <= 0
    # 确保虚部是 *确切* 的零
    if on_branch_cut:
        z = z.real + 0.0j
    # 如果在分支切割处并且不满足条件1或条件2，则将 z_mpmath 设置为 z.real + 1.0e-15j
    if on_branch_cut and not (cond1 or cond2):
        z_mpmath = z.real + 1.0e-15j
    else:
        z_mpmath = z
    # 返回 mpmath 中 hyp2f1 的计算结果，返回值为复数
    return complex(mpmath.hyp2f1(a, b, c, z_mpmath))


class Hyp2f1TestCase(NamedTuple):
    """NamedTuple for storing hyp2f1 test case parameters."""
    a: float
    b: float
    c: float
    z: complex
    expected: complex
    rtol: float


class TestHyp2f1:
    """Tests for hyp2f1 for complex values.

    Expected values for test cases were computed using mpmath. See
    `scipy.special._precompute.hyp2f1_data`. The verbose style of specifying
    test cases is used for readability and to make it easier to mark individual
    cases as expected to fail. Expected failures are used to highlight cases
    where improvements are needed. See
    `scipy.special._precompute.hyp2f1_data.make_hyp2f1_test_cases` for a
    function to generate the boilerplate for the test cases.

    Assertions have been added to each test to ensure that the test cases match
    the situations that are intended. A final test `test_test_hyp2f1` checks
    that the expected values in the test cases actually match what is computed
    by mpmath. This test is marked slow even though it isn't particularly slow
    so that it won't run by default on continuous integration builds.
    """

    # 测试函数，检查对于非正整数 c 的情况
    def test_c_non_positive_int(self, hyp2f1_test_case):
        # 从 hyp2f1_test_case 中解包参数
        a, b, c, z, expected, rtol = hyp2f1_test_case
        # 使用 assert_allclose 检查 hyp2f1 的计算结果是否接近期望值
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)
    def test_unital_argument(self, hyp2f1_test_case):
        """Tests for case z = 1, c - a - b > 0.

        Expected answers computed using mpmath.
        """
        a, b, c, z, expected, rtol = hyp2f1_test_case
        # 断言 z 等于 1 并且 c - a - b 大于 0，用于验证测试条件
        assert z == 1 and c - a - b > 0  # Tests the test
        # 调用 hyp2f1 函数，验证计算结果是否接近预期值
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)

    @pytest.mark.parametrize(
        "hyp2f1_test_case",
        [
            pytest.param(
                Hyp2f1TestCase(
                    a=0.5,
                    b=0.2,
                    c=1.3,
                    z=-1 + 0j,
                    expected=0.9428846409614143 + 0j,
                    rtol=1e-15),
            ),
            pytest.param(
                Hyp2f1TestCase(
                    a=12.3,
                    b=8.0,
                    c=5.300000000000001,
                    z=-1 + 0j,
                    expected=-4.845809986595704e-06 + 0j,
                    rtol=1e-15
                ),
            ),
            pytest.param(
                Hyp2f1TestCase(
                    a=221.5,
                    b=90.2,
                    c=132.3,
                    z=-1 + 0j,
                    expected=2.0490488728377282e-42 + 0j,
                    rtol=1e-7,
                ),
            ),
            pytest.param(
                Hyp2f1TestCase(
                    a=-102.1,
                    b=-20.3,
                    c=-80.8,
                    z=-1 + 0j,
                    expected=45143784.46783885 + 0j,
                    rtol=1e-7,
                ),
                marks=pytest.mark.xfail(
                    condition=sys.maxsize < 2**32,
                    reason="Fails on 32 bit.",
                )
            ),
        ],
    )
    def test_special_case_z_near_minus_1(self, hyp2f1_test_case):
        """Tests for case z ~ -1, c ~ 1 + a - b

        Expected answers computed using mpmath.
        """
        a, b, c, z, expected, rtol = hyp2f1_test_case
        # 断言 1 + a - b - c 的绝对值小于 1e-15 并且 z + 1 的绝对值小于 1e-15，用于验证测试条件
        assert abs(1 + a - b - c) < 1e-15 and abs(z + 1) < 1e-15
        # 调用 hyp2f1 函数，验证计算结果是否接近预期值
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)

    def test_a_b_negative_int(self, hyp2f1_test_case):
        a, b, c, z, expected, rtol = hyp2f1_test_case
        # 断言 a 是负整数或者 b 是负整数，用于验证测试条件
        assert a == int(a) and a < 0 or b == int(b) and b < 0  # Tests the test
        # 调用 hyp2f1 函数，验证计算结果是否接近预期值
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)
    @pytest.mark.parametrize(
        "hyp2f1_test_case",
    def test_region3(self, hyp2f1_test_case):
        """Test case for region 3 of hyp2f1 function.

        Parameters:
        - hyp2f1_test_case: Tuple containing (a, b, c, z, expected, rtol)
        
        This test verifies the function behavior when z satisfies 0.9 <= |z| <= 1
        and |1 - z| < 0.9.

        It asserts that the computed result using hyp2f1 function matches the expected
        result within a relative tolerance (rtol).
        """
        a, b, c, z, expected, rtol = hyp2f1_test_case
        assert 0.9 <= abs(z) <= 1 and abs(1 - z) < 0.9  # Tests the test
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)

    )

    def test_region4(self, hyp2f1_test_case):
        """Test case for region 4 of hyp2f1 function.

        Parameters:
        - hyp2f1_test_case: Tuple containing (a, b, c, z, expected, rtol)
        
        This test handles cases where z satisfies 0.9 <= |z| <= 1 and |1 - z| >= 0.9,
        which requires special handling beyond standard transformations.

        It asserts that the computed result using hyp2f1 function matches the expected
        result within a relative tolerance (rtol).
        """
        a, b, c, z, expected, rtol = hyp2f1_test_case
        assert 0.9 <= abs(z) <= 1 and abs(1 - z) >= 0.9  # Tests the test
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)

    )

    def test_region5(self, hyp2f1_test_case):
        """Test case for region 5 of hyp2f1 function.

        Parameters:
        - hyp2f1_test_case: Tuple containing (a, b, c, z, expected, rtol)
        
        This test covers cases where z satisfies 1 < |z| < 1.1, |1 - z| >= 0.9, and
        real(z) >= 0.

        It asserts that the computed result using hyp2f1 function matches the expected
        result within a relative tolerance (rtol).
        """
        a, b, c, z, expected, rtol = hyp2f1_test_case
        assert 1 < abs(z) < 1.1 and abs(1 - z) >= 0.9 and z.real >= 0
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)

    )

    def test_region6(self, hyp2f1_test_case):
        """Test case for region 6 of hyp2f1 function.

        Parameters:
        - hyp2f1_test_case: Tuple containing (a, b, c, z, expected, rtol)
        
        This test handles cases where |z| > 1 but does not satisfy the conditions of
        region 5.

        It asserts that the computed result using hyp2f1 function matches the expected
        result within a relative tolerance (rtol).
        """
        a, b, c, z, expected, rtol = hyp2f1_test_case
        assert (
            abs(z) > 1 and
            not (1 < abs(z) < 1.1 and abs(1 - z) >= 0.9 and z.real >= 0)
        )
        assert_allclose(hyp2f1(a, b, c, z), expected, rtol=rtol)

    @pytest.mark.slow
    @check_version(mpmath, "1.0.0")
    def test_test_hyp2f1(self):
        """Comprehensive test for hyp2f1 function using mpmath.

        This test ensures that the computed values of hyp2f1(a, b, c, z) match
        the expected values computed with mpmath. It iterates over all test cases
        defined in the class, retrieves parameters, and compares results with a
        high relative tolerance.

        It runs as a slow test due to the potentially intensive computation using mpmath.
        """
        test_methods = [
            test_method for test_method in dir(self)
            if test_method.startswith('test') and
            # Filter properties and attributes (futureproofing).
            callable(getattr(self, test_method)) and
            # Filter out this test
            test_method != 'test_test_hyp2f1'
        ]
        for test_method in test_methods:
            params = self._get_test_parameters(getattr(self, test_method))
            for a, b, c, z, expected, _ in params:
                assert_allclose(mp_hyp2f1(a, b, c, z), expected, rtol=2.25e-16)

    def _get_test_parameters(self, test_method):
        """Retrieve test parameters for a given test method.

        Parameters:
        - test_method: Method object representing the test case
        
        Returns:
        - List of test case parameters extracted from the parametrize marker of the test.
        """
        return [
            case.values[0] for mark in test_method.pytestmark
            if mark.name == 'parametrize'
            for case in mark.args[1]
        ]
```