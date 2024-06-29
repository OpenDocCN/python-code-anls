# `.\numpy\numpy\f2py\tests\test_pyf_src.py`

```
# 导入必要的模块和函数，这段代码从numpy.distutils移植而来
from numpy.f2py._src_pyf import process_str  # 导入process_str函数，用于处理字符串
from numpy.testing import assert_equal  # 导入assert_equal函数，用于比较两个值是否相等


# 定义输入的Fortran语言接口代码字符串
pyf_src = """
python module foo
    <_rd=real,double precision>
    interface
        subroutine <s,d>foosub(tol)
            <_rd>, intent(in,out) :: tol
        end subroutine <s,d>foosub
    end interface
end python module foo
"""

# 预期的Fortran语言接口代码字符串
expected_pyf = """
python module foo
    interface
        subroutine sfoosub(tol)
            real, intent(in,out) :: tol
        end subroutine sfoosub
        subroutine dfoosub(tol)
            double precision, intent(in,out) :: tol
        end subroutine dfoosub
    end interface
end python module foo
"""


def normalize_whitespace(s):
    """
    去除字符串首尾的空白字符，并将内部连续的空白字符转换为单个空格。
    """
    return ' '.join(s.split())


def test_from_template():
    """
    gh-10712的回归测试函数。
    """
    # 使用process_str函数处理pyf_src字符串
    pyf = process_str(pyf_src)
    # 对处理后的字符串进行空白字符标准化处理
    normalized_pyf = normalize_whitespace(pyf)
    # 对预期字符串进行空白字符标准化处理
    normalized_expected_pyf = normalize_whitespace(expected_pyf)
    # 断言处理后的字符串与预期字符串相等
    assert_equal(normalized_pyf, normalized_expected_pyf)
```