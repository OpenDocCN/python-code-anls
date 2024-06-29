# `.\numpy\numpy\distutils\tests\test_from_template.py`

```py
# 从 numpy.distutils.from_template 模块中导入 process_str 函数
from numpy.distutils.from_template import process_str
# 从 numpy.testing 模块中导入 assert_equal 函数
from numpy.testing import assert_equal

# 定义一个字符串变量 pyf_src，包含了一个 Fortran 的声明
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

# 定义一个字符串变量 expected_pyf，包含了与 pyf_src 对应的预期 Fortran 声明
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

# 定义函数 normalize_whitespace，用于规范化字符串的空白字符
def normalize_whitespace(s):
    """
    Remove leading and trailing whitespace, and convert internal
    stretches of whitespace to a single space.
    """
    return ' '.join(s.split())

# 定义测试函数 test_from_template，用于验证 process_str 函数的输出是否符合预期
def test_from_template():
    """Regression test for gh-10712."""
    # 调用 process_str 函数处理 pyf_src，生成处理后的字符串 pyf
    pyf = process_str(pyf_src)
    # 规范化处理后的字符串 pyf
    normalized_pyf = normalize_whitespace(pyf)
    # 规范化预期的字符串 expected_pyf
    normalized_expected_pyf = normalize_whitespace(expected_pyf)
    # 使用 assert_equal 函数比较规范化后的 pyf 和 expected_pyf，确保它们相等
    assert_equal(normalized_pyf, normalized_expected_pyf)
```