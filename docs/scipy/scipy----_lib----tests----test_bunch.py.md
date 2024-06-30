# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_bunch.py`

```
import pytest  # 导入 pytest 测试框架
import pickle  # 导入 pickle 用于对象序列化
from numpy.testing import assert_equal  # 从 numpy.testing 中导入 assert_equal 断言函数
from scipy._lib._bunch import _make_tuple_bunch  # 导入 _make_tuple_bunch 函数用于创建命名元组

# `Result` is defined at the top level of the module so it can be
# used to test pickling.
# 在模块顶层定义 `Result`，以便用于测试序列化。

Result = _make_tuple_bunch('Result', ['x', 'y', 'z'], ['w', 'beta'])


class TestMakeTupleBunch:

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Tests with Result
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def setup_method(self):
        # Set up an instance of Result.
        # 创建一个 Result 的实例。
        self.result = Result(x=1, y=2, z=3, w=99, beta=0.5)

    def test_attribute_access(self):
        assert_equal(self.result.x, 1)  # 断言 self.result.x 等于 1
        assert_equal(self.result.y, 2)  # 断言 self.result.y 等于 2
        assert_equal(self.result.z, 3)  # 断言 self.result.z 等于 3
        assert_equal(self.result.w, 99)  # 断言 self.result.w 等于 99
        assert_equal(self.result.beta, 0.5)  # 断言 self.result.beta 等于 0.5

    def test_indexing(self):
        assert_equal(self.result[0], 1)  # 断言 self.result[0] 等于 1
        assert_equal(self.result[1], 2)  # 断言 self.result[1] 等于 2
        assert_equal(self.result[2], 3)  # 断言 self.result[2] 等于 3
        assert_equal(self.result[-1], 3)  # 断言 self.result[-1] 等于 3
        with pytest.raises(IndexError, match='index out of range'):
            self.result[3]  # 断言 self.result[3] 抛出 IndexError 异常

    def test_unpacking(self):
        x0, y0, z0 = self.result  # 对 self.result 进行解包操作
        assert_equal((x0, y0, z0), (1, 2, 3))  # 断言解包后的值与预期相符
        assert_equal(self.result, (1, 2, 3))  # 断言 self.result 等于 (1, 2, 3)

    def test_slice(self):
        assert_equal(self.result[1:], (2, 3))  # 断言切片操作的结果与预期相符
        assert_equal(self.result[::2], (1, 3))  # 断言切片操作的结果与预期相符
        assert_equal(self.result[::-1], (3, 2, 1))  # 断言切片操作的结果与预期相符

    def test_len(self):
        assert_equal(len(self.result), 3)  # 断言 len(self.result) 等于 3

    def test_repr(self):
        s = repr(self.result)  # 获取 self.result 的字符串表示
        assert_equal(s, 'Result(x=1, y=2, z=3, w=99, beta=0.5)')  # 断言字符串表示与预期相符

    def test_hash(self):
        assert_equal(hash(self.result), hash((1, 2, 3)))  # 断言 self.result 的哈希值与预期相符

    def test_pickle(self):
        s = pickle.dumps(self.result)  # 序列化 self.result 对象
        obj = pickle.loads(s)  # 反序列化得到的对象
        assert isinstance(obj, Result)  # 断言反序列化后的对象是 Result 类型
        assert_equal(obj.x, self.result.x)  # 断言反序列化后的对象的属性与原对象相符
        assert_equal(obj.y, self.result.y)  # 断言反序列化后的对象的属性与原对象相符
        assert_equal(obj.z, self.result.z)  # 断言反序列化后的对象的属性与原对象相符
        assert_equal(obj.w, self.result.w)  # 断言反序列化后的对象的属性与原对象相符
        assert_equal(obj.beta, self.result.beta)  # 断言反序列化后的对象的属性与原对象相符

    def test_read_only_existing(self):
        with pytest.raises(AttributeError, match="can't set attribute"):
            self.result.x = -1  # 尝试修改只读属性 self.result.x，预期引发 AttributeError 异常

    def test_read_only_new(self):
        self.result.plate_of_shrimp = "lattice of coincidence"  # 给 self.result 添加新的只读属性
        assert self.result.plate_of_shrimp == "lattice of coincidence"  # 断言新属性的值符合预期

    def test_constructor_missing_parameter(self):
        with pytest.raises(TypeError, match='missing'):
            # `w` is missing.
            Result(x=1, y=2, z=3, beta=0.75)  # 测试缺少必需参数 `w`，预期引发 TypeError 异常

    def test_constructor_incorrect_parameter(self):
        with pytest.raises(TypeError, match='unexpected'):
            # `foo` is not an existing field.
            Result(x=1, y=2, z=3, w=123, beta=0.75, foo=999)  # 测试传入不存在的参数 `foo`，预期引发 TypeError 异常

    def test_module(self):
        m = 'scipy._lib.tests.test_bunch'
        assert_equal(Result.__module__, m)  # 断言 Result 的模块名与预期相符
        assert_equal(self.result.__module__, m)  # 断言 self.result 的模块名与预期相符
    def test_extra_fields_per_instance(self):
        # This test exists to ensure that instances of the same class
        # store their own values for the extra fields. That is, the values
        # are stored per instance and not in the class.

        # Create an instance of Result with specific attributes
        result1 = Result(x=1, y=2, z=3, w=-1, beta=0.0)
        result2 = Result(x=4, y=5, z=6, w=99, beta=1.0)

        # Assert statements to check values of attributes for result1
        assert_equal(result1.w, -1)
        assert_equal(result1.beta, 0.0)
        
        # Additional checks (not essential for the main purpose of the test)
        assert_equal(result1[:], (1, 2, 3))
        
        # Assert statements to check values of attributes for result2
        assert_equal(result2.w, 99)
        assert_equal(result2.beta, 1.0)
        assert_equal(result2[:], (4, 5, 6))


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Other tests
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def test_extra_field_names_is_optional(self):
        # Define a tuple-like class Square with attributes 'width' and 'height'
        Square = _make_tuple_bunch('Square', ['width', 'height'])
        
        # Create an instance of Square with specified attributes
        sq = Square(width=1, height=2)
        
        # Assert statements to verify attribute values
        assert_equal(sq.width, 1)
        assert_equal(sq.height, 2)
        
        # Get the string representation of sq and assert its correctness
        s = repr(sq)
        assert_equal(s, 'Square(width=1, height=2)')


    def test_tuple_like(self):
        # Define a tuple-like class Tup with attributes 'a' and 'b'
        Tup = _make_tuple_bunch('Tup', ['a', 'b'])
        
        # Create an instance of Tup with specified attributes
        tu = Tup(a=1, b=2)
        
        # Assert statements to check instance type and tuple concatenation
        assert isinstance(tu, tuple)
        assert isinstance(tu + (1,), tuple)


    def test_explicit_module(self):
        # Define a class Foo with attribute 'x' and extra fields 'a' and 'b'
        m = 'some.module.name'
        Foo = _make_tuple_bunch('Foo', ['x'], ['a', 'b'], module=m)
        
        # Create an instance of Foo with specified attributes
        foo = Foo(x=1, a=355, b=113)
        
        # Assert statements to check module assignment for Foo and foo instances
        assert_equal(Foo.__module__, m)
        assert_equal(foo.__module__, m)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -```
    # Argument validation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @pytest.mark.parametrize('args', [('123', ['a'], ['b']),
                                      ('Foo', ['-3'], ['x']),
                                      ('Foo', ['a'], ['+-*/'])])
    def test_identifiers_not_allowed(self, args):
        # Test case for checking identifiers in field names
        with pytest.raises(ValueError, match='identifiers'):
            _make_tuple_bunch(*args)

    @pytest.mark.parametrize('args', [('Foo', ['a', 'b', 'a'], ['x']),
                                      ('Foo', ['a', 'b'], ['b', 'x'])])
    def test_repeated_field_names(self, args):
        # Test case for checking duplicate field names
        with pytest.raises(ValueError, match='Duplicate'):
            _make_tuple_bunch(*args)

    @pytest.mark.parametrize('args', [('Foo', ['_a'], ['x']),
                                      ('Foo', ['a'], ['_x'])])
    def test_leading_underscore_not_allowed(self, args):
        # Test case for checking leading underscore in field names
        with pytest.raises(ValueError, match='underscore'):
            _make_tuple_bunch(*args)

    @pytest.mark.parametrize('args', [('Foo', ['def'], ['x']),
                                      ('Foo', ['a'], ['or']),
                                      ('and', ['a'], ['x'])])


这些注释详细解释了每个测试函数的目的和测试内容，确保了代码的清晰和易于理解。
    # 定义一个测试方法，用于验证不允许使用关键字作为字段名时是否抛出 ValueError 异常，
    # 并且异常消息中包含 'keyword' 字符串。
    def test_keyword_not_allowed_in_fields(self, args):
        # 使用 pytest 的断言来验证调用 _make_tuple_bunch 函数时是否会抛出 ValueError 异常，
        # 并检查异常消息是否包含 'keyword' 字符串。
        with pytest.raises(ValueError, match='keyword'):
            _make_tuple_bunch(*args)
    
    # 定义另一个测试方法，用于验证至少需要一个字段名时是否抛出 ValueError 异常，
    # 并且异常消息中包含 'at least one name' 字符串。
    def test_at_least_one_field_name_required(self):
        # 使用 pytest 的断言来验证调用 _make_tuple_bunch 函数时是否会抛出 ValueError 异常，
        # 并检查异常消息是否包含 'at least one name' 字符串。
        with pytest.raises(ValueError, match='at least one name'):
            _make_tuple_bunch('Qwerty', [], ['a', 'b'])
```