# `basic-computer-games\53_King\csharp\ValidityTest.cs`

```

namespace King; // 命名空间声明

internal class ValidityTest // 内部类 ValidityTest 声明
{
    private readonly Predicate<float> _isValid; // 只读字段，存储 float 类型的断言
    private readonly Func<string> _getError; // 只读字段，存储返回字符串的函数

    public ValidityTest(Predicate<float> isValid, string error) // 构造函数，接受断言和错误信息
        : this(isValid, () => error) // 调用另一个构造函数
    {
    }

    public ValidityTest(Predicate<float> isValid, Func<string> getError) // 构造函数，接受断言和返回错误信息的函数
    {
        _isValid = isValid; // 将传入的断言赋值给 _isValid
        _getError = getError; // 将传入的返回错误信息的函数赋值给 _getError
    }

    public bool IsValid(float value, IReadWrite io) // 公共方法，用于检查值的有效性
    {
        if (_isValid(value)) { return true; } // 如果值有效，返回 true
        
        io.Write(_getError()); // 否则，将错误信息写入 IReadWrite 接口
        return false; // 返回 false
    }
}

```