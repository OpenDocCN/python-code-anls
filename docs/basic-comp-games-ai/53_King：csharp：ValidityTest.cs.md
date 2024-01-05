# `53_King\csharp\ValidityTest.cs`

```
namespace King;  # 命名空间声明

internal class ValidityTest  # 内部类 ValidityTest 声明
{
    private readonly Predicate<float> _isValid;  # 只读字段 _isValid，类型为 Predicate<float>
    private readonly Func<string> _getError;  # 只读字段 _getError，类型为 Func<string>

    public ValidityTest(Predicate<float> isValid, string error)  # ValidityTest 类的构造函数，接受 Predicate<float> 和 string 类型的参数
        : this(isValid, () => error)  # 调用另一个构造函数
    {
    }

    public ValidityTest(Predicate<float> isValid, Func<string> getError)  # ValidityTest 类的构造函数，接受 Predicate<float> 和 Func<string> 类型的参数
    {
        _isValid = isValid;  # 将参数 isValid 赋值给 _isValid 字段
        _getError = getError;  # 将参数 getError 赋值给 _getError 字段
    }

    public bool IsValid(float value, IReadWrite io)  # 公共方法 IsValid，接受 float 和 IReadWrite 类型的参数
    {
        # 如果值有效，则返回 true
        if (_isValid(value)) { return true; }
        
        # 向输出流写入错误信息
        io.Write(_getError());
        # 返回 false
        return false;
    }
}
```