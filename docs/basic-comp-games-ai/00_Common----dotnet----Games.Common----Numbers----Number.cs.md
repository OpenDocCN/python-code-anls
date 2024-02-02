# `basic-computer-games\00_Common\dotnet\Games.Common\Numbers\Number.cs`

```py
// 声明一个命名空间，用于组织代码
namespace Games.Common.Numbers;

/// <summary>
/// 一个单精度浮点数，其字符串格式化等效于 BASIC 解释器。
/// </summary>
// 声明一个结构体 Number
public struct Number
{
    // 声明一个只读的单精度浮点数变量
    private readonly float _value;

    // 构造函数，用于初始化 Number 结构体
    public Number (float value)
    {
        _value = value;
    }

    // 隐式转换，将 Number 转换为 float
    public static implicit operator float(Number value) => value._value;

    // 隐式转换，将 float 转换为 Number
    public static implicit operator Number(float value) => new Number(value);

    // 重写 ToString 方法，根据 _value 的值返回对应的字符串
    public override string ToString() => _value < 0 ? $"{_value} " : $" {_value} ";
}
```