# `basic-computer-games\00_Common\dotnet\Games.Common\Numbers\Number.cs`

```

// 命名空间声明，表示该代码属于 Games.Common.Numbers 命名空间
namespace Games.Common.Numbers;

/// <summary>
/// 表示一个具有与 BASIC 解释器等效的字符串格式的单精度浮点数。
/// </summary>
// 定义一个名为 Number 的结构体
public struct Number
{
    // 只读字段，存储单精度浮点数的值
    private readonly float _value;

    // 构造函数，用于初始化 Number 结构体的值
    public Number (float value)
    {
        _value = value;
    }

    // 隐式转换操作符，将 Number 结构体转换为 float 类型
    public static implicit operator float(Number value) => value._value;

    // 隐式转换操作符，将 float 类型转换为 Number 结构体
    public static implicit operator Number(float value) => new Number(value);

    // 重写 ToString 方法，根据 _value 的值返回对应的字符串
    public override string ToString() => _value < 0 ? $"{_value} " : $" {_value} ";
}

```