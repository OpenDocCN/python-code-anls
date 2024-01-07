# `basic-computer-games\62_Mugwump\csharp\Distance.cs`

```

// 命名空间声明，表示该结构体属于Mugwump命名空间
namespace Mugwump;

// 结构体声明，表示Distance结构体是内部的，只能在当前命名空间内部访问
internal struct Distance
{
    // 只读字段，表示距离的数值
    private readonly float _value;

    // 构造函数，接受两个参数，计算欧几里得距离并赋值给_value字段
    public Distance(float deltaX, float deltaY)
    {
        _value = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    // 重写ToString方法，返回_value字段的字符串表示形式，保留一位小数
    public override string ToString() => _value.ToString("0.0");
}

```