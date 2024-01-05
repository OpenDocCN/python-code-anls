# `62_Mugwump\csharp\Distance.cs`

```
namespace Mugwump;  // 命名空间声明，用于组织和管理代码

internal struct Distance  // 定义一个结构体 Distance，表示距离
{
    private readonly float _value;  // 声明一个只读的浮点数变量 _value，用于存储距离值

    public Distance(float deltaX, float deltaY)  // 定义 Distance 结构体的构造函数，接受 deltaX 和 deltaY 作为参数
    {
        _value = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);  // 计算欧几里得距离并赋值给 _value
    }

    public override string ToString() => _value.ToString("0.0");  // 重写 ToString 方法，返回距离值的字符串表示形式，保留一位小数
}
```