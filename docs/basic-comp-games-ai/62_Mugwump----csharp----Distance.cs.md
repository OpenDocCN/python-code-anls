# `basic-computer-games\62_Mugwump\csharp\Distance.cs`

```
# 在Mugwump命名空间下定义一个内部结构Distance
namespace Mugwump;

# 定义一个只能在当前程序集内部访问的结构Distance
internal struct Distance
{
    # 只读字段，存储距离值
    private readonly float _value;

    # 构造函数，接受两个参数，计算欧几里得距离并存储到_value字段中
    public Distance(float deltaX, float deltaY)
    {
        _value = (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
    }

    # 重写ToString方法，返回_value字段的值，保留一位小数
    public override string ToString() => _value.ToString("0.0");
}
```