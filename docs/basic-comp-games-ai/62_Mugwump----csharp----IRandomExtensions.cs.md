# `basic-computer-games\62_Mugwump\csharp\IRandomExtensions.cs`

```

# 创建一个名为Mugwump的命名空间
namespace Mugwump;

# 创建一个名为IRandomExtensions的静态类，用于扩展IRandom接口
internal static class IRandomExtensions
{
    # 创建一个名为NextPosition的静态方法，用于生成一个Position对象
    internal static Position NextPosition(this IRandom random, int maxX, int maxY) =>
        # 使用IRandom接口的Next方法生成一个随机X坐标和一个随机Y坐标，创建一个Position对象并返回
        new(random.Next(maxX), random.Next(maxY));
}

```