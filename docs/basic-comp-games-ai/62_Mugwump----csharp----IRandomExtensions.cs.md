# `basic-computer-games\62_Mugwump\csharp\IRandomExtensions.cs`

```
# 创建一个名为Mugwump的命名空间，用于组织相关的类和接口
namespace Mugwump;

# 创建一个名为IRandomExtensions的静态类，用于扩展IRandom接口的功能
internal static class IRandomExtensions
{
    # 创建一个名为NextPosition的静态方法，用于生成一个Position对象
    # 该方法接受一个IRandom对象和两个整数参数maxX和maxY
    internal static Position NextPosition(this IRandom random, int maxX, int maxY) =>
        # 使用IRandom对象的Next方法生成一个x坐标和y坐标，创建一个新的Position对象并返回
        new(random.Next(maxX), random.Next(maxY));
}
```