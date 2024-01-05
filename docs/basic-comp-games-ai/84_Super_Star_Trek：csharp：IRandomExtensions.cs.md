# `84_Super_Star_Trek\csharp\IRandomExtensions.cs`

```
using Games.Common.Randomness; // 导入 Games.Common.Randomness 命名空间
using SuperStarTrek.Space; // 导入 SuperStarTrek.Space 命名空间

namespace SuperStarTrek; // 定义 SuperStarTrek 命名空间

internal static class IRandomExtensions // 定义 IRandomExtensions 静态类
{
    internal static Coordinates NextCoordinate(this IRandom random) => // 定义 NextCoordinate 方法，返回 Coordinates 类型
        new Coordinates(random.Next1To8Inclusive() - 1, random.Next1To8Inclusive() - 1); // 使用 random.Next1To8Inclusive() 方法获取坐标值

    // Duplicates the algorithm used in the original code to get an integer value from 1 to 8, inclusive:
    //     475 DEF FNR(R)=INT(RND(R)*7.98+1.01)
    // Returns a value from 1 to 8, inclusive.
    // Note there's a slight bias away from the extreme values, 1 and 8.
    internal static int Next1To8Inclusive(this IRandom random) => (int)(random.NextFloat() * 7.98 + 1.01); // 定义 Next1To8Inclusive 方法，返回一个介于1到8之间的整数值
}
```