# `basic-computer-games\84_Super_Star_Trek\csharp\IRandomExtensions.cs`

```py
// 引入 Games.Common.Randomness 命名空间
using Games.Common.Randomness;
// 引入 SuperStarTrek.Space 命名空间
using SuperStarTrek.Space;

// 定义 SuperStarTrek 命名空间
namespace SuperStarTrek
{
    // 定义 IRandomExtensions 静态类
    internal static class IRandomExtensions
    {
        // 定义 NextCoordinate 方法，返回一个 Coordinates 对象
        internal static Coordinates NextCoordinate(this IRandom random) =>
            // 使用 IRandom 对象的 Next1To8Inclusive 方法获取 x 和 y 坐标，然后创建 Coordinates 对象
            new Coordinates(random.Next1To8Inclusive() - 1, random.Next1To8Inclusive() - 1);

        // Duplicates the algorithm used in the original code to get an integer value from 1 to 8, inclusive:
        //     475 DEF FNR(R)=INT(RND(R)*7.98+1.01)
        // Returns a value from 1 to 8, inclusive.
        // Note there's a slight bias away from the extreme values, 1 and 8.
        // 定义 Next1To8Inclusive 方法，返回一个介于 1 到 8 之间的整数
        internal static int Next1To8Inclusive(this IRandom random) => (int)(random.NextFloat() * 7.98 + 1.01);
    }
}
```