# `basic-computer-games\84_Super_Star_Trek\csharp\IRandomExtensions.cs`

```

// 引入随机数生成和空间坐标的命名空间
using Games.Common.Randomness;
using SuperStarTrek.Space;

// 定义静态类 IRandomExtensions
namespace SuperStarTrek;

internal static class IRandomExtensions
{
    // 扩展方法，用于生成下一个坐标
    internal static Coordinates NextCoordinate(this IRandom random) =>
        new Coordinates(random.Next1To8Inclusive() - 1, random.Next1To8Inclusive() - 1);

    // 重复了原始代码中用于获取1到8之间整数值的算法：
    //     475 DEF FNR(R)=INT(RND(R)*7.98+1.01)
    // 返回一个1到8之间的值，包括1和8
    // 注意，对于极端值1和8有轻微的偏差
    internal static int Next1To8Inclusive(this IRandom random) => (int)(random.NextFloat() * 7.98 + 1.01);
}

```