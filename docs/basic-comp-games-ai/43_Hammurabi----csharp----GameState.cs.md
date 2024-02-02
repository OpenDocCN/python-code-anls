# `basic-computer-games\43_Hammurabi\csharp\GameState.cs`

```py
# 命名空间声明，指定代码所属的命名空间
namespace Hammurabi
{
    /// <summary>
    /// 存储游戏的状态信息
    /// </summary>
    # 声明一个记录类型，用于存储游戏的状态信息
    public record GameState
    /// <summary>
    /// 获取当前游戏年份。
    /// </summary>
    public int Year { get; init; }

    /// <summary>
    /// 获取城市的人口。
    /// </summary>
    public int Population { get; init; }

    /// <summary>
    /// 获取今年的人口增长。
    /// </summary>
    public int PopulationIncrease { get; init; }

    /// <summary>
    /// 获取挨饿的人数。
    /// </summary>
    public int Starvation { get; init; }

    /// <summary>
    /// 获取城市的土地面积（英亩）。
    /// </summary>
    public int Acres { get; init; }

    /// <summary>
    /// 获取每英亩土地的价格（以蒲式耳计）。
    /// </summary>
    public int LandPrice { get; init; }

    /// <summary>
    /// 获取城市仓库中的粮食数量（以蒲式耳计）。
    /// </summary>
    public int Stores { get; init; }

    /// <summary>
    /// 获取分发给人民的食物数量。
    /// </summary>
    public int FoodDistributed { get; init; }

    /// <summary>
    /// 获取种植的土地面积。
    /// </summary>
    public int AcresPlanted { get; init; }

    /// <summary>
    /// 获取每英亩土地产出的粮食数量。
    /// </summary>
    public int Productivity { get; init; }

    /// <summary>
    /// 获取被老鼠吃掉的粮食数量。
    /// </summary>
    public int Spoilage { get; init; }

    /// <summary>
    /// 获取一个指示当前年份是否为瘟疫年的标志。
    /// </summary>
    public bool IsPlagueYear { get; init; }

    /// <summary>
    /// 获取一个指示玩家是否被弹劾的标志。
    /// </summary>
    public bool IsPlayerImpeached { get; init; }
# 闭合前面的函数定义
```