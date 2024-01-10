# `basic-computer-games\43_Hammurabi\csharp\GameResult.cs`

```
// 存储最终游戏结果的记录
public record GameResult
{
    // 获取玩家的表现评级
    public PerformanceRating Rating { get; init; }

    // 获取城市每人的土地面积
    public int AcresPerPerson { get; init; }

    // 获取最后一年在任期内挨饿的人数
    public int FinalStarvation { get; init; }

    // 获取总共挨饿的人数
    public int TotalStarvation { get; init; }

    // 获取每年平均挨饿率（作为人口的百分比）
    public int AverageStarvationRate { get; init; }

    // 获取想要暗杀玩家的人数
    public int Assassins { get; init; }

    // 获取一个指示玩家是否因为饿死太多人而被弹劾的标志
    public bool WasPlayerImpeached { get; init; }
}
```