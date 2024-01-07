# `basic-computer-games\43_Hammurabi\csharp\ActionResult.cs`

```

// 声明一个枚举类型，用于列举游戏中尝试各种行动的不同可能结果
public enum ActionResult
{
    // 行动成功
    Success,

    // 城市粮食储备不足，无法完成行动
    InsufficientStores,

    // 城市土地面积不足，无法完成行动
    InsufficientLand,

    // 城市人口不足，无法完成行动
    InsufficientPopulation,

    // 请求的行动冒犯了城市的官员
    Offense
}

```