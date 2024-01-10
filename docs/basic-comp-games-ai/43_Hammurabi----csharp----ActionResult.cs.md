# `basic-computer-games\43_Hammurabi\csharp\ActionResult.cs`

```
// 命名空间 Hammurabi
namespace Hammurabi
{
    /// <summary>
    /// 枚举了尝试游戏中各种行动的不同可能结果。
    /// </summary>
    public enum ActionResult
    {
        /// <summary>
        /// 行动成功。
        /// </summary>
        Success,

        /// <summary>
        /// 由于城市没有足够的谷物，无法完成行动。
        /// </summary>
        InsufficientStores,

        /// <summary>
        /// 由于城市土地不足，无法完成行动。
        /// </summary>
        InsufficientLand,

        /// <summary>
        /// 由于城市人口不足，无法完成行动。
        /// </summary>
        InsufficientPopulation,

        /// <summary>
        /// 请求的行动冒犯了城市管家。
        /// </summary>
        Offense
    }
}
```