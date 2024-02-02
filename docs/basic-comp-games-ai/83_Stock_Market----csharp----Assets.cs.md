# `basic-computer-games\83_Stock_Market\csharp\Assets.cs`

```py
// 引入不可变集合的命名空间
using System.Collections.Immutable;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 存储玩家的资产。
    /// </summary>
    // 定义一个记录类型 Assets
    public record Assets
    {
        /// <summary>
        /// 获取玩家现金金额。
        /// </summary>
        // 定义一个可获取且可初始化的属性 Cash
        public double Cash { get; init; }

        /// <summary>
        /// 获取每家公司拥有的股票数量。
        /// </summary>
        // 定义一个不可变整数数组属性 Portfolio
        public ImmutableArray<int> Portfolio { get; init; }
    }
}
```