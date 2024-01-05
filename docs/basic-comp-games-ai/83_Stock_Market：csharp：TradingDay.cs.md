# `83_Stock_Market\csharp\TradingDay.cs`

```
// 使用不可变集合命名空间
using System.Collections.Immutable;
// 使用 LINQ 命名空间
using System.Linq;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 表示单个交易日
    /// </summary>
    // 定义记录类型 TradingDay
    public record TradingDay
    {
        /// <summary>
        /// 获取当天市场上所有公司的平均股价
        /// </summary>
        // 定义属性 AverageSharePrice，使用 LINQ 求平均值
        public double AverageSharePrice =>
            Companies.Average (company => company.SharePrice);

        /// <summary>
        /// 获取当天股市上公开上市的公司集合
        /// </summary>
```
```csharp
        // 定义属性 Companies，使用不可变集合
        public ImmutableArray<Company> Companies { get; init; }
    }
}
# 定义一个不可变的数组属性 Companies，用于存储 Company 对象
public ImmutableArray<Company> Companies { get; init; }
# 结束类定义
```
在这段代码中，我们定义了一个不可变的数组属性 Companies，用于存储 Company 对象。这个属性是只读的，即初始化后不能再修改。
```