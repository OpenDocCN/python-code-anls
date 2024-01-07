# `basic-computer-games\83_Stock_Market\csharp\TradingDay.cs`

```

// 引入不可变集合和 LINQ 查询
using System.Collections.Immutable;
using System.Linq;

namespace Game
{
    /// <summary>
    /// 表示单个交易日。
    /// </summary>
    public record TradingDay
    {
        /// <summary>
        /// 获取当天市场上所有公司的平均股价。
        /// </summary>
        public double AverageSharePrice =>
            Companies.Average (company => company.SharePrice);

        /// <summary>
        /// 获取当天股市上公开上市的公司集合。
        /// </summary>
        public ImmutableArray<Company> Companies { get; init; }
    }
}

```