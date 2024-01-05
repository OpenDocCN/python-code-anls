# `d:/src/tocomm/basic-computer-games\83_Stock_Market\csharp\StockMarket.cs`

```
using System;  // 导入 System 命名空间
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间
using System.Collections.Immutable;  // 导入 System.Collections.Immutable 命名空间
using System.Linq;  // 导入 System.Linq 命名空间
using Game.Extensions;  // 导入 Game.Extensions 命名空间

namespace Game  // 声明 Game 命名空间
{
    /// <summary>
    /// 提供模拟股票市场的方法。
    /// </summary>
    public static class StockMarket  // 声明 StockMarket 静态类
    {
        /// <summary>
        /// 模拟股票市场随时间的变化。
        /// </summary>
        /// <param name="companies">
        /// 将参与市场的公司集合。
        /// </param>
        /// <returns>
        /// An infinite sequence of trading days.  Each day represents the
        /// state of the stock market at the start of that day.
        /// </returns>
        public static IEnumerable<TradingDay> Simulate(ImmutableArray<Company> companies)
        {
            // 创建一个随机数生成器
            var random = new Random();

            // 使用Zip方法将三个序列合并成一个新的序列
            var cyclicParameters = EnumerableExtensions.Zip(
                Trends(random, 1, 5),  // 生成趋势序列
                PriceSpikes(random, companies.Length, 1, 5),  // 生成价格尖峰序列
                PriceSpikes(random, companies.Length, 1, 5),  // 生成价格尖峰序列
                (trend, company1, company2) => (trend, positiveSpike: company1, negativeSpike: company2));  // 将三个序列合并成一个元组序列

            // 对合并后的序列进行处理，生成TradingDay对象序列
            return cyclicParameters.SelectAndAggregate(
                new TradingDay
                {
                    Companies = companies  // 设置TradingDay对象的Companies属性
                },
                (parameters, previousDay) => previousDay with  // 对每个参数进行处理，生成新的TradingDay对象
                {
                    // 使用前一天的公司数据进行映射，调整股价并创建新的公司列表
                    Companies = previousDay.Companies.Map(
                        (company, index) => AdjustSharePrice(
                            random,
                            company,
                            parameters.trend,
                            parameters.positiveSpike == index,
                            parameters.negativeSpike == index))
                });
        }

        /// <summary>
        /// 根据给定的参数，创建一个公司的副本，并随机调整股价。
        /// </summary>
        /// <param name="random">
        /// 随机数生成器。
        /// </param>
        /// <param name="company">
        /// 要调整的公司。
        /// </param>
        /// <param name="trend">
        /// The slope of the overall market price trend.
        /// </param>
        /// <param name="positiveSpike">
        /// True if the function should simulate a positive spike in the
        /// company's share price.
        /// </param>
        /// <param name="negativeSpike">
        /// True if the function should simulate a negative spike in the
        /// company's share price.
        /// </param>
        /// <returns>
        /// The adjusted company.
        /// </returns>
        private static Company AdjustSharePrice(Random random, Company company, double trend, bool positiveSpike, bool negativeSpike)
        {
            // 生成一个0到4之间的随机整数，乘以0.25得到一个0到1之间的随机浮点数
            var boost = random.Next(4) * 0.25;

            // 初始化一个变量用于存储价格波动的数量
            var spikeAmount = 0.0;
            // 如果市场趋势为正向，则设置价格波动值为10
            if (positiveSpike)
                spikeAmount = 10;

            // 如果市场趋势为负向，则将价格波动值减去10
            if (negativeSpike)
                spikeAmount = spikeAmount - 10;

            // 计算价格变动，包括趋势、公司股价、增益、随机波动和价格波动值
            var priceChange = (int)(trend * company.SharePrice) + boost + (int)(3.5 - (6 * random.NextDouble())) + spikeAmount;

            // 计算新的股价
            var newPrice = company.SharePrice + priceChange;
            // 如果新的股价小于0，则将其设置为0
            if (newPrice < 0)
                newPrice = 0;

            // 返回更新后的公司对象，其中股价已更新
            return company with { SharePrice = newPrice };
        }

        /// <summary>
        /// 生成一个无限序列的市场趋势。
        /// </summary>
        /// <param name="random">
        /// 随机数生成器。
        /// <summary>
        /// 生成市场趋势的随机值。
        /// </summary>
        /// <param name="random">
        /// 随机数生成器。
        /// </param>
        /// <returns>
        /// 在范围[-0.1, 0.1]内的趋势值。
        /// </returns>
        private static double GenerateTrend(Random random) =>
            ((int)(random.NextDouble() * 10 + 0.5) / 100.0) * (random.Next(2) == 0 ? 1 : -1) ;
```
这行代码是一个数学表达式，它生成一个随机的价格波动值。

```csharp
        /// <summary>
        /// Generates an infinite sequence of price spikes.
        /// </summary>
```
这是一个文档注释，用于描述下面的函数的作用。

```csharp
        /// <param name="random">
        /// The random number generator.
        /// </param>
```
这是一个参数注释，用于描述函数的参数random的作用。

```csharp
        /// <param name="companyCount">
        /// The number of companies.
        /// </param>
```
这是一个参数注释，用于描述函数的参数companyCount的作用。

```csharp
        /// <param name="minDays">
        /// The minimum number of days in between price spikes.
        /// </param>
```
这是一个参数注释，用于描述函数的参数minDays的作用。

```csharp
        /// <param name="maxDays">
        /// The maximum number of days in between price spikes.
        /// </param>
```
这是一个参数注释，用于描述函数的参数maxDays的作用。

```csharp
        /// <returns>
        /// An infinite sequence of random company indexes and null values.
        /// A non-null value means that the corresponding company should
```
这是一个返回值注释，用于描述函数的返回值。
# 生成价格波动的函数，返回一个包含价格波动的序列
def PriceSpikes(random, companyCount, minDays, maxDays):
    # 生成一个包含 minDays 到 maxDays 之间的随机整数序列
    daysInCycle = random.Integers(minDays, maxDays + 1)
    # 对每个随机整数进行遍历
    for days in daysInCycle:
        # 对每个周期内的每一天进行遍历
        for dayNumber in range(days):
            # 如果是周期的第一天，随机生成一个公司的价格
            if dayNumber == 0:
                yield random.Next(companyCount)
            # 如果不是周期的第一天，返回空值
            else:
                yield None
```