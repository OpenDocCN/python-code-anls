# `83_Stock_Market\csharp\Program.cs`

```
// 引入必要的命名空间
using System;
using System.Collections.Immutable;
using System.Linq;

namespace Game
{
    class Program
    {
        /// <summary>
        /// 定义将在游戏中模拟的公司集合。
        /// </summary>
        private readonly static ImmutableArray<Company> Companies = ImmutableArray.CreateRange(new[]
        {
            // 创建并初始化公司对象，包括公司名称、股票代码和股价
            new Company("INT. BALLISTIC MISSILES",     "IBM", sharePrice:100),
            new Company("RED CROSS OF AMERICA",        "RCA", sharePrice:85 ),
            new Company("LICHTENSTEIN, BUMRAP & JOKE", "LBJ", sharePrice:150),
            new Company("AMERICAN BANKRUPT CO.",       "ABC", sharePrice:140),
            new Company("CENSURED BOOKS STORE",        "CBS", sharePrice:110)
        });
        static void Main()
        {
            // 创建一个资产对象，设置现金为10000.0，投资组合为指定长度的不可变数组
            var assets = new Assets
            {
                Cash      = 10000.0,
                Portfolio = ImmutableArray.CreateRange(Enumerable.Repeat(0, Companies.Length))
            };

            // 初始化前一交易日对象
            var previousDay = default(TradingDay);

            // 启动游戏控制器
            Controller.StartGame();

            // 对每个交易日模拟股市交易
            foreach (var day in StockMarket.Simulate(Companies))
            {
                // 如果前一交易日为空，则显示公司信息
                if (previousDay is null)
                    View.ShowCompanies(day.Companies);
                // 否则显示交易结果
                else
                    View.ShowTradeResults(day, previousDay, assets);

                // 显示资产信息和公司信息
                View.ShowAssets(assets, day.Companies);
# 如果前一天的数据不为空，并且用户选择不继续，则跳出循环
if (previousDay is not null && !Controller.GetYesOrNo(View.PromptContinue))
    break;

# 更新资产信息，传入当前资产和当天公司数据，返回更新后的资产信息
assets = Controller.UpdateAssets(assets, day.Companies);

# 将当前天的数据赋值给previousDay，用于下一次循环的比较
previousDay = day;
```
这段代码是一个循环体内的逻辑，根据前一天的数据和用户的选择来决定是否继续循环，然后更新资产信息和保存当前天的数据。
```