# `basic-computer-games\83_Stock_Market\csharp\Program.cs`

```

// 引入必要的命名空间
using System;
using System.Collections.Immutable;
using System.Linq;

// 定义游戏类
namespace Game
{
    class Program
    {
        /// <summary>
        /// 定义将在游戏中模拟的公司集合
        /// </summary>
        private readonly static ImmutableArray<Company> Companies = ImmutableArray.CreateRange(new[]
        {
            new Company("INT. BALLISTIC MISSILES",     "IBM", sharePrice:100),
            new Company("RED CROSS OF AMERICA",        "RCA", sharePrice:85 ),
            new Company("LICHTENSTEIN, BUMRAP & JOKE", "LBJ", sharePrice:150),
            new Company("AMERICAN BANKRUPT CO.",       "ABC", sharePrice:140),
            new Company("CENSURED BOOKS STORE",        "CBS", sharePrice:110)
        });

        static void Main()
        {
            // 初始化资产
            var assets = new Assets
            {
                Cash      = 10000.0,
                Portfolio = ImmutableArray.CreateRange(Enumerable.Repeat(0, Companies.Length))
            };

            var previousDay = default(TradingDay);

            // 启动游戏控制器
            Controller.StartGame();

            // 模拟股市交易
            foreach (var day in StockMarket.Simulate(Companies))
            {
                // 如果是第一天，则显示公司信息
                if (previousDay is null)
                    View.ShowCompanies(day.Companies);
                else
                    View.ShowTradeResults(day, previousDay, assets);

                // 显示资产信息和公司信息
                View.ShowAssets(assets, day.Companies);

                // 如果不是第一天并且玩家选择不继续，则退出循环
                if (previousDay is not null && !Controller.GetYesOrNo(View.PromptContinue))
                    break;

                // 更新资产信息
                assets      = Controller.UpdateAssets(assets, day.Companies);
                previousDay = day;
            }

            // 显示结束语
            View.ShowFarewell();
        }
    }
}

```