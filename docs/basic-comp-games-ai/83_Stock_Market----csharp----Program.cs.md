# `basic-computer-games\83_Stock_Market\csharp\Program.cs`

```
using System;  // 引入 System 命名空间
using System.Collections.Immutable;  // 引入 Immutable 命名空间
using System.Linq;  // 引入 Linq 命名空间

namespace Game  // 命名空间 Game
{
    class Program  // 类 Program
    {
        /// <summary>
        /// Defines the set of companies that will be simulated in the game.
        /// </summary>
        private readonly static ImmutableArray<Company> Companies = ImmutableArray.CreateRange(new[]  // 定义将在游戏中模拟的公司集合
        {
            new Company("INT. BALLISTIC MISSILES",     "IBM", sharePrice:100),  // 创建公司对象并添加到集合中
            new Company("RED CROSS OF AMERICA",        "RCA", sharePrice:85 ),  // 创建公司对象并添加到集合中
            new Company("LICHTENSTEIN, BUMRAP & JOKE", "LBJ", sharePrice:150),  // 创建公司对象并添加到集合中
            new Company("AMERICAN BANKRUPT CO.",       "ABC", sharePrice:140),  // 创建公司对象并添加到集合中
            new Company("CENSURED BOOKS STORE",        "CBS", sharePrice:110)   // 创建公司对象并添加到集合中
        });

        static void Main()  // 主函数
        {
            var assets = new Assets  // 创建资产对象
            {
                Cash      = 10000.0,  // 设置现金
                Portfolio = ImmutableArray.CreateRange(Enumerable.Repeat(0, Companies.Length))  // 设置投资组合
            };

            var previousDay = default(TradingDay);  // 初始化前一天的交易日

            Controller.StartGame();  // 调用控制器的开始游戏方法

            foreach (var day in StockMarket.Simulate(Companies))  // 遍历模拟股市交易日
            {
                if (previousDay is null)  // 如果前一天为空
                    View.ShowCompanies(day.Companies);  // 显示公司信息
                else
                    View.ShowTradeResults(day, previousDay, assets);  // 显示交易结果

                View.ShowAssets(assets, day.Companies);  // 显示资产信息

                if (previousDay is not null && !Controller.GetYesOrNo(View.PromptContinue))  // 如果前一天不为空且用户选择不继续
                    break;  // 跳出循环

                assets      = Controller.UpdateAssets(assets, day.Companies);  // 更新资产信息
                previousDay = day;  // 更新前一天的交易日
            }

            View.ShowFarewell();  // 显示结束语
        }
    }
}
```