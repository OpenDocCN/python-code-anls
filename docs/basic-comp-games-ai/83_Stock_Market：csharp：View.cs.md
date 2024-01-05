# `d:/src/tocomm/basic-computer-games\83_Stock_Market\csharp\View.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using Game.Extensions;

namespace Game
{
    /// <summary>
    /// 包含向用户显示信息的函数。
    /// </summary>
    public static class View
    {
        // 显示横幅
        public static void ShowBanner()
        {
            Console.WriteLine("                             STOCK MARKET");
            Console.WriteLine("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }
```

在这个示例中，我们为给定的 C# 代码添加了注释，解释了每个语句的作用。
// 显示程序的使用说明
public static void ShowInstructions()
{
    // 打印程序玩股票市场的说明
    Console.WriteLine("THIS PROGRAM PLAYS THE STOCK MARKET.  YOU WILL BE GIVEN");
    Console.WriteLine("$10,000 AND MAY BUY OR SELL STOCKS.  THE STOCK PRICES WILL");
    Console.WriteLine("BE GENERATED RANDOMLY AND THEREFORE THIS MODEL DOES NOT");
    Console.WriteLine("REPRESENT EXACTLY WHAT HAPPENS ON THE EXCHANGE.  A TABLE");
    Console.WriteLine("OF AVAILABLE STOCKS, THEIR PRICES, AND THE NUMBER OF SHARES");
    Console.WriteLine("IN YOUR PORTFOLIO WILL BE PRINTED.  FOLLOWING THIS, THE");
    Console.WriteLine("INITIALS OF EACH STOCK WILL BE PRINTED WITH A QUESTION");
    Console.WriteLine("MARK.  HERE YOU INDICATE A TRANSACTION.  TO BUY A STOCK");
    Console.WriteLine("TYPE +NNN, TO SELL A STOCK TYPE -NNN, WHERE NNN IS THE");
    Console.WriteLine("NUMBER OF SHARES.  A BROKERAGE FEE OF 1% WILL BE CHARGED");
    Console.WriteLine("ON ALL TRANSACTIONS.  NOTE THAT IF A STOCK'S VALUE DROPS");
    Console.WriteLine("TO ZERO IT MAY REBOUND TO A POSITIVE VALUE AGAIN.  YOU");
    Console.WriteLine("HAVE $10,000 TO INVEST.  USE INTEGERS FOR ALL YOUR INPUTS.");
    Console.WriteLine("(NOTE:  TO GET A 'FEEL' FOR THE MARKET RUN FOR AT LEAST");
    Console.WriteLine("10 DAYS)");
    Console.WriteLine("-----GOOD LUCK!-----");
}
            // 显示公司信息
            public static void ShowCompanies(IEnumerable<Company> companies)
            {
                // 计算公司名称的最大长度
                var maxNameLength = companies.Max(company => company.Name.Length);

                // 打印表头
                Console.WriteLine($"{"STOCK".PadRight(maxNameLength)} INITIALS      PRICE/SHARE");
                // 遍历每个公司，打印公司信息
                foreach (var company in companies)
                    Console.WriteLine($"{company.Name.PadRight(maxNameLength)}   {company.StockSymbol}          {company.SharePrice:0.00}");

                // 打印空行
                Console.WriteLine();
                // 打印纽约证券交易所平均股价
                Console.WriteLine($"NEW YORK STOCK EXCHANGE AVERAGE: {companies.Average(company => company.SharePrice):0.00}");
                // 打印空行
                Console.WriteLine();
            }

            // 显示交易结果
            public static void ShowTradeResults(TradingDay day, TradingDay previousDay, Assets assets)
            {
                // 使用自定义的 Zip 方法将公司、前一交易日的公司、资产组合进行组合
                var results = EnumerableExtensions.Zip(
                    day.Companies,
                    previousDay.Companies,
                    assets.Portfolio,
                (company, previous, shares) =>
                (
                    stockSymbol: company.StockSymbol,  // 获取公司的股票代码
                    price: company.SharePrice,  // 获取公司的股价
                    shares,  // 获取持有的股票数量
                    value: shares * company.SharePrice,  // 计算持有股票的价值
                    change: company.SharePrice - previous.SharePrice  // 计算股价变化
                )).ToList();  // 将结果转换为列表

            Console.WriteLine();  // 输出空行
            Console.WriteLine();  // 输出空行
            Console.WriteLine("**********     END OF DAY'S TRADING     **********");  // 输出交易结束标志
            Console.WriteLine();  // 输出空行
            Console.WriteLine();

            Console.WriteLine("STOCK\tPRICE/SHARE\tHOLDINGS\tVALUE\tNET PRICE CHANGE");  // 输出表头
            foreach (var result in results)  // 遍历结果列表
                Console.WriteLine($"{result.stockSymbol}\t{result.price}\t\t{result.shares}\t\t{result.value:0.00}\t\t{result.change:0.00}");  // 输出每条结果的股票代码、股价、持有数量、价值和股价变化

            Console.WriteLine();  // 输出空行
            Console.WriteLine();
            Console.WriteLine();
            // 获取当日平均股价
            var averagePrice = day.AverageSharePrice;
            // 计算平均股价变化
            var averagePriceChange = averagePrice - previousDay.AverageSharePrice;

            // 打印纽约证券交易所平均股价和净变化
            Console.WriteLine($"NEW YORK STOCK EXCHANGE AVERAGE: {averagePrice:0.00} NET CHANGE {averagePriceChange:0.00}");
            Console.WriteLine();
        }

        public static void ShowAssets(Assets assets, IEnumerable<Company> companies)
        {
            // 计算总股票价值
            var totalStockValue = Enumerable.Zip(
                assets.Portfolio,
                companies,
                (shares, company) => shares * company.SharePrice).Sum();

            // 打印总股票资产
            Console.WriteLine($"TOTAL STOCK ASSETS ARE   ${totalStockValue:0.00}");
            // 打印总现金资产
            Console.WriteLine($"TOTAL CASH ASSETS ARE    ${assets.Cash:0.00}");
            // 打印总资产
            Console.WriteLine($"TOTAL ASSETS ARE         ${totalStockValue + assets.Cash:0.00}");
# 打印空行
            Console.WriteLine();
        }

        # 打印已经超卖股票的消息
        public static void ShowOversold()
        {
            Console.WriteLine();
            Console.WriteLine("YOU HAVE OVERSOLD A STOCK; TRY AGAIN.");
        }

        # 打印已经超支的金额
        public static void ShowOverspent(double amount)
        {
            Console.WriteLine();
            Console.WriteLine($"YOU HAVE USED ${amount:0.00} MORE THAN YOU HAVE.");
        }

        # 打印道别信息
        public static void ShowFarewell()
        {
            Console.WriteLine("HOPE YOU HAD FUN!!");
        }
# 打印两行空白
public static void ShowSeparator()
{
    Console.WriteLine();
    Console.WriteLine();
}

# 打印单个字符
public static void ShowChar(char c)
{
    Console.WriteLine(c);
}

# 提示用户是否需要显示说明
public static void PromptShowInstructions()
{
    Console.Write("DO YOU WANT THE INSTRUCTIONS (YES-TYPE 1, NO-TYPE 0)? ");
}

# 提示用户是否需要继续
public static void PromptContinue()
{
    Console.Write("DO YOU WISH TO CONTINUE (YES-TYPE 1, NO-TYPE 0)? ");
}
# 提示用户输入交易信息
public static void PromptEnterTransactions()
{
    Console.WriteLine("WHAT IS YOUR TRANSACTION IN");
}

# 提示用户买卖公司股票
public static void PromptBuySellCompany(Company company)
{
    Console.Write($"{company.StockSymbol}? ");
}

# 提示用户输入有效的整数
public static void PromptValidInteger()
{
    Console.WriteLine("PLEASE ENTER A VALID INTEGER");
}
```