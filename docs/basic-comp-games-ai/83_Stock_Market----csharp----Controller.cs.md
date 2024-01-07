# `basic-computer-games\83_Stock_Market\csharp\Controller.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;

// 创建名为 Controller 的静态类
namespace Game
{
    public static class Controller
    {
        /// <summary>
        /// 管理与用户的初始交互。
        /// </summary>
        public static void StartGame()
        {
            // 显示游戏横幅
            View.ShowBanner();

            // 获取用户是否显示游戏说明的选择
            var showInstructions = GetYesOrNo(View.PromptShowInstructions);
            View.ShowSeparator();
            if (showInstructions)
                View.ShowInstructions();

            View.ShowSeparator();
        }

        /// <summary>
        /// 从用户获取是或否的答案。
        /// </summary>
        /// <param name="prompt">
        /// 显示提示。
        /// </param>
        /// <returns>
        /// 如果用户回答是，则为 true，如果回答否，则为 false。
        /// </returns>
        public static bool GetYesOrNo(Action prompt)
        {
            prompt();

            var response = default(char);
            do
            {
                response = Console.ReadKey(intercept: true).KeyChar;
            }
            while (response != '0' && response != '1');

            View.ShowChar(response);
            return response == '1';
        }

        /// <summary>
        /// 获取给定公司集合中每家公司的交易金额，并返回更新后的资产。
        /// </summary>
        /// <param name="assets">
        /// 要更新的资产。
        /// </param>
        /// <param name="companies">
        /// 公司集合。
        /// </param>
        /// <returns>
        /// 更新后的资产。
        /// </returns>
        public static Assets UpdateAssets(Assets assets, IEnumerable<Company> companies)
        {
            while (true)
            {
                View.PromptEnterTransactions();

                var result = Broker.Apply (
                    assets,
                    companies.Select(GetTransactionAmount).ToList(),
                    companies);

                switch (result)
                {
                    case (Assets newAssets, TransactionResult.Ok):
                        return newAssets;
                    case (_, TransactionResult.Oversold):
                        View.ShowOversold();
                        break;
                    case (Assets newAssets, TransactionResult.Overspent):
                        View.ShowOverspent(-newAssets.Cash);
                        break;
                }
            }
        }

        /// <summary>
        /// 获取给定公司的交易金额。
        /// </summary>
        /// <param name="company">
        /// 要买入或卖出的公司。
        /// </param>
        /// <returns>
        /// 要买入或卖出的股票数量。
        /// </returns>
        public static int GetTransactionAmount(Company company)
        {
            while (true)
            {
                View.PromptBuySellCompany(company);

                var input = Console.ReadLine();
                if (input is null)
                    Environment.Exit(0);
                else
                if (!Int32.TryParse(input, out var amount))
                    View.PromptValidInteger();
                else
                    return amount;
            }
        }
    }
}

```