# `basic-computer-games\83_Stock_Market\csharp\Broker.cs`

```

// 引入必要的命名空间
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

// 创建名为 Broker 的静态类
namespace Game
{
    /// <summary>
    /// 包含用于交换资产的函数。
    /// </summary>
    public static class Broker
    {
        /// <summary>
        /// 将给定的交易集应用于给定的资产集。
        /// </summary>
        /// <param name="assets">
        /// 要更新的资产。
        /// </param>
        /// <param name="transactions">
        /// 要购买或出售的股票集。正值表示购买，负值表示销售。
        /// </param>
        /// <param name="companies">
        /// 公司集合。
        /// </param>
        /// <returns>
        /// 返回卖方的新资产和表示交易结果的代码。
        /// </returns>
        public static (Assets newAssets, TransactionResult result) Apply(Assets assets, IEnumerable<int> transactions, IEnumerable<Company> companies)
        {
            // 计算交易的净成本和交易规模
            var (netCost, transactionSize) = Enumerable.Zip(
                    transactions,
                    companies,
                    (amount, company) => (amount * company.SharePrice))
                .Aggregate(
                    (netCost: 0.0, transactionSize: 0.0),
                    (accumulated, amount) => (accumulated.netCost + amount, accumulated.transactionSize + Math.Abs(amount)));

            // 计算经纪费
            var brokerageFee = 0.01 * transactionSize;

            // 更新资产
            var newAssets = assets with
            {
                Cash      = assets.Cash - netCost - brokerageFee,
                Portfolio = ImmutableArray.CreateRange(Enumerable.Zip(
                    assets.Portfolio,
                    transactions,
                    (sharesOwned, delta) => sharesOwned + delta))
            };

            // 检查是否有超卖或超支
            if (newAssets.Portfolio.Any(amount => amount < 0))
                return (newAssets, TransactionResult.Oversold);
            else if (newAssets.Cash < 0)
                return (newAssets, TransactionResult.Overspent);
            else
                return (newAssets, TransactionResult.Ok);
        }
    }
}

```