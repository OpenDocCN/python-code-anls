# `basic-computer-games\83_Stock_Market\csharp\Broker.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 包含用于交换资产的函数
    /// </summary>
    // 创建名为 Broker 的静态类
    public static class Broker
    {
        /// <summary>
        /// Applies the given set of transactions to the given set of assets.
        /// </summary>
        /// <param name="assets">
        /// The assets to update.
        /// </param>
        /// <param name="transactions">
        /// The set of stocks to purchase or sell.  Positive values indicate
        /// purchaes and negative values indicate sales.
        /// </param>
        /// <param name="companies">
        /// The collection of companies.
        /// </param>
        /// <returns>
        /// Returns the sellers new assets and a code indicating the result
        /// of the transaction.
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
    
            // 计算交易的佣金费用
            var brokerageFee = 0.01 * transactionSize;
    
            // 更新资产信息
            var newAssets = assets with
            {
                Cash      = assets.Cash - netCost - brokerageFee,
                Portfolio = ImmutableArray.CreateRange(Enumerable.Zip(
                    assets.Portfolio,
                    transactions,
                    (sharesOwned, delta) => sharesOwned + delta))
            };
    
            // 检查是否有超卖的情况
            if (newAssets.Portfolio.Any(amount => amount < 0))
                return (newAssets, TransactionResult.Oversold);
            else
            // 检查是否有超支的情况
            if (newAssets.Cash < 0)
                return (newAssets, TransactionResult.Overspent);
            else
                return (newAssets, TransactionResult.Ok);
        }
    }
# 闭合前面的函数定义
```