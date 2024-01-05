# `83_Stock_Market\csharp\Broker.cs`

```
/// <summary>
        /// Applies the given set of transactions to the given set of assets.
        /// </summary>
        /// <param name="assets">
        /// The assets to update.
        /// </param>
        /// <param name="transactions">
        /// The set of stocks to purchase or sell.  Positive values indicate
        /// purchases, negative values indicate sales.
        /// </param>
        public static void ApplyTransactions(ref Dictionary<string, int> assets, IEnumerable<KeyValuePair<string, int>> transactions)
        {
            // Loop through each transaction
            foreach (var transaction in transactions)
            {
                // If the asset exists in the dictionary, update the quantity
                if (assets.ContainsKey(transaction.Key))
                {
                    assets[transaction.Key] += transaction.Value;
                }
                // If the asset does not exist, add it to the dictionary
                else
                {
                    assets.Add(transaction.Key, transaction.Value);
                }
            }
        }
    }
}
        /// <summary>
        /// Apply the transactions to the assets of the seller.
        /// </summary>
        /// <param name="assets">
        /// The current assets of the seller.
        /// </param>
        /// <param name="transactions">
        /// The collection of transaction amounts, where positive values indicate purchases and negative values indicate sales.
        /// </param>
        /// <param name="companies">
        /// The collection of companies.
        /// </param>
        /// <returns>
        /// Returns the seller's new assets and a code indicating the result of the transaction.
        /// </returns>
        public static (Assets newAssets, TransactionResult result) Apply(Assets assets, IEnumerable<int> transactions, IEnumerable<Company> companies)
        {
            // Calculate the net cost and total transaction size by combining the transaction amounts with the share prices of the corresponding companies
            var (netCost, transactionSize) = Enumerable.Zip(
                    transactions,
                    companies,
                    (amount, company) => (amount * company.SharePrice))
                .Aggregate(
                    (netCost: 0.0, transactionSize: 0.0),
                    (accumulated, amount) => (accumulated.netCost + amount, accumulated.transactionSize + Math.Abs(amount)));

            // Calculate the brokerage fee as 1% of the total transaction size
            var brokerageFee = 0.01 * transactionSize;
# 创建一个新的资产对象，其中现金减去净成本和经纪费，投资组合更新为原有股票数量加上交易数量
var newAssets = assets with
{
    Cash      = assets.Cash - netCost - brokerageFee,
    Portfolio = ImmutableArray.CreateRange(Enumerable.Zip(
        assets.Portfolio,
        transactions,
        (sharesOwned, delta) => sharesOwned + delta))
};

# 如果新的投资组合中有任何股票数量小于0，则返回（新资产对象，交易结果为过度卖出）
if (newAssets.Portfolio.Any(amount => amount < 0))
    return (newAssets, TransactionResult.Oversold);
# 如果新的现金数量小于0，则返回（新资产对象，交易结果为过度支出）
else if (newAssets.Cash < 0)
    return (newAssets, TransactionResult.Overspent);
# 否则返回（新资产对象，交易结果为正常）
else
    return (newAssets, TransactionResult.Ok);
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```