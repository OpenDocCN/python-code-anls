# `basic-computer-games\83_Stock_Market\csharp\TransactionResult.cs`

```py
// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 枚举了应用交易的不同可能结果
    /// </summary>
    public enum TransactionResult
    {
        /// <summary>
        /// 交易成功
        /// </summary>
        Ok,

        /// <summary>
        /// 交易失败，因为卖家试图出售比自己拥有的股份更多的股份
        /// </summary>
        Oversold,

        /// <summary>
        /// 交易失败，因为净成本大于卖家可用现金
        /// </summary>
        Overspent
    }
}
```