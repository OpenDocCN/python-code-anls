# `basic-computer-games\83_Stock_Market\csharp\TransactionResult.cs`

```

// 命名空间 Game，用于组织和管理相关的类和枚举
namespace Game
{
    /// <summary>
    /// 枚举了应用交易时可能出现的不同结果
    /// </summary>
    public enum TransactionResult
    {
        /// <summary>
        /// 交易成功
        /// </summary>
        Ok,

        /// <summary>
        /// 交易失败，因为卖家试图出售比其拥有的股份更多的股份
        /// </summary>
        Oversold,

        /// <summary>
        /// 交易失败，因为净成本大于卖家可用现金
        /// </summary>
        Overspent
    }
}

```