# `basic-computer-games\83_Stock_Market\csharp\Company.cs`

```

// 命名空间声明，表示代码所属的命名空间
namespace Game
{
    /// <summary>
    /// 表示一个公司。
    /// </summary>
    // 定义一个记录类型 Company
    public record Company
    {
        /// <summary>
        /// 获取公司的名称。
        /// </summary>
        // 公开的字符串属性 Name，用于获取公司名称
        public string Name { get; }

        /// <summary>
        /// 获取公司的三个字母的股票代码。
        /// </summary>
        // 公开的字符串属性 StockSymbol，用于获取公司的股票代码
        public string StockSymbol { get; }

        /// <summary>
        /// 获取公司的当前股价。
        /// </summary>
        // 公开的双精度浮点数属性 SharePrice，用于获取公司的股价，并且可以进行初始化
        public double SharePrice { get; init; }

        /// <summary>
        /// 初始化一个新的 Company 记录。
        /// </summary>
        // 公开的构造函数，用于初始化 Company 记录
        public Company(string name, string stockSymbol, double sharePrice) =>
            // 使用元组赋值语法初始化属性值
            (Name, StockSymbol, SharePrice) = (name, stockSymbol, sharePrice);
    }
}

```