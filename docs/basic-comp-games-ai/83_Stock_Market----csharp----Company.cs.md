# `basic-computer-games\83_Stock_Market\csharp\Company.cs`

```
// 命名空间声明，表示代码所属的命名空间
namespace Game
{
    /// <summary>
    /// 表示一个公司
    /// </summary>
    // 定义一个记录类型，表示一个公司
    public record Company
    {
        /// <summary>
        /// 获取公司的名称
        /// </summary>
        // 公司的名称属性
        public string Name { get; }

        /// <summary>
        /// 获取公司的三个字母股票代码
        /// </summary>
        // 公司的股票代码属性
        public string StockSymbol { get; }

        /// <summary>
        /// 获取公司的当前股价
        /// </summary>
        // 公司的股价属性，可以初始化但只读
        public double SharePrice { get; init; }

        /// <summary>
        /// 初始化一个新的公司记录
        /// </summary>
        // 公司记录的构造函数，接受名称、股票代码和股价作为参数
        public Company(string name, string stockSymbol, double sharePrice) =>
            // 使用元组赋值语法初始化属性
            (Name, StockSymbol, SharePrice) = (name, stockSymbol, sharePrice);
    }
}
```