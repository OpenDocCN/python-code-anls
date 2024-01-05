# `83_Stock_Market\csharp\Company.cs`

```
        public decimal SharePrice { get; }

        /// <summary>
        /// Initializes a new instance of the Company class.
        /// </summary>
        /// <param name="name">The company's name.</param>
        /// <param name="stockSymbol">The company's stock symbol.</param>
        /// <param name="sharePrice">The company's share price.</param>
        public Company(string name, string stockSymbol, decimal sharePrice)
        {
            // 设置公司名称
            Name = name;
            // 设置公司股票代码
            StockSymbol = stockSymbol;
            // 设置公司股价
            SharePrice = sharePrice;
        }
    }
}
        public double SharePrice { get; init; }  // 定义一个公共的双精度类型属性SharePrice，可以被获取和初始化

        /// <summary>
        /// Initializes a new Company record.
        /// </summary>
        public Company(string name, string stockSymbol, double sharePrice) =>
            (Name, StockSymbol, SharePrice) = (name, stockSymbol, sharePrice);  // 创建一个Company类的构造函数，用于初始化公司的名称、股票代码和股价
    }
}
```