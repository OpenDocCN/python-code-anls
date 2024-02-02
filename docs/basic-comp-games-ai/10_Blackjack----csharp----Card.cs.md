# `basic-computer-games\10_Blackjack\csharp\Card.cs`

```py
namespace Blackjack
{
    // 定义一个扑克牌类
    public class Card
    {
        // 定义一个只读的扑克牌名称数组
        private static readonly string[] _names = new[] {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};

        // 构造函数，初始化扑克牌的索引
        public Card(int index)
        {
            Index = index;
        }

        // 扑克牌的索引属性
        public int Index { get; private set; }

        // 获取扑克牌的名称
        public string Name => _names[Index];

        // 获取扑克牌的不定冠词
        public string IndefiniteArticle => (Index == 0 || Index == 7) ? "an" : "a";

        // 判断扑克牌是否为Ace
        public bool IsAce => Index == 0;

        // 获取扑克牌的点数
        public int Value
        {
            get
            {
                // 如果是Ace，则返回11
                if (IsAce)
                    return 11;
                // 如果是10、J、Q、K，则返回10
                if (Index > 8)
                    return 10;
                // 否则返回索引加1
                return Index + 1;
            }
        }
    }
}
```