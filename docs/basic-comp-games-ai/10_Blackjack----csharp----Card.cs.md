# `basic-computer-games\10_Blackjack\csharp\Card.cs`

```

namespace Blackjack
{
    // 定义一个名为 Card 的类
    public class Card
    {
        // 定义一个只读的字符串数组，包含扑克牌的名称
        private static readonly string[] _names = new[] {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};

        // 构造函数，接受一个整数参数作为索引
        public Card(int index)
        {
            Index = index;
        }

        // 索引属性，只能在类内部进行设置
        public int Index { get; private set; }

        // 名称属性，根据索引返回对应的扑克牌名称
        public string Name => _names[Index];

        // 不定冠词属性，根据索引返回对应的不定冠词
        public string IndefiniteArticle => (Index == 0 || Index == 7) ? "an" : "a";

        // 是否为 Ace 的属性，判断当前扑克牌是否为 Ace
        public bool IsAce => Index == 0;

        // 值属性，根据扑克牌的索引返回对应的点数值
        public int Value
        {
            get
            {
                if (IsAce)
                    return 11;
                if (Index > 8)
                    return 10;
                return Index + 1;
            }
        }
    }
}

```