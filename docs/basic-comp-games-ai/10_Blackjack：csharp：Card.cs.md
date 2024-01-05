# `d:/src/tocomm/basic-computer-games\10_Blackjack\csharp\Card.cs`

```
namespace Blackjack
{
    public class Card
    {
        // 定义一个只读的字符串数组，包含扑克牌的名称
        private static readonly string[] _names = new[] {"A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};

        // 构造函数，接受一个索引参数
        public Card(int index)
        {
            Index = index;
        }

        // 索引属性，只能在类内部设置
        public int Index { get; private set; }

        // 名称属性，根据索引返回对应的扑克牌名称
        public string Name => _names[Index];

        // 不定冠词属性，根据索引返回对应的不定冠词
        public string IndefiniteArticle => (Index == 0 || Index == 7) ? "an" : "a";

        // 是否为Ace牌属性，根据索引判断是否为Ace牌
        public bool IsAce => Index == 0;

        // 值属性，根据索引返回对应的扑克牌点数
        {
            # 如果牌是 A，则返回 11
            get
            {
                if (IsAce)
                    return 11;
                # 如果牌面值大于 8，则返回 10
                if (Index > 8)
                    return 10;
                # 否则返回牌面值加1
                return Index + 1;
            }
        }
    }
}
```