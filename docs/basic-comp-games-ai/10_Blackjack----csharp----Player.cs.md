# `basic-computer-games\10_Blackjack\csharp\Player.cs`

```py
namespace Blackjack
{
    // 定义一个名为 Player 的命名空间
    public class Player
    {
        // 构造函数，初始化玩家的索引、名称、手牌和第二手牌
        public Player(int index)
        {
            // 设置玩家的索引
            Index = index;
            // 设置玩家的名称为索引加一的字符串形式
            Name = (index + 1).ToString();
            // 初始化玩家的手牌
            Hand = new Hand();
            // 初始化玩家的第二手牌
            SecondHand = new Hand();
        }

        // 玩家的索引属性，只读
        public int Index { get; private set; }

        // 玩家的名称属性，只读
        public string Name { get; private set; }

        // 玩家的手牌属性，只读
        public Hand Hand { get; private set; }

        // 玩家的第二手牌属性，只读
        public Hand SecondHand { get; private set;}

        // 玩家在本轮下注的筹码数，可读写
        public int RoundBet { get; set; }

        // 玩家在本轮赢得的筹码数，可读写
        public int RoundWinnings { get; set; }

        // 玩家总共赢得的筹码数，可读写
        public int TotalWinnings { get; set; }
    }
}
```