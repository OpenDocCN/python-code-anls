# `10_Blackjack\csharp\Player.cs`

```
namespace Blackjack
{
    public class Player
    {
        // 构造函数，初始化玩家的索引
        public Player(int index)
        {
            Index = index;
            // 根据索引设置玩家的名字
            Name = (index + 1).ToString();
            // 初始化玩家的手牌
            Hand = new Hand();
            // 初始化玩家的第二手牌
            SecondHand = new Hand();
        }

        // 玩家的索引属性
        public int Index { get; private set; }

        // 玩家的名字属性
        public string Name { get; private set; }

        // 玩家的手牌属性
        public Hand Hand { get; private set; }

        // 玩家的第二手牌属性
        public Hand SecondHand { get; private set;}
# 定义一个公共整型属性 RoundBet，用于存储每轮的下注金额
public int RoundBet { get; set; }

# 定义一个公共整型属性 RoundWinnings，用于存储每轮的赢取金额
public int RoundWinnings { get; set; }

# 定义一个公共整型属性 TotalWinnings，用于存储总的赢取金额
public int TotalWinnings { get; set; }
```