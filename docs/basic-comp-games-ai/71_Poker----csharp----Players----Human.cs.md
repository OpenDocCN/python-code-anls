# `basic-computer-games\71_Poker\csharp\Players\Human.cs`

```py
// 引入扑克牌和策略命名空间
using Poker.Cards;
using Poker.Strategies;

// 定义 Human 类，继承自 Player 类
namespace Poker.Players
{
    // 声明 Human 类为内部类
    internal class Human : Player
    {
        // 声明私有字段 _io，类型为 IReadWrite 接口
        private readonly IReadWrite _io;

        // Human 类的构造函数，接受银行金额和 IReadWrite 接口实例作为参数
        public Human(int bank, IReadWrite io)
            : base(bank)
        {
            // 设置 HasWatch 属性为 true
            HasWatch = true;
            // 将传入的 io 参数赋值给 _io 字段
            _io = io;
        }

        // 声明 HasWatch 属性，可读可写
        public bool HasWatch { get; set; }

        // 重写父类的 DrawCards 方法，接受 Deck 对象作为参数
        protected override void DrawCards(Deck deck)
        {
            // 从玩家输入中读取要抽取的卡片数量
            var count = _io.ReadNumber("How many cards do you want", 3, "You can't draw more than three cards.");
            // 如果 count 为 0，则直接返回
            if (count == 0) { return; }

            // 输出提示信息，要求玩家输入要抽取的卡片的数字
            _io.WriteLine("What are their numbers:");
            // 循环抽取卡片，并替换手中的卡片
            for (var i = 1; i <= count; i++)
            {
                Hand = Hand.Replace((int)_io.ReadNumber(), deck.DealCard());
            }

            // 输出玩家新的手牌
            _io.WriteLine("Your new hand:");
            _io.Write(Hand);
        }

        // 设置赌注的方法
        internal bool SetWager()
        {
            // 从玩家输入中读取策略
            var strategy = _io.ReadHumanStrategy(Table.Computer.Bet == 0 && Bet == 0);
            // 如果策略是 Bet 或者 Check
            if (strategy is Strategies.Bet or Check)
            {
                // 如果下注加上策略值小于电脑的下注
                if (Bet + strategy.Value < Table.Computer.Bet)
                {
                    // 输出提示信息，要求玩家跟注或者弃牌
                    _io.WriteLine("If you can't see my bet, then fold.");
                    return false;
                }
                // 如果余额减去下注和策略值大于等于 0
                if (Balance - Bet - strategy.Value >= 0)
                {
                    // 设置 HasBet 为 true，增加下注金额，并返回 true
                    HasBet = true;
                    Bet += strategy.Value;
                    return true;
                }
                // 调用 RaiseFunds 方法
                RaiseFunds();
            }
            else
            {
                // 弃牌，并收集赌注
                Fold();
                Table.CollectBets();
            }
            return false;
        }

        // 增加资金的方法
        public void RaiseFunds()
        {
            // 输出空行和提示信息
            _io.WriteLine();
            _io.WriteLine("You can't bet with what you haven't got.");

            // 如果电脑尝试购买手表成功，则直接返回
            if (Table.Computer.TryBuyWatch()) { return; }

            // 标记玩家破产
            IsBroke = true;
        }

        // 收到手表的方法
        public void ReceiveWatch()
        {
            // 在原始代码中，玩家收到手表时不需要支付任何费用
            // 设置 HasWatch 为 true
            HasWatch = true;
        }
    }
}
    # 结束 SellWatch 方法的定义
    }

    # 出售手表的方法，接受一个参数 amount
    public void SellWatch(int amount)
    {
        # 将 HasWatch 设为 false，表示手表已售出
        HasWatch = false;
        # 增加余额
        Balance += amount;
    }

    # 重写 TakeWinnings 方法
    public override void TakeWinnings()
    {
        # 输出 "You win." 到输出流
        _io.WriteLine("You win.");
        # 调用基类的 TakeWinnings 方法
        base.TakeWinnings();
    }
# 闭合前面的函数定义
```