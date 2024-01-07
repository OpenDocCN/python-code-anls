# `basic-computer-games\71_Poker\csharp\Players\Human.cs`

```

// 引入扑克牌和策略相关的命名空间
using Poker.Cards;
using Poker.Strategies;

// 定义一个名为 Human 的内部类，继承自 Player 类
namespace Poker.Players
{
    internal class Human : Player
    {
        // 声明一个只读的 IReadWrite 接口类型的字段 _io
        private readonly IReadWrite _io;

        // 构造函数，初始化 Human 类的实例
        public Human(int bank, IReadWrite io)
            : base(bank)
        {
            // 设置 HasWatch 属性为 true
            HasWatch = true;
            // 将传入的 io 参数赋值给 _io 字段
            _io = io;
        }

        // 声明一个公共的属性 HasWatch
        public bool HasWatch { get; set; }

        // 重写父类的 DrawCards 方法
        protected override void DrawCards(Deck deck)
        {
            // 从玩家输入中读取要抽取的牌的数量
            var count = _io.ReadNumber("How many cards do you want", 3, "You can't draw more than three cards.");
            // 如果 count 为 0，则直接返回
            if (count == 0) { return; }

            // 输出提示信息
            _io.WriteLine("What are their numbers:");
            // 循环读取玩家输入的牌号，并替换手中的牌
            for (var i = 1; i <= count; i++)
            {
                Hand = Hand.Replace((int)_io.ReadNumber(), deck.DealCard());
            }

            // 输出玩家新的手牌
            _io.WriteLine("Your new hand:");
            _io.Write(Hand);
        }

        // 内部方法，设置赌注
        internal bool SetWager()
        {
            // 从玩家输入中读取策略
            var strategy = _io.ReadHumanStrategy(Table.Computer.Bet == 0 && Bet == 0);
            // 如果策略是 Bet 或者 Check
            if (strategy is Strategies.Bet or Check)
            {
                // 如果下注加上策略值小于电脑的赌注，则输出提示信息并返回 false
                if (Bet + strategy.Value < Table.Computer.Bet)
                {
                    _io.WriteLine("If you can't see my bet, then fold.");
                    return false;
                }
                // 如果余额减去下注和策略值大于等于 0，则设置 HasBet 为 true，增加赌注，并返回 true
                if (Balance - Bet - strategy.Value >= 0)
                {
                    HasBet = true;
                    Bet += strategy.Value;
                    return true;
                }
                // 调用 RaiseFunds 方法
                RaiseFunds();
            }
            else
            {
                // 调用 Fold 方法，收集赌注
                Fold();
                Table.CollectBets();
            }
            return false;
        }

        // 增加资金的方法
        public void RaiseFunds()
        {
            // 输出提示信息
            _io.WriteLine();
            _io.WriteLine("You can't bet with what you haven't got.");

            // 如果电脑尝试购买手表成功，则直接返回
            if (Table.Computer.TryBuyWatch()) { return; }

            // 设置 IsBroke 为 true
            IsBroke = true;
        }

        // 接收手表的方法
        public void ReceiveWatch()
        {
            // 将 HasWatch 属性设置为 true
            HasWatch = true;
        }

        // 出售手表的方法
        public void SellWatch(int amount)
        {
            // 将 HasWatch 属性设置为 false，增加余额
            HasWatch = false;
            Balance += amount;
        }

        // 重写父类的 TakeWinnings 方法
        public override void TakeWinnings()
        {
            // 输出提示信息
            _io.WriteLine("You win.");
            // 调用父类的 TakeWinnings 方法
            base.TakeWinnings();
        }
    }
}

```