# `basic-computer-games\71_Poker\csharp\Players\Computer.cs`

```py
{
    // 使用 Poker.Cards 命名空间中的类
    using Poker.Cards;
    // 使用 Poker.Strategies 命名空间中的类
    using Poker.Strategies;
    // 使用 System.StringComparison 枚举中的成员
    using static System.StringComparison;

    // 定义 Poker.Players 命名空间中的内部类 Computer，继承自 Player 类
    internal class Computer : Player
    {
        // 声明只读字段 _io，类型为 IReadWrite 接口
        private readonly IReadWrite _io;
        // 声明只读字段 _random，类型为 IRandom 接口
        private readonly IRandom _random;

        // 构造函数，接受 bank、io、random 三个参数
        public Computer(int bank, IReadWrite io, IRandom random)
            : base(bank)
        {
            // 初始化 _io 字段
            _io = io;
            // 初始化 _random 字段
            _random = random;
            // 设置 Strategy 属性为 Strategy.None
            Strategy = Strategy.None;
        }

        // 公共属性 Strategy，可读可写
        public Strategy Strategy { get; set; }

        // 重写父类的 NewHand 方法
        public override void NewHand()
        {
            // 调用父类的 NewHand 方法
            base.NewHand();

            // 根据不同的条件设置 Strategy 属性的值
            Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
            {
                // 当 Hand.IsWeak 为 true 且随机数小于 2 时，设置 Strategy 为 Bluff(23, 0b11100)
                (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11100),
                // 当 Hand.IsWeak 为 true 且随机数小于 2 时，设置 Strategy 为 Bluff(23, 0b11110)
                (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11110),
                // 当 Hand.IsWeak 为 true 且随机数小于 1 时，设置 Strategy 为 Bluff(23, 0b11111)
                (true, _, _) when _random.Next(10) < 1 => Strategy.Bluff(23, 0b11111),
                // 当 Hand.IsWeak 为 true 时，设置 Strategy 为 Fold
                (true, _, _) => Strategy.Fold,
                // 当 Hand.IsWeak 为 false 且 Hand.Rank < HandRank.Three 为 true 时，根据随机数设置 Strategy
                (false, true, _) => _random.Next(10) < 2 ? Strategy.Bluff(23) : Strategy.Check,
                // 当 Hand.IsWeak 为 false 且 Hand.Rank < HandRank.FullHouse 为 true 时，设置 Strategy 为 Bet(35)
                (false, false, true) => Strategy.Bet(35),
                // 当 Hand.IsWeak 为 false 且 Hand.Rank < HandRank.FullHouse 为 false 时，根据随机数设置 Strategy
                (false, false, false) => _random.Next(10) < 1 ? Strategy.Bet(35) : Strategy.Raise
            };
        }

        // 重写父类的 DrawCards 方法
        protected override void DrawCards(Deck deck)
    {
        // 获取保留的牌的掩码，如果没有指定则使用默认的手牌保留掩码
        var keepMask = Strategy.KeepMask ?? Hand.KeepMask;
        // 计数器初始化
        var count = 0;
        // 遍历5张牌
        for (var i = 1; i <= 5; i++)
        {
            // 如果该位置的牌不需要保留
            if ((keepMask & (1 << (i - 1))) == 0)
            {
                // 替换手牌中不需要保留的牌
                Hand = Hand.Replace(i, deck.DealCard());
                // 计数器加一
                count++;
            }
        }

        // 输出空行
        _io.WriteLine();
        // 输出取牌的数量
        _io.Write($"I am taking {count} card");
        // 如果取牌数量不为1，则输出复数形式
        if (count != 1)
        {
            _io.WriteLine("s");
        }

        // 根据手牌的强度和牌型选择策略
        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            // 如果当前策略是 Bluff，则使用 Bluff 策略并设定参数
            _ when Strategy is Bluff => Strategy.Bluff(28),
            // 如果手牌弱，则选择 Fold 策略
            (true, _, _) => Strategy.Fold,
            // 如果手牌不弱且牌型小于 Three of a Kind，则根据随机数选择 Bet 或 Raise 策略
            (false, true, _) => _random.Next(10) == 0 ? Strategy.Bet(19) : Strategy.Raise,
            // 如果手牌不弱且牌型小于 Full House，则根据随机数选择 Bet 或 Raise 策略
            (false, false, true) => _random.Next(10) == 0 ? Strategy.Bet(11) : Strategy.Bet(19),
            // 如果手牌不弱且牌型大于 Full House，则选择 Raise 策略
            (false, false, false) => Strategy.Raise
        };
    }

    // 获取赌注
    public int GetWager(int wager)
    {
        // 随机增加赌注
        wager += _random.Next(10);
        // 如果余额不足以跟注
        if (Balance < Table.Human.Bet + wager)
        {
            // 如果对手没有下注，则全押
            if (Table.Human.Bet == 0) { return Balance; }
            // 如果余额足够跟注，则跟注并收集赌注
            if (Balance >= Table.Human.Bet)
            {
                _io.WriteLine("I'll see you.");
                Bet = Table.Human.Bet;
                Table.CollectBets();
            }
            // 如果余额不足以跟注，则筹码不够，需要增加筹码
            else
            {
                RaiseFunds();
            }
        }

        return wager;
    }

    // 尝试购买手表
    {
        // 如果玩家没有手表，返回 false
        if (!Table.Human.HasWatch) { return false; }

        // 询问玩家是否想要卖掉手表
        var response = _io.ReadString("Would you like to sell your watch");
        // 如果回答以 "N" 开头（不区分大小写），返回 false
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return false; }

        // 根据随机数生成的值，决定购买手表的价格和消息
        var (value, message) = (_random.Next(10) < 7) switch
        {
            true => (75, "I'll give you $75 for it."),
            false => (25, "That's a pretty crummy watch - I'll give you $25.")
        };

        // 输出消息
        _io.WriteLine(message);
        // 玩家卖掉手表
        Table.Human.SellWatch(value);
        // 原始代码中没有计算机部分有任何钱

        // 返回 true
        return true;
    }

    // 筹集资金
    public void RaiseFunds()
    {
        // 如果玩家没有手表，直接返回
        if (Table.Human.HasWatch) { return; }

        // 询问玩家是否愿意以 $50 的价格买回手表
        var response = _io.ReadString("Would you like to buy back your watch for $50");
        // 如果回答以 "N" 开头（不区分大小写），直接返回
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return; }

        // 原始代码中没有从玩家扣除 $50
        // 余额增加 $50
        Balance += 50;
        // 玩家收回手表
        Table.Human.ReceiveWatch();
        // 破产状态设为 true
        IsBroke = true;
    }

    // 检查资金
    public void CheckFunds() { IsBroke = Balance <= Table.Ante; }

    // 赢得游戏
    public override void TakeWinnings()
    {
        // 输出消息 "I win."
        _io.WriteLine("I win.");
        // 调用基类的 TakeWinnings 方法
        base.TakeWinnings();
    }
# 闭合前面的函数定义
```