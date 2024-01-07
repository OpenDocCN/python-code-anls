# `basic-computer-games\71_Poker\csharp\Players\Computer.cs`

```

// 导入所需的命名空间
using Poker.Cards;
using Poker.Strategies;
using static System.StringComparison;

// 创建一个名为Computer的类，继承自Player类
internal class Computer : Player
{
    // 声明私有字段_io和_random
    private readonly IReadWrite _io;
    private readonly IRandom _random;

    // 创建Computer类的构造函数，接受bank、io和random作为参数
    public Computer(int bank, IReadWrite io, IRandom random)
        : base(bank)
    {
        _io = io;
        _random = random;
        Strategy = Strategy.None;
    }

    // 创建公共属性Strategy
    public Strategy Strategy { get; set; }

    // 重写父类的NewHand方法
    public override void NewHand()
    {
        base.NewHand();

        // 根据不同的手牌情况选择不同的策略
        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            // 根据随机数选择Bluff策略
            (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11100),
            // 根据随机数选择Bluff策略
            (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11110),
            // 根据随机数选择Bluff策略
            (true, _, _) when _random.Next(10) < 1 => Strategy.Bluff(23, 0b11111),
            // 选择Fold策略
            (true, _, _) => Strategy.Fold,
            // 根据随机数选择Bluff或Check策略
            (false, true, _) => _random.Next(10) < 2 ? Strategy.Bluff(23) : Strategy.Check,
            // 选择Bet策略
            (false, false, true) => Strategy.Bet(35),
            // 根据随机数选择Bet或Raise策略
            (false, false, false) => _random.Next(10) < 1 ? Strategy.Bet(35) : Strategy.Raise
        };
    }

    // 重写父类的DrawCards方法
    protected override void DrawCards(Deck deck)
    {
        // 根据策略选择保留哪些牌
        var keepMask = Strategy.KeepMask ?? Hand.KeepMask;
        var count = 0;
        for (var i = 1; i <= 5; i++)
        {
            // 根据keepMask决定是否保留牌
            if ((keepMask & (1 << (i - 1))) == 0)
            {
                Hand = Hand.Replace(i, deck.DealCard());
                count++;
            }
        }

        _io.WriteLine();
        _io.Write($"I am taking {count} card");
        if (count != 1)
        {
            _io.WriteLine("s");
        }

        // 根据不同的手牌情况选择不同的策略
        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            _ when Strategy is Bluff => Strategy.Bluff(28),
            (true, _, _) => Strategy.Fold,
            (false, true, _) => _random.Next(10) == 0 ? Strategy.Bet(19) : Strategy.Raise,
            (false, false, true) => _random.Next(10) == 0 ? Strategy.Bet(11) : Strategy.Bet(19),
            (false, false, false) => Strategy.Raise
        };
    }

    // 根据传入的wager计算下注金额
    public int GetWager(int wager)
    {
        wager += _random.Next(10);
        if (Balance < Table.Human.Bet + wager)
        {
            if (Table.Human.Bet == 0) { return Balance; }

            if (Balance >= Table.Human.Bet)
            {
                _io.WriteLine("I'll see you.");
                Bet = Table.Human.Bet;
                Table.CollectBets();
            }
            else
            {
                RaiseFunds();
            }
        }

        return wager;
    }

    // 尝试购买手表
    public bool TryBuyWatch()
    {
        if (!Table.Human.HasWatch) { return false; }

        var response = _io.ReadString("Would you like to sell your watch");
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return false; }

        var (value, message) = (_random.Next(10) < 7) switch
        {
            true => (75, "I'll give you $75 for it."),
            false => (25, "That's a pretty crummy watch - I'll give you $25.")
        };

        _io.WriteLine(message);
        Table.Human.SellWatch(value);

        return true;
    }

    // 筹集资金
    public void RaiseFunds()
    {
        if (Table.Human.HasWatch) { return; }

        var response = _io.ReadString("Would you like to buy back your watch for $50");
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return; }

        Balance += 50;
        Table.Human.ReceiveWatch();
        IsBroke = true;
    }

    // 检查资金是否足够
    public void CheckFunds() { IsBroke = Balance <= Table.Ante; }

    // 重写父类的TakeWinnings方法
    public override void TakeWinnings()
    {
        _io.WriteLine("I win.");
        base.TakeWinnings();
    }
}

```