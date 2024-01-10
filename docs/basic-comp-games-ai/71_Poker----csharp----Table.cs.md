# `basic-computer-games\71_Poker\csharp\Table.cs`

```
# 引入所需的命名空间
using Poker.Cards;
using Poker.Players;
using Poker.Strategies;

# 定义名为 Table 的内部类
namespace Poker;

internal class Table
{
    # 声明私有只读字段 _io 和 _random
    private readonly IReadWrite _io;
    private readonly IRandom _random;
    # 声明公共整型字段 Pot
    public int Pot;

    # 定义构造函数，接受 io、random、deck、human、computer 五个参数
    public Table(IReadWrite io, IRandom random, Deck deck, Human human, Computer computer)
    {
        # 初始化 _io 和 _random 字段
        _io = io;
        _random = random;
        # 初始化 Deck、Human、Computer 属性
        Deck = deck;
        Human = human;
        Computer = computer;

        # 让 human 和 computer 分别加入到当前牌桌
        human.Sit(this);
        computer.Sit(this);
    }

    # 声明只读整型属性 Ante，并初始化为 5
    public int Ante { get; } = 5;
    # 声明 Deck、Human、Computer 属性
    public Deck Deck { get; }
    public Human Human { get; }
    public Computer Computer { get; }

    # 定义 PlayHand 方法
    internal void PlayHand()
    {
        # 循环执行以下代码块
        while (true)
        {
            # 输出空行
            _io.WriteLine();
            # 检查 computer 的资金
            Computer.CheckFunds();
            # 如果 computer 破产，则返回
            if (Computer.IsBroke) { return; }

            # 输出当前底注
            _io.WriteLine($"The ante is ${Ante}.  I will deal:");
            _io.WriteLine();
            # 如果 human 的余额小于等于底注，则增加资金
            if (Human.Balance <= Ante)
            {
                Human.RaiseFunds();
                # 如果 human 破产，则返回
                if (Human.IsBroke) { return; }
            }

            # 发牌
            Deal(_random);

            # 输出空行
            _io.WriteLine();
            # 获取赌注
            GetWagers("I'll open with ${0}", "I check.", allowRaiseAfterCheck: true);
            # 如果有人破产或有人弃牌，则返回
            if (SomeoneIsBroke() || SomeoneHasFolded()) { return; }

            # 换牌
            Draw();

            # 获取赌注
            GetWagers();
            # 如果有人破产，则返回
            if (SomeoneIsBroke()) { return; }
            # 如果 human 没有下注，则获取赌注
            if (!Human.HasBet)
            {
                GetWagers("I'll bet ${0}", "I'll check");
            }
            # 如果有人破产或有人弃牌，则返回
            if (SomeoneIsBroke() || SomeoneHasFolded()) { return; }
            # 获取赢家，并将赢家的赌注加入到其余额中
            if (GetWinner() is { } winner)
            {
                winner.TakeWinnings();
                return;
            }
        }
    }

    # 定义 Deal 方法，接受 random 参数
    private void Deal(IRandom random)
    {
        # 洗牌
        Deck.Shuffle(random);

        # 计算当前奖池的金额
        Pot = Human.AnteUp() + Computer.AnteUp();

        # human 和 computer 分别获得新的手牌
        Human.NewHand();
        Computer.NewHand();

        # 输出"Your hand:"，并显示 human 的手牌
        _io.WriteLine("Your hand:");
        _io.Write(Human.Hand);
    }

    # 定义 Draw 方法
    {
        # 输出空行
        _io.WriteLine();
        # 输出"Now we draw -- "
        _io.Write("Now we draw -- ");
        # 人类玩家抽牌
        Human.DrawCards();
        # 电脑玩家抽牌
        Computer.DrawCards();
        # 输出空行
        _io.WriteLine();
    }

    # 获取赌注
    private void GetWagers(string betFormat, string checkMessage, bool allowRaiseAfterCheck = false)
    {
        # 如果电脑玩家的策略是下注
        if (Computer.Strategy is Bet)
        {
            # 电脑玩家下注
            Computer.Bet = Computer.GetWager(Computer.Strategy.Value);
            # 如果电脑玩家破产，则返回
            if (Computer.IsBroke) { return; }

            # 输出电脑玩家下注的信息
            _io.WriteLine(betFormat, Computer.Bet);
        }
        else
        {
            # 输出检查信息
            _io.WriteLine(checkMessage);
            # 如果不允许在检查后加注，则返回
            if (!allowRaiseAfterCheck) { return; }
        }

        # 递归调用GetWagers方法
        GetWagers();
    }

    # 获取赌注
    private void GetWagers()
    {
        # 无限循环
        while (true)
        {
            # 人类玩家尚未下注
            Human.HasBet = false;
            # 无限循环
            while (true)
            {
                # 如果人类玩家下注成功，则跳出循环
                if (Human.SetWager()) { break; }
                # 如果人类玩家破产或弃牌，则返回
                if (Human.IsBroke || Human.HasFolded) { return; }
            }
            # 如果人类玩家下注与电脑玩家下注相等
            if (Human.Bet == Computer.Bet)
            {
                # 收集赌注
                CollectBets();
                return;
            }
            # 如果电脑玩家的策略是弃牌
            if (Computer.Strategy is Fold)
            {
                # 如果人类玩家下注大于5
                if (Human.Bet > 5)
                {
                    # 电脑玩家弃牌
                    Computer.Fold();
                    # 输出"I fold."
                    _io.WriteLine("I fold.");
                    return;
                }
            }
            # 如果人类玩家下注大于3倍电脑玩家策略的值
            if (Human.Bet > 3 * Computer.Strategy.Value)
            {
                # 如果电脑玩家的策略不是加注
                if (Computer.Strategy is not Raise)
                {
                    # 输出"I'll see you."
                    _io.WriteLine("I'll see you.");
                    # 电脑玩家下注等于人类玩家下注
                    Computer.Bet = Human.Bet;
                    # 收集赌注
                    CollectBets();
                    return;
                }
            }

            # 计算加注的金额
            var raise = Computer.GetWager(Human.Bet - Computer.Bet);
            # 如果电脑玩家破产，则返回
            if (Computer.IsBroke) { return; }
            # 输出"I'll see you, and raise you {raise}"
            _io.WriteLine($"I'll see you, and raise you {raise}");
            # 电脑玩家下注等于人类玩家下注加上加注的金额
            Computer.Bet = Human.Bet + raise;
        }
    }

    # 收集赌注
    internal void CollectBets()
    {
        // 从玩家和电脑的余额中扣除下注金额
        Human.Balance -= Human.Bet;
        Computer.Balance -= Computer.Bet;
        // 将下注金额加入奖池
        Pot += Human.Bet + Computer.Bet;
    }

    // 检查是否有玩家放弃
    private bool SomeoneHasFolded()
    {
        // 如果玩家放弃
        if (Human.HasFolded)
        {
            _io.WriteLine();
            // 让电脑获得奖金
            Computer.TakeWinnings();
        }
        // 如果电脑放弃
        else if (Computer.HasFolded)
        {
            _io.WriteLine();
            // 让玩家获得奖金
            Human.TakeWinnings();
        }
        else
        {
            return false;
        }

        // 重置奖池
        Pot = 0;
        return true;
    }

    // 检查是否有玩家破产
    private bool SomeoneIsBroke() => Human.IsBroke || Computer.IsBroke;

    // 获取赢家
    private Player? GetWinner()
    {
        _io.WriteLine();
        _io.WriteLine("Now we compare hands:");
        _io.WriteLine("My hand:");
        _io.Write(Computer.Hand);
        _io.WriteLine();
        _io.WriteLine($"You have {Human.Hand.Name}");
        _io.WriteLine($"and I have {Computer.Hand.Name}");
        // 比较玩家和电脑的手牌，返回赢家
        if (Computer.Hand > Human.Hand) { return Computer; }
        if (Human.Hand > Computer.Hand) { return Human; }
        _io.WriteLine("The hand is drawn.");
        _io.WriteLine($"All $ {Pot} remains in the pot.");
        return null;
    }

    // 判断是否继续下一局
    internal bool ShouldPlayAnotherHand()
    {
        // 如果电脑破产
        if (Computer.IsBroke)
        {
            _io.WriteLine("I'm busted.  Congratulations!");
            return true;
        }
        // 如果玩家破产
        if (Human.IsBroke)
        {
            _io.WriteLine("Your wad is shot.  So long, sucker!");
            return true;
        }
        // 显示玩家和电脑的余额，并询问是否继续下一局
        _io.WriteLine($"Now I have $ {Computer.Balance} and you have $ {Human.Balance}");
        return _io.ReadYesNo("Do you wish to continue");
    }
# 闭合了一个代码块
```