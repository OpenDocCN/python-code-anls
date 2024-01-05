# `71_Poker\csharp\Table.cs`

```
using Poker.Cards;  # 导入Poker.Cards模块，用于处理扑克牌相关的操作
using Poker.Players;  # 导入Poker.Players模块，用于处理玩家相关的操作
using Poker.Strategies;  # 导入Poker.Strategies模块，用于处理策略相关的操作

namespace Poker;  # 定义Poker命名空间

internal class Table  # 定义Table类
{
    private readonly IReadWrite _io;  # 声明私有的只读属性_io，类型为IReadWrite接口
    private readonly IRandom _random;  # 声明私有的只读属性_random，类型为IRandom接口
    public int Pot;  # 声明公共的整型属性Pot

    public Table(IReadWrite io, IRandom random, Deck deck, Human human, Computer computer)  # 定义Table类的构造函数，接受io、random、deck、human、computer五个参数
    {
        _io = io;  # 将构造函数参数io赋值给私有属性_io
        _random = random;  # 将构造函数参数random赋值给私有属性_random
        Deck = deck;  # 将构造函数参数deck赋值给Deck属性
        Human = human;  # 将构造函数参数human赋值给Human属性
        Computer = computer;  # 将构造函数参数computer赋值给Computer属性
        human.Sit(this);  // 调用 Human 类的 Sit 方法，将当前对象作为参数传入
        computer.Sit(this);  // 调用 Computer 类的 Sit 方法，将当前对象作为参数传入
    }

    public int Ante { get; } = 5;  // 定义一个只读属性 Ante，初始值为 5
    public Deck Deck { get; }  // 定义一个只读属性 Deck
    public Human Human { get; }  // 定义一个只读属性 Human
    public Computer Computer { get; }  // 定义一个只读属性 Computer

    internal void PlayHand()  // 定义一个内部方法 PlayHand
    {
        while (true)  // 进入无限循环
        {
            _io.WriteLine();  // 输出空行
            Computer.CheckFunds();  // 调用 Computer 类的 CheckFunds 方法
            if (Computer.IsBroke) { return; }  // 如果 Computer 的 IsBroke 属性为 true，则返回

            _io.WriteLine($"The ante is ${Ante}.  I will deal:");  // 输出赌注金额
            _io.WriteLine();  // 输出空行
            if (Human.Balance <= Ante)  // 如果 Human 的 Balance 属性小于等于赌注金额
            {
                # 调用 Human 对象的 RaiseFunds 方法，筹集资金
                Human.RaiseFunds();
                # 如果 Human 对象已经破产，则返回
                if (Human.IsBroke) { return; }
            }

            # 进行交易
            Deal(_random);

            # 输出空行
            _io.WriteLine();
            # 获取玩家的赌注，可以选择以指定的金额开局，或者选择跟注
            GetWagers("I'll open with ${0}", "I check.", allowRaiseAfterCheck: true);
            # 如果有人破产或者有人弃牌，则返回
            if (SomeoneIsBroke() || SomeoneHasFolded()) { return; }

            # 进行抽牌
            Draw();

            # 获取玩家的赌注
            GetWagers();
            # 如果有人破产，则返回
            if (SomeoneIsBroke()) { return; }
            # 如果 Human 对象还没有下注
            if (!Human.HasBet)
            {
                # 获取玩家的赌注，可以选择下注指定金额，或者选择跟注
                GetWagers("I'll bet ${0}", "I'll check");
            }
            # 如果有人破产或者有人弃牌，则返回
            if (SomeoneIsBroke() || SomeoneHasFolded()) { return; }
            if (GetWinner() is { } winner)  # 如果GetWinner()返回的对象不为空，则将其赋值给winner
            {
                winner.TakeWinnings();  # 调用winner对象的TakeWinnings()方法
                return;  # 结束当前方法的执行
            }
        }
    }

    private void Deal(IRandom random)  # 定义一个私有方法Deal，接受一个IRandom类型的参数random
    {
        Deck.Shuffle(random);  # 调用Deck对象的Shuffle方法，传入random参数进行洗牌

        Pot = Human.AnteUp() + Computer.AnteUp();  # 计算玩家下注的总额，赋值给Pot

        Human.NewHand();  # 调用Human对象的NewHand方法，发放新的手牌
        Computer.NewHand();  # 调用Computer对象的NewHand方法，发放新的手牌

        _io.WriteLine("Your hand:");  # 在_io对象上调用WriteLine方法，输出"Your hand:"
        _io.Write(Human.Hand);  # 在_io对象上调用Write方法，输出Human对象的Hand属性
    }
    # 定义一个名为 Draw 的私有方法
    def Draw():
        # 输出空行
        _io.WriteLine()
        # 输出"Now we draw -- "
        _io.Write("Now we draw -- ")
        # 调用 Human 对象的 DrawCards 方法
        Human.DrawCards()
        # 调用 Computer 对象的 DrawCards 方法
        Computer.DrawCards()
        # 输出空行
        _io.WriteLine()

    # 定义一个名为 GetWagers 的私有方法，接受三个参数：betFormat、checkMessage、allowRaiseAfterCheck（默认值为 False）
    def GetWagers(betFormat, checkMessage, allowRaiseAfterCheck = False):
        # 如果 Computer 对象的 Strategy 属性是 Bet 类型
        if (Computer.Strategy is Bet):
            # 将 Computer 对象的 Bet 属性设置为调用 Computer 对象的 GetWager 方法并传入 Computer 对象的 Strategy 属性值
            Computer.Bet = Computer.GetWager(Computer.Strategy.Value)
            # 如果 Computer 对象已破产，则返回
            if (Computer.IsBroke) { return; }

            # 输出格式化后的字符串，格式为 betFormat，参数为 Computer 对象的 Bet 属性值
            _io.WriteLine(betFormat, Computer.Bet)
        # 如果 Computer 对象的 Strategy 属性不是 Bet 类型
        else
        {
            _io.WriteLine(checkMessage);  # 输出检查消息
            if (!allowRaiseAfterCheck) { return; }  # 如果不允许在检查后加注，则返回
        }

        GetWagers();  # 调用 GetWagers() 函数
    }

    private void GetWagers()  # 定义 GetWagers() 函数
    {
        while (true)  # 进入无限循环
        {
            Human.HasBet = false;  # 将 Human.HasBet 设为 false
            while (true)  # 进入内部无限循环
            {
                if (Human.SetWager()) { break; }  # 如果 Human.SetWager() 返回 true，则跳出内部循环
                if (Human.IsBroke || Human.HasFolded) { return; }  # 如果 Human.IsBroke 为 true 或 Human.HasFolded 为 true，则返回
            }
            if (Human.Bet == Computer.Bet)  # 如果 Human.Bet 等于 Computer.Bet
            {
                # 收集下注
                CollectBets();
                # 返回
                return;
            }
            # 如果电脑的策略是弃牌
            if (Computer.Strategy is Fold)
            {
                # 如果人类下注大于5
                if (Human.Bet > 5)
                {
                    # 电脑弃牌
                    Computer.Fold();
                    # 输出信息
                    _io.WriteLine("I fold.");
                    # 返回
                    return;
                }
            }
            # 如果人类下注大于3倍电脑策略的值
            if (Human.Bet > 3 * Computer.Strategy.Value)
            {
                # 如果电脑的策略不是加注
                if (Computer.Strategy is not Raise)
                {
                    # 输出信息
                    _io.WriteLine("I'll see you.");
                    # 电脑下注等于人类下注
                    Computer.Bet = Human.Bet;
                    # 收集下注
                    CollectBets();
                    # 返回
                    return;
            }
        }

        // 根据玩家下注情况确定是否有人加注
        var raise = Computer.GetWager(Human.Bet - Computer.Bet);
        // 如果电脑玩家破产，则结束游戏
        if (Computer.IsBroke) { return; }
        // 输出电脑玩家的决定
        _io.WriteLine($"I'll see you, and raise you {raise}");
        // 电脑玩家下注更新
        Computer.Bet = Human.Bet + raise;
    }

    // 收集玩家的下注金额
    internal void CollectBets()
    {
        // 玩家余额减去下注金额
        Human.Balance -= Human.Bet;
        // 电脑玩家余额减去下注金额
        Computer.Balance -= Computer.Bet;
        // 奖池金额增加
        Pot += Human.Bet + Computer.Bet;
    }

    // 判断是否有玩家弃牌
    private bool SomeoneHasFolded()
    {
        // 如果玩家弃牌，则返回 true
        if (Human.HasFolded)
        {
            _io.WriteLine();  # 输出空行
            Computer.TakeWinnings();  # 让电脑玩家取得赢利
        }
        else if (Computer.HasFolded)  # 如果电脑玩家已经弃牌
        {
            _io.WriteLine();  # 输出空行
            Human.TakeWinnings();  # 让玩家取得赢利
        }
        else  # 其他情况
        {
            return false;  # 返回 false
        }

        Pot = 0;  # 将奖池设为 0
        return true;  # 返回 true
    }

    private bool SomeoneIsBroke() => Human.IsBroke || Computer.IsBroke;  # 检查是否有玩家破产
    private Player? GetWinner()
    {
        # 输出空行
        _io.WriteLine();
        # 输出比较手牌的提示信息
        _io.WriteLine("Now we compare hands:");
        # 输出计算机的手牌
        _io.WriteLine("My hand:");
        _io.Write(Computer.Hand);
        # 输出换行
        _io.WriteLine();
        # 输出玩家的手牌名称
        _io.WriteLine($"You have {Human.Hand.Name}");
        # 输出计算机的手牌名称
        _io.WriteLine($"and I have {Computer.Hand.Name}");
        # 如果计算机的手牌大于玩家的手牌，返回计算机
        if (Computer.Hand > Human.Hand) { return Computer; }
        # 如果玩家的手牌大于计算机的手牌，返回玩家
        if (Human.Hand > Computer.Hand) { return Human; }
        # 输出手牌相同的提示信息
        _io.WriteLine("The hand is drawn.");
        # 输出剩余筹码数量
        _io.WriteLine($"All $ {Pot} remains in the pot.");
        # 返回空值
        return null;
    }

    internal bool ShouldPlayAnotherHand()
    {
        # 如果计算机破产，返回false
        if (Computer.IsBroke)
        {
# 如果计算机破产了，输出信息并返回 true
if (Computer.IsBroke)
{
    _io.WriteLine("I'm busted.  Congratulations!");
    return true;
}

# 如果玩家破产了，输出信息并返回 true
if (Human.IsBroke)
{
    _io.WriteLine("Your wad is shot.  So long, sucker!");
    return true;
}

# 输出计算机和玩家的余额信息，并根据玩家输入判断是否继续游戏
_io.WriteLine($"Now I have $ {Computer.Balance} and you have $ {Human.Balance}");
return _io.ReadYesNo("Do you wish to continue");
```