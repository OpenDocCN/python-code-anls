# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Players\Computer.cs`

```
using Poker.Cards;  // 导入Poker.Cards命名空间，以便使用其中的类
using Poker.Strategies;  // 导入Poker.Strategies命名空间，以便使用其中的类
using static System.StringComparison;  // 导入System.StringComparison枚举类型，以便在代码中直接使用其成员

namespace Poker.Players;  // 声明Poker.Players命名空间

internal class Computer : Player  // 声明一个名为Computer的类，继承自Player类
{
    private readonly IReadWrite _io;  // 声明一个只读的IReadWrite类型的私有字段_io
    private readonly IRandom _random;  // 声明一个只读的IRandom类型的私有字段_random

    public Computer(int bank, IReadWrite io, IRandom random)  // 声明一个名为Computer的构造函数，接受bank、io和random三个参数
        : base(bank)  // 调用基类Player的构造函数，传入bank参数
    {
        _io = io;  // 将io参数赋值给私有字段_io
        _random = random;  // 将random参数赋值给私有字段_random
        Strategy = Strategy.None;  // 将Strategy属性初始化为Strategy.None
    }

    public Strategy Strategy { get; set; }  // 声明一个公共的Strategy类型的属性Strategy，可读可写
}
    public override void NewHand()
    {
        base.NewHand(); // 调用基类的 NewHand 方法

        // 使用三元运算符根据手牌的强度和等级选择策略
        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            // 当手牌弱，且随机数小于2时，选择 Bluff 策略
            (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11100),
            (true, _, _) when _random.Next(10) < 2 => Strategy.Bluff(23, 0b11110),
            (true, _, _) when _random.Next(10) < 1 => Strategy.Bluff(23, 0b11111),
            (true, _, _) => Strategy.Fold, // 当手牌弱时，选择 Fold 策略
            (false, true, _) => _random.Next(10) < 2 ? Strategy.Bluff(23) : Strategy.Check, // 当手牌不弱且等级小于 Three 时，根据随机数选择 Bluff 或 Check 策略
            (false, false, true) => Strategy.Bet(35), // 当手牌不弱且等级不小于 Three 且不小于 FullHouse 时，选择 Bet 策略
            (false, false, false) => _random.Next(10) < 1 ? Strategy.Bet(35) : Strategy.Raise // 当手牌不弱且等级不小于 Three 且不小于 FullHouse 且不小于 FullHouse 时，根据随机数选择 Bet 或 Raise 策略
        };
    }

    protected override void DrawCards(Deck deck)
    {
        var keepMask = Strategy.KeepMask ?? Hand.KeepMask; // 根据策略的保留掩码或者手牌的保留掩码选择保留的牌
        # 初始化计数器
        var count = 0;
        # 循环遍历1到5的数字
        for (var i = 1; i <= 5; i++)
        {
            # 检查keepMask中第i位是否为0
            if ((keepMask & (1 << (i - 1))) == 0)
            {
                # 用新的牌替换手牌中的第i张牌
                Hand = Hand.Replace(i, deck.DealCard());
                # 计数器加1
                count++;
            }
        }

        # 输出空行
        _io.WriteLine();
        # 输出拿取了多少张牌
        _io.Write($"I am taking {count} card");
        # 如果拿取的牌不止一张，输出"s"
        if (count != 1)
        {
            _io.WriteLine("s");
        }

        # 根据手牌的强度和牌型进行策略选择
        Strategy = (Hand.IsWeak, Hand.Rank < HandRank.Three, Hand.Rank < HandRank.FullHouse) switch
        {
            # 当策略是Bluff时，选择Bluff策略并传入参数28
            _ when Strategy is Bluff => Strategy.Bluff(28),
            (true, _, _) => Strategy.Fold,  // 如果对手已经下注，放弃
            (false, true, _) => _random.Next(10) == 0 ? Strategy.Bet(19) : Strategy.Raise,  // 如果对手没有下注但有可能加注，有10%的概率下注19，否则加注
            (false, false, true) => _random.Next(10) == 0 ? Strategy.Bet(11) : Strategy.Bet(19),  // 如果对手没有下注也没有可能加注，有10%的概率下注11，否则下注19
            (false, false, false) => Strategy.Raise  // 如果对手没有下注也没有可能加注，加注
        };
    }

    public int GetWager(int wager)
    {
        wager += _random.Next(10);  // 下注额度加上一个0到9的随机数
        if (Balance < Table.Human.Bet + wager)  // 如果余额小于对手下注额度加上当前下注额度
        {
            if (Table.Human.Bet == 0) { return Balance; }  // 如果对手没有下注，返回当前余额

            if (Balance >= Table.Human.Bet)  // 如果余额大于等于对手下注额度
            {
                _io.WriteLine("I'll see you.");  // 输出信息
                Bet = Table.Human.Bet;  // 下注额度等于对手下注额度
                Table.CollectBets();  // 收集下注
            }
            else
            {
                RaiseFunds();  # 如果条件不满足，则调用RaiseFunds()函数
            }
        }

        return wager;  # 返回wager变量的值
    }

    public bool TryBuyWatch()  # 定义一个名为TryBuyWatch的公共函数
    {
        if (!Table.Human.HasWatch) { return false; }  # 如果Table.Human.HasWatch为false，则返回false

        var response = _io.ReadString("Would you like to sell your watch");  # 从_io对象中读取用户输入的字符串
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return false; }  # 如果用户输入的字符串以"N"开头（不区分大小写），则返回false

        var (value, message) = (_random.Next(10) < 7) switch  # 根据_random生成的随机数判断条件
        {
            true => (75, "I'll give you $75 for it."),  # 如果条件为true，则value为75，message为"I'll give you $75 for it."
            false => (25, "That's a pretty crummy watch - I'll give you $25.")  # 如果条件为false，则value为25，message为"That's a pretty crummy watch - I'll give you $25."
        };

        _io.WriteLine(message);  // 输出消息
        Table.Human.SellWatch(value);  // 卖掉手表

        // 原始代码中的计算机部分没有任何钱

        return true;  // 返回 true
    }

    public void RaiseFunds()
    {
        if (Table.Human.HasWatch) { return; }  // 如果玩家有手表，则返回

        var response = _io.ReadString("Would you like to buy back your watch for $50");  // 询问玩家是否愿意以50美元买回手表
        if (response.StartsWith("N", InvariantCultureIgnoreCase)) { return; }  // 如果玩家回答以"N"开头（不区分大小写），则返回

        // 原始代码中没有从玩家扣除50美元
        Balance += 50;  // 余额增加50美元
        Table.Human.ReceiveWatch();  // 玩家收回手表
        IsBroke = true;  // 破产标志设为 true
    }  # 结束 CheckFunds 方法的定义

    public void CheckFunds() { IsBroke = Balance <= Table.Ante; }  # 定义 CheckFunds 方法，检查余额是否小于或等于赌注

    public override void TakeWinnings()  # 重写 TakeWinnings 方法
    {
        _io.WriteLine("I win.");  # 在控制台输出 "I win."
        base.TakeWinnings();  # 调用基类的 TakeWinnings 方法
    }
}
```