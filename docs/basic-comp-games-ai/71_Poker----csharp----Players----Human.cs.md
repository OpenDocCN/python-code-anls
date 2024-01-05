# `71_Poker\csharp\Players\Human.cs`

```
using Poker.Cards;  # 导入Poker.Cards模块
using Poker.Strategies;  # 导入Poker.Strategies模块

namespace Poker.Players;  # 定义Poker.Players命名空间

internal class Human : Player  # 定义一个名为Human的类，继承自Player类
{
    private readonly IReadWrite _io;  # 声明一个私有的只读属性_io，类型为IReadWrite接口

    public Human(int bank, IReadWrite io)  # 定义一个名为Human的构造函数，接受bank和io两个参数
        : base(bank)  # 调用基类Player的构造函数，传入bank参数
    {
        HasWatch = true;  # 设置HasWatch属性为true
        _io = io;  # 将io参数赋值给_io属性
    }

    public bool HasWatch { get; set; }  # 定义一个名为HasWatch的公共属性，可读写

    protected override void DrawCards(Deck deck)  # 重写基类Player的DrawCards方法，接受一个Deck类型的参数deck
    {
        # 从用户输入中读取要抽取的卡片数量，限定范围为1到3，如果超出范围则提示错误信息
        var count = _io.ReadNumber("How many cards do you want", 3, "You can't draw more than three cards.");
        # 如果用户选择不抽取卡片，则直接返回
        if (count == 0) { return; }

        # 提示用户输入要替换的卡片号码
        _io.WriteLine("What are their numbers:");
        # 循环读取用户输入的卡片号码，并替换手中的卡片
        for (var i = 1; i <= count; i++)
        {
            Hand = Hand.Replace((int)_io.ReadNumber(), deck.DealCard());
        }

        # 显示替换后的新手牌
        _io.WriteLine("Your new hand:");
        _io.Write(Hand);
    }

    # 设置赌注
    internal bool SetWager()
    {
        # 从用户输入中读取策略，如果电脑和玩家都没有下注，则提示用户输入策略
        var strategy = _io.ReadHumanStrategy(Table.Computer.Bet == 0 && Bet == 0);
        # 如果策略是下注或者检查
        if (strategy is Strategies.Bet or Check)
        {
            # 如果下注加上策略值小于电脑的赌注，则执行下面的操作
            if (Bet + strategy.Value < Table.Computer.Bet)
            {
                _io.WriteLine("If you can't see my bet, then fold.");  # 在控制台输出消息
                return false;  # 返回 false
            }
            if (Balance - Bet - strategy.Value >= 0)  # 如果余额减去当前下注金额再减去策略值大于等于0
            {
                HasBet = true;  # 设置 HasBet 为 true
                Bet += strategy.Value;  # 将下注金额增加策略值
                return true;  # 返回 true
            }
            RaiseFunds();  # 调用 RaiseFunds 方法
        }
        else  # 否则
        {
            Fold();  # 调用 Fold 方法
            Table.CollectBets();  # 调用 Table 类的 CollectBets 方法
        }
        return false;  # 返回 false
    }

    public void RaiseFunds()  # 定义 RaiseFunds 方法
    {
        _io.WriteLine();  # 输出空行
        _io.WriteLine("You can't bet with what you haven't got.");  # 输出提示信息

        if (Table.Computer.TryBuyWatch()) { return; }  # 如果电脑尝试购买手表成功，则返回

        // The original program had some code about selling a tie tack, but due to a fault
        // in the logic the code was unreachable. I've omitted it in this port.
        // 原始程序中有关出售领带别针的代码，但由于逻辑错误，该代码无法执行。在此端口中我已省略了它。

        IsBroke = true;  # 设置玩家破产状态为真
    }

    public void ReceiveWatch()
    {
        // In the original code the player does not pay any money to receive the watch back.
        // 在原始代码中，玩家不需要支付任何费用就可以收回手表。
        HasWatch = true;  # 设置玩家拥有手表状态为真
    }

    public void SellWatch(int amount)
    {
        HasWatch = false;  // 将 HasWatch 变量设置为 false，表示玩家失去了手表
        Balance += amount;  // 将玩家的余额增加 amount，表示玩家赢得了游戏的奖金
    }

    public override void TakeWinnings()
    {
        _io.WriteLine("You win.");  // 在控制台输出 "You win."，表示玩家赢得了游戏
        base.TakeWinnings();  // 调用基类的 TakeWinnings 方法，执行基类中的赢取奖金的操作
    }
}
```