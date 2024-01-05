# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Players\Player.cs`

```
using Poker.Cards;  # 导入Poker.Cards模块

namespace Poker.Players;  # 命名空间Poker.Players

internal abstract class Player  # 内部抽象类Player
{
    private Table? _table;  # 私有Table类型的_table变量
    private bool _hasFolded;  # 私有布尔类型的_hasFolded变量

    protected Player(int bank)  # 受保护的构造函数，参数为bank
    {
        Hand = Hand.Empty;  # 初始化Hand属性为空手
        Balance = bank;  # 初始化Balance属性为bank
    }

    public Hand Hand { get; set; }  # 公共Hand属性，可读写
    public int Balance { get; set; }  # 公共整型Balance属性，可读写
    public bool HasBet { get; set; }  # 公共布尔类型HasBet属性，可读写
    public int Bet { get; set; }  # 公共整型Bet属性，可读写
    public bool HasFolded => _hasFolded;  # 公共只读属性HasFolded，返回_hasFolded的值
    public bool IsBroke { get; protected set; }  # 定义一个公共属性IsBroke，用于表示玩家是否破产

    protected Table Table =>  # 定义一个受保护的属性Table，用于获取玩家所在的桌子
        _table ?? throw new InvalidOperationException("The player must be sitting at the table.");  # 如果_table为空，则抛出异常，要求玩家必须坐在桌子上

    public void Sit(Table table) => _table = table;  # 定义一个公共方法Sit，用于让玩家坐在指定的桌子上

    public virtual void NewHand()  # 定义一个虚方法NewHand，用于开始新的一手牌
    {
        Bet = 0;  # 下注金额初始化为0
        Hand = Table.Deck.DealHand();  # 玩家手中的牌由桌子的牌堆发牌得到
        _hasFolded = false;  # 玩家未弃牌
    }

    public int AnteUp()  # 定义一个公共方法AnteUp，用于下底注
    {
        Balance -= Table.Ante;  # 玩家的余额减去桌子的底注
        return Table.Ante;  # 返回桌子的底注
    }
    public void DrawCards()
    {
        Bet = 0;  // 将赌注设为0
        DrawCards(Table.Deck);  // 调用DrawCards方法，传入Table.Deck作为参数
    }

    protected abstract void DrawCards(Deck deck);  // 声明一个抽象方法DrawCards，接受一个Deck类型的参数

    public virtual void TakeWinnings()
    {
        Balance += Table.Pot;  // 将赢得的筹码加到余额上
        Table.Pot = 0;  // 将奖池设为0
    }

    public void Fold()
    {
        _hasFolded = true;  // 将_hasFolded标记为true，表示放弃
    }
}
```