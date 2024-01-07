# `basic-computer-games\71_Poker\csharp\Players\Player.cs`

```

// 使用扑克牌命名空间中的卡牌类
using Poker.Cards;

// 定义扑克牌玩家命名空间
namespace Poker.Players;

// 内部抽象类 Player
internal abstract class Player
{
    // 私有字段 _table，表示玩家所在的桌子
    private Table? _table;
    // 私有字段 _hasFolded，表示玩家是否已经弃牌
    private bool _hasFolded;

    // 受保护的构造函数，初始化玩家手牌为空，余额为指定的初始金额
    protected Player(int bank)
    {
        Hand = Hand.Empty;
        Balance = bank;
    }

    // 公共属性 Hand，表示玩家手牌
    public Hand Hand { get; set; }
    // 公共属性 Balance，表示玩家余额
    public int Balance { get; set; }
    // 公共属性 HasBet，表示玩家是否已下注
    public bool HasBet { get; set; }
    // 公共属性 Bet，表示玩家的下注金额
    public int Bet { get; set; }
    // 公共只读属性 HasFolded，表示玩家是否已经弃牌
    public bool HasFolded => _hasFolded;
    // 公共属性 IsBroke，表示玩家是否破产
    public bool IsBroke { get; protected set; }

    // 受保护的 Table 属性，表示玩家所在的桌子
    protected Table Table =>
        _table ?? throw new InvalidOperationException("The player must be sitting at the table.");

    // 公共方法 Sit，让玩家坐下到指定的桌子
    public void Sit(Table table) => _table = table;

    // 虚拟方法 NewHand，表示开始新一轮游戏
    public virtual void NewHand()
    {
        Bet = 0;
        Hand = Table.Deck.DealHand();
        _hasFolded = false;
    }

    // AnteUp 方法，表示玩家下底注
    public int AnteUp()
    {
        Balance -= Table.Ante;
        return Table.Ante;
    }

    // DrawCards 方法，表示玩家摸牌
    public void DrawCards()
    {
        Bet = 0;
        DrawCards(Table.Deck);
    }

    // 受保护的抽象方法 DrawCards，表示玩家摸牌
    protected abstract void DrawCards(Deck deck);

    // 虚拟方法 TakeWinnings，表示玩家获得赢利
    public virtual void TakeWinnings()
    {
        Balance += Table.Pot;
        Table.Pot = 0;
    }

    // Fold 方法，表示玩家弃牌
    public void Fold()
    {
        _hasFolded = true;
    }
}

```