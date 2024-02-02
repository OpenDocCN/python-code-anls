# `basic-computer-games\71_Poker\csharp\Players\Player.cs`

```py
// 使用 Poker.Cards 命名空间
namespace Poker.Players
{
    // 内部抽象类 Player
    internal abstract class Player
    {
        // 私有字段 _table，可空类型
        private Table? _table;
        // 私有字段 _hasFolded
        private bool _hasFolded;

        // 受保护的构造函数，传入初始银行金额
        protected Player(int bank)
        {
            // 初始化 Hand 为空手牌
            Hand = Hand.Empty;
            // 初始化余额为银行金额
            Balance = bank;
        }

        // 公共属性 Hand
        public Hand Hand { get; set; }
        // 公共属性 Balance
        public int Balance { get; set; }
        // 公共属性 HasBet
        public bool HasBet { get; set; }
        // 公共属性 Bet
        public int Bet { get; set; }
        // 公共只读属性 HasFolded
        public bool HasFolded => _hasFolded;
        // 公共属性 IsBroke，受保护的设置器
        public bool IsBroke { get; protected set; }

        // 受保护的 Table 属性，如果为空则抛出异常
        protected Table Table =>
            _table ?? throw new InvalidOperationException("The player must be sitting at the table.");

        // 公共方法 Sit，传入 Table 对象
        public void Sit(Table table) => _table = table;

        // 虚拟方法 NewHand
        public virtual void NewHand()
        {
            // 下注金额归零
            Bet = 0;
            // 发牌给玩家
            Hand = Table.Deck.DealHand();
            // 重置 _hasFolded 为 false
            _hasFolded = false;
        }

        // 公共方法 AnteUp，返回下注金额
        public int AnteUp()
        {
            // 余额减去底注
            Balance -= Table.Ante;
            // 返回底注
            return Table.Ante;
        }

        // 公共方法 DrawCards
        public void DrawCards()
        {
            // 下注金额归零
            Bet = 0;
            // 从牌堆中抽牌
            DrawCards(Table.Deck);
        }

        // 受保护的抽象方法 DrawCards，传入 Deck 对象
        protected abstract void DrawCards(Deck deck);

        // 虚拟方法 TakeWinnings
        public virtual void TakeWinnings()
        {
            // 余额增加奖池金额
            Balance += Table.Pot;
            // 奖池金额归零
            Table.Pot = 0;
        }

        // 公共方法 Fold，设置 _hasFolded 为 true
        public void Fold()
        {
            _hasFolded = true;
        }
    }
}
```