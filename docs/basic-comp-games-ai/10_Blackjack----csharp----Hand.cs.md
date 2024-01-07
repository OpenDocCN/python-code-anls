# `basic-computer-games\10_Blackjack\csharp\Hand.cs`

```

// 引入系统和集合类库
using System;
using System.Collections.Generic;

// 命名空间为Blackjack
namespace Blackjack
{
    // 定义一个名为Hand的公共类
    public class Hand
    {
        // 创建一个私有的名为_cards的Card类型列表，初始容量为12
        private readonly List<Card> _cards = new List<Card>(12);
        // 创建一个私有的整型变量_cachedTotal，初始值为0

        // 添加一张卡片到手牌中
        public Card AddCard(Card card)
        {
            _cards.Add(card); // 将卡片添加到_cards列表中
            _cachedTotal = 0; // 重置_cachedTotal为0
            return card; // 返回添加的卡片
        }

        // 丢弃手牌中的卡片
        public void Discard(Deck deck)
        {
            deck.Discard(_cards); // 调用Deck类的Discard方法，将_cards列表中的卡片丢弃
            _cards.Clear(); // 清空_cards列表
            _cachedTotal = 0; // 重置_cachedTotal为0
        }

        // 分牌
        public void SplitHand(Hand secondHand)
        {
            // 如果手牌数量不为2或者第二手牌数量不为0，则抛出InvalidOperationException异常
            if (Count != 2 || secondHand.Count != 0)
                throw new InvalidOperationException();
            secondHand.AddCard(_cards[1]); // 将_cards列表中的第二张卡片添加到第二手牌中
            _cards.RemoveAt(1); // 移除_cards列表中的第二张卡片
            _cachedTotal = 0; // 重置_cachedTotal为0
        }

        // 获取_cards列表的只读副本
        public IReadOnlyList<Card> Cards => _cards;

        // 获取_cards列表的数量
        public int Count => _cards.Count;

        // 判断_cards列表是否存在卡片
        public bool Exists => _cards.Count > 0;

        // 计算手牌的总点数
        public int Total
        {
            get
            {
                if (_cachedTotal == 0)
                {
                    var aceCount = 0; // 创建一个名为aceCount的变量，初始值为0
                    foreach (var card in _cards) // 遍历_cards列表中的每张卡片
                    {
                        _cachedTotal += card.Value; // 将卡片的点数加到_cachedTotal中
                        if (card.IsAce) // 如果卡片是Ace
                            aceCount++; // aceCount加1
                    }
                    while (_cachedTotal > 21 && aceCount > 0) // 当_cachedTotal大于21且aceCount大于0时
                    {
                        _cachedTotal -= 10; // _cachedTotal减去10
                        aceCount--; // aceCount减1
                    }
                }
                return _cachedTotal; // 返回_cachedTotal
            }
        }

        // 判断手牌是否为21点
        public bool IsBlackjack => Total == 21 && Count == 2;

        // 判断手牌是否爆牌
        public bool IsBusted => Total > 21;
    }
}

```