# `basic-computer-games\10_Blackjack\csharp\Hand.cs`

```py
using System;
using System.Collections.Generic;

namespace Blackjack
{
    public class Hand
    {
        // 创建私有的卡片列表，最多包含12张卡片
        private readonly List<Card> _cards = new List<Card>(12);
        // 缓存手牌的总点数
        private int _cachedTotal = 0;

        // 添加一张卡片到手牌中
        public Card AddCard(Card card)
        {
            _cards.Add(card);
            // 重置缓存的总点数
            _cachedTotal = 0;
            return card;
        }

        // 丢弃手牌中的所有卡片
        public void Discard(Deck deck)
        {
            // 将手牌中的卡片放回牌堆中
            deck.Discard(_cards);
            // 清空手牌
            _cards.Clear();
            // 重置缓存的总点数
            _cachedTotal = 0;
        }

        // 将手牌分成两份
        public void SplitHand(Hand secondHand)
        {
            // 如果手牌数量不为2或者第二手牌数量不为0，则抛出异常
            if (Count != 2 || secondHand.Count != 0)
                throw new InvalidOperationException();
            // 将第二手牌添加一张卡片
            secondHand.AddCard(_cards[1]);
            // 移除当前手牌的第二张卡片
            _cards.RemoveAt(1);
            // 重置缓存的总点数
            _cachedTotal = 0;
        }

        // 返回只读的卡片列表
        public IReadOnlyList<Card> Cards => _cards;

        // 返回手牌中卡片的数量
        public int Count => _cards.Count;

        // 判断手牌是否存在卡片
        public bool Exists => _cards.Count > 0;

        // 计算手牌的总点数
        public int Total
        {
            get
            {
                if (_cachedTotal == 0)
                {
                    var aceCount = 0;
                    foreach (var card in _cards)
                    {
                        _cachedTotal += card.Value;
                        if (card.IsAce)
                            aceCount++;
                    }
                    // 如果总点数大于21且有A，则将A的点数减少10
                    while (_cachedTotal > 21 && aceCount > 0)
                    {
                        _cachedTotal -= 10;
                        aceCount--;
                    }
                }
                return _cachedTotal;
            }
        }

        // 判断手牌是否为21点
        public bool IsBlackjack => Total == 21 && Count == 2;

        // 判断手牌是否爆牌
        public bool IsBusted => Total > 21;
    }
}
```