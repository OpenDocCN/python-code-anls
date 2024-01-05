# `10_Blackjack\csharp\Hand.cs`

```
        // 使用给定的卡片添加到手牌中
        public Card AddCard(Card card)
        {
            _cards.Add(card); // 将卡片添加到手牌列表中
            _cachedTotal = 0; // 重置缓存的总点数
            return card; // 返回添加的卡片
        }

        // 丢弃手牌中的所有卡片到牌堆中
        public void Discard(Deck deck)
        {
            deck.Discard(_cards); // 将手牌中的所有卡片丢弃到牌堆中
            _cards.Clear();  # 清空_cards列表中的所有元素
            _cachedTotal = 0;  # 将_cachedTotal变量的值设为0
        }

        public void SplitHand(Hand secondHand)
        {
            if (Count != 2 || secondHand.Count != 0)  # 如果当前手牌数量不等于2或者第二手牌数量不等于0
                throw new InvalidOperationException();  # 抛出无效操作异常
            secondHand.AddCard(_cards[1]);  # 将当前手牌的第二张牌添加到第二手牌中
            _cards.RemoveAt(1);  # 移除当前手牌的第二张牌
            _cachedTotal = 0;  # 将_cachedTotal变量的值设为0
        }

        public IReadOnlyList<Card> Cards => _cards;  # 返回_cards列表的只读版本

        public int Count => _cards.Count;  # 返回_cards列表中元素的数量

        public bool Exists => _cards.Count > 0;  # 返回_cards列表中是否存在元素

        public int Total  # 定义一个名为Total的属性
        {
            get
            {
                if (_cachedTotal == 0)  # 如果缓存的总数为0
                {
                    var aceCount = 0;  # 初始化A的数量为0
                    foreach (var card in _cards)  # 遍历_cards列表中的每张牌
                    {
                        _cachedTotal += card.Value;  # 将每张牌的点数加到缓存的总数中
                        if (card.IsAce)  # 如果这张牌是A
                            aceCount++;  # A的数量加1
                    }
                    while (_cachedTotal > 21 && aceCount > 0)  # 当缓存的总数大于21且A的数量大于0时
                    {
                        _cachedTotal -= 10;  # 缓存的总数减去10
                        aceCount--;  # A的数量减1
                    }
                }
                return _cachedTotal;  # 返回缓存的总数
            }
        }  # 结束类定义

        public bool IsBlackjack => Total == 21 && Count == 2;  # 如果牌的总点数为21且牌的数量为2，返回true，否则返回false

        public bool IsBusted => Total > 21;  # 如果牌的总点数大于21，返回true，否则返回false
    }
}
```