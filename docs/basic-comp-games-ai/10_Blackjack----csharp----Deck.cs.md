# `10_Blackjack\csharp\Deck.cs`

```
        # 创建一个新的牌堆对象
        def __init__(self):
            # 创建一个私有的静态只读的随机数生成器对象
            self._random = Random()
            # 创建一个包含52张卡牌的列表对象
            self._cards = List[Card](52)
            # 创建一个包含52张弃牌的列表对象
            self._discards = List[Card](52)

            # 循环12次，表示12种不同的牌面
            for index in range(12):
                # 循环4次，表示4种不同的花色
                for suit in range(4):
                    # 将新创建的卡牌对象添加到弃牌列表中
                    self._discards.append(Card(index))
            }
            Reshuffle();  # 调用Reshuffle()方法，重新洗牌
        }

        private void Reshuffle()  # 定义一个私有方法Reshuffle()，用于重新洗牌
        {
            Console.WriteLine("Reshuffling");  # 在控制台打印输出"Reshuffling"

            _cards.AddRange(_discards);  # 将_discards列表中的牌添加到_cards列表中
            _discards.Clear();  # 清空_discards列表

            for (var index1 = _cards.Count - 1; index1 > 0; index1--)  # 从最后一张牌开始向前遍历_cards列表
            {
                var index2 = _random.Next(0, index1);  # 生成一个随机数index2，范围在0到index1之间
                var swapCard = _cards[index1];  # 将_cards[index1]的值赋给swapCard
                _cards[index1] = _cards[index2];  # 将_cards[index2]的值赋给_cards[index1]
                _cards[index2] = swapCard;  # 将swapCard的值赋给_cards[index2]
            }
        }
        public Card DrawCard()
        {
            // 如果牌堆中的牌少于2张，则重新洗牌
            if (_cards.Count < 2)
                Reshuffle();

            // 从牌堆中取出最后一张牌
            var card = _cards[_cards.Count - 1];
            // 从牌堆中移除最后一张牌
            _cards.RemoveAt(_cards.Count - 1);
            // 返回取出的牌
            return card;
        }

        public void Discard(IEnumerable<Card> cards)
        {
            // 将要弃掉的牌添加到弃牌堆中
            _discards.AddRange(cards);
        }
    }
}
```