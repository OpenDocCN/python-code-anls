# `basic-computer-games\10_Blackjack\csharp\Deck.cs`

```
using System;
using System.Collections.Generic;

namespace Blackjack
{
    public class Deck
    {
        private static readonly Random _random = new Random();  // 创建一个静态只读的 Random 对象

        private readonly List<Card> _cards = new List<Card>(52);  // 创建一个只读的 Card 对象列表，初始容量为 52
        private readonly List<Card> _discards = new List<Card>(52);  // 创建一个只读的 Card 对象列表，初始容量为 52

        public Deck()
        {
            for (var index = 0; index < 12; index++)  // 循环12次
            {
                for (var suit = 0; suit < 4; suit++)  // 循环4次
                {
                    _discards.Add(new Card(index));  // 向 _discards 列表中添加一个新的 Card 对象
                }
            }
            Reshuffle();  // 调用 Reshuffle 方法
        }

        private void Reshuffle()
        {
            Console.WriteLine("Reshuffling");  // 打印 "Reshuffling" 到控制台

            _cards.AddRange(_discards);  // 将 _discards 列表中的元素添加到 _cards 列表中
            _discards.Clear();  // 清空 _discards 列表

            for (var index1 = _cards.Count - 1; index1 > 0; index1--)  // 从 _cards 列表的倒数第二个元素开始循环到第一个元素
            {
                var index2 = _random.Next(0, index1);  // 生成一个随机数 index2，范围在 [0, index1) 之间
                var swapCard = _cards[index1];  // 获取 _cards 列表中的第 index1 个元素
                _cards[index1] = _cards[index2];  // 将 _cards 列表中的第 index2 个元素赋值给第 index1 个元素
                _cards[index2] = swapCard;  // 将 swapCard 赋值给 _cards 列表中的第 index2 个元素
            }
        }

        public Card DrawCard()
        {
            if (_cards.Count < 2)  // 如果 _cards 列表中的元素个数小于2
                Reshuffle();  // 调用 Reshuffle 方法

            var card = _cards[_cards.Count - 1];  // 获取 _cards 列表中的最后一个元素
            _cards.RemoveAt(_cards.Count - 1);  // 移除 _cards 列表中的最后一个元素
            return card;  // 返回获取的卡片
        }

        public void Discard(IEnumerable<Card> cards)
        {
            _discards.AddRange(cards);  // 将 cards 中的元素添加到 _discards 列表中
        }
    }
}
```