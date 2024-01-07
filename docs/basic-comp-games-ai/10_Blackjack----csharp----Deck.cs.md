# `basic-computer-games\10_Blackjack\csharp\Deck.cs`

```

// 引入系统和集合类库
using System;
using System.Collections.Generic;

// 命名空间为Blackjack
namespace Blackjack
{
    // 定义Deck类
    public class Deck
    {
        // 创建一个静态只读的Random对象
        private static readonly Random _random = new Random();

        // 创建一个包含52张卡牌的列表和一个用于存放弃牌的列表
        private readonly List<Card> _cards = new List<Card>(52);
        private readonly List<Card> _discards = new List<Card>(52);

        // 构造函数，初始化卡牌并洗牌
        public Deck()
        {
            for (var index = 0; index < 12; index++)
            {
                for (var suit = 0; suit < 4; suit++)
                {
                    _discards.Add(new Card(index));
                }
            }
            Reshuffle();
        }

        // 洗牌方法
        private void Reshuffle()
        {
            Console.WriteLine("Reshuffling");

            // 将弃牌列表中的卡牌加入到卡牌列表中，并清空弃牌列表
            _cards.AddRange(_discards);
            _discards.Clear();

            // 遍历卡牌列表，随机交换卡牌的位置
            for (var index1 = _cards.Count - 1; index1 > 0; index1--)
            {
                var index2 = _random.Next(0, index1);
                var swapCard = _cards[index1];
                _cards[index1] = _cards[index2];
                _cards[index2] = swapCard;
            }
        }

        // 抽牌方法
        public Card DrawCard()
        {
            // 如果卡牌列表中的卡牌数量小于2，则重新洗牌
            if (_cards.Count < 2)
                Reshuffle();

            // 抽取最后一张卡牌并从列表中移除，然后返回该卡牌
            var card = _cards[_cards.Count - 1];
            _cards.RemoveAt(_cards.Count - 1);
            return card;
        }

        // 弃牌方法，将一组卡牌加入到弃牌列表中
        public void Discard(IEnumerable<Card> cards)
        {
            _discards.AddRange(cards);
        }
    }
}

```