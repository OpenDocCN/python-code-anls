# `basic-computer-games\94_War\csharp\War\Cards.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;

// 定义花色的枚举类型
namespace War
{
    public enum Suit
    {
        clubs,      // 梅花
        diamonds,   // 方块
        hearts,     // 红桃
        spades      // 黑桃
    }

    // 定义牌面大小的枚举类型
    public enum Rank
    {
        // 从2到10的牌面大小
        two = 2,
        three,
        four,
        five,
        six,
        seven,
        eight,
        nine,
        ten,
        jack,       // J
        queen,      // Q
        king,       // K
        ace         // A
    }

    // 代表一张扑克牌的类
    public class Card
    {
        // 代表一副扑克牌的类
        public class Deck
        {
            // 一副牌的大小
            public const int deckSize = 52;

            // 一副牌的数组
            private Card[] theDeck = new Card[deckSize];

            // 构造函数，用于初始化一副牌
            public Deck()
            {
                // 用循环填充theDeck数组，包含所有的扑克牌
                int i = 0;
                for (Suit suit = Suit.clubs; suit <= Suit.spades; suit++)
                {
                    for (Rank rank = Rank.two; rank <= Rank.ace; rank++)
                    {
                        theDeck[i] = new Card(suit, rank);
                        i++;
                    }
                }
            }

            // 返回牌堆中特定位置的牌
            // 由于这是一个非常简短的方法，我们将其定义为表达式体方法
            public Card GetCard(int i) => theDeck[i];

            // 洗牌，使用现代版本的Fisher-Yates洗牌算法
            // 参考：https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
            public void Shuffle()
            {
                var rand = new Random();

                // 从牌堆末尾向前迭代
                for (int i = deckSize - 1; i >= 1; i--)
                {
                    int j = rand.Next(0, i);

                    // 交换位置i和j的牌
                    Card temp = theDeck[j];
                    theDeck[j] = theDeck[i];
                    theDeck[i] = temp;
                }
            }
        }
    }
}
```