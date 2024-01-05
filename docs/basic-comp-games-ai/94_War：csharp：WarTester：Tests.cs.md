# `d:/src/tocomm/basic-computer-games\94_War\csharp\WarTester\Tests.cs`

```
        {
            // 使用断言判断 c1 是否小于 c2
            Assert.IsTrue(c1 < c2);
            // 使用断言判断 c2 是否小于 c3
            Assert.IsFalse(c2 < c3);
            // 使用断言判断 c3 是否小于等于 c4
            Assert.IsTrue(c3 <= c4);
        }

        [TestMethod]
        public void GreaterThanIsValid()
        {
            // 使用断言判断 c2 是否大于 c1
            Assert.IsTrue(c2 > c1);
            // 使用断言判断 c3 是否大于 c2
            Assert.IsFalse(c3 > c2);
            // 使用断言判断 c4 是否大于等于 c3
            Assert.IsTrue(c4 >= c3);
        }

        [TestMethod]
        public void EqualToIsValid()
        {
            // 使用断言判断 c3 是否等于 c4
            Assert.IsTrue(c3 == c4);
            // 使用断言判断 c1 是否不等于 c2
            Assert.IsTrue(c1 != c2);
        }
    }
}
        {
            Assert.IsTrue(c1 < c2, "c1 < c2");  // 检查c1是否小于c2，即检查两张牌的等级是否不同但花色相同
            Assert.IsFalse(c2 < c1, "c2 < c1"); // 检查c2是否小于c1，即检查两张牌的等级是否不同但花色相同

            Assert.IsFalse(c3 < c4, "c3 < c4"); // 检查c3是否小于c4，即检查两张牌的等级和花色是否相同

            Assert.IsTrue(c1 < c3, "c1 < c3");  // 检查c1是否小于c3，即检查两张牌的等级和花色是否不同
            Assert.IsFalse(c3 < c1, "c3 < c1"); // 检查c3是否小于c1，即检查两张牌的等级和花色是否不同

            Assert.IsFalse(c2 < c4, "c2 < c4"); // 检查c2是否小于c4，即检查两张牌的等级相同但花色不同
            Assert.IsFalse(c4 < c2, "c4 < c2"); // 检查c4是否小于c2，即检查两张牌的等级相同但花色不同
        }

        [TestMethod]
        public void GreaterThanIsValid()
        {
            Assert.IsFalse(c1 > c2, "c1 > c2"); // 检查c1是否大于c2，即检查两张牌的等级是否不同但花色相同
            Assert.IsTrue(c2 > c1, "c2 > c1");  // 检查c2是否大于c1，即检查两张牌的等级是否不同但花色相同

            Assert.IsFalse(c3 > c4, "c3 > c4"); // 检查c3是否大于c4，即检查两张牌的等级和花色是否相同
            Assert.IsFalse(c1 > c3, "c1 > c3"); // 检查 c1 是否大于 c3，即检查两张牌的大小关系，不同花色，不同点数。
            Assert.IsTrue(c3 > c1, "c3 > c1");  // 检查 c3 是否大于 c1，即检查两张牌的大小关系，不同花色，不同点数。

            Assert.IsFalse(c2 > c4, "c2 > c4"); // 检查 c2 是否大于 c4，即检查两张牌的大小关系，不同花色，相同点数。
            Assert.IsFalse(c4 > c2, "c4 > c2"); // 检查 c4 是否大于 c2，即检查两张牌的大小关系，不同花色，相同点数。
        }

        [TestMethod]
        public void LessThanEqualsIsValid()
        {
            Assert.IsTrue(c1 <= c2, "c1 <= c2");  // 检查 c1 是否小于等于 c2，即检查两张牌的大小关系，相同花色，不同点数。
            Assert.IsFalse(c2 <= c1, "c2 <= c1"); // 检查 c2 是否小于等于 c1，即检查两张牌的大小关系，相同花色，不同点数。

            Assert.IsTrue(c3 <= c4, "c3 <= c4");  // 检查 c3 是否小于等于 c4，即检查两张牌的大小关系，相同花色，相同点数。

            Assert.IsTrue(c1 <= c3, "c1 <= c3");  // 检查 c1 是否小于等于 c3，即检查两张牌的大小关系，不同花色，不同点数。
            Assert.IsFalse(c3 <= c1, "c3 <= c1"); // 检查 c3 是否小于等于 c1，即检查两张牌的大小关系，不同花色，不同点数。

            Assert.IsTrue(c2 <= c4, "c2 <= c4");  // 检查 c2 是否小于等于 c4，即检查两张牌的大小关系，不同花色，相同点数。
            Assert.IsTrue(c4 <= c2, "c4 <= c2");  # 检查 c4 是否小于等于 c2，如果是则测试通过，否则测试失败，并输出错误信息 "c4 <= c2"

        }

        [TestMethod]
        public void GreaterThanEqualsIsValid()
        {
            Assert.IsFalse(c1 >= c2, "c1 >= c2");  # 检查 c1 是否大于等于 c2，如果是则测试失败，否则测试通过，并输出错误信息 "c1 >= c2" 
            Assert.IsTrue(c2 >= c1, "c2 >= c1");  # 检查 c2 是否大于等于 c1，如果是则测试通过，否则测试失败，并输出错误信息 "c2 >= c1"

            Assert.IsTrue(c3 >= c4, "c3 >= c4");  # 检查 c3 是否大于等于 c4，如果是则测试通过，否则测试失败，并输出错误信息 "c3 >= c4"

            Assert.IsFalse(c1 >= c3, "c1 >= c3");  # 检查 c1 是否大于等于 c3，如果是则测试失败，否则测试通过，并输出错误信息 "c1 >= c3"
            Assert.IsTrue(c3 >= c1, "c3 >= c1");  # 检查 c3 是否大于等于 c1，如果是则测试通过，否则测试失败，并输出错误信息 "c3 >= c1"

            Assert.IsTrue(c2 >= c4, "c2 >= c4");  # 检查 c2 是否大于等于 c4，如果是则测试通过，否则测试失败，并输出错误信息 "c2 >= c4"
            Assert.IsTrue(c4 >= c2, "c4 >= c2");  # 检查 c4 是否大于等于 c2，如果是则测试通过，否则测试失败，并输出错误信息 "c4 >= c2"
        }

        [TestMethod]
        public void ToStringIsValid()
        {
            // 将花色和点数转换为字符串
            var s1 = c1.ToString();
            var s2 = c3.ToString();
            // 创建新的卡牌对象，并将其转换为字符串
            var s3 = new Card(Suit.hearts, Rank.queen).ToString();
            var s4 = new Card(Suit.spades, Rank.ace).ToString();

            // 断言每个字符串是否符合预期值，如果不符合则输出错误信息
            Assert.IsTrue(s1 == "C-2", "s1 invalid");
            Assert.IsTrue(s2 == "D-10", "s2 invalid");
            Assert.IsTrue(s3 == "H-Q", "s3 invalid");
            Assert.IsTrue(s4 == "S-A", "s4 invalid");
        }
    }

    [TestClass]
    public class DeckTest
    {

        //Helper method. Adds the names of all the cards together into a single string.
        // 辅助方法。将所有卡牌的名称连接成一个字符串。
        private string ConcatenateTheDeck(Deck d)
        {
            # 创建一个 StringBuilder 对象，用于拼接字符串
            sb = StringBuilder()

            # 遍历 Deck 对象中的卡片，将每张卡片的内容添加到 StringBuilder 中
            for i in range(Deck.deckSize):
                sb.Append(d.GetCard(i))

            # 将 StringBuilder 中的内容转换为字符串并返回
            return sb.ToString()
        }

        # 测试初始的 Deck 对象中是否包含按顺序排列的卡片
        [TestMethod]
        public void InitialDeckContainsCardsInOrder()
        {
            # 创建一个 Deck 对象
            Deck d = new Deck()
            # 将 Deck 对象中的所有卡片内容拼接成一个字符串
            string allTheCards = ConcatenateTheDeck(d)

            # 断言拼接后的字符串与预期的卡片顺序字符串相等
            Assert.IsTrue(allTheCards == cardNamesInOrder)
        }

        # 其他测试方法
# 定义一个公共方法，用于测试洗牌操作是否成功
public void ShufflingChangesDeck()
{
    // 我不确定如何测试洗牌是否成功，除非检查卡片是否不再处于初始顺序。
    // 创建一个新的牌组对象
    Deck d = new Deck();
    // 对牌组进行洗牌操作
    d.Shuffle();
    // 将洗牌后的牌组连接成一个字符串
    string allTheCards = ConcatenateTheDeck(d);
    // 断言，确保洗牌后的牌组不再是初始顺序
    Assert.IsTrue(allTheCards != cardNamesInOrder);
}
```