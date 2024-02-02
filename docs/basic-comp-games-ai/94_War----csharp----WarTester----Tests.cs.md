# `basic-computer-games\94_War\csharp\WarTester\Tests.cs`

```py
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Text;
using War;

// 命名空间 WarTester
namespace WarTester
{
    // 定义 CardTest 类
    [TestClass]
    public class CardTest
    }

    // 定义 DeckTest 类
    [TestClass]
    public class DeckTest
    {
        // 存储卡片名称的字符串，按顺序排列
        private readonly string cardNamesInOrder = "C-2C-3C-4C-5C-6C-7C-8C-9C-10C-JC-QC-KC-AD-2D-3D-4D-5D-6D-7D-8D-9D-10D-JD-QD-KD-AH-2H-3H-4H-5H-6H-7H-8H-9H-10H-JH-QH-KH-AS-2S-3S-4S-5S-6S-7S-8S-9S-10S-JS-QS-KS-A";

        // 辅助方法，将整副牌的名称连接成一个字符串
        private string ConcatenateTheDeck(Deck d)
        {
            // 创建 StringBuilder 对象
            StringBuilder sb = new StringBuilder();

            // 遍历整副牌，将每张牌的名称添加到 StringBuilder 对象中
            for (int i = 0; i < Deck.deckSize; i++)
            {
                sb.Append(d.GetCard(i));
            }

            // 返回连接后的字符串
            return sb.ToString();
        }

        // 测试初始牌组是否按顺序包含所有牌
        [TestMethod]
        public void InitialDeckContainsCardsInOrder()
        {
            // 创建 Deck 对象
            Deck d = new Deck();
            // 获取整副牌的名称连接成的字符串
            string allTheCards = ConcatenateTheDeck(d);

            // 断言整副牌的名称连接成的字符串是否等于按顺序排列的字符串
            Assert.IsTrue(allTheCards == cardNamesInOrder);
        }

        // 测试洗牌是否改变了牌组
        [TestMethod]
        public void ShufflingChangesDeck()
        {
            // 我不确定如何测试洗牌是否成功，除了检查牌是否不再按初始顺序排列
            // 创建 Deck 对象
            Deck d = new Deck();
            // 洗牌
            d.Shuffle();
            // 获取整副牌的名称连接成的字符串
            string allTheCards = ConcatenateTheDeck(d);

            // 断言整副牌的名称连接成的字符串是否不等于按顺序排列的字符串
            Assert.IsTrue(allTheCards != cardNamesInOrder);
        }
    }
}
```