# `d:/src/tocomm/basic-computer-games\10_Blackjack\java\test\ScoringUtilsTest.java`

```
import org.junit.jupiter.api.Test;  // 导入 JUnit 的 Test 类
import org.junit.jupiter.api.DisplayName;  // 导入 JUnit 的 DisplayName 类
import static org.junit.jupiter.api.Assertions.assertEquals;  // 导入 JUnit 的 assertEquals 静态方法
import java.util.LinkedList;  // 导入 Java 的 LinkedList 类

public class ScoringUtilsTest {  // 定义 ScoringUtilsTest 类

    @Test  // 声明该方法是一个测试方法
    @DisplayName("scoreHand should score aces as 1 when using 11 would bust")  // 设置测试方法的显示名称
    public void scoreHandHardAce() {  // 定义 scoreHandHardAce 方法
        // Given
        LinkedList<Card> hand = new LinkedList<>();  // 创建一个 LinkedList 对象
        hand.add(new Card(10, Card.Suit.SPADES));  // 向 hand 中添加一个 Card 对象
        hand.add(new Card(9, Card.Suit.SPADES));  // 向 hand 中添加一个 Card 对象
        hand.add(new Card(1, Card.Suit.SPADES));  // 向 hand 中添加一个 Card 对象

        // When
        int result = ScoringUtils.scoreHand(hand); // 调用ScoringUtils类的scoreHand方法，传入手牌参数，得到分数结果

        // Then
        assertEquals(20, result); // 断言结果应该等于20
    }

    @Test
    @DisplayName("scoreHand should score 3 aces as 13")
    public void scoreHandMultipleAces() {
        // Given
        LinkedList<Card> hand = new LinkedList<>(); // 创建一个LinkedList对象来存储手牌
        hand.add(new Card(1, Card.Suit.SPADES)); // 向手牌中添加一张点数为1花色为SPADES的牌
        hand.add(new Card(1, Card.Suit.CLUBS)); // 向手牌中添加一张点数为1花色为CLUBS的牌
        hand.add(new Card(1, Card.Suit.HEARTS)); // 向手牌中添加一张点数为1花色为HEARTS的牌

        // When
        int result = ScoringUtils.scoreHand(hand); // 调用ScoringUtils类的scoreHand方法，传入手牌参数，得到分数结果

        // Then
        assertEquals(13, result); // 断言结果应该等于13
    }

    @Test
    @DisplayName("compareHands should return 1 meaning A beat B, 20 to 12")
    public void compareHandsAWins() {
        // 创建手牌 A 的链表
        LinkedList<Card> handA = new LinkedList<>();
        // 向手牌 A 添加两张牌
        handA.add(new Card(10, Card.Suit.SPADES));
        handA.add(new Card(10, Card.Suit.CLUBS));

        // 创建手牌 B 的链表
        LinkedList<Card> handB = new LinkedList<>();
        // 向手牌 B 添加两张牌
        handB.add(new Card(1, Card.Suit.SPADES));
        handB.add(new Card(1, Card.Suit.CLUBS));

        // 调用比较手牌的方法，将结果保存在 result 变量中
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 1
        assertEquals(1, result);
    }

    @Test
    @DisplayName("compareHands should return -1 meaning B beat A, 18 to 4")
    # 定义一个名为 compareHandsBwins 的方法
    def compareHandsBwins():
        # 创建一个名为 handA 的链表，添加两张牌
        handA = LinkedList()
        handA.add(Card(2, Card.Suit.SPADES))
        handA.add(Card(2, Card.Suit.CLUBS))

        # 创建一个名为 handB 的链表，添加三张牌
        handB = LinkedList()
        handB.add(Card(5, Card.Suit.SPADES))
        handB.add(Card(6, Card.Suit.HEARTS))
        handB.add(Card(7, Card.Suit.CLUBS))

        # 调用 ScoringUtils.compareHands 方法比较 handA 和 handB 的大小，将结果保存在 result 变量中
        result = ScoringUtils.compareHands(handA, handB)

        # 断言 result 的值为 -1
        assertEquals(-1, result)
    ```

    ```python
    # 定义一个名为 compareHandsAWinsWithNaturalBlackJack 的测试方法
    @Test
    @DisplayName("compareHands should return 1 meaning A beat B, natural Blackjack to Blackjack")
    def compareHandsAWinsWithNaturalBlackJack():
        # Hand A 以 natural BlackJack 胜出，Hand B 以 Blackjack 胜出
        handA = LinkedList()
    ```
        // 创建一个手牌对象 handA，添加一张点数为 10，花色为 SPADES 的牌
        handA.add(new Card(10, Card.Suit.SPADES));
        // 继续向手牌对象 handA 添加一张点数为 1，花色为 CLUBS 的牌

        // 创建一个手牌对象 handB
        LinkedList<Card> handB = new LinkedList<>();
        // 向手牌对象 handB 添加一张点数为 6，花色为 SPADES 的牌
        handB.add(new Card(6, Card.Suit.SPADES));
        // 继续向手牌对象 handB 添加一张点数为 7，花色为 HEARTS 的牌
        handB.add(new Card(7, Card.Suit.HEARTS));
        // 继续向手牌对象 handB 添加一张点数为 8，花色为 CLUBS 的牌

        // 调用 ScoringUtils 类的 compareHands 方法比较 handA 和 handB 的大小，将结果保存在 result 变量中
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言 result 的值为 1
        assertEquals(1, result);
    }

    @Test
    @DisplayName("compareHands should return -1 meaning B beat A, natural Blackjack to Blackjack")
    public void compareHandsBWinsWithNaturalBlackJack() {
        // 创建一个手牌对象 handA
        LinkedList<Card> handA = new LinkedList<>();
        // 向手牌对象 handA 添加一张点数为 6，花色为 SPADES 的牌
        handA.add(new Card(6, Card.Suit.SPADES));
        // 继续向手牌对象 handA 添加一张点数为 7，花色为 HEARTS 的牌
        handA.add(new Card(7, Card.Suit.HEARTS));
        // 继续向手牌对象 handA 添加一张点数为 8，花色为 CLUBS 的牌
        // 创建一个名为 handB 的 LinkedList 对象，用于存储 Card 对象
        LinkedList<Card> handB = new LinkedList<>();
        // 向 handB 中添加一张牌，点数为 10，花色为 SPADES
        handB.add(new Card(10, Card.Suit.SPADES));
        // 向 handB 中添加一张牌，点数为 1，花色为 CLUBS
        handB.add(new Card(1, Card.Suit.CLUBS);

        // 调用 ScoringUtils 类的 compareHands 方法，比较 handA 和 handB 的牌面大小，将结果存储在 result 变量中
        int result = ScoringUtils.compareHands(handA,handB);

        // 使用断言来验证 result 是否等于 -1
        assertEquals(-1, result);
    }

    // 定义一个名为 compareHandsTieBothBlackJack 的测试方法
    @Test
    @DisplayName("compareHands should return 0, hand A and B tied with a Blackjack")
    public void compareHandsTieBothBlackJack() {
        // 创建一个名为 handA 的 LinkedList 对象，用于存储 Card 对象
        LinkedList<Card> handA = new LinkedList<>();
        // 向 handA 中添加一张牌，点数为 11，花色为 SPADES
        handA.add(new Card(11, Card.Suit.SPADES));
        // 向 handA 中添加一张牌，点数为 10，花色为 CLUBS
        handA.add(new Card(10, Card.Suit.CLUBS));
        
        // 创建一个名为 handB 的 LinkedList 对象，用于存储 Card 对象
        LinkedList<Card> handB = new LinkedList<>();
        // 向 handB 中添加一张牌，点数为 10，花色为 SPADES
        handB.add(new Card(10, Card.Suit.SPADES));
        // 向 handB 中添加一张牌，点数为 11，花色为 CLUBS
        handB.add(new Card(11, Card.Suit.CLUBS));
        int result = ScoringUtils.compareHands(handA,handB); // 调用ScoringUtils类的compareHands方法，比较handA和handB的牌面大小，将结果赋值给result变量

        assertEquals(0, result); // 断言result的值为0，即handA和handB的牌面大小相同
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tie without a Blackjack")
    public void compareHandsTieNoBlackJack() {
        LinkedList<Card> handA = new LinkedList<>(); // 创建一个LinkedList对象handA，用于存储牌
        handA.add(new Card(10, Card.Suit.DIAMONDS)); // 向handA中添加一张点数为10花色为方块的牌
        handA.add(new Card(10, Card.Suit.HEARTS)); // 向handA中添加一张点数为10花色为红桃的牌
        
        LinkedList<Card> handB = new LinkedList<>(); // 创建一个LinkedList对象handB，用于存储牌
        handB.add(new Card(10, Card.Suit.SPADES)); // 向handB中添加一张点数为10花色为黑桃的牌
        handB.add(new Card(10, Card.Suit.CLUBS)); // 向handB中添加一张点数为10花色为梅花的牌

        int result = ScoringUtils.compareHands(handA,handB); // 调用ScoringUtils类的compareHands方法，比较handA和handB的牌面大小，将结果赋值给result变量

        assertEquals(0, result); // 断言result的值为0，即handA和handB的牌面大小相同
    }

    // 定义测试用例，测试当手牌 A 和 B 都爆牌时，比较结果应为平局（返回 0）
    @Test
    @DisplayName("compareHands should return 0, hand A and B tie when both bust")
    public void compareHandsTieBust() {
        // 创建手牌 A
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        // 创建手牌 B
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(11, Card.Suit.SPADES));
        handB.add(new Card(4, Card.Suit.SPADES));

        // 调用比较方法，比较手牌 A 和 B 的结果
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 0
        assertEquals(0, result);
    }
    @Test
    // 定义测试方法，用于比较两手牌的大小，A玩家爆牌
    @DisplayName("compareHands should return -1, meaning B beat A, A busted")
    public void compareHandsABusted() {
        // 创建A玩家的手牌
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        // 创建B玩家的手牌
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.SPADES));

        // 调用比较方法，比较两手牌的大小
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为-1
        assertEquals(-1, result);
    }

    // 定义测试方法，用于比较两手牌的大小，B玩家爆牌
    @Test
    @DisplayName("compareHands should return 1, meaning A beat B, B busted")
    public void compareHandsBBusted() {
        // 创建A玩家的手牌
        LinkedList<Card> handA = new LinkedList<>();
        // 创建一个手牌对象 handA，并向其中添加一张点数为 10，花色为方块的牌
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        // 向手牌对象 handA 中添加一张点数为 3，花色为红桃的牌
        
        // 创建一个手牌对象 handB，并向其中添加两张点数为 10，花色为黑桃的牌以及一张点数为 5，花色为黑桃的牌
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(5, Card.Suit.SPADES));

        // 调用 ScoringUtils 类中的 compareHands 方法比较 handA 和 handB 的大小，将结果保存在 result 变量中
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言 result 的值为 1
        assertEquals(1, result);
    }
}
```