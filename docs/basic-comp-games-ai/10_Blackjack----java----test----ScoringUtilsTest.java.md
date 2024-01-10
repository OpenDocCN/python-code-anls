# `basic-computer-games\10_Blackjack\java\test\ScoringUtilsTest.java`

```
import org.junit.jupiter.api.Test;  // 导入 JUnit 的 Test 类
import org.junit.jupiter.api.DisplayName;  // 导入 JUnit 的 DisplayName 类
import static org.junit.jupiter.api.Assertions.assertEquals;  // 导入 JUnit 的 assertEquals 静态方法
import java.util.LinkedList;  // 导入 LinkedList 类

public class ScoringUtilsTest {  // 定义 ScoringUtilsTest 类

    @Test  // 标记该方法为测试方法
    @DisplayName("scoreHand should score aces as 1 when using 11 would bust")  // 设置测试方法的显示名称
    public void scoreHandHardAce() {  // 定义 scoreHandHardAce 方法
        // Given
        LinkedList<Card> hand = new LinkedList<>();  // 创建 LinkedList 对象 hand
        hand.add(new Card(10, Card.Suit.SPADES));  // 向 hand 中添加 Card 对象
        hand.add(new Card(9, Card.Suit.SPADES));  // 向 hand 中添加 Card 对象
        hand.add(new Card(1, Card.Suit.SPADES));  // 向 hand 中添加 Card 对象

        // When
        int result = ScoringUtils.scoreHand(hand);  // 调用 ScoringUtils 的 scoreHand 方法，计算结果赋值给 result

        // Then
        assertEquals(20, result);  // 使用 JUnit 的 assertEquals 方法进行断言
    }

    @Test  // 标记该方法为测试方法
    @DisplayName("scoreHand should score 3 aces as 13")  // 设置测试方法的显示名称
    public void scoreHandMultipleAces() {  // 定义 scoreHandMultipleAces 方法
        // Given
        LinkedList<Card> hand = new LinkedList<>();  // 创建 LinkedList 对象 hand
        hand.add(new Card(1, Card.Suit.SPADES));  // 向 hand 中添加 Card 对象
        hand.add(new Card(1, Card.Suit.CLUBS));  // 向 hand 中添加 Card 对象
        hand.add(new Card(1, Card.Suit.HEARTS));  // 向 hand 中添加 Card 对象

        // When
        int result = ScoringUtils.scoreHand(hand);  // 调用 ScoringUtils 的 scoreHand 方法，计算结果赋值给 result

        // Then
        assertEquals(13, result);  // 使用 JUnit 的 assertEquals 方法进行断言
    }

    @Test  // 标记该方法为测试方法
    @DisplayName("compareHands should return 1 meaning A beat B, 20 to 12")  // 设置测试方法的显示名称
    public void compareHandsAWins() {  // 定义 compareHandsAWins 方法
        LinkedList<Card> handA = new LinkedList<>();  // 创建 LinkedList 对象 handA
        handA.add(new Card(10, Card.Suit.SPADES));  // 向 handA 中添加 Card 对象
        handA.add(new Card(10, Card.Suit.CLUBS));  // 向 handA 中添加 Card 对象

        LinkedList<Card> handB = new LinkedList<>();  // 创建 LinkedList 对象 handB
        handB.add(new Card(1, Card.Suit.SPADES));  // 向 handB 中添加 Card 对象
        handB.add(new Card(1, Card.Suit.CLUBS));  // 向 handB 中添加 Card 对象

        int result = ScoringUtils.compareHands(handA,handB);  // 调用 ScoringUtils 的 compareHands 方法，计算结果赋值给 result

        assertEquals(1, result);  // 使用 JUnit 的 assertEquals 方法进行断言
    }

    @Test  // 标记该方法为测试方法
    @DisplayName("compareHands should return -1 meaning B beat A, 18 to 4")  // 设置测试方法的显示名称
    public void compareHandsBwins() {
        // 创建手牌 A，添加两张牌
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(2, Card.Suit.SPADES));
        handA.add(new Card(2, Card.Suit.CLUBS);

        // 创建手牌 B，添加三张牌
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(5, Card.Suit.SPADES));
        handB.add(new Card(6, Card.Suit.HEARTS));
        handB.add(new Card(7, Card.Suit.CLUBS));

        // 比较手牌 A 和手牌 B 的大小
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 -1
        assertEquals(-1, result);
    }

    @Test
    @DisplayName("compareHands should return 1 meaning A beat B, natural Blackjack to Blackjack")
    public void compareHandsAWinsWithNaturalBlackJack() {
        // 手牌 A 获胜，手牌 A 为自然二十一点，手牌 B 为二十一点
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.SPADES));
        handA.add(new Card(1, Card.Suit.CLUBS));

        // 创建手牌 B
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(6, Card.Suit.SPADES));
        handB.add(new Card(7, Card.Suit.HEARTS));
        handB.add(new Card(8, Card.Suit.CLUBS));

        // 比较手牌 A 和手牌 B 的大小
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 1
        assertEquals(1, result);
    }

    @Test
    @DisplayName("compareHands should return -1 meaning B beat A, natural Blackjack to Blackjack")
    public void compareHandsBWinsWithNaturalBlackJack() {
        // 创建手牌 A
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(6, Card.Suit.SPADES));
        handA.add(new Card(7, Card.Suit.HEARTS));
        handA.add(new Card(8, Card.Suit.CLUBS));
        
        // 手牌 B 获胜，手牌 B 为自然二十一点，手牌 A 为二十一点
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(1, Card.Suit.CLUBS));

        // 比较手牌 A 和手牌 B 的大小
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 -1
        assertEquals(-1, result);
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tied with a Blackjack")
    // 比较两手牌，如果都是黑杰克，则平局
    public void compareHandsTieBothBlackJack() {
        // 创建手牌 A，包含黑杰克牌和10点牌
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(11, Card.Suit.SPADES));
        handA.add(new Card(10, Card.Suit.CLUBS));
        
        // 创建手牌 B，包含10点牌和黑杰克牌
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(11, Card.Suit.CLUBS);

        // 比较两手牌，返回比较结果
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为平局
        assertEquals(0, result);
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tie without a Blackjack")
    public void compareHandsTieNoBlackJack() {
        // 创建手牌 A，包含两张10点牌
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        
        // 创建手牌 B，包含两张10点牌
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.CLUBS));

        // 比较两手牌，返回比较结果
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为平局
        assertEquals(0, result);
    }

    @Test
    @DisplayName("compareHands should return 0, hand A and B tie when both bust")
    public void compareHandsTieBust() {
        // 创建手牌 A，总点数超过21
        LinkedList<Card> handA = new LinkedList<>();
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        // 创建手牌 B，总点数超过21
        LinkedList<Card> handB = new LinkedList<>();
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(11, Card.Suit.SPADES));
        handB.add(new Card(4, Card.Suit.SPADES));

        // 比较两手牌，返回比较结果
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为平局
        assertEquals(0, result);
    }
    @Test
    @DisplayName("compareHands should return -1, meaning B beat A, A busted")
    // 比较两手牌的大小，其中手牌 A 爆牌
    public void compareHandsABusted() {
        // 创建手牌 A 的链表
        LinkedList<Card> handA = new LinkedList<>();
        // 向手牌 A 中添加三张牌
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(10, Card.Suit.HEARTS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        // 创建手牌 B 的链表
        LinkedList<Card> handB = new LinkedList<>();
        // 向手牌 B 中添加两张牌
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.SPADES));

        // 调用 ScoringUtils 类的 compareHands 方法比较两手牌的大小
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 -1
        assertEquals(-1, result);
    }

    @Test
    @DisplayName("compareHands should return 1, meaning A beat B, B busted")
    // 比较两手牌的大小，其中手牌 B 爆牌
    public void compareHandsBBusted() {
        // 创建手牌 A 的链表
        LinkedList<Card> handA = new LinkedList<>();
        // 向手牌 A 中添加两张牌
        handA.add(new Card(10, Card.Suit.DIAMONDS));
        handA.add(new Card(3, Card.Suit.HEARTS));
        
        // 创建手牌 B 的链表
        LinkedList<Card> handB = new LinkedList<>();
        // 向手牌 B 中添加三张牌
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(10, Card.Suit.SPADES));
        handB.add(new Card(5, Card.Suit.SPADES));

        // 调用 ScoringUtils 类的 compareHands 方法比较两手牌的大小
        int result = ScoringUtils.compareHands(handA,handB);

        // 断言比较结果为 1
        assertEquals(1, result);
    }
# 闭合前面的函数定义
```