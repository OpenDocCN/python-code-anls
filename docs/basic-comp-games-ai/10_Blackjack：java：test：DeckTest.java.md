# `d:/src/tocomm/basic-computer-games\10_Blackjack\java\test\DeckTest.java`

```
import static org.junit.jupiter.api.Assertions.assertEquals;  // 导入静态的 assertEquals 方法
import static org.junit.jupiter.api.Assertions.assertAll;  // 导入静态的 assertAll 方法
import org.junit.jupiter.api.Test;  // 导入 Test 类

public class DeckTest {  // 定义 DeckTest 类

    @Test  // 声明该方法为测试方法
    void testInit() {  // 定义 testInit 方法
        // When
        Deck deck = new Deck((cards) -> cards);  // 创建一个 Deck 对象，传入一个 lambda 表达式
        deck.reshuffle();  // 调用 Deck 对象的 reshuffle 方法，重新洗牌

        // Then
        long nCards = deck.size();  // 获取 Deck 对象的牌数
        long nSuits = deck.getCards().stream()  // 获取 Deck 对象的牌列表，转换为流
                .map(card -> card.suit())  // 对每张牌获取花色
                .distinct()  // 去重
                .count();  // 统计数量
        long nValues = deck.getCards().stream()  // 获取 Deck 对象的牌列表，转换为流
                .map(card -> card.value())  // 对每张牌获取点数
                .distinct() // 从流中去除重复的元素
                .count(); // 计算流中元素的个数

        assertAll("deck", // 使用给定的组名执行所有断言
            () -> assertEquals(52, nCards, "Expected 52 cards in a deck, but got " + nCards), // 断言nCards的值为52，如果不是则抛出异常
            () -> assertEquals(4, nSuits, "Expected 4 suits, but got " + nSuits), // 断言nSuits的值为4，如果不是则抛出异常
            () -> assertEquals(13, nValues, "Expected 13 values, but got " + nValues) // 断言nValues的值为13，如果不是则抛出异常
        );
        
    }

}
```