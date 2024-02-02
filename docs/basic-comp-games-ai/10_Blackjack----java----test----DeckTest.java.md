# `basic-computer-games\10_Blackjack\java\test\DeckTest.java`

```py
import static org.junit.jupiter.api.Assertions.assertEquals;  // 导入静态的断言方法，用于断言测试结果是否符合预期
import static org.junit.jupiter.api.Assertions.assertAll;  // 导入静态的断言方法，用于同时验证多个断言
import org.junit.jupiter.api.Test;  // 导入 JUnit 的测试注解

public class DeckTest {

    @Test
    void testInit() {  // 测试初始化方法
        // 创建一个新的牌堆对象，使用 lambda 表达式对牌进行洗牌
        Deck deck = new Deck((cards) -> cards);
        deck.reshuffle();  // 洗牌

        // 获取牌堆中的卡牌数量
        long nCards = deck.size();
        // 获取牌堆中的花色数量
        long nSuits = deck.getCards().stream()
                .map(card -> card.suit())  // 获取每张卡牌的花色
                .distinct()  // 去重
                .count();  // 统计数量
        // 获取牌堆中的点数数量
        long nValues = deck.getCards().stream()
                .map(card -> card.value())  // 获取每张卡牌的点数
                .distinct()  // 去重
                .count();  // 统计数量

        // 使用 assertAll 方法同时验证多个断言
        assertAll("deck",
            () -> assertEquals(52, nCards, "Expected 52 cards in a deck, but got " + nCards),  // 验证卡牌数量是否为 52
            () -> assertEquals(4, nSuits, "Expected 4 suits, but got " + nSuits),  // 验证花色数量是否为 4
            () -> assertEquals(13, nValues, "Expected 13 values, but got " + nValues)  // 验证点数数量是否为 13
        );
        
    }

}
```