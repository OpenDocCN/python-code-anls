# `basic-computer-games\10_Blackjack\java\test\DeckTest.java`

```

# 导入必要的断言方法
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertAll;
import org.junit.jupiter.api.Test;

# 创建测试类 DeckTest
public class DeckTest {

    # 测试初始化方法
    @Test
    void testInit() {
        # 创建一个新的牌堆对象，并进行洗牌
        Deck deck = new Deck((cards) -> cards);
        deck.reshuffle();

        # 获取牌堆中的卡片数量、花色数量和数值数量
        long nCards = deck.size();
        long nSuits = deck.getCards().stream()
                .map(card -> card.suit())
                .distinct()
                .count();
        long nValues = deck.getCards().stream()
                .map(card -> card.value())
                .distinct()
                .count();

        # 使用断言方法验证测试结果
        assertAll("deck",
            () -> assertEquals(52, nCards, "Expected 52 cards in a deck, but got " + nCards),
            () -> assertEquals(4, nSuits, "Expected 4 suits, but got " + nSuits),
            () -> assertEquals(13, nValues, "Expected 13 values, but got " + nValues)
        );
        
    }

}

```