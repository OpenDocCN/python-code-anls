# `basic-computer-games\10_Blackjack\java\src\Card.java`

```
/**
 * 这是 Java 中的一个“record”类的示例。这只是一种说法，即属性（value 和 suit）在对象创建后不能更改（它没有 'setter' 方法，并且属性隐式为 'final'）。
 *
 * 不可变性通常使得更容易推理代码逻辑并避免某些类别的错误。
 *
 * 由于在游戏中卡牌在中途更改永远没有意义，因此这是不可变性的一个很好的候选对象。
 */
record Card(int value, Suit suit) {

    public enum Suit {
        HEARTS, DIAMONDS, SPADES, CLUBS;
    }

    public Card {
        if(value < 1 || value > 13) {
            throw new IllegalArgumentException("Invalid card value " + value);
        }
        if(suit == null) {
            throw new IllegalArgumentException("Card suit must be non-null");
        }
    }

    public String toString() {
        StringBuilder result = new StringBuilder(2); 
        if(value == 1) {
            result.append("A");
        } else if(value < 11) {
            result.append(value);
        } else if(value == 11) {
            result.append('J');
        } else if(value == 12) {
            result.append('Q');
        } else if(value == 13) {
            result.append('K');
        }
        // 取消注释以在输出中包含花色。用于调试很有用，但不符合原始 BASIC 的行为。
        // result.append(suit.name().charAt(0));
        return result.toString();
    }

    /**
     * 返回 {@link #toString()} 的值，前面加上 "AN " 或 "A "，取决于语法是否正确。
     * 
     * @return 当 [x] 是 "an" ace 或 "an" 8 时返回 "AN [x]"，否则返回 "A [X]"。
     */
    public String toProseString() {
        if(value == 1 || value == 8) {
            return "AN " + toString();
        } else {
            return "A " + toString();
        }
    }

}
```