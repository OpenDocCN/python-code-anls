# `10_Blackjack\java\src\Card.java`

```
/**
 * This is an example of an "record" class in Java. That's just a fancy way
 * of saying the properties (value and suit) can't change after the object has
 * been created (it has no 'setter' methods and the properties are implicitly 'final'). 
 *
 * Immutability often makes it easier to reason about code logic and avoid
 * certain classes of bugs.
 *
 * Since it would never make sense for a card to change in the middle of a game,
 * this is a good candidate for immutability.
 */
record Card(int value, Suit suit) { // 定义一个名为Card的record类，包含value和suit两个属性

	public enum Suit { // 定义一个枚举类型Suit，包含HEARTS, DIAMONDS, SPADES, CLUBS四个值
		HEARTS, DIAMONDS, SPADES, CLUBS;
	}

	public Card { // 定义构造函数，用于初始化Card对象
        if(value < 1 || value > 13) { // 检查value的取值范围是否合法
            throw new IllegalArgumentException("Invalid card value " + value); // 如果不合法，抛出IllegalArgumentException异常
        }
        if(suit == null) {  # 如果花色为空，则抛出非法参数异常
            throw new IllegalArgumentException("Card suit must be non-null");
        }
	}

    public String toString() {  # 重写toString方法
        StringBuilder result = new StringBuilder(2);  # 创建一个StringBuilder对象，初始容量为2
        if(value == 1) {  # 如果牌面值为1，追加"A"到result
            result.append("A");
        } else if(value < 11) {  # 如果牌面值小于11，追加牌面值到result
            result.append(value);
        } else if(value == 11) {  # 如果牌面值为11，追加'J'到result
            result.append('J');
        } else if(value == 12) {  # 如果牌面值为12，追加'Q'到result
            result.append('Q');
        } else if(value == 13) {  # 如果牌面值为13，追加'K'到result
            result.append('K');
        }
        // Uncomment to include the suit in output. Useful for debugging, but  # 取消注释以在输出中包含花色。用于调试很有用，但
    // 返回 toString() 方法的值，前面加上 "AN " 或 "A "，取决于语法上的正确性
    // 当值为 "an" ace 或 "an" 8 时，返回 "AN [x]"，否则返回 "A [X]"
    public String toProseString() {
		if(value == 1 || value == 8) {
            return "AN " + toString();
        } else {
            return "A " + toString();
        }
    }
```
这段代码是一个方法，用于返回一个字符串，该字符串是调用 toString() 方法的结果，前面加上 "AN " 或 "A "，取决于语法上的正确性。当值为 "an" ace 或 "an" 8 时，返回 "AN [x]"，否则返回 "A [X]"。
```