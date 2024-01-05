# `d:/src/tocomm/basic-computer-games\10_Blackjack\java\src\Deck.java`

```
import java.util.Collections; // 导入 Collections 类，用于操作集合
import java.util.LinkedList; // 导入 LinkedList 类，用于创建链表
import java.util.List; // 导入 List 接口，用于操作列表
import java.util.function.Function; // 导入 Function 函数式接口，用于定义函数

public class Deck {

    private LinkedList<Card> cards; // 创建名为 cards 的链表，存储 Card 对象
    private Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm; // 创建名为 shuffleAlgorithm 的函数，用于洗牌算法
    
    /**
     * Initialize the game deck with the given number of standard decks.
     * e.g. if you want to play with 2 decks, then {@code new Decks(2)} will
     * initialize 'cards' with 2 copies of a standard 52 card deck.
     * 
     * @param shuffleAlgorithm A function that takes the initial sorted card
     * list and returns a shuffled list ready to deal.
     * 
     */
    public Deck(Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm) { // 构造函数，接受一个洗牌算法作为参数
        this.shuffleAlgorithm = shuffleAlgorithm;  // 设置洗牌算法

    }

    /**
     * 从牌堆中发一张牌，从牌堆中移除它。如果牌堆为空，在发牌之前将重新洗牌。
     * 
     * @return 被发出的牌。
     */
    public Card deal() {
        if(cards == null || cards.isEmpty()) {  // 如果牌堆为空
            reshuffle();  // 重新洗牌
        }
        return cards.pollFirst();  // 返回被发出的牌
    }

    /**
     * 使用洗牌算法重新洗牌。
     */
    public void reshuffle() {
        LinkedList<Card> newCards = new LinkedList<>();  // 创建一个新的链表用于存储卡牌
        for(Card.Suit suit : Card.Suit.values()) {  // 遍历所有花色
            for(int value = 1; value < 14; value++) {  // 遍历每种花色的牌值
                newCards.add(new Card(value, suit));  // 向新的链表中添加新的卡牌
            }
        }
        this.cards = this.shuffleAlgorithm.apply(newCards);  // 使用洗牌算法对新的卡牌进行洗牌
    }

    /**
     * Get the number of cards in this deck.
     * @return The number of cards in this deck. For example, 52 for a single deck.
     */
    public int size() {
        return cards.size();  // 返回卡牌的数量
    }

    /**
     * Returns the cards in this deck.
     * @return An immutable view of the cards in this deck.
    */
    public List<Card> getCards() {
        // 返回的列表是不可变的，因为我们不希望其他代码修改牌组。
        return Collections.unmodifiableList(cards);
    }
}
```

在这段代码中，`getCards` 方法返回了一个不可变的牌组列表，以防止其他代码修改牌组。
```