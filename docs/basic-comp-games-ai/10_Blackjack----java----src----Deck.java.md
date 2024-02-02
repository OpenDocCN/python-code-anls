# `basic-computer-games\10_Blackjack\java\src\Deck.java`

```py
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

public class Deck {

    private LinkedList<Card> cards; // 用于存储卡牌的链表
    private Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm; // 用于洗牌算法的函数

    /**
     * 初始化游戏牌组，使用给定数量的标准牌组。
     * 例如，如果要使用2副牌进行游戏，则 {@code new Decks(2)} 将使用2副标准的52张牌初始化 'cards'。
     * 
     * @param shuffleAlgorithm 一个函数，接受初始排序的卡牌列表，并返回一个洗好的列表以便发牌。
     * 
     */
    public Deck(Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm) {
        this.shuffleAlgorithm = shuffleAlgorithm; // 初始化洗牌算法
    }

    /**
     * 从牌组中发一张牌，从对象的状态中移除它。如果牌组为空，则在发牌之前将重新洗牌。
     * 
     * @return 被发出的牌。
     */
    public Card deal() {
        if(cards == null || cards.isEmpty()) { // 如果牌组为空
            reshuffle(); // 重新洗牌
        }
        return cards.pollFirst(); // 返回被发出的牌
    }

    /**
     * 使用洗牌算法对牌组进行洗牌。
     */
    public void reshuffle() {
        LinkedList<Card> newCards = new LinkedList<>(); // 创建一个新的牌组
        for(Card.Suit suit : Card.Suit.values()) { // 遍历花色
            for(int value = 1; value < 14; value++) { // 遍历牌面值
                newCards.add(new Card(value, suit)); // 向新牌组中添加牌
            }
        }
        this.cards = this.shuffleAlgorithm.apply(newCards); // 使用洗牌算法对牌组进行洗牌
    }

    /**
     * 获取牌组中的牌数。
     * @return 牌组中的牌数。例如，单副牌为52张。
     */
    public int size() {
        return cards.size(); // 返回牌组中的牌数
    }

    /**
     * 返回牌组中的牌。
     * @return 牌组中的牌的不可变视图。
     */
}
    # 获取卡片列表的方法
    public List<Card> getCards() {
        # 返回的列表是不可变的，因为我们不希望其他代码修改卡片组
        return Collections.unmodifiableList(cards);
    }
# 闭合前面的函数定义
```