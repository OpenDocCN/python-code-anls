# `basic-computer-games\10_Blackjack\java\src\Deck.java`

```

// 导入所需的类
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

// 创建 Deck 类
public class Deck {

    // 用于存储卡牌的链表
    private LinkedList<Card> cards;
    // 用于洗牌算法的函数
    private Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm;
    
    /**
     * 用给定数量的标准牌组初始化游戏牌组。
     * 例如，如果要使用2副牌进行游戏，则 {@code new Decks(2)} 将使用2副标准的52张牌初始化 'cards'。
     * 
     * @param shuffleAlgorithm 一个函数，接受初始排序的卡牌列表，并返回一个洗好的列表以便发牌。
     * 
     */
    public Deck(Function<LinkedList<Card>, LinkedList<Card>> shuffleAlgorithm) {
        this.shuffleAlgorithm = shuffleAlgorithm;
    }

    /**
     * 从牌组中发一张牌，从该对象的状态中移除它。如果牌组为空，则在发牌之前将重新洗牌。
     * 
     * @return 被发出的牌。
     */
    public Card deal() {
        if(cards == null || cards.isEmpty()) {
            reshuffle();
        }
        return cards.pollFirst();
    }

    /**
     * 使用 shuffleAlgorithm 对该牌组中的牌进行洗牌。
     */
    public void reshuffle() {
        LinkedList<Card> newCards = new LinkedList<>();
        for(Card.Suit suit : Card.Suit.values()) {
            for(int value = 1; value < 14; value++) {
                newCards.add(new Card(value, suit));
            }
        }
        this.cards = this.shuffleAlgorithm.apply(newCards);
    }

    /**
     * 获取该牌组中的牌的数量。
     * @return 该牌组中的牌的数量。例如，单副牌为52张。
     */
    public int size() {
        return cards.size();
    }

    /**
     * 返回该牌组中的牌。
     * @return 该牌组中的牌的不可变视图。
     */
    public List<Card> getCards() {
        // 返回的列表是不可变的，因为我们不希望其他代码干扰牌组。
        return Collections.unmodifiableList(cards);
    }
}

```