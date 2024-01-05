# `10_Blackjack\java\src\Player.java`

```
import java.util.Collections; // 导入 Collections 类，用于操作集合
import java.util.LinkedList; // 导入 LinkedList 类，用于创建链表
import java.util.List; // 导入 List 接口，用于操作列表

/**
 * Represents a player and data related to them (number, bets, cards).
 */
public class Player {

    private int playerNumber;     // 玩家编号，例如 playerNumber = 1 表示 "这是玩家 1"
    private double currentBet; // 当前下注金额
    private double insuranceBet; // 保险下注金额，当玩家没有进行保险下注时为 0（要么不适用，要么选择不下注）
    private double splitBet; // 分牌下注金额，当手牌没有分牌时为 0
    private double total; // 总金额
    private LinkedList<Card> hand; // 手牌
    private LinkedList<Card> splitHand; // 分牌的手牌，当手牌没有分牌时为 null

    /**
    * Represents a player in the game with cards, bets, total and a playerNumber. 
    */
}
    # 初始化玩家对象，设置玩家编号，当前下注金额，保险下注金额，分牌下注金额，总金额，手牌和分牌
    public Player(int playerNumber) {
        this.playerNumber = playerNumber;
        currentBet = 0;
        insuranceBet = 0;
        splitBet = 0;
        total = 0;
        hand = new LinkedList<>();
        splitHand = null;
    }

    # 获取玩家编号
    public int getPlayerNumber() {
        return this.playerNumber;
    }
    
    # 获取当前下注金额
    public double getCurrentBet() {
        return this.currentBet;
    }

    # 设置当前下注金额
    public void setCurrentBet(double currentBet) {
        this.currentBet = currentBet;
    }

    public double getSplitBet() {
        return splitBet;
    }
    // 返回分牌赌注的值

    public double getInsuranceBet() {
        return insuranceBet;
    }
    // 返回保险赌注的值

    public void setInsuranceBet(double insuranceBet) {
        this.insuranceBet = insuranceBet;
    }
    // 设置保险赌注的值

    /**
    * RecordRound adds input paramater 'totalBet' to 'total' and then 
    * sets 'currentBet', 'splitBet', and 'insuranceBet' to zero
    */
    public void recordRound(double totalBet) {
        this.total = this.total + totalBet;
        // 将输入参数'totalBet'添加到'total'，然后将'currentBet'、'splitBet'和'insuranceBet'设置为零
        this.currentBet = 0;
        this.splitBet = 0;
        this.insuranceBet = 0;
    }
    // 记录一轮游戏，将输入参数'totalBet'添加到'total'，然后将'currentBet'、'splitBet'和'insuranceBet'设置为零
        this.currentBet = 0; // 初始化当前赌注为0
        this.splitBet = 0; // 初始化分牌赌注为0
        this.insuranceBet = 0; // 初始化保险赌注为0
    }

    /**
     * 返回所有赌注赢得/失去的总额。
     * @return 总值
     */
    public double getTotal() {
        return this.total; // 返回总值
    }

    /**
     * 将给定的牌添加到玩家的主手中。
     * 
     * @param card 要添加的牌。
     */
    public void dealCard(Card card) {
        dealCard(card, 1); // 调用重载的dealCard方法，传入牌和数量1
    /**
     * Adds the given card to the players hand or split hand depending on the handNumber.
     * 
     * @param card The card to add
     * @param handNumber 1 for the "first" hand and 2 for the "second" hand in a split hand scenario.
     */
    public void dealCard(Card card, int handNumber) {
        if(handNumber == 1) {  # 如果handNumber等于1
            hand.add(card);  # 将卡片添加到手牌
        } else if (handNumber == 2) {  # 如果handNumber等于2
            splitHand.add(card);  # 将卡片添加到分牌
        } else {  # 否则
            throw new IllegalArgumentException("Invalid hand number " + handNumber);  # 抛出异常，手牌编号无效
        }
    }

    /**
     * Determines whether the player is eligible to split.
    /**
     * Determines whether the player can split their hand.
     * @return True if the player has not already split, and their hand is a pair. False otherwise.
     */
    public boolean canSplit() {
        if(isSplit()) {
            // Can't split twice
            return false;
        } else {
            boolean isPair = this.hand.get(0).value() == this.hand.get(1).value();
            return isPair;
        }
    }

    /**
     * Determines whether the player has already split their hand.
     * @return false if splitHand is null, true otherwise.
     */
    public boolean isSplit() {
        return this.splitHand != null;
    }
    /**
     * Removes first card from hand to add it to new split hand
     */
    public void split() {
        // 设置分牌赌注为当前赌注
        this.splitBet = this.currentBet;
        // 创建新的分牌手牌
        this.splitHand = new LinkedList<>();
        // 从原手牌中移除第一张牌并加入到新的分牌手牌中
        splitHand.add(hand.pop());
    }

    /**
     * Determines whether the player can double down.
     * 
     * @param handNumber
     * @return
     */
    public boolean canDoubleDown(int handNumber) {
        // 如果是第一手牌
        if(handNumber == 1){
            // 判断手牌是否只有两张牌
            return this.hand.size() == 2;
        } 
        // 如果是第二手牌
        else if(handNumber == 2){
            // 判断分牌手牌是否只有两张牌
            return this.splitHand.size() == 2;
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }
    // 在给定手牌上加倍下注。具体来说，这个方法会为给定手牌加倍下注并发放给定的牌。
    // 
    // @param card 要发放的牌
    // @param handNumber 要发放并加倍下注的手牌
    public void doubleDown(Card card, int handNumber) {
        if(handNumber == 1){
            this.currentBet = this.currentBet * 2;  // 如果手牌号为1，则将当前下注金额翻倍
        } else if(handNumber == 2){
            this.splitBet = this.splitBet * 2;  // 如果手牌号为2，则将分牌下注金额翻倍
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);  // 如果手牌号不是1或2，则抛出异常
        }
        this.dealCard(card, handNumber);  // 发放给定的牌到指定的手牌上
    }

    /**
     * 重置手牌为一个空列表，并将分牌手牌设置为null。
     */
    public void resetHand() {
        this.hand = new LinkedList<>();
        this.splitHand = null;
    }

    /**
     * 返回手牌列表
     */
    public List<Card> getHand() {
        return getHand(1);
    }

    /**
     * 返回指定的手牌
     * @param handNumber 1表示分牌后的“第一”手牌（或者当没有分牌时表示主手牌），2表示分牌后的“第二”手牌。
     * @return 由handNumber指定的手牌
     */
    public List<Card> getHand(int handNumber) {
        if(handNumber == 1){  # 如果手牌编号为1
            return Collections.unmodifiableList(this.hand);  # 返回不可修改的手牌列表
        } else if(handNumber == 2){  # 否则，如果手牌编号为2
            return Collections.unmodifiableList(this.splitHand);  # 返回不可修改的分牌列表
        } else {  # 否则
            throw new IllegalArgumentException("Invalid hand number " + handNumber);  # 抛出参数异常，提示无效的手牌编号
        }
    }
}
```