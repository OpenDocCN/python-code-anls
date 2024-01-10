# `basic-computer-games\10_Blackjack\java\src\Player.java`

```
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

/**
 * Represents a player and data related to them (number, bets, cards).
 */
public class Player {

    private int playerNumber;     // e.g. playerNumber = 1 means "this is Player 1"
    private double currentBet;
    private double insuranceBet; // 0 when the player has not made an insurance bet (either it does not apply or they chose not to)
    private double splitBet; // 0 whenever the hand is not split
    private double total;
    private LinkedList<Card> hand;
    private LinkedList<Card> splitHand; // null whenever the hand is not split

    /**
    * Represents a player in the game with cards, bets, total and a playerNumber. 
    */
    public Player(int playerNumber) {
        this.playerNumber = playerNumber;
        currentBet = 0;
        insuranceBet = 0;
        splitBet = 0;
        total = 0;
        hand = new LinkedList<>();  // 创建一个空的卡牌列表
        splitHand = null;  // 将分牌的卡牌列表初始化为null
    }

    public int getPlayerNumber() {
        return this.playerNumber;  // 返回玩家编号
    }
    
    public double getCurrentBet() {
        return this.currentBet;  // 返回当前赌注
    }

    public void setCurrentBet(double currentBet) {
        this.currentBet = currentBet;  // 设置当前赌注
    }

    public double getSplitBet() {
        return splitBet;  // 返回分牌赌注
    }

    public double getInsuranceBet() {
        return insuranceBet;  // 返回保险赌注
    }

    public void setInsuranceBet(double insuranceBet) {
        this.insuranceBet = insuranceBet;  // 设置保险赌注
    }

    /**
    * RecordRound adds input paramater 'totalBet' to 'total' and then 
    * sets 'currentBet', 'splitBet', and 'insuranceBet' to zero
    */
    public void recordRound(double totalBet) {
        this.total = this.total + totalBet;  // 将总赌注加上本轮的赌注
        this.currentBet = 0;  // 将当前赌注设为0
        this.splitBet = 0;  // 将分牌赌注设为0
        this.insuranceBet = 0;  // 将保险赌注设为0
    }

    /**
     * Returns the total of all bets won/lost.
     * @return Total value
     */
    public double getTotal() {
        return this.total;  // 返回总赌注
    }
}
    /**
     * 将给定的卡牌添加到玩家的主手中。
     * 
     * @param card 要添加的卡牌。
     */
    public void dealCard(Card card) {
        dealCard(card, 1);
    }
    
    /**
     * 根据手号将给定的卡牌添加到玩家的手或分牌手中。
     * 
     * @param card 要添加的卡牌
     * @param handNumber 在分牌情况下，1表示“第一”手，2表示“第二”手。
     */
    public void dealCard(Card card, int handNumber) {
        if(handNumber == 1) {
            hand.add(card);
        } else if (handNumber == 2) {
            splitHand.add(card);
        } else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }

    /**
     * 确定玩家是否有资格分牌。
     * @return 如果玩家尚未分牌，并且他们的手是一对，则返回true。否则返回false。
     */
    public boolean canSplit() {
        if(isSplit()) {
            // 不能再次分牌
            return false;
        } else {
            boolean isPair = this.hand.get(0).value() == this.hand.get(1).value();
            return isPair;
        }
    }

    /**
     * 确定玩家是否已经分牌。
     * @return 如果splitHand为null，则返回false，否则返回true。
     */
    public boolean isSplit() {
        return this.splitHand != null;
    }

    /**
     * 从手中移除第一张卡牌，将其添加到新的分牌手中
     */
    public void split() {
        this.splitBet = this.currentBet;
        this.splitHand = new LinkedList<>();
        splitHand.add(hand.pop());
    }

    /**
     * 确定玩家是否可以加倍。
     * 
     * @param handNumber
     * @return
     */
    // 检查是否可以加倍下注
    public boolean canDoubleDown(int handNumber) {
        // 如果是第一手牌，检查手牌是否为两张
        if(handNumber == 1){
            return this.hand.size() == 2;
        } 
        // 如果是第二手牌，检查分牌后的手牌是否为两张
        else if(handNumber == 2){
            return this.splitHand.size() == 2;
        } 
        // 如果手牌编号既不是1也不是2，抛出异常
        else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }

    /**
     * 对给定的手牌加倍下注。具体来说，这个方法会为给定的手牌加倍下注并发放给定的牌。
     * 
     * @param card 要发放的牌
     * @param handNumber 要发放并加倍下注的手牌
     */
    public void doubleDown(Card card, int handNumber) {
        // 如果是第一手牌，将当前下注额翻倍
        if(handNumber == 1){
            this.currentBet = this.currentBet * 2;
        } 
        // 如果是第二手牌，将分牌后的下注额翻倍
        else if(handNumber == 2){
            this.splitBet = this.splitBet * 2;
        } 
        // 如果手牌编号既不是1也不是2，抛出异常
        else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
        // 发放牌给指定的手牌
        this.dealCard(card, handNumber);
    }

    /**
     * 将手牌重置为空列表，并将分牌后的手牌重置为null。
     */
    public void resetHand() {
        this.hand = new LinkedList<>();
        this.splitHand = null;
    }

    // 获取第一手牌
    public List<Card> getHand() {
        return getHand(1);
    }

    /**
     * 返回指定的手牌
     * @param handNumber 1表示分牌后的“第一”手牌（或者当没有分牌时表示主手牌），2表示分牌后的“第二”手牌。
     * @return 由handNumber指定的手牌
     */
    public List<Card> getHand(int handNumber) {
        // 如果是第一手牌，返回不可修改的手牌列表
        if(handNumber == 1){
            return Collections.unmodifiableList(this.hand);
        } 
        // 如果是第二手牌，返回不可修改的分牌后的手牌列表
        else if(handNumber == 2){
            return Collections.unmodifiableList(this.splitHand);
        } 
        // 如果手牌编号既不是1也不是2，抛出异常
        else {
            throw new IllegalArgumentException("Invalid hand number " + handNumber);
        }
    }
# 闭合函数或代码块的结束
```