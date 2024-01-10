# `basic-computer-games\10_Blackjack\java\src\Game.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Collection;  // 导入 Collection 类
import java.util.List;  // 导入 List 类
import java.text.DecimalFormat;  // 导入 DecimalFormat 类

/**
 * 这是运行游戏本身的主要类。
 */
public class Game {
    
    private Deck deck;  // 创建一个 Deck 对象
    private UserIo userIo;  // 创建一个 UserIo 对象

    public Game(Deck deck, UserIo userIo) {  // 游戏类的构造函数，接受一个 Deck 对象和一个 UserIo 对象作为参数
        this.deck = deck;  // 将传入的 Deck 对象赋值给类的 deck 属性
        this.userIo = userIo;  // 将传入的 UserIo 对象赋值给类的 userIo 属性
    }

    /**
     * 运行游戏，进行多轮游戏直到用户按下 CTRL+D/CTRL+Z 或 CTRL+C 结束
     */
    }

    protected void collectInsurance(Iterable<Player> players) {  // 保护方法，用于收集玩家的保险
        boolean isInsurance = userIo.promptBoolean("ANY INSURANCE");  // 询问用户是否购买保险
        if(isInsurance) {  // 如果用户购买了保险
            userIo.println("INSURANCE BETS");  // 输出信息提示用户进行保险下注
            for(Player player : players) {  // 遍历玩家集合
                while(true) {  // 进入循环，直到用户输入合法的保险下注金额
                    double insuranceBet = userIo.promptDouble("# " + player.getPlayerNumber() + " ");  // 询问玩家输入保险下注金额
                    // 0 表示该玩家不购买保险
                    if(insuranceBet >= 0 && insuranceBet <= (player.getCurrentBet() / 2)) {  // 如果输入的保险下注金额合法
                        player.setInsuranceBet(insuranceBet);  // 设置玩家的保险下注金额
                        break;  // 退出循环
                    }
                }
            }
        }
    }

    /**
     * 打印每个玩家的牌和庄家的明牌。
     * 以以下格式打印初始发牌：
     *        
     *    玩家1     2    庄家
     *         7    10     4   
     *         2     A   
     */
    // 打印初始发牌情况，包括玩家和庄家的牌
    private void printInitialDeal(List<Player> players, Player dealer) {
    
        // 创建一个字符串构建器
        StringBuilder output = new StringBuilder(); 
        // 添加玩家编号到输出字符串
        output.append("PLAYERS ");
        // 遍历玩家列表，添加每个玩家的编号到输出字符串
        for (Player player : players) {
            output.append(player.getPlayerNumber() + "\t");
        }
        // 添加庄家到输出字符串
        output.append("DEALER\n");
        // 循环两次，分别处理两行牌
        for (int j = 0; j < 2; j++) {
            output.append("\t");
            // 遍历玩家列表，添加每个玩家当前行的牌到输出字符串
            for (Player player : players) {
                output.append(player.getHand().get(j).toString()).append("\t");
            }
            // 如果是第一行，添加庄家当前行的牌到输出字符串
            if(j == 0 ){
                output.append(dealer.getHand().get(j).toString());
            }
            // 添加换行符到输出字符串
            output.append("\n");
        }
        // 打印输出字符串
        userIo.print(output.toString());
    }

    /**
     * 玩家的回合。提示玩家是否要牌（H）、停牌（S），或者如果适用，分牌（/）或加倍（D），然后执行相应的操作。在要牌时，打印"RECEIVED A  [x]  HIT? "
     * 
     * @param player
     */
    protected void play(Player player) {
        play(player, 1);
    }

    /**
     * 检查庄家的手牌是否需要继续打牌。如果每个玩家要么爆牌要么自然21点获胜，庄家就不需要继续打牌。
     * 
     * @param players
     * @return 庄家是否需要继续打牌的布尔值
     */
    protected boolean shouldPlayDealer(List<Player> players){
        for(Player player : players){
            // 计算玩家手牌的分数
            int score = ScoringUtils.scoreHand(player.getHand());
            // 如果分数小于21或者等于21且手牌数量大于2，返回true
            if(score < 21 || (score == 21 && player.getHand().size() > 2)){
                return true;
            }
            // 如果玩家分牌
            if(player.isSplit()){                
                // 计算分牌后的手牌分数
                int splitScore = ScoringUtils.scoreHand(player.getHand(2));
                // 如果分数小于21或者等于21且手牌数量大于2，返回true
                if(splitScore < 21 || (splitScore == 21 && player.getHand(2).size() > 2)){
                    return true;
                }
            }
        }
        // 如果以上条件都不满足，返回false
        return false;
    }    
    /**
     * 播放庄家的手牌。庄家抽牌直到手牌总点数>=17或爆牌。打印每次抽牌的结果，如下例所示：
     * 
     * DEALER HAS A  5 CONCEALED FOR A TOTAL OF 11 
     * DRAWS 10   ---TOTAL IS 21
     *  
     * @param dealerHand
     */
    protected void playDealer(Player dealer) {
        // 计算庄家手牌的总点数
        int score = ScoringUtils.scoreHand(dealer.getHand());
        // 打印庄家的一张暗牌和总点数
        userIo.println("DEALER HAS " + dealer.getHand().get(1).toProseString() + " CONCEALED FOR A TOTAL OF " + score);

        // 如果庄家手牌总点数小于17，则继续抽牌
        if(score < 17){
            userIo.print("DRAWS");
        }
        // 当庄家手牌总点数小于17时，持续抽牌直到总点数>=17
        while(score < 17) {
            // 从牌堆中抽一张牌
            Card dealtCard = deck.deal();
            // 将抽到的牌加入庄家手牌
            dealer.dealCard(dealtCard);
            // 重新计算庄家手牌的总点数
            score = ScoringUtils.scoreHand(dealer.getHand());
            // 打印抽到的牌
            userIo.print("  " + String.format("%-4s", dealtCard.toString()));
        }
        
        // 如果庄家手牌总点数>21，则爆牌
        if(score > 21) {
            userIo.println("...BUSTED\n");
        } else {
            // 否则打印庄家手牌的总点数
            userIo.println("---TOTAL IS " + score + "\n");
        }
    }

    /**
     * 评估本轮的结果，打印结果，并更新玩家/庄家的总分。
     * 
     *    PLAYER 1 LOSES   100 TOTAL=-100 
     *    PLAYER 2  WINS   150 TOTAL= 150
     *    DEALER'S TOTAL= 200
      *
     * @param players
     * @param dealerHand
     */
    // 评估每一轮玩家的表现
    protected void evaluateRound(List<Player> players, Player dealer) {
        // 格式化数字，去除尾随的零
        DecimalFormat formatter = new DecimalFormat("0.#"); 
        // 遍历玩家列表
        for(Player player : players){
            // 比较玩家手牌和庄家手牌的结果
            int result = ScoringUtils.compareHands(player.getHand(), dealer.getHand());
            // 初始化总下注金额
            double totalBet = 0;
            // 根据比较结果更新总下注金额
            if(result > 0) {
                totalBet += player.getCurrentBet();
            } else if(result < 0){
                totalBet -= player.getCurrentBet();
            }
            // 如果玩家分牌
            if(player.isSplit()) {
                // 比较分牌后的手牌和庄家手牌的结果
                int splitResult = ScoringUtils.compareHands(player.getHand(2), dealer.getHand());
                // 根据比较结果更新总下注金额
                if(splitResult > 0){
                    totalBet += player.getSplitBet();
                } else if(splitResult < 0){
                    totalBet -= player.getSplitBet();
                } 
            }
            // 如果玩家购买了保险
            if(player.getInsuranceBet() != 0){
                // 计算庄家手牌的得分
                int dealerResult = ScoringUtils.scoreHand(dealer.getHand());
                // 根据庄家手牌的得分和数量更新总下注金额
                if(dealerResult == 21 && dealer.getHand().size() == 2){
                    totalBet += (player.getInsuranceBet() * 2);
                } else {
                    totalBet -= player.getInsuranceBet();
                }
            }
            
            // 输出玩家编号和结果
            userIo.print("PLAYER " + player.getPlayerNumber());
            if(totalBet < 0) {
                userIo.print(" LOSES " + String.format("%6s", formatter.format(Math.abs(totalBet))); 
            } else if(totalBet > 0) {
                userIo.print("  WINS " + String.format("%6s", formatter.format(totalBet))); 
            } else {
                userIo.print(" PUSHES      ");
            }
            // 记录玩家这一轮的结果
            player.recordRound(totalBet);
            // 记录庄家这一轮的结果
            dealer.recordRound(totalBet * (-1));
            // 输出玩家的总金额
            userIo.println(" TOTAL= " + formatter.format(player.getTotal()));
            // 重置玩家手牌
            player.resetHand();
        }
        // 输出庄家的总金额
        userIo.println("DEALER'S TOTAL= " + formatter.format(dealer.getTotal()) + "\n");
        // 重置庄家手牌
        dealer.resetHand();
    }
    /**
     * 验证所有的赌注是否在0（不包括）到500（包括）之间。分数赌注也是有效的。
     * 
     * @param players 拥有其当前赌注的玩家集合
     * @return 如果所有赌注都有效则返回true，否则返回false
     */
    public boolean betsAreValid(Collection<Player> players) {
        // 使用流处理每个玩家的当前赌注，检查是否都在0到500之间
        return players.stream()
            .map(Player::getCurrentBet)
            .allMatch(bet -> bet > 0 && bet <= 500);
    }
# 闭合前面的函数定义
```