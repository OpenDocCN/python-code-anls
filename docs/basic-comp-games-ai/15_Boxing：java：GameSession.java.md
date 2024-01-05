# `d:/src/tocomm/basic-computer-games\15_Boxing\java\GameSession.java`

```
/**
 * Game Session
 * The session store the state of the game
 */
public class GameSession {
    // 存储玩家和对手的信息
    private final Player player;
    private final Player opponent;
    // 对手获胜回合数和玩家获胜回合数
    private int opponentRoundWins = 0;
    private int playerRoundWins = 0;

    // 玩家和对手的得分
    int playerPoints = 0;
    int opponentPoints = 0;
    // 玩家是否被击倒
    boolean knocked = false;

    // 初始化游戏会话，传入玩家和对手信息
    GameSession(Player player, Player opponent) {
        this.player = player;
        this.opponent = opponent;
    }

    // 获取玩家信息
    public Player getPlayer() { return player;}
    public Player getOpponent() { return opponent;}  // 返回对手对象

    public void setKnocked() {  // 设置被击倒状态为真
        knocked = true;
    }

    public void resetPoints() {  // 重置玩家和对手的得分为0
        playerPoints = 0;
        opponentPoints = 0;
    }

    public void addPlayerPoints(int ptos) { playerPoints+=ptos;}  // 增加玩家的得分
    public void addOpponentPoints(int ptos) { opponentPoints+=ptos;}  // 增加对手的得分

    public int getPoints(Player player) {  // 获取指定玩家的得分
        if(player.isPlayer())  // 如果是玩家
            return playerPoints;  // 返回玩家的得分
        else
            return opponentPoints;  // 返回对手的得分
    }
    # 增加玩家或对手的回合获胜次数
    public void addRoundWind(Player player) {
        if(player.isPlayer()) playerRoundWins++; else opponentRoundWins++;
    }

    # 判断游戏是否结束
    public boolean isOver() {
        return (opponentRoundWins >= 2 || playerRoundWins >= 2);
    }

    # 判断当前回合是否获胜
    public boolean isRoundWinner(Player player) {
        if (player.isPlayer())
            return playerPoints > opponentPoints;
        else
            return opponentPoints > playerPoints;
    }

    # 判断游戏是否获胜
    public boolean isGameWinner(Player player) {
        if (player.isPlayer())
            return playerRoundWins > 2;
        else
```
在这个示例中，我们为给定的代码添加了注释，解释了每个方法的作用。这样做可以帮助其他程序员更容易地理解代码的功能和逻辑。
# 返回对手获胜回合数是否大于2
return opponentRoundWins > 2;
}

# 返回玩家是否被击倒
public boolean isPlayerKnocked() {
    return knocked;
}
```
在这个示例中，第一个方法返回对手获胜回合数是否大于2，第二个方法返回玩家是否被击倒。
```