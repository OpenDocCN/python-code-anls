# `basic-computer-games\15_Boxing\java\GameSession.java`

```py
/**
 * Game Session
 * The session store the state of the game
 */
public class GameSession {
    // 玩家对象
    private final Player player;
    // 对手对象
    private final Player opponent;
    // 对手获胜回合数
    private int opponentRoundWins = 0;
    // 玩家获胜回合数
    private int playerRoundWins = 0;

    // 玩家得分
    int playerPoints = 0;
    // 对手得分
    int opponentPoints = 0;
    // 是否被击倒
    boolean knocked = false;

    // 游戏会话构造函数
    GameSession(Player player, Player opponent) {
        this.player = player;
        this.opponent = opponent;
    }

    // 获取玩家对象
    public Player getPlayer() { return player;}
    // 获取对手对象
    public Player getOpponent() { return opponent;}

    // 设置被击倒状态
    public void setKnocked() {
        knocked = true;
    }

    // 重置得分
    public void resetPoints() {
        playerPoints = 0;
        opponentPoints = 0;
    }

    // 增加玩家得分
    public void addPlayerPoints(int ptos) { playerPoints+=ptos;}
    // 增加对手得分
    public void addOpponentPoints(int ptos) { opponentPoints+=ptos;}

    // 获取玩家得分或对手得分
    public int getPoints(Player player) {
        if(player.isPlayer())
            return playerPoints;
        else
            return opponentPoints;
    }

    // 增加获胜回合数
    public void addRoundWind(Player player) {
        if(player.isPlayer()) playerRoundWins++; else opponentRoundWins++;
    }

    // 判断游戏是否结束
    public boolean isOver() {
        return (opponentRoundWins >= 2 || playerRoundWins >= 2);
    }

    // 判断当前回合是否获胜
    public boolean isRoundWinner(Player player) {
        if (player.isPlayer())
            return playerPoints > opponentPoints;
        else
            return opponentPoints > playerPoints;
    }

    // 判断游戏是否获胜
    public boolean isGameWinner(Player player) {
        if (player.isPlayer())
            return playerRoundWins > 2;
        else
            return opponentRoundWins > 2;
    }

    // 判断玩家是否被击倒
    public boolean isPlayerKnocked() {
        return knocked;
    }
}
```