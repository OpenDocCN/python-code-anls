# `basic-computer-games\15_Boxing\java\GameSession.java`

```

/**
 * Game Session
 * The session store the state of the game
 */
public class GameSession {
    // 存储玩家和对手的信息
    private final Player player;
    private final Player opponent;
    // 对手和玩家的回合获胜次数
    private int opponentRoundWins = 0;
    private int playerRoundWins = 0;

    // 玩家和对手的得分
    int playerPoints = 0;
    int opponentPoints = 0;
    // 玩家是否被击倒
    boolean knocked = false;

    // 构造函数，初始化玩家和对手
    GameSession(Player player, Player opponent) {
        this.player = player;
        this.opponent = opponent;
    }

    // 获取玩家和对手的方法
    public Player getPlayer() { return player;}
    public Player getOpponent() { return opponent;}

    // 设置玩家被击倒的方法
    public void setKnocked() {
        knocked = true;
    }

    // 重置玩家和对手的得分
    public void resetPoints() {
        playerPoints = 0;
        opponentPoints = 0;
    }

    // 增加玩家和对手的得分
    public void addPlayerPoints(int ptos) { playerPoints+=ptos;}
    public void addOpponentPoints(int ptos) { opponentPoints+=ptos;}

    // 获取玩家和对手的得分
    public int getPoints(Player player) {
        if(player.isPlayer())
            return playerPoints;
        else
            return opponentPoints;
    }

    // 增加玩家和对手的回合获胜次数
    public void addRoundWind(Player player) {
        if(player.isPlayer()) playerRoundWins++; else opponentRoundWins++;
    }

    // 判断游戏是否结束
    public boolean isOver() {
        return (opponentRoundWins >= 2 || playerRoundWins >= 2);
    }

    // 判断当前回合的赢家
    public boolean isRoundWinner(Player player) {
        if (player.isPlayer())
            return playerPoints > opponentPoints;
        else
            return opponentPoints > playerPoints;
    }

    // 判断游戏的赢家
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