# `basic-computer-games\15_Boxing\java\Boxing.java`

```py
/**
 * 拳击游戏
 *
 * <p>
 * 基于这里的基本游戏BatNum
 * https://github.com/coding-horror/basic-computer-games/tree/main/15%20Boxing
 * <p>
 */
public class Boxing {

    private static final Basic.Console console = new Basic.Console();

    private GameSession session;

    public void play() {
        showIntro(); // 显示游戏介绍

        loadPlayers(); // 加载玩家信息

        console.print("%s'S ADVANTAGE IS %d AND VULNERABILITY IS SECRET.\n", session.getOpponent().getName(), session.getOpponent().getBestPunch().getCode()); // 打印对手的优势和脆弱性

        for (int roundNro = 1; roundNro <= 3; roundNro++) { // 进行3轮比赛
            if (session.isOver()) // 如果比赛结束，跳出循环
                break;

            session.resetPoints(); // 重置比分
            console.print("\nROUND %d BEGINS...%n", roundNro); // 打印当前轮次开始

            for (int majorPunches = 1; majorPunches <= 7; majorPunches++) { // 进行7次主要出拳
                long i = Basic.randomOf(10); // 生成一个0-9的随机数

                if (i > 5) { // 如果随机数大于5
                    boolean stopPunches = opponentPunch(); // 对手出拳
                    if (stopPunches ) break; // 如果对手出拳后比赛结束，跳出循环
                } else {
                    playerPunch(); // 玩家出拳
                }
            }
            showRoundWinner(roundNro); // 显示当前轮次的获胜者
        }
        showWinner(); // 显示比赛的最终获胜者
    }

    private void showRoundWinner(int roundNro) {
        if (session.isRoundWinner(session.getPlayer())) { // 如果玩家赢得了这一轮
            console.print("\n %s WINS ROUND %d\n", session.getPlayer().getName(), roundNro); // 打印玩家赢得这一轮
            session.addRoundWind(session.getPlayer()); // 将这一轮的胜利记录给玩家
        } else {
            console.print("\n %s WINS ROUND %d\n", session.getOpponent().getName(), roundNro); // 打印对手赢得这一轮
            session.addRoundWind(session.getOpponent()); // 将这一轮的胜利记录给对手
        }
    }
}
    // 显示获胜者的信息
    private void showWinner() {
        // 如果对手是游戏的获胜者
        if (session.isGameWinner(session.getOpponent())) {
            console.print("%s WINS (NICE GOING, " + session.getOpponent().getName() + ").", session.getOpponent().getName());
        } 
        // 如果玩家是游戏的获胜者
        else if (session.isGameWinner(session.getPlayer())) {
            console.print("%s AMAZINGLY WINS!!", session.getPlayer().getName());
        } 
        // 如果玩家被击倒
        else if (session.isPlayerKnocked()) {
            console.print("%s IS KNOCKED COLD AND %s IS THE WINNER AND CHAMP!", session.getPlayer().getName(), session.getOpponent().getName());
        } 
        // 如果对手被击倒
        else {
            console.print("%s IS KNOCKED COLD AND %s IS THE WINNER AND CHAMP!", session.getOpponent().getName(), session.getPlayer().getName());
        }

        console.print("\n\nAND NOW GOODBYE FROM THE OLYMPIC ARENA.\n");
    }

    // 加载玩家信息
    private void loadPlayers() {
        console.print("WHAT IS YOUR OPPONENT'S NAME? ");
        final String opponentName = console.readLine();

        console.print("INPUT YOUR MAN'S NAME? ");
        final String playerName = console.readLine();

        console.print("DIFFERENT PUNCHES ARE: (1) FULL SWING; (2) HOOK; (3) UPPERCUT; (4) JAB.\n");
        console.print("WHAT IS YOUR MANS BEST? ");

        final int b = console.readInt();

        console.print("WHAT IS HIS VULNERABILITY? ");
        final int d = console.readInt();

        // 创建玩家对象
        final Player player = new Player(playerName, Punch.fromCode(b), Punch.fromCode(d));
        final Player opponent = new Player(opponentName);

        // 创建游戏会话
        session = new GameSession(player, opponent);
    }

    // 显示游戏介绍
    private void showIntro () {
        console.print("                                 BOXING\n");
        console.print("               CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n");
        console.print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)\n\n");
    }
# 闭合前面的函数定义
```