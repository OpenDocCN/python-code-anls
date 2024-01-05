# `d:/src/tocomm/basic-computer-games\18_Bullseye\java\src\Bullseye.java`

```
import java.util.ArrayList;  // 导入 ArrayList 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Game of Bullseye
 * <p>
 * Based on the Basic game of Bullseye here
 * https://github.com/coding-horror/basic-computer-games/blob/main/18%20Bullseye/bullseye.bas
 * <p>
 * Note:  The idea was to create a version of 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Bullseye {

    // Used for formatting output
    public static final int FIRST_IDENT = 10;  // 定义常量 FIRST_IDENT 为 10，用于格式化输出
    public static final int SECOND_IDENT = 30;  // 定义常量 SECOND_IDENT 为 30，用于格式化输出
    public static final int THIRD_INDENT = 30;  // 定义常量 THIRD_INDENT 为 30，用于格式化输出

    // Used to decide throw result
    public static final double[] SHOT_ONE = new double[]{.65, .55, .5, .5}; // 定义名为SHOT_ONE的常量数组，存储四个double类型的数值
    public static final double[] SHOT_TWO = new double[]{.99, .77, .43, .01}; // 定义名为SHOT_TWO的常量数组，存储四个double类型的数值
    public static final double[] SHOT_THREE = new double[]{.95, .75, .45, .05}; // 定义名为SHOT_THREE的常量数组，存储四个double类型的数值

    private enum GAME_STATE { // 定义一个枚举类型GAME_STATE，包含四个状态：STARTING, START_GAME, PLAYING, GAME_OVER
        STARTING,
        START_GAME,
        PLAYING,
        GAME_OVER
    }

    private GAME_STATE gameState; // 声明一个私有的GAME_STATE类型变量gameState

    private final ArrayList<Player> players; // 声明一个私有的ArrayList类型变量players，存储Player对象

    private final Shot[] shots; // 声明一个私有的数组变量shots，存储Shot对象

    // Used for keyboard input
    private final Scanner kbScanner; // 声明一个私有的Scanner类型变量kbScanner，用于处理键盘输入
    private int round; // 声明一个整型变量 round

    public Bullseye() { // Bullseye 类的构造函数

        gameState = GAME_STATE.STARTING; // 设置游戏状态为开始状态
        players = new ArrayList<>(); // 创建一个空的玩家列表

        // Save the random chances of points based on shot type
        // 根据射击类型保存随机得分的机会

        shots = new Shot[3]; // 创建一个长度为3的 Shot 数组
        shots[0] = new Shot(SHOT_ONE); // 初始化第一个 Shot 对象
        shots[1] = new Shot(SHOT_TWO); // 初始化第二个 Shot 对象
        shots[2] = new Shot(SHOT_THREE); // 初始化第三个 Shot 对象

        // Initialise kb scanner
        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in); // 创建一个 Scanner 对象，用于从键盘输入
    }

    /**
     * Main game loop
     * 主游戏循环
    */
    public void play() {

        do {
            switch (gameState) {

                // Show an introduction the first time the game is played.
                case STARTING:
                    intro();  #调用intro()函数，显示游戏介绍
                    gameState = GAME_STATE.START_GAME;  #将游戏状态设置为开始游戏
                    break;

                // Start the game, set the number of players, names and round
                case START_GAME:

                    int numberOfPlayers = chooseNumberOfPlayers();  #调用chooseNumberOfPlayers()函数，选择玩家数量

                    for (int i = 0; i < numberOfPlayers; i++) {
                        String name = displayTextAndGetInput("NAME OF PLAYER #" + (i + 1) + "? ");  #显示并获取玩家名称
                        Player player = new Player(name);  #创建一个新的Player对象，传入玩家名称作为参数
                    }

                    this.round = 1;  // 设置当前游戏回合为第一回合

                    gameState = GAME_STATE.PLAYING;  // 设置游戏状态为进行中
                    break;

                // Playing round by round until we have a winner
                case PLAYING:  // 当游戏状态为进行中时
                    System.out.println();  // 打印空行
                    System.out.println("ROUND " + this.round);  // 打印当前回合数
                    System.out.println("=======");  // 打印分隔线

                    // Each player takes their turn
                    for (Player player : players) {  // 遍历每位玩家
                        int playerThrow = getPlayersThrow(player);  // 获取玩家的投掷结果
                        int points = calculatePlayerPoints(playerThrow);  // 计算玩家得分
                        player.addScore(points);  // 将得分加到玩家的总分上
                        System.out.println("TOTAL SCORE = " + player.getScore());  // 打印玩家的总分
                    }

                    boolean foundWinner = false;  // 声明一个布尔变量，用于标记是否找到了赢家

                    // 检查是否有任何玩家获胜
                    for (Player thePlayer : players) {  // 遍历玩家列表
                        int score = thePlayer.getScore();  // 获取玩家的分数
                        if (score >= 200) {  // 如果玩家的分数大于等于200
                            if (!foundWinner) {  // 如果还没有找到赢家
                                System.out.println("WE HAVE A WINNER!!");  // 打印出“我们有一个赢家！”
                                System.out.println();  // 打印空行
                                foundWinner = true;  // 将找到赢家的标记设为true
                            }
                            System.out.println(thePlayer.getName() + " SCORED "
                                    + thePlayer.getScore() + " POINTS");  // 打印出玩家的名字和分数
                        }
                    }

                    if (foundWinner) {  // 如果找到了赢家
                        System.out.println("THANKS FOR THE GAME.");  // 打印出“谢谢参与游戏。”
                        gameState = GAME_STATE.GAME_OVER;  // 设置游戏状态为游戏结束
                    } else {
                        // 没有找到赢家，继续下一轮
                        this.round++;  // 增加回合数
                    }

                    break;  // 跳出循环
            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时继续循环
    }

    /**
     * 显示游戏信息
     */
    private void intro() {
        System.out.println("BULLSEYE");  // 打印"BULLSEYE"
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
        System.out.println();  // 打印空行
        System.out.println("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET");  // 打印"IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET"
        System.out.println("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS");  // 打印"WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS"
        // 打印获取200分的提示
        System.out.println("TO GET 200 POINTS.");
        
        // 打印空行
        System.out.println();
        
        // 打印格式化后的字符串，包括投掷类型、描述和可能得分
        System.out.println(paddedString("THROW", "DESCRIPTION", "PROBABLE SCORE"));
        
        // 打印格式化后的字符串，包括投掷类型、描述和可能得分
        System.out.println(paddedString("1", "FAST OVERARM", "BULLSEYE OR COMPLETE MISS"));
        
        // 打印格式化后的字符串，包括投掷类型、描述和可能得分
        System.out.println(paddedString("2", "CONTROLLED OVERARM", "10, 20 OR 30 POINTS"));
        
        // 打印格式化后的字符串，包括投掷类型、描述和可能得分
        System.out.println(paddedString("3", "UNDERARM", "ANYTHING"));
    }

    /**
     * 计算玩家得分
     * 得分基于投掷类型加上一个随机因素
     *
     * @param playerThrow 表示投掷类型的1、2或3
     * @return 玩家得分
     */
    private int calculatePlayerPoints(int playerThrow) {

        // -1是因为Java数组是从0开始的
        double p1 = this.shots[playerThrow - 1].getShot(0);
        double p2 = this.shots[playerThrow - 1].getShot(1);
        // 获取玩家投掷的第三次和第四次投篮的命中率
        double p3 = this.shots[playerThrow - 1].getShot(2);
        double p4 = this.shots[playerThrow - 1].getShot(3);

        // 生成一个随机数
        double random = Math.random();

        // 初始化得分变量
        int points;

        // 如果随机数大于等于p1，则打印"BULLSEYE!!  40 POINTS!"，并设置得分为40
        // 如果投掷是1（靶心或未命中），则将其设置为未命中
        // 注意：这是对基本代码的修复，对于投篮类型1，允许靶心，但如果未命中靶心，则得分不应为零（但实际上应该为零）。
        if (random >= p1) {
            System.out.println("BULLSEYE!!  40 POINTS!");
            points = 40;
        } else if (playerThrow == 1) {
            // 如果玩家投掷是1，则打印"MISSED THE TARGET!  TOO BAD."，并将得分设置为0
            System.out.println("MISSED THE TARGET!  TOO BAD.");
            points = 0;
        } else if (random >= p2) {
            // 如果随机数大于等于p2，则打印"30-POINT ZONE!"，并将得分设置为30
            System.out.println("30-POINT ZONE!");
            points = 30;
        } else if (random >= p3) {  # 如果随机数大于等于p3
            System.out.println("20-POINT ZONE");  # 打印"20-POINT ZONE"
            points = 20;  # 将points设为20
        } else if (random >= p4) {  # 如果随机数大于等于p4
            System.out.println("WHEW!  10 POINTS.");  # 打印"WHEW!  10 POINTS."
            points = 10;  # 将points设为10
        } else {  # 否则
            System.out.println("MISSED THE TARGET!  TOO BAD.");  # 打印"MISSED THE TARGET!  TOO BAD."
            points = 0;  # 将points设为0
        }

        return points;  # 返回points
    }

    /**
     * Get players shot 1,2, or 3 - ask again if invalid input
     *
     * @param player the player we are calculating the throw on  # 参数player表示我们正在计算投掷的玩家
     * @return 1, 2, or 3 indicating the players shot  # 返回1、2或3，表示玩家的投掷
     */
    # 定义一个方法，用于获取玩家投掷的结果
    private int getPlayersThrow(Player player) {
        # 初始化输入正确标志为假
        boolean inputCorrect = false;
        # 初始化玩家投掷结果的字符串
        String theThrow;
        # 循环直到输入正确
        do {
            # 获取玩家的输入
            theThrow = displayTextAndGetInput(player.getName() + "'S THROW ");
            # 如果输入为1、2或3，则标志输入正确
            if (theThrow.equals("1") || theThrow.equals("2") || theThrow.equals("3")) {
                inputCorrect = true;
            } else {
                # 如果输入不为1、2或3，则提示重新输入
                System.out.println("INPUT 1, 2, OR 3!");
            }

        } while (!inputCorrect);

        # 将玩家的输入结果转换为整数并返回
        return Integer.parseInt(theThrow);
    }


    /**
     * Get players guess from kb
     *
    /**
     * 选择玩家数量的方法
     * @return 玩家猜测的数量（整数）
     */
    private int chooseNumberOfPlayers() {
        return Integer.parseInt((displayTextAndGetInput("HOW MANY PLAYERS? ")));
    }

    /*
     * 在屏幕上打印一条消息，然后从键盘接受输入。
     *
     * @param text 要在屏幕上显示的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 将三个字符串格式化为给定数量的空格
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```