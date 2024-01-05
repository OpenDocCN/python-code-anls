# `d:/src/tocomm/basic-computer-games\11_Bombardment\java\src\Bombardment.java`

```
import java.util.HashSet;  // 导入 HashSet 类
import java.util.Scanner;  // 导入 Scanner 类

/**
 * Game of Bombardment
 * <p>
 * Based on the Basic game of Bombardment here
 * https://github.com/coding-horror/basic-computer-games/blob/main/11%20Bombardment/bombardment.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Bombardment {

    public static final int MAX_GRID_SIZE = 25;  // 定义最大网格大小为 25
    public static final int PLATOONS = 4;  // 定义连队数量为 4

    private enum GAME_STATE {  // 定义游戏状态枚举
        STARTING,  // 开始状态
        DRAW_BATTLEFIELD,  // 绘制战场状态
        GET_PLAYER_CHOICES,  // 定义了游戏状态中玩家可以选择的选项
        PLAYERS_TURN,  // 定义了游戏状态中轮到玩家的状态
        COMPUTER_TURN,  // 定义了游戏状态中轮到电脑的状态
        PLAYER_WON,  // 定义了游戏状态中玩家获胜的状态
        PLAYER_LOST,  // 定义了游戏状态中玩家失败的状态
        GAME_OVER  // 定义了游戏状态中游戏结束的状态
    }

    private GAME_STATE gameState;  // 定义了游戏的状态变量

    public static final String[] PLAYER_HIT_MESSAGES = {"ONE DOWN, THREE TO GO.",
            "TWO DOWN, TWO TO GO.", "THREE DOWN, ONE TO GO."};  // 定义了玩家击中敌方的消息数组

    public static final String[] COMPUTER_HIT_MESSAGES = {"YOU HAVE ONLY THREE OUTPOSTS LEFT.",
            "YOU HAVE ONLY TWO OUTPOSTS LEFT.", "YOU HAVE ONLY ONE OUTPOST LEFT."};  // 定义了电脑击中玩家的消息数组

    private HashSet<Integer> computersPlatoons;  // 定义了电脑的部队集合
    private HashSet<Integer> playersPlatoons;  // 定义了玩家的部队集合

    private HashSet<Integer> computersGuesses;  // 定义了电脑猜测的集合
    // 用于键盘输入
    private final Scanner kbScanner;  // 声明一个键盘输入的 Scanner 对象

    public Bombardment() {
        gameState = GAME_STATE.STARTING;  // 设置游戏状态为开始状态

        // 初始化键盘输入的 Scanner 对象
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    public void play() {
        do {
            switch (gameState) {
                // 在游戏第一次进行时显示介绍
                case STARTING:
                    init(); // 初始化游戏
                    intro(); // 显示游戏介绍
                    gameState = GAME_STATE.DRAW_BATTLEFIELD; // 设置游戏状态为绘制战场
                    break;

                // 绘制战场
                case DRAW_BATTLEFIELD:
                    drawBattlefield(); // 调用绘制战场的函数
                    gameState = GAME_STATE.GET_PLAYER_CHOICES; // 设置游戏状态为获取玩家选择
                    break;

                // 获取玩家的4个位置选择
                case GET_PLAYER_CHOICES:
                    String playerChoices = displayTextAndGetInput("WHAT ARE YOUR FOUR POSITIONS? "); // 显示文本并获取玩家输入

                    // 存储玩家输入的4个位置选择，用逗号分隔
                    for (int i = 0; i < PLATOONS; i++) {
                        playersPlatoons.add(getDelimitedValue(playerChoices, i)); // 将玩家输入的位置选择添加到列表中
                    }

                    gameState = GAME_STATE.PLAYERS_TURN;  # 设置游戏状态为玩家回合
                    break;  # 结束当前的 switch 语句

                // 玩家回合选择位置
                case PLAYERS_TURN:

                    int firePosition = getDelimitedValue(
                            displayTextAndGetInput("WHERE DO YOU WISH TO FIRE YOUR MISSILE? "), 0);  # 获取玩家输入的位置

                    if (didPlayerHitComputerPlatoon(firePosition)) {  # 判断玩家是否击中了计算机的编队
                        // 玩家击中了计算机的编队
                        int hits = updatePlayerHits(firePosition);  # 更新玩家的击中次数
                        // 玩家一共击中了多少次？
                        if (hits != PLATOONS) {  # 如果击中次数不等于编队数
                            showPlayerProgress(hits);  # 展示玩家的进度
                            gameState = GAME_STATE.COMPUTER_TURN;  # 设置游戏状态为计算机回合
                        } else {
                            // 玩家已经获得了4次击中，他们赢了
                            // 如果计算机击中了玩家的位置
                            if (hits == 3) {
                                // 如果计算机击中了三次，玩家输了
                                System.out.println("I GOT YOU! YOU LOSE!");
                                gameState = GAME_STATE.COMPUTER_WON;
                            } else {
                                // 如果计算机击中了但还没赢，继续计算机的回合
                                System.out.println("I HIT YOUR PLATOON! MY TURN AGAIN:");
                                System.out.println();
                                gameState = GAME_STATE.COMPUTER_TURN;
                            }
                        } else {
                            // 如果计算机没有击中，轮到玩家回合
                            System.out.println("HA, HA YOU MISSED. MY TURN NOW:");
                            System.out.println();
                            gameState = GAME_STATE.PLAYER_TURN;
                        }

                        break;
                        if (hits != PLATOONS) { // 如果击中数不等于 PLATOONS
                            showComputerProgress(hits, computerFirePosition); // 显示计算机的进度
                            gameState = GAME_STATE.PLAYERS_TURN; // 设置游戏状态为玩家回合
                        } else {
                            // Computer has obtained 4 hits, they win
                            System.out.println("YOU'RE DEAD. YOUR LAST OUTPOST WAS AT " + computerFirePosition
                                    + ". HA, HA, HA."); // 打印玩家失败的消息
                            gameState = GAME_STATE.PLAYER_LOST; // 设置游戏状态为玩家失败
                        }
                    } else {
                        // Computer missed
                        System.out.println("I MISSED YOU, YOU DIRTY RAT. I PICKED " + computerFirePosition
                                + ". YOUR TURN:"); // 打印计算机未命中的消息
                        System.out.println();
                        gameState = GAME_STATE.PLAYERS_TURN; // 设置游戏状态为玩家回合
                    }

                    break;

                // The player won
                case PLAYER_WON:  // 如果玩家赢了
                    System.out.println("YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN");  // 打印玩家赢了的消息
                    System.out.println("MY TRANSISTO&S RECUP%RA*E!");  // 打印玩家赢了的消息
                    gameState = GAME_STATE.GAME_OVER;  // 将游戏状态设置为游戏结束
                    break;  // 结束该case的执行

                case PLAYER_LOST:  // 如果玩家输了
                    System.out.println("BETTER LUCK NEXT TIME.");  // 打印玩家输了的消息
                    gameState = GAME_STATE.GAME_OVER;  // 将游戏状态设置为游戏结束
                    break;  // 结束该case的执行

                // GAME_OVER State does not specifically have a case  // 游戏结束状态没有特定的case

            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时，继续执行循环
    }

    /**
     * Calculate computer guess.  Make that the computer does not guess the same
     * location twice
     *
    /**
     * 生成一个不重复的计算机猜测的位置
     * @return 之前未被猜测过的计算机猜测位置
     */
    private int uniqueComputerGuess() {
        // 初始化变量，用于判断猜测是否有效
        boolean validGuess = false;
        // 初始化计算机猜测的位置
        int computerGuess;
        // 循环直到生成一个不重复的猜测位置
        do {
            // 生成一个随机的猜测位置
            computerGuess = randomNumber();
            // 如果该位置之前未被猜测过，则将validGuess设置为true
            if (!computersGuesses.contains(computerGuess)) {
                validGuess = true;
            }
        } while (!validGuess);

        // 将该位置添加到已猜测位置的列表中
        computersGuesses.add(computerGuess);

        // 返回计算机猜测的位置
        return computerGuess;
    }
```
在这段代码中，我们定义了一个方法`uniqueComputerGuess()`，用于生成一个不重复的计算机猜测的位置。方法中使用了循环来确保生成的位置是之前未被猜测过的。同时，还添加了注释来解释每个语句的作用。
    * 创建计算机的四个独特的位置
    * 我们使用哈希集确保唯一性，所以
    * 我们只需要不断尝试添加随机数
    * 直到所有四个位置都在哈希集中
    *
    * @return 计算机的四个位置
    */
    private HashSet<Integer> computersChosenPlatoons() {

        // 初始化内容
        HashSet<Integer> tempPlatoons = new HashSet<>();

        boolean allPlatoonsAdded = false;

        do {
            tempPlatoons.add(randomNumber());

            // 是否已创建了所有四个位置？
            if (tempPlatoons.size() == PLATOONS) {
                // 当我们创建了四个位置时退出
                allPlatoonsAdded = true;  # 设置变量 allPlatoonsAdded 为 true，表示所有的排都已经添加
            }

        } while (!allPlatoonsAdded);  # 当所有的排都已经添加时退出循环

        return tempPlatoons;  # 返回排的列表
    }

    /**
     * Shows a different message for each number of hits
     *
     * @param hits total number of hits by player on computer
     */
    private void showPlayerProgress(int hits) {

        System.out.println("YOU GOT ONE OF MY OUTPOSTS!");  # 打印玩家击中电脑的输出信息
        showProgress(hits, PLAYER_HIT_MESSAGES);  # 调用 showProgress 方法显示玩家的进度信息
    }

    /**
    # 显示计算机对玩家的每次命中情况的不同消息
    # @param hits 计算机对玩家的总命中次数
    # @param lastGuess 上一次猜测的位置
    def showComputerProgress(hits, lastGuess):
        # 打印消息，根据上一次猜测的位置和命中次数显示不同的消息
        print("I GOT YOU. IT WON'T BE LONG NOW. POST " + str(lastGuess) + " WAS HIT.")
        # 调用showProgress函数，根据命中次数显示不同的消息
        showProgress(hits, COMPUTER_HIT_MESSAGES)

    # 根据命中次数打印传入数组中的消息
    # @param hits 玩家或计算机的命中次数
    # @param messages 包含消息的字符串数组
    def showProgress(hits, messages):
        # 打印数组中根据命中次数对应的消息
        print(messages[hits - 1])
    /**
     * Update a player hit - adds a hit the player made on the computers platoon.
     *
     * @param fireLocation - computer location that got hit
     * @return number of hits the player has inflicted on the computer in total
     */
    private int updatePlayerHits(int fireLocation) {

        // N.B. only removes if present, so its redundant to check if it exists first
        computersPlatoons.remove(fireLocation); // 从计算机的位置列表中移除被击中的位置

        // return number of hits in total
        return PLATOONS - computersPlatoons.size(); // 返回玩家对计算机总共造成的击中数
    }

    /**
     * Update a computer hit - adds a hit the computer made on the players platoon.
     *
     * @param fireLocation - player location that got hit
     * @return number of hits the player has inflicted on the computer in total
     */
    */
    // 更新计算机被击中的次数
    private int updateComputerHits(int fireLocation) {

        // 注意：只有在存在时才会移除，因此先检查是否存在是多余的
        playersPlatoons.remove(fireLocation);

        // 返回总共击中的次数
        return PLATOONS - playersPlatoons.size();
    }

    /**
     * 确定玩家是否击中了计算机的一个排
     *
     * @param fireLocation 玩家选择的射击位置
     * @return 如果计算机的一个排在该位置上则返回true
     */
    private boolean didPlayerHitComputerPlatoon(int fireLocation) {
        return computersPlatoons.contains(fireLocation);
    }
    /**
     * Determine if the computer hit one of the players platoons
     *
     * @param fireLocation the computers choice of location to fire on
     * @return true if a players platoon was at that position
     */
    private boolean didComputerHitPlayerPlatoon(int fireLocation) {
        return playersPlatoons.contains(fireLocation); // 检查玩家的位置列表中是否包含计算机选择的位置，如果包含则返回true
    }

    /**
     * Draw the battlefield grid
     */
    private void drawBattlefield() {
        for (int i = 1; i < MAX_GRID_SIZE + 1; i += 5) { // 循环遍历战场网格的行
            System.out.printf("%-2s %-2s %-2s %-2s %-2s %n", i, i + 1, i + 2, i + 3, i + 4); // 打印每行的网格位置
        }
    }

    /**
    /**
     * 游戏的基本信息
     */
    private void intro() {
        System.out.println("BOMBARDMENT"); // 打印游戏名称
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"); // 打印游戏开发者信息
        System.out.println();
        System.out.println("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU"); // 打印游戏背景信息
        System.out.println("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED."); // 打印游戏背景信息
        System.out.println("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST."); // 打印游戏规则
        System.out.println("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS."); // 打印游戏规则
        System.out.println();
        System.out.println("THE OBJECT OF THE GAME IS TO FIRE MISSILES AT THE"); // 打印游戏目标
        System.out.println("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU."); // 打印游戏目标
        System.out.println("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS"); // 打印游戏目标
        System.out.println("FIRST IS THE WINNER."); // 打印游戏目标
        System.out.println();
        System.out.println("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!"); // 打印祝福语
        System.out.println();
        System.out.println("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS."); // 打印提示信息
        System.out.println();
    }
        System.out.println();
    }
```
这行代码是一个空行，没有实际的代码功能，可能是为了提高代码的可读性或者分隔不同功能块的作用。

```
    private void init() {
```
这是一个私有方法init()的定义，用于初始化程序的一些变量或者对象。

```
        // Create four locations for the computers platoons.
        computersPlatoons = computersChosenPlatoons();
```
这行代码创建了计算机的platoons（编队）的四个位置，并将其赋值给computersPlatoons变量。computersChosenPlatoons()是一个方法，可能用于计算机选择编队的逻辑。

```
        // Players platoons.
        playersPlatoons = new HashSet<>();
```
这行代码创建了一个空的HashSet对象，并将其赋值给playersPlatoons变量，用于存储玩家的编队信息。

```
        computersGuesses = new HashSet<>();
```
这行代码创建了一个空的HashSet对象，并将其赋值给computersGuesses变量，用于存储计算机的猜测信息。

```
    /**
     * Accepts a string delimited by comma's and returns the nth delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
```
这是一个方法的注释部分，说明了该方法的作用和参数的含义。该方法接受一个由逗号分隔的字符串，返回第n个被逗号分隔的值（从0开始计数）。参数text是被逗号分隔的值，参数pos是要返回值的位置。
    /**
     * 从给定的文本中获取指定位置的值的整数表示
     * @param text 给定的文本
     * @param pos 指定位置
     * @return 值的整数表示
     */
    private int getDelimitedValue(String text, int pos) {
        String[] tokens = text.split(",");
        return Integer.parseInt(tokens[pos]);
    }

    /**
     * 在屏幕上打印消息，然后从键盘接受输入
     * @param text 要在屏幕上显示的消息
     * @return 玩家输入的内容
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }
# 生成随机数
def randomNumber():
    # 使用 Math.random() 生成 0 到 1 之间的随机小数，乘以 MAX_GRID_SIZE，再加上 1，取整数部分作为随机数
    return int(Math.random() * (MAX_GRID_SIZE) + 1)
```