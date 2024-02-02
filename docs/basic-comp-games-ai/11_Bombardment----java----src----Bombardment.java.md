# `basic-computer-games\11_Bombardment\java\src\Bombardment.java`

```py
import java.util.HashSet;
import java.util.Scanner;

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

    public static final int MAX_GRID_SIZE = 25;  // 定义最大网格大小为25
    public static final int PLATOONS = 4;  // 定义连队数量为4

    private enum GAME_STATE {  // 定义游戏状态枚举
        STARTING,
        DRAW_BATTLEFIELD,
        GET_PLAYER_CHOICES,
        PLAYERS_TURN,
        COMPUTER_TURN,
        PLAYER_WON,
        PLAYER_LOST,
        GAME_OVER
    }

    private GAME_STATE gameState;  // 游戏状态变量

    public static final String[] PLAYER_HIT_MESSAGES = {"ONE DOWN, THREE TO GO.",  // 玩家击中消息数组
            "TWO DOWN, TWO TO GO.", "THREE DOWN, ONE TO GO."};

    public static final String[] COMPUTER_HIT_MESSAGES = {"YOU HAVE ONLY THREE OUTPOSTS LEFT.",  // 计算机击中消息数组
            "YOU HAVE ONLY TWO OUTPOSTS LEFT.", "YOU HAVE ONLY ONE OUTPOST LEFT."};

    private HashSet<Integer> computersPlatoons;  // 计算机连队集合
    private HashSet<Integer> playersPlatoons;  // 玩家连队集合

    private HashSet<Integer> computersGuesses;  // 计算机猜测集合

    // Used for keyboard input
    private final Scanner kbScanner;  // 键盘输入扫描器

    public Bombardment() {

        gameState = GAME_STATE.STARTING;  // 初始化游戏状态为开始

        // Initialise kb scanner
        kbScanner = new Scanner(System.in);  // 初始化键盘输入扫描器
    }

    /**
     * Main game loop
     */
    }

    /**
     * Calculate computer guess.  Make that the computer does not guess the same
     * location twice
     *
     * @return location of the computers guess that has not been guessed previously
     */
    # 生成一个唯一的计算机猜测数字
    private int uniqueComputerGuess() {
        
        # 初始化变量
        boolean validGuess = false;
        int computerGuess;
        # 循环直到生成一个唯一的猜测数字
        do {
            # 生成一个随机数作为计算机的猜测数字
            computerGuess = randomNumber();

            # 如果计算机的猜测数字不在已有的猜测列表中，则为有效猜测
            if (!computersGuesses.contains(computerGuess)) {
                validGuess = true;
            }
        } while (!validGuess);

        # 将计算机的猜测数字添加到猜测列表中
        computersGuesses.add(computerGuess);

        # 返回计算机的猜测数字
        return computerGuess;
    }

    /**
     * 为计算机创建四个唯一的位置
     * 使用哈希集合保证唯一性，只需不断尝试添加随机数，直到有四个位置在哈希集合中
     *
     * @return 计算机的四个位置
     */
    private HashSet<Integer> computersChosenPlatoons() {

        # 初始化哈希集合
        HashSet<Integer> tempPlatoons = new HashSet<>();

        # 是否已经添加了所有位置的标志
        boolean allPlatoonsAdded = false;

        # 循环直到添加了所有位置
        do {
            # 添加一个随机数作为位置
            tempPlatoons.add(randomNumber());

            # 是否已经创建了四个位置
            if (tempPlatoons.size() == PLATOONS) {
                # 当创建了四个位置时退出循环
                allPlatoonsAdded = true;
            }

        } while (!allPlatoonsAdded);

        # 返回计算机的位置集合
        return tempPlatoons;
    }

    /**
     * 根据玩家的击中次数显示不同的消息
     *
     * @param hits 玩家对计算机的总击中次数
     */
    private void showPlayerProgress(int hits) {

        # 显示玩家击中了计算机的前哨站的消息
        System.out.println("YOU GOT ONE OF MY OUTPOSTS!");
        # 根据击中次数显示不同的消息
        showProgress(hits, PLAYER_HIT_MESSAGES);
    }

    /**
     * 根据计算机的击中次数显示不同的消息
     *
     * @param hits 计算机对玩家的总击中次数
     */
    private void showComputerProgress(int hits, int lastGuess) {

        # 显示计算机击中了玩家的消息，并显示最后一次猜测的位置
        System.out.println("I GOT YOU. IT WON'T BE LONG NOW. POST " + lastGuess + " WAS HIT.");
        # 根据击中次数显示不同的消息
        showProgress(hits, COMPUTER_HIT_MESSAGES);
    }
    /**
     * Prints a message from the passed array based on the value of hits
     *
     * @param hits     - number of hits the player or computer has made
     * @param messages - an array of string with messages
     */
    private void showProgress(int hits, String[] messages) {
        // 打印基于命中次数的消息
        System.out.println(messages[hits - 1]);
    }

    /**
     * Update a player hit - adds a hit the player made on the computers platoon.
     *
     * @param fireLocation - computer location that got hit
     * @return number of hits the player has inflicted on the computer in total
     */
    private int updatePlayerHits(int fireLocation) {

        // N.B. only removes if present, so its redundant to check if it exists first
        // 移除计算机位置上的部队，如果存在的话
        computersPlatoons.remove(fireLocation);

        // 返回玩家总共对计算机造成的命中次数
        return PLATOONS - computersPlatoons.size();
    }

    /**
     * Update a computer hit - adds a hit the computer made on the players platoon.
     *
     * @param fireLocation - player location that got hit
     * @return number of hits the player has inflicted on the computer in total
     */
    private int updateComputerHits(int fireLocation) {

        // N.B. only removes if present, so its redundant to check if it exists first
        // 移除玩家位置上的部队，如果存在的话
        playersPlatoons.remove(fireLocation);

        // 返回计算机总共对玩家造成的命中次数
        return PLATOONS - playersPlatoons.size();
    }

    /**
     * Determine if the player hit one of the computers platoons
     *
     * @param fireLocation the players choice of location to fire on
     * @return true if a computer platoon was at that position
     */
    private boolean didPlayerHitComputerPlatoon(int fireLocation) {
        // 判断玩家是否击中了计算机的部队
        return computersPlatoons.contains(fireLocation);
    }

    /**
     * Determine if the computer hit one of the players platoons
     *
     * @param fireLocation the computers choice of location to fire on
     * @return true if a players platoon was at that position
     */
    // 检查计算机是否击中了玩家的一个位置
    private boolean didComputerHitPlayerPlatoon(int fireLocation) {
        return playersPlatoons.contains(fireLocation);
    }

    /**
     * 绘制战场网格
     */
    private void drawBattlefield() {
        for (int i = 1; i < MAX_GRID_SIZE + 1; i += 5) {
            System.out.printf("%-2s %-2s %-2s %-2s %-2s %n", i, i + 1, i + 2, i + 3, i + 4);
        }
    }

    /**
     * 游戏的基本信息
     */
    private void intro() {
        System.out.println("BOMBARDMENT");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU");
        System.out.println("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.");
        System.out.println("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.");
        System.out.println("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.");
        System.out.println();
        System.out.println("THE OBJECT OF THE GAME IS TO FIRE MISSILES AT THE");
        System.out.println("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.");
        System.out.println("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS");
        System.out.println("FIRST IS THE WINNER.");
        System.out.println();
        System.out.println("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!");
        System.out.println();
        System.out.println("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.");
        System.out.println();
        System.out.println();
    }

    private void init() {

        // 为计算机的四个位置创建位置
        computersPlatoons = computersChosenPlatoons();

        // 玩家的位置
        playersPlatoons = new HashSet<>();

        computersGuesses = new HashSet<>();
    }
    /**
     * Accepts a string delimited by comma's and returns the nth delimited
     * value (starting at count 0).
     *
     * @param text - text with values separated by comma's
     * @param pos  - which position to return a value for
     * @return the int representation of the value
     */
    private int getDelimitedValue(String text, int pos) {
        // 将文本按逗号分隔成数组
        String[] tokens = text.split(",");
        // 返回指定位置的值的整数表示
        return Integer.parseInt(tokens[pos]);
    }


    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        // 在屏幕上打印消息
        System.out.print(text);
        // 从键盘接收输入并返回
        return kbScanner.next();
    }

    /**
     * Generate random number
     *
     * @return random number
     */
    private int randomNumber() {
        // 生成一个随机数
        return (int) (Math.random()
                * (MAX_GRID_SIZE) + 1);
    }
# 闭合前面的函数定义
```