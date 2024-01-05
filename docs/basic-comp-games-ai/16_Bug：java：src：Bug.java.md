# `d:/src/tocomm/basic-computer-games\16_Bug\java\src\Bug.java`

```
import java.util.ArrayList;  # 导入 ArrayList 类
import java.util.Scanner;  # 导入 Scanner 类

/**
 * Game of Bug
 * <p>
 * Based on the Basic game of Bug here
 * https://github.com/coding-horror/basic-computer-games/blob/main/16%20Bug/bug.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Bug {

    // Dice roll
    public static final int SIX = 6;  # 定义常量 SIX 为 6

    private enum GAME_STATE {  # 定义枚举类型 GAME_STATE
        START,  # 开始状态
        PLAYER_TURN,  # 玩家回合
        COMPUTER_TURN,  // 定义游戏状态为电脑回合
        CHECK_FOR_WINNER,  // 定义游戏状态为检查是否有获胜者
        GAME_OVER  // 定义游戏状态为游戏结束
    }

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private GAME_STATE gameState;

    // 玩家的昆虫
    private final Insect playersBug;

    // 电脑的昆虫
    private final Insect computersBug;

    // 用于显示骰子投掷结果
    private final String[] ROLLS = new String[]{"BODY", "NECK", "HEAD", "FEELERS", "TAIL", "LEGS"};

    public Bug() {
        // 创建一个新的玩家角色对象
        playersBug = new PlayerBug();
        // 创建一个新的电脑角色对象
        computersBug = new ComputerBug();
        // 设置游戏状态为开始状态
        gameState = GAME_STATE.START;

        // 初始化键盘输入扫描器
        kbScanner = new Scanner(System.in);
    }

    /**
     * 主游戏循环
     */
    public void play() {

        do {
            switch (gameState) {

                // 第一次玩游戏时显示介绍信息
                // 可选地显示游戏说明
                case START:  # 开始游戏状态
                    intro();  # 调用intro函数，显示游戏介绍
                    if (!noEntered(displayTextAndGetInput("DO YOU WANT INSTRUCTIONS? "))) {  # 如果玩家选择需要说明
                        instructions();  # 调用instructions函数，显示游戏说明
                    }

                    gameState = GAME_STATE.PLAYER_TURN;  # 设置游戏状态为玩家回合
                    break;

                case PLAYER_TURN:  # 玩家回合状态
                    int playersRoll = randomNumber();  # 生成一个随机数作为玩家的掷骰子结果
                    System.out.println("YOU ROLLED A " + playersRoll + "=" + ROLLS[playersRoll - 1]);  # 打印玩家掷骰子的结果
                    switch (playersRoll) {  # 根据掷骰子的结果进行不同的操作
                        case 1:  # 如果掷骰子结果为1
                            System.out.println(playersBug.addBody());  # 打印并执行添加身体部位的操作
                            break;
                        case 2:  # 如果掷骰子结果为2
                            System.out.println(playersBug.addNeck());  # 打印并执行添加脖子部位的操作
                            break;
                        case 3:  # 如果掷骰子结果为3
                    System.out.println(playersBug.addHead());  // 调用 playersBug 对象的 addHead 方法，并打印结果
                    break;  // 结束 switch 语句

                case 4:
                    System.out.println(playersBug.addFeelers());  // 调用 playersBug 对象的 addFeelers 方法，并打印结果
                    break;  // 结束 switch 语句

                case 5:
                    System.out.println(playersBug.addTail());  // 调用 playersBug 对象的 addTail 方法，并打印结果
                    break;  // 结束 switch 语句

                case 6:
                    System.out.println(playersBug.addLeg());  // 调用 playersBug 对象的 addLeg 方法，并打印结果
                    break;  // 结束 switch 语句
            }

            gameState = GAME_STATE.COMPUTER_TURN;  // 将游戏状态设置为 COMPUTER_TURN
            break;  // 结束 switch 语句

        case COMPUTER_TURN:
            int computersRoll = randomNumber();  // 调用 randomNumber 方法生成一个随机数，并赋值给 computersRoll
            System.out.println("I ROLLED A " + computersRoll + "=" + ROLLS[computersRoll - 1]);  // 打印计算机掷骰子的结果
            switch (computersRoll) {  // 根据计算机掷骰子的结果进行不同的操作
# 选择不同的情况并打印出相应的计算机错误信息
case 1:
    打印出计算机错误信息中添加身体的结果
    break;
case 2:
    打印出计算机错误信息中添加颈部的结果
    break;
case 3:
    打印出计算机错误信息中添加头部的结果
    break;
case 4:
    打印出计算机错误信息中添加触角的结果
    break;
case 5:
    打印出计算机错误信息中添加尾部的结果
    break;
case 6:
    打印出计算机错误信息中添加腿部的结果
    break;
                    # 设置游戏状态为检查是否有获胜者
                    gameState = GAME_STATE.CHECK_FOR_WINNER;
                    # 跳出 switch 语句
                    break;

                case CHECK_FOR_WINNER:
                    # 初始化游戏结束标志为假
                    boolean gameOver = false;

                    # 如果玩家的虫完成了
                    if (playersBug.complete()) {
                        # 打印消息提示玩家虫已完成
                        System.out.println("YOUR BUG IS FINISHED.");
                        # 设置游戏结束标志为真
                        gameOver = true;
                    } 
                    # 如果计算机的虫完成了
                    else if (computersBug.complete()) {
                        # 打印消息提示计算机虫已完成
                        System.out.println("MY BUG IS FINISHED.");
                        # 设置游戏结束标志为真
                        gameOver = true;
                    }

                    # 如果玩家不想要图片
                    if (noEntered(displayTextAndGetInput("DO YOU WANT THE PICTURES? "))) {
                        # 设置游戏状态为玩家回合
                        gameState = GAME_STATE.PLAYER_TURN;
                    } 
                    # 如果玩家想要图片
                    else {
                        # 打印消息提示玩家虫
                        System.out.println("*****YOUR BUG*****");
                        # 打印空行
                        System.out.println();
                        # 绘制玩家的虫
                        draw(playersBug);
                        System.out.println();
                        System.out.println("*****MY BUG*****");  // 打印出错误信息
                        System.out.println();  // 打印空行
                        draw(computersBug);  // 调用draw函数，传入computersBug参数
                        gameState = GAME_STATE.PLAYER_TURN;  // 设置游戏状态为玩家回合
                    }
                    if (gameOver) {  // 如果游戏结束
                        System.out.println("I HOPE YOU ENJOYED THE GAME, PLAY IT AGAIN SOON!!");  // 打印出游戏结束信息
                        gameState = GAME_STATE.GAME_OVER;  // 设置游戏状态为游戏结束
                    }
            }
        } while (gameState != GAME_STATE.GAME_OVER);  // 当游戏状态不是游戏结束时继续循环
    }

    /**
     * Draw the bug (player or computer) based on what has
     * already been added to it.
     *
     * @param bug The bug to be drawn.  // 参数bug表示要绘制的bug
    */
    # 根据给定的昆虫对象绘制图像
    private void draw(Insect bug) {
        # 调用昆虫对象的draw方法，获取绘制图像的字符串列表
        ArrayList<String> insectOutput = bug.draw();
        # 遍历字符串列表，逐行打印图像
        for (String s : insectOutput) {
            System.out.println(s);
        }
    }

    /**
     * Display an intro
     */
    # 显示游戏介绍
    private void intro() {
        System.out.println("BUG");
        System.out.println("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("THE GAME BUG");
        System.out.println("I HOPE YOU ENJOY THIS GAME.");
    }

    # 显示游戏指令
    private void instructions() {
# 输出以下信息
System.out.println("THE OBJECT OF BUG IS TO FINISH YOUR BUG BEFORE I FINISH");
System.out.println("MINE. EACH NUMBER STANDS FOR A PART OF THE BUG BODY.");
System.out.println("I WILL ROLL THE DIE FOR YOU, TELL YOU WHAT I ROLLED FOR YOU");
System.out.println("WHAT THE NUMBER STANDS FOR, AND IF YOU CAN GET THE PART.");
System.out.println("IF YOU CAN GET THE PART I WILL GIVE IT TO YOU.");
System.out.println("THE SAME WILL HAPPEN ON MY TURN.");
System.out.println("IF THERE IS A CHANGE IN EITHER BUG I WILL GIVE YOU THE");
System.out.println("OPTION OF SEEING THE PICTURES OF THE BUGS.");
System.out.println("THE NUMBERS STAND FOR PARTS AS FOLLOWS:");
System.out.println("NUMBER\tPART\tNUMBER OF PART NEEDED");
System.out.println("1\tBODY\t1");
System.out.println("2\tNECK\t1");
System.out.println("3\tHEAD\t1");
System.out.println("4\tFEELERS\t2");
System.out.println("5\tTAIL\t1");
System.out.println("6\tLEGS\t6");
System.out.println();
    /**
     * 检查玩家是否输入了N或NO作为答案
     *
     * @param text 玩家从键盘输入的字符串
     * @return 如果输入了N或NO，则返回true，否则返回false
     */
    private boolean noEntered(String text) {
        return stringIsAnyValue(text, "N", "NO");
    }

    /**
     * 检查一个字符串是否等于一系列变量值中的任意一个
     * 用于检查例如Y或YES
     * 比较是不区分大小写的
     *
     * @param text   源字符串
     * @param values 要与源字符串进行比较的一系列值
     * @return 如果在传递的一系列字符串中找到了匹配，则返回true
     */
    private boolean stringIsAnyValue(String text, String... values) {
        // 遍历变量 values 中的每个值，并测试每个值
        for (String val : values) {
            if (text.equalsIgnoreCase(val)) { // 如果文本与值相等（忽略大小写）
                return true; // 返回 true
            }
        }

        // 没有匹配项
        return false; // 返回 false
    }

    /*
     * 在屏幕上打印消息，然后接受键盘输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text); // 打印消息在屏幕上
        return kbScanner.next();  # 从键盘输入中获取下一个输入值并返回

    /**
     * Generate random number
     *
     * @return random number  # 生成一个随机数并返回
     */
    private int randomNumber() {  # 定义一个私有方法，用于生成随机数
        return (int) (Math.random()  # 使用Math.random()方法生成一个0到1之间的随机数
                * (SIX) + 1);  # 将随机数乘以6并加1，得到1到6之间的随机整数并返回
    }
}
```