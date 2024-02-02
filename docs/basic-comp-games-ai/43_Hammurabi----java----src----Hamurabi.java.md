# `basic-computer-games\43_Hammurabi\java\src\Hamurabi.java`

```py
# 导入必要的类
import java.util.Arrays;
import java.util.Scanner;

/**
 * Hamurabi 游戏
 * <p>
 * 基于 Basic 版本的 Hamurabi 游戏，链接在这里
 * https://github.com/coding-horror/basic-computer-games/blob/main/43%20Hammurabi/hammurabi.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 Basic 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 */
public class Hamurabi {

    public static final int INITIAL_POPULATION = 95;  # 初始人口
    public static final int INITIAL_BUSHELS = 2800;  # 初始小麦
    public static final int INITIAL_HARVEST = 3000;  # 初始收成
    public static final int INITIAL_LAND_TRADING_AT = 3;  # 初始土地交易价格
    public static final int INITIAL_CAME_TO_CITY = 5;  # 初始来到城市的人数
    public static final int MAX_GAME_YEARS = 10;  # 最大游戏年限
    public static final double MAX_STARVATION_IN_A_YEAR = .45d;  # 一年内最大饥饿比例

    private int year;  # 年份
    private int population;  # 人口
    private int acres;  # 土地面积
    private int bushels;  # 小麦数量
    private int harvest;  # 收成
    private int landTradingAt;  # 土地交易价格
    private int cameToCity;  # 来到城市的人数
    private int starvedInAYear;  # 一年内饿死的人数
    private int starvedOverall;  # 总共饿死的人数
    private boolean chanceOfPlague;  # 是否有瘟疫的机会
    private int ratsAte;  # 老鼠吃掉的小麦数量
    private double peopleFed;  # 被喂饱的人数
    private double percentageStarved;  # 饿死的人数比例
    private int bushelsToFeedPeople;  # 喂饱人们所需的小麦数量

    // 用于键盘输入
    private final Scanner kbScanner;

    // 当前游戏状态
    private enum GAME_STATE {
        STARTUP,
        INIT,
        YEAR_CYCLE,
        BUY_ACRES,
        SELL_ACRES,
        FEED_PEOPLE,
        PLANT_SEED,
        CALCULATE_HARVEST,
        CALCULATE_BABIES,
        RESULTS,
        FINISH_GAME,
        GAME_OVER
    }

    // 游戏的主要逻辑
    private GAME_STATE gameState;

    public Hamurabi() {
        kbScanner = new Scanner(System.in);  # 初始化键盘输入扫描器
        gameState = GAME_STATE.STARTUP;  # 设置游戏状态为启动
    }

    /**
     * 主游戏循环
     */
}
    // 输出空行
    System.out.println();
    // 输出星星人口过多的信息
    System.out.println("YOU STARVED " + starved + " PEOPLE IN ONE YEAR!!!");
    System.out.println("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY");
    System.out.println("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE");
    System.out.println("ALSO BEEN DECLARED NATIONAL FINK!!!!");

    // 输出过于严厉的信息
    System.out.println("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY");
    System.out.println("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE");
    System.out.println("ALSO BEEN DECLARED NATIONAL FINK!!!!");

    // 输出表现可以更好的信息
    System.out.println("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT");
    System.out.println("REALLY WASN'T TOO BAD AT ALL. " + (int) (Math.random() * (population * .8)) + " PEOPLE");
    System.out.println("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR");
    System.out.println("TRIVIAL PROBLEMS.");

    // 输出表现极好的信息
    System.out.println("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND");
    System.out.println("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!");

    // 输出人口不足的信息
    System.out.println("BUT YOU HAVE ONLY " + population + " PEOPLE TO TEND THE FIELDS!  NOW THEN,");

    // 输出粮食不足的信息
    System.out.println("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY");
    System.out.println(bushels + " BUSHELS OF GRAIN.  NOW THEN,");

    // 输出土地不足的信息
    System.out.println("HAMURABI:  THINK AGAIN.  YOU OWN ONLY " + acres + " ACRES.  NOW THEN,");
    // 打印游戏介绍信息
    private void intro() {
        System.out.println(simulateTabs(32) + "HAMURABI");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA");
        System.out.println("FOR A TEN-YEAR TERM OF OFFICE.");
        System.out.println();
    }

    /*
     * 打印屏幕上的消息，然后从键盘接受输入。
     * 将输入转换为整数
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private int displayTextAndGetNumber(String text) {
        return Integer.parseInt(displayTextAndGetInput(text));
    }

    /*
     * 打印屏幕上的消息，然后从键盘接受输入。
     *
     * @param text 要显示在屏幕上的消息。
     * @return 玩家输入的内容。
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * 模拟旧的基本tab(xx)命令，通过xx个空格缩进文本。
     *
     * @param spaces 需要的空格数
     * @return 包含指定空格数的字符串
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
# 闭合前面的函数定义
```