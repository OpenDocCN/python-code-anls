# `07_Basketball\java\Basketball.java`

```
# 导入 java.lang.Math 包
import java.lang.Math;
# 导入 java.util 包
import java.util.*;
# 导入 java.util.Scanner 类
import java.util.Scanner;

/* 篮球类是一个电脑游戏，允许你扮演达特茅斯学院的队长和组织者
   该游戏使用设定的概率来模拟每次进攻的结果
   你可以选择投篮类型以及防守阵型 */

public class Basketball {
    # 初始化时间为 0
    int time = 0;
    # 初始化比分数组为 [0, 0]
    int[] score = {0, 0};
    # 初始化防守值为 -1
    double defense = -1;
    # 初始化防守选择列表
    List<Double> defense_choices = Arrays.asList(6.0, 6.5, 7.0, 7.5);
    # 初始化投篮值为 -1
    int shot = -1;
    # 初始化投篮选择列表
    List<Integer> shot_choices = Arrays.asList(0, 1, 2, 3, 4);
    # 初始化对手概率为 0
    double opponent_chance = 0;
    # 初始化对手名称为 null
    String opponent = null;

    # 构造函数
    public Basketball() {
        // 解释键盘输入
        System.out.println("\t\t\t Basketball");
        System.out.println("\t Creative Computing  Morristown, New Jersey\n\n\n");
        System.out.println("This is Dartmouth College basketball. ");
        System.out.println("Υou will be Dartmouth captain and playmaker.");
        System.out.println("Call shots as follows:");
        System.out.println("1. Long (30ft.) Jump Shot; 2. Short (15 ft.) Jump Shot; "
              + "3. Lay up; 4. Set Shot");
        System.out.println("Both teams will use the same defense. Call Defense as follows:");
        System.out.println("6. Press; 6.5 Man-to-Man; 7. Zone; 7.5 None.");
        System.out.println("To change defense, just type 0 as your next shot.");
        System.out.print("Your starting defense will be? ");

        Scanner scanner = new Scanner(System.in); // 创建一个扫描器

        // 获取防守方式的输入
        if (scanner.hasNextDouble()) {
            defense = scanner.nextDouble();
        }
        else {
            scanner.next(); // 如果输入不是double类型，跳过该输入
        }

        // 确保输入合法
        while (!defense_choices.contains(defense)) {
            System.out.print("Your new defensive allignment is? "); // 提示用户输入新的防守方式
            if (scanner.hasNextDouble()) { // 如果下一个输入是double类型
                defense = scanner.nextDouble(); // 将输入的double类型赋值给defense变量
            }
            else {
                scanner.next(); // 如果输入不是double类型，跳过该输入
                continue; // 继续循环，直到输入合法
            }
        }

        // 输入对手的名字
        System.out.print("\nChoose your opponent? "); // 提示用户选择对手

        opponent = scanner.next(); // 获取用户输入的对手名字
        start_of_period();
    }
    // 开始新的比赛周期

    // adds points to the score
    // team can take 0 or 1, for opponent or Dartmouth, respectively
    // 将得分加到比分中
    // team 参数可以取0或1，分别代表对手或达特茅斯队
    private void add_points(int team, int points) {
        score[team] += points;
        print_score();
    }
    // 将得分加到指定队伍的比分中
    // team 参数可以取0或1，分别代表对手或达特茅斯队
    // points 参数代表要加的分数
    // 调用 print_score() 函数打印当前比分

    private void ball_passed_back() {
        System.out.print("Ball passed back to you. ");
        dartmouth_ball();
    }
    // 球传回给你
    // 调用 dartmouth_ball() 函数

    // change defense, called when the user enters 0 for their shot
    // 改变防守方式，当用户输入0时调用
    private void change_defense() {
        defense = -1;
        Scanner scanner = new Scanner(System.in); // creates a scanner
    }
    // 将 defense 变量设为-1
    // 创建一个 Scanner 对象用于用户输入
        while (!defense_choices.contains(defense)) {  # 当防守选择不在给定的防守选择列表中时
            System.out.println("Your new defensive allignment is? ");  # 打印消息要求输入新的防守选择
            if (scanner.hasNextDouble()) {  # 如果输入是一个双精度浮点数
                defense = (double)(scanner.nextDouble());  # 将输入的双精度浮点数赋值给defense变量
            }
            else {  # 如果输入不是双精度浮点数
                continue;  # 继续循环，等待正确的输入
            }
        }

        dartmouth_ball();  # 调用dartmouth_ball函数
    }

    // simulates two foul shots for a player and adds the points
    private void foul_shots(int team) {  # 定义一个私有函数foul_shots，接受一个整数参数team
        System.out.println("Shooter fouled.  Two shots.");  # 打印消息表示球员被犯规，有两次罚球机会

        if (Math.random() > .49) {  # 如果随机生成的数大于0.49
            if (Math.random() > .75) {  # 如果随机生成的数大于0.75
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
    // 打印当前得分
    private void print_score() {
        System.out.println("Score:  " + score[1] + " to " + score[0] + "\n");
    }

    // 模拟比赛开始时的中心跳球，确定谁控球
    private void start_of_period() {
        System.out.println("Center jump");
        if (Math.random() > .6) {
            System.out.println("Dartmouth controls the tap.\n");
            dartmouth_ball();
        }
        else {
            System.out.println(opponent + " controls the tap.\n");
            opponent_ball();
        }
    }
    // 当 t = 92 时调用
    private void two_minute_warning() {
        System.out.println("   *** 比赛还剩两分钟 ***");
    }

    // 当用户输入1或2时调用，表示他们的投篮动作
    private void dartmouth_jump_shot() {
        time ++;
        if (time == 50) {
            halftime();
        }
        else if (time == 92) {
            two_minute_warning();
        }

        System.out.println("跳投。");
        // 模拟不同可能结果的机会
        if (Math.random() > .341 * defense / 8) {
            if (Math.random() > .682 * defense / 8) {
                if (Math.random() > .782 * defense / 8) {
                    # 如果随机数大于 .843 乘以防守值除以 8，表示犯规
                    if (Math.random() > .843 * defense / 8) {
                        # 打印信息并让对方控球
                        System.out.println("Charging foul. Dartmouth loses ball.\n");
                        opponent_ball();
                    }
                    else:
                        # 球员被犯规
                        foul_shots(1)
                        opponent_ball()
                    }
                else:
                    # 如果随机数大于 0.5，表示投篮被封堵
                    if (Math.random() > .5) {
                        # 打印信息并让对方控球
                        System.out.println("Shot is blocked. Ball controlled by " +
                              opponent + ".\n");
                        opponent_ball();
                    }
                    else:
                        # 打印信息并让达特茅斯控球
                        System.out.println("Shot is blocked. Ball controlled by Dartmouth.");
                        dartmouth_ball();
                }
            }
            else {
                System.out.println("Shot is off target.");  // 打印输出“投篮偏离目标”
                if (defense / 6 * Math.random() > .45) {  // 如果防守/6 * 随机数 大于 0.45
                    System.out.println("Rebound to " + opponent + "\n");  // 打印输出“篮板球给对手”
                    opponent_ball();  // 调用opponent_ball函数
                }
                else {
                    System.out.println("Dartmouth controls the rebound.");  // 打印输出“达特茅斯控制篮板球”
                    if (Math.random() > .4) {  // 如果随机数大于0.4
                        if (defense == 6 && Math.random() > .6) {  // 如果防守等于6且随机数大于0.6
                            System.out.println("Pass stolen by " + opponent
                                  + ", easy lay up");  // 打印输出“被对手抢断，轻松上篮”
                            add_points(0, 2);  // 调用add_points函数，传入参数0和2
                            dartmouth_ball();  // 调用dartmouth_ball函数
                        }
                        else {
                            // ball is passed back to you
                            ball_passed_back();  // 调用ball_passed_back函数
    }
                    }
                    else {
                        System.out.println("");
                        dartmouth_non_jump_shot();
                    }
                }
            }
        }
        else {
            System.out.println("Shot is good.");
            add_points(1, 2);
            opponent_ball();
        }
    }
```

这部分代码是一个嵌套的条件语句，根据不同的条件执行不同的操作。在这里需要添加注释来解释每个条件语句的作用和执行的操作。
        // 如果时间等于50，则调用halftime()函数
        if (time == 50) {
            halftime();
        }
        // 如果时间等于92，则调用two_minute_warning()函数
        else if (time == 92) {
            two_minute_warning();
        }

        // 如果投篮次数等于4，则打印"Set shot."
        if (shot == 4) {
            System.out.println("Set shot.");
        }
        // 如果投篮次数等于3，则打印"Lay up."
        else if (shot == 3) {
            System.out.println("Lay up.");
        }
        // 如果投篮次数等于0，则调用change_defense()函数
        else if (shot == 0) {
            change_defense();
        }

        // 模拟投篮或防守不同结果的情况
        if (7/defense*Math.random() > .4) {
            // 如果随机数大于0.7，则执行以下代码
            if (7/defense*Math.random() > .7) {
                # 如果防守者的7分之1乘以一个随机数大于0.875
                if (7/defense*Math.random() > .875) {
                    # 如果防守者的7分之1乘以一个随机数大于0.925
                    if (7/defense*Math.random() > .925) {
                        # 打印信息并让对手控球
                        System.out.println("Charging foul. Dartmouth loses the ball.\n");
                        opponent_ball();
                    }
                    else {
                        # 打印信息并让对手控球
                        System.out.println("Shot blocked. " + opponent + "'s ball.\n");
                        opponent_ball();
                    }
                }
                else {
                    # 进行犯规投篮，然后让对手控球
                    foul_shots(1);
                    opponent_ball();
                }
            }
            else {
                # 打印信息
                System.out.println("Shot is off the rim.");
                # 如果随机数大于2/3
                if (Math.random() > 2/3) {
                    # 打印信息
                    System.out.println("Dartmouth controls the rebound.");
                    # 如果随机数大于0.4
                    if (Math.random() > .4) {
# 打印信息，表示球传回给对方
System.out.println("Ball passed back to you.\n");
# 调用 dartmouth_ball() 函数
dartmouth_ball();
# 如果条件不满足，执行下面的代码
else:
    # 调用 dartmouth_non_jump_shot() 函数
    dartmouth_non_jump_shot();
# 如果条件不满足，执行下面的代码
else:
    # 打印信息，表示对手控制篮板
    System.out.println(opponent + " controls the rebound.\n");
    # 调用 opponent_ball() 函数
    opponent_ball();
# 如果条件不满足，执行下面的代码
else:
    # 打印信息，表示投篮命中，得到两分
    System.out.println("Shot is good. Two points.");
    # 调用 add_points() 函数，增加对方得分
    add_points(1, 2);
    # 调用 opponent_ball() 函数
    opponent_ball();
    // plays out a Dartmouth posession, starting with your choice of shot
    private void dartmouth_ball() {
        Scanner scanner = new Scanner(System.in); // creates a scanner to read user input from the console
        System.out.print("Your shot? "); // prompts the user to input their choice of shot
        shot = -1; // initializes the variable 'shot' to -1
        if (scanner.hasNextInt()) { // checks if the next input is an integer
            shot = scanner.nextInt(); // reads the next integer input and assigns it to the variable 'shot'
        }
        else {
            System.out.println(""); // prints an empty line
            scanner.next(); // reads the next input
        }

        while (!shot_choices.contains(shot)) { // loops until the user's shot choice is in the list of valid shot choices
            System.out.print("Incorrect answer. Retype it. Your shot?"); // prompts the user to retype their shot choice
            if (scanner.hasNextInt()) { // checks if the next input is an integer
                shot = scanner.nextInt(); // reads the next integer input and assigns it to the variable 'shot'
            }
            else {
                System.out.println("");  // 打印空行
                scanner.next();  // 从控制台读取下一个输入
            }
        }

        if (time < 100 || Math.random() < .5) {  // 如果时间小于100或者随机数小于0.5
            if (shot == 1 || shot == 2) {  // 如果投篮方式是1或2
                dartmouth_jump_shot();  // 执行达特茅斯大学跳投动作
            }
            else {
                dartmouth_non_jump_shot();  // 否则执行达特茅斯大学非跳投动作
            }
        }
        else {
            if (score[0] != score[1]) {  // 如果得分不相等
                System.out.println("\n   ***** End Of Game *****");  // 打印比赛结束信息
                System.out.println("Final Score: Dartmouth: " + score[1] + "  "
                      + opponent + ": " + score[0]);  // 打印最终比分
                System.exit(0);  // 退出程序
            }
            else {
                System.out.println("\n   ***** End Of Second Half *****");  # 打印输出比赛下半场结束
                System.out.println("Score at end of regulation time:");  # 打印输出常规时间结束时的比分
                System.out.println("     Dartmouth: " + score[1] + " " +  # 打印输出达特茅斯队的得分
                      opponent + ": " + score[0]);  # 打印输出对手队的得分
                System.out.println("Begin two minute overtime period");  # 打印输出开始两分钟的加时赛
                time = 93;  # 设置时间为93分钟
                start_of_period();  # 调用开始新的比赛阶段的函数
            }
        }
    }

    // simulates the opponents jumpshot  # 模拟对手的跳投
    private void opponent_jumpshot() {  # 定义模拟对手跳投的函数
        System.out.println("Jump Shot.");  # 打印输出跳投
        if (8/defense*Math.random() > .35) {  # 如果（8/防守值*随机数）大于0.35
            if (8/defense*Math.random() > .75) {  # 如果（8/防守值*随机数）大于0.75
                if (8/defense*Math.random() > .9) {  # 如果（8/防守值*随机数）大于0.9
                    System.out.println("Offensive foul. Dartmouth's ball.\n");  # 打印输出进攻犯规，达特茅斯队球权
                    dartmouth_ball();  # 调用达特茅斯队持球函数
                }
                else {
                    foul_shots(0);  # 调用foul_shots函数，参数为0
                    dartmouth_ball();  # 调用dartmouth_ball函数
                }
            }
            else {
                System.out.println("Shot is off the rim.");  # 打印输出"Shot is off the rim."
                if (defense/6*Math.random() > .5) {  # 如果defense/6乘以一个随机数大于0.5
                    System.out.println(opponent + " controls the rebound.");  # 打印输出对手控制篮板
                    if (defense == 6) {  # 如果defense等于6
                        if (Math.random() > .75) {  # 如果随机数大于0.75
                            System.out.println("Ball stolen. Easy lay up for Dartmouth.");  # 打印输出"Ball stolen. Easy lay up for Dartmouth."
                            add_points(1, 2);  # 调用add_points函数，参数为1和2
                            opponent_ball();  # 调用opponent_ball函数
                        }
                        else {
                            if (Math.random() > .5) {  # 如果随机数大于0.5
                                System.out.println("");  # 打印输出空行
                                opponent_non_jumpshot();  # 调用opponent_non_jumpshot函数
                            }
                            else {
                                System.out.println("Pass back to " + opponent +
                                      " guard.\n");
                                opponent_ball();
                            }
                        }
                    }
                    else {
                        if (Math.random() > .5) {
                            opponent_non_jumpshot();
                        }
                        else {
                            System.out.println("Pass back to " + opponent +
                                  " guard.\n");
                            opponent_ball();
                        }
                    }
                }
                else {
```

这段代码是一个嵌套的条件语句，其中包含了多个if-else语句。根据条件的不同，会执行不同的操作，比如打印信息或者调用其他函数。在具体情境下，这段代码可能是一个篮球比赛的模拟程序，根据球员的动作和随机数的结果来决定下一步的操作。
                    System.out.println("Dartmouth controls the rebound.\n");
                    // 打印输出“达特茅斯控制篮板”
                    dartmouth_ball();
                    // 调用函数dartmouth_ball()，表示达特茅斯队控球
                }
            }
        }
        else {
            System.out.println("Shot is good.");
            // 打印输出“投篮命中”
            add_points(0, 2);
            // 调用函数add_points()，给对方队伍加2分
            dartmouth_ball();
            // 调用函数dartmouth_ball()，表示达特茅斯队控球
        }
    }

    // 模拟对手的上篮或定点投篮
    private void opponent_non_jumpshot() {
        if (opponent_chance > 3) {
            System.out.println("Set shot.");
            // 如果对手机会大于3，打印输出“定点投篮”
        }
        else {
            System.out.println("Lay up");
            // 否则，打印输出“上篮”
        }
        # 如果防守者的7除以defense乘以一个随机数大于0.413
        if (7/defense*Math.random() > .413) {
            # 打印出投篮未中的消息
            System.out.println("Shot is missed.");
            # 如果防守者除以6乘以一个随机数大于0.5
            if (defense/6*Math.random() > .5) {
                # 打印出对手控制篮板的消息
                System.out.println(opponent + " controls the rebound.");
                # 如果防守者等于6
                if (defense == 6) {
                    # 如果一个随机数大于0.75
                    if (Math.random() > .75) {
                        # 打印出球被抢断的消息，达特茅斯轻松上篮得分
                        System.out.println("Ball stolen. Easy lay up for Dartmouth.");
                        # 添加得分1分，对手得分2分
                        add_points(1, 2);
                        # 对手控球
                        opponent_ball();
                    }
                    else {
                        # 如果一个随机数大于0.5
                        if (Math.random() > .5) {
                            # 什么也不做，对手非跳投
                            System.out.println("");
                            opponent_non_jumpshot();
                        }
                        else {
                            # 打印出传球回给对手的消息
                            System.out.println("Pass back to " + opponent +
                                  " guard.\n");
                            # 对手控球
                            opponent_ball();
                        }
                    }
                }
                else {
                    if (Math.random() > .5) {  // 如果随机数大于0.5
                        System.out.println("");  // 输出空行
                        opponent_non_jumpshot();  // 调用对手的非跳投动作
                    }
                    else {
                        System.out.println("Pass back to " + opponent + " guard\n");  // 输出将球传回给对手后卫的信息
                        opponent_ball();  // 对手控球
                    }
                }
            }
            else {
                System.out.println("Dartmouth controls the rebound.\n");  // 输出达特茅斯控制篮板的信息
                dartmouth_ball();  // 达特茅斯控球
            }
        }
        else {
            System.out.println("Shot is good.");  // 输出投篮命中的信息
            add_points(0, 2);  // 调用 add_points 函数，给当前队伍加 2 分
            dartmouth_ball();  // 调用 dartmouth_ball 函数
        }
    }

    // 模拟对手的进攻
    // 随机选择跳投、上篮或定点投篮
    private void opponent_ball() {
        time ++;  // 时间加一
        if (time == 50) {  // 如果时间达到 50 分钟
            halftime();  // 进入中场休息
        }
        opponent_chance = 10/4*Math.random()+1;  // 对手的进攻机会，随机生成一个数
        if (opponent_chance > 2) {  // 如果对手的进攻机会大于 2
            opponent_non_jumpshot();  // 对手进行非跳投进攻
        }
        else {  // 否则
            opponent_jumpshot();  // 对手进行跳投进攻
        }
    }
# 创建一个名为main的函数，参数为args
def main(args):
    # 创建一个名为new_game的Basketball对象
    new_game = Basketball()
```
在这段代码中，我们定义了一个名为main的函数，它接受一个参数args。在函数内部，我们创建了一个名为new_game的Basketball对象。
```