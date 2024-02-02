# `basic-computer-games\07_Basketball\java\Basketball.java`

```py
// 导入数学类和相关工具类
import java.lang.Math;
import java.util.*;
import java.util.Scanner;

/* 篮球类是一个电脑游戏，允许你扮演达特茅斯学院的队长和组织者
   该游戏使用设定的概率来模拟每次进攻的结果
   你可以选择投篮类型以及防守阵型 */

public class Basketball {
    // 时间初始化为0
    int time = 0;
    // 比分初始化为{0, 0}
    int[] score = {0, 0};
    // 防守初始化为-1
    double defense = -1;
    // 防守选择列表
    List<Double> defense_choices = Arrays.asList(6.0, 6.5, 7.0, 7.5);
    // 投篮初始化为-1
    int shot = -1;
    // 投篮选择列表
    List<Integer> shot_choices = Arrays.asList(0, 1, 2, 3, 4);
    // 对手机会初始化为0
    double opponent_chance = 0;
    // 对手初始化为null
    String opponent = null;
}
    // 构造函数，初始化篮球游戏
    public Basketball() {

        // 输出游戏标题和说明
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

        // 获取输入的防守方式
        if (scanner.hasNextDouble()) {
            defense = scanner.nextDouble();
        }
        else {
            scanner.next();
        }

        // 确保输入的防守方式合法
        while (!defense_choices.contains(defense)) {
            System.out.print("Your new defensive allignment is? ");
            if (scanner.hasNextDouble()) {
                defense = scanner.nextDouble();
            }
            else {
                scanner.next();
                continue;
            }
        }

        // 获取对手的名字
        System.out.print("\nChoose your opponent? ");
        opponent = scanner.next();
        start_of_period();
    }

    // 增加得分
    // team 参数表示队伍，0 表示对手，1 表示 Dartmouth
    private void add_points(int team, int points) {
        score[team] += points;
        print_score();
    }

    // 球传回
    private void ball_passed_back() {
        System.out.print("Ball passed back to you. ");
        dartmouth_ball();
    }
    // 当用户输入0时调用，改变防守方式
    private void change_defense() {
        // 将防守方式设置为-1
        defense = -1;
        // 创建一个Scanner对象
        Scanner scanner = new Scanner(System.in);

        // 当用户输入的防守方式不在可选范围内时，循环询问用户输入
        while (!defense_choices.contains(defense)) {
            System.out.println("Your new defensive allignment is? ");
            // 如果用户输入的是double类型的数据
            if (scanner.hasNextDouble()) {
                defense = (double)(scanner.nextDouble());
            }
            else {
                continue;
            }
        }

        // 调用dartmouth_ball()方法
        dartmouth_ball();
    }

    // 为球员模拟两次罚球，并计分
    private void foul_shots(int team) {
        System.out.println("Shooter fouled.  Two shots.");

        // 如果随机数大于0.49
        if (Math.random() > .49) {
            // 如果第一次罚球的随机数大于0.75
            if (Math.random() > .75) {
                System.out.println("Both shots missed.");
            }
            else {
                System.out.println("Shooter makes one shot and misses one.");
                score[team] += 1;
            }
        }
        else {
            System.out.println("Shooter makes both shots.");
            score[team] += 2;
        }

        // 调用print_score()方法
        print_score();
    }

    // 当时间 = 50 时调用，开始新的半场
    private void halftime() {
        System.out.println("\n   ***** End of first half *****\n");
        // 调用print_score()方法
        print_score();
        // 调用start_of_period()方法
        start_of_period();
    }

    // 打印当前比分
    private void print_score() {
        System.out.println("Score:  " + score[1] + " to " + score[0] + "\n");
    }

    // 模拟一个中心跳球，确定比赛开始时的球权
    private void start_of_period() {
        System.out.println("Center jump");
        // 如果随机数大于0.6
        if (Math.random() > .6) {
            System.out.println("Dartmouth controls the tap.\n");
            // 调用dartmouth_ball()方法
            dartmouth_ball();
        }
        else {
            System.out.println(opponent + " controls the tap.\n");
            // 调用opponent_ball()方法
            opponent_ball();
        }
    }

    // 当t = 92时调用
    // 输出游戏中还剩两分钟的提醒信息
    private void two_minute_warning() {
        System.out.println("   *** Two minutes left in the game ***");
    }

    // 当用户输入1或2时调用，表示他们的投篮选择
    }

    // 当用户输入0、3或4时调用，表示上篮、定点投篮或防守变化
    }


    // 模拟达特茅斯队的进攻，从你选择的投篮开始
    // 模拟达特茅斯大学的投篮动作
    private void dartmouth_ball() {
        // 创建一个用于从标准输入读取数据的 Scanner 对象
        Scanner scanner = new Scanner(System.in);
        // 打印提示信息
        System.out.print("Your shot? ");
        // 初始化 shot 变量为 -1
        shot = -1;
        // 如果输入的是整数，则将其赋值给 shot
        if (scanner.hasNextInt()) {
            shot = scanner.nextInt();
        }
        // 如果输入不是整数，则打印空行并清空输入缓冲区
        else {
            System.out.println("");
            scanner.next();
        }

        // 当输入的 shot 不在合法的选择范围内时，循环提示重新输入
        while (!shot_choices.contains(shot)) {
            System.out.print("Incorrect answer. Retype it. Your shot?");
            if (scanner.hasNextInt()) {
                shot = scanner.nextInt();
            }
            else {
                System.out.println("");
                scanner.next();
            }
        }

        // 如果时间小于 100 或者随机数小于 0.5
        if (time < 100 || Math.random() < .5) {
            // 如果 shot 为 1 或 2，则模拟达特茅斯大学的跳投动作
            if (shot == 1 || shot == 2) {
                dartmouth_jump_shot();
            }
            // 否则模拟达特茅斯大学的非跳投动作
            else {
                dartmouth_non_jump_shot();
            }
        }
        // 如果时间大于等于 100 且随机数大于等于 0.5
        else {
            // 如果达特茅斯大学的得分不等于对手的得分
            if (score[0] != score[1]) {
                // 打印比赛结束信息和最终比分，然后退出程序
                System.out.println("\n   ***** End Of Game *****");
                System.out.println("Final Score: Dartmouth: " + score[1] + "  "
                      + opponent + ": " + score[0]);
                System.exit(0);
            }
            // 如果达特茅斯大学的得分等于对手的得分
            else {
                // 打印上半场结束信息和比分，然后进入两分钟的加时赛
                System.out.println("\n   ***** End Of Second Half *****");
                System.out.println("Score at end of regulation time:");
                System.out.println("     Dartmouth: " + score[1] + " " +
                      opponent + ": " + score[0]);
                System.out.println("Begin two minute overtime period");
                time = 93;
                start_of_period();
            }
        }
    }

    // 模拟对手的跳投动作

    // 模拟对手的上篮或定点投篮动作
    // 模拟对手的进攻
    private void opponent_non_jumpshot() {
        // 如果对手的机会大于3，输出"定点投篮"
        if (opponent_chance > 3) {
            System.out.println("Set shot.");
        }
        // 否则输出"上篮"
        else {
            System.out.println("Lay up");
        }
        // 如果防守/7 * 随机数 大于 0.413
        if (7/defense*Math.random() > .413) {
            // 输出"投篮不中"
            System.out.println("Shot is missed.");
            // 如果防守/6 * 随机数 大于 0.5
            if (defense/6*Math.random() > .5) {
                // 输出对手控球
                System.out.println(opponent + " controls the rebound.");
                // 如果防守等于6
                if (defense == 6) {
                    // 如果随机数大于0.75
                    if (Math.random() > .75) {
                        // 输出"球被抢断。达特茅斯轻松上篮得分"
                        System.out.println("Ball stolen. Easy lay up for Dartmouth.");
                        // 加分
                        add_points(1, 2);
                        // 对手控球
                        opponent_ball();
                    }
                    else {
                        // 如果随机数大于0.5
                        if (Math.random() > .5) {
                            System.out.println("");
                            // 重新进行非跳投
                            opponent_non_jumpshot();
                        }
                        else {
                            // 输出"传球回" + 对手 + "后卫\n"
                            System.out.println("Pass back to " + opponent +
                                  " guard.\n");
                            // 对手控球
                            opponent_ball();
                        }
                    }
                }
                else {
                    // 如果随机数大于0.5
                    if (Math.random() > .5) {
                        System.out.println("");
                        // 重新进行非跳投
                        opponent_non_jumpshot();
                    }
                    else {
                        // 输出"传球回" + 对手 + "后卫\n"
                        System.out.println("Pass back to " + opponent + " guard\n");
                        // 对手控球
                        opponent_ball();
                    }
                }
            }
            else {
                // 输出"达特茅斯控球\n"
                System.out.println("Dartmouth controls the rebound.\n");
                // 达特茅斯控球
                dartmouth_ball();
            }
        }
        else {
            // 输出"投篮命中"
            System.out.println("Shot is good.");
            // 加分
            add_points(0, 2);
            // 达特茅斯控球
            dartmouth_ball();
        }
    }

    // 模拟对手的进攻
    // 随机选择跳投或上篮/定点投篮
    # 对手出手动作
    private void opponent_ball() {
        # 时间增加
        time ++;
        # 如果时间达到50，执行中场休息
        if (time == 50) {
            halftime();
        }
        # 生成对手出手机会
        opponent_chance = 10/4*Math.random()+1;
        # 如果对手出手机会大于2，执行非跳投动作
        if (opponent_chance > 2) {
            opponent_non_jumpshot();
        }
        # 否则执行跳投动作
        else {
            opponent_jumpshot();
        }
    }

    # 主程序入口
    public static void main(String[] args) {
        # 创建篮球游戏对象
        Basketball new_game = new Basketball();
    }
# 闭合前面的函数定义
```