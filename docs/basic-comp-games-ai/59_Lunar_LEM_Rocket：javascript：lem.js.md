# `59_Lunar_LEM_Rocket\javascript\lem.js`

```
# 定义函数print，用于向页面输出内容
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义函数input，用于获取用户输入
def input():
    # 声明变量
    var input_element
    var input_str

    # 返回一个Promise对象，用于异步处理用户输入
    return new Promise(function (resolve) {
        # 创建一个input元素
        input_element = document.createElement("INPUT')
        # 输出提示符
        print("? ")
        # 设置input元素的类型为文本
        input_element.setAttribute("type", "text")
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
        str += " ";  # 将一个空格添加到字符串末尾
    return str;  # 返回修改后的字符串

// Main program
async function main()
{
    print(tab(34) + "LEM\n");  # 在指定位置打印字符串 "LEM"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 在指定位置打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    // ROCKT2 is an interactive game that simulates a lunar
    // landing is similar to that of the Apollo program.
    // There is absolutely no chance involved
    zs = "GO";  # 初始化变量 zs 为字符串 "GO"
    b1 = 1;  # 初始化变量 b1 为整数 1
    while (1) {  # 进入无限循环
        m = 17.95;  # 初始化变量 m 为浮点数 17.95
        f1 = 5.25;  # 初始化变量 f1 为浮点数 5.25
        n = 7.5;  # 初始化变量 n 为浮点数 7.5
        r0 = 926;  # 初始化变量 r0 为整数 926
        v0 = 1.29;  # 初始化变量 v0 为浮点数 1.29
        t = 0;  # 初始化变量 t 为 0
        h0 = 60;  # 初始化变量 h0 为 60
        r = r0 + h0;  # 计算变量 r 的值为 r0 + h0
        a = -3,425;  # 初始化变量 a 为 -3,425
        r1 = 0;  # 初始化变量 r1 为 0
        a1 = 8.84361e-4;  # 初始化变量 a1 为 8.84361e-4
        r3 = 0;  # 初始化变量 r3 为 0
        a3 = 0;  # 初始化变量 a3 为 0
        m1 = 7.45;  # 初始化变量 m1 为 7.45
        m0 = m1;  # 变量 m0 的值等于 m1
        b = 750;  # 初始化变量 b 为 750
        t1 = 0;  # 初始化变量 t1 为 0
        f = 0;  # 初始化变量 f 为 0
        p = 0;  # 初始化变量 p 为 0
        n = 1;  # 初始化变量 n 为 1
        m2 = 0;  # 初始化变量 m2 为 0
        s = 0;  # 初始化变量 s 为 0
        c = 0;  # 初始化变量 c 为 0
        if (zs == "YES") {  # 如果变量 zs 的值为 "YES"，则执行以下代码
            print("\n");  # 打印换行符
            # 打印提示信息，询问用户是否需要完整的指令或输入输出语句
            print("OK, DO YOU WANT THE COMPLETE INSTRUCTIONS OR THE INPUT -\n");
            print("OUTPUT STATEMENTS?\n");
            # 进入循环，等待用户输入选择
            while (1) {
                # 打印选项1
                print("1=COMPLETE INSTRUCTIONS\n");
                # 打印选项2
                print("2=INPUT-OUTPUT STATEMENTS\n");
                # 打印选项3
                print("3=NEITHER\n");
                # 获取用户输入的选择
                b1 = parseInt(await input());
                # 初始化qs变量
                qs = "NO";
                # 如果用户选择了1，则跳出循环
                if (b1 == 1)
                    break;
                # 设置qs为"YES"
                qs = "YES";
                # 如果用户选择了2或3，则跳出循环
                if (b1 == 2 || b1 == 3)
                    break;
            }
        } else {
            # 打印提示信息
            print("\n");
            print("LUNAR LANDING SIMULATION\n");
            print("\n");
            # 打印提示信息，询问用户是否之前飞行过阿波罗/LEM任务
            print("HAVE YOU FLOWN AN APOLLO/LEM MISSION BEFORE");
            # 进入循环，等待用户输入选择
            while (1) {
                # 打印提示信息，要求用户输入 YES 或者 NO
                print(" (YES OR NO)");
                # 等待用户输入
                qs = await input();
                # 如果用户输入的是 YES 或者 NO，则跳出循环
                if (qs == "YES" || qs == "NO")
                    break;
                # 如果用户输入的不是 YES 或者 NO，则提示用户重新回答问题
                print("JUST ANSWER THE QUESTION, PLEASE, ");
            }
        }
        # 如果用户输入的是 YES，则提示用户输入测量选项的数字
        if (qs == "YES") {
            print("\n");
            print("INPUT MEASUREMENT OPTION NUMBER");
        } else {
            # 如果用户输入的是 NO，则提示用户选择使用哪种测量系统
            print("\n");
            print("WHICH SYSTEM OF MEASUREMENT DO YOU PREFER?\n");
            print(" 1=METRIC     0=ENGLISH\n");
            print("ENTER THE APPROPRIATE NUMBER");
        }
        # 循环等待用户输入数字，直到用户输入的是 0 或者 1
        while (1) {
            k = parseInt(await input());
            if (k == 0 || k == 1)
                break;
            print("ENTER THE APPROPRIATE NUMBER");  # 打印提示信息，要求输入适当的数字
        }
        if (k == 1) {  # 如果 k 的值等于 1
            z = 1852.8;  # 将 z 赋值为 1852.8
            ms = "METERS";  # 将 ms 赋值为 "METERS"
            g3 = 3.6;  # 将 g3 赋值为 3.6
            ns = " KILOMETERS";  # 将 ns 赋值为 " KILOMETERS"
            g5 = 1000;  # 将 g5 赋值为 1000
        } else {  # 否则
            z = 6080;  # 将 z 赋值为 6080
            ms = "FEET";  # 将 ms 赋值为 "FEET"
            g3 = 0.592;  # 将 g3 赋值为 0.592
            ns = "N.MILES";  # 将 ns 赋值为 "N.MILES"
            g5 = z;  # 将 g5 赋值为 z 的值
        }
        if (b1 != 3) {  # 如果 b1 的值不等于 3
            if (qs != "YES") {  # 如果 qs 的值不等于 "YES"
                print("\n");  # 打印换行
                print("  YOU ARE ON A LUNAR LANDING MISSION.  AS THE PILOT OF\n");  # 打印提示信息
                print("THE LUNAR EXCURSION MODULE, YOU WILL BE EXPECTED TO\n");  # 打印提示信息
# 打印给定命令到导航系统模块
print("GIVE CERTAIN COMMANDS TO THE MODULE NAVIGATION SYSTEM.\n");
# 打印需要导航飞船的信息
print("THE ON-BOARD COMPUTER WILL GIVE A RUNNING ACCOUNT\n");
print("OF INFORMATION NEEDED TO NAVIGATE THE SHIP.\n");
print("\n");
print("\n");
# 打印所需的姿态角度描述
print("THE ATTITUDE ANGLE CALLED FOR IS DESCRIBED AS FOLLOWS.\n");
print("+ OR -180 DEGREES IS DIRECTLY AWAY FROM THE MOON\n");
print("-90 DEGREES IS ON A TANGENT IN THE DIRECTION OF ORBIT\n");
print("+90 DEGREES IS ON A TANGENT FROM THE DIRECTION OF ORBIT\n");
print("0 (ZERO) DEGREES IS DIRECTLY TOWARD THE MOON\n");
print("\n");
# 打印姿态角度示意图
print(tab(30) + "-180|+180\n");
print(tab(34) + "^\n");
print(tab(27) + "-90 < -+- > +90\n");
print(tab(34) + "!\n");
print(tab(34) + "0\n");
print(tab(21) + "<<<< DIRECTION OF ORBIT <<<<\n");
print("\n");
print(tab(20) + "------ SURFACE OF MOON ------\n");
print("\n");
                # 打印空行
                print("\n");
                # 打印提示信息
                print("ALL ANGLES BETWEEN -180 AND +180 DEGREES ARE ACCEPTED.\n");
                print("\n");
                # 打印提示信息
                print("1 FUEL UNIT = 1 SEC. AT MAX THRUST\n");
                # 打印提示信息
                print("ANY DISCREPANCIES ARE ACCOUNTED FOR IN THE USE OF FUEL\n");
                # 打印提示信息
                print("FOR AN ATTITUDE CHANGE.\n");
                # 打印提示信息
                print("AVAILABLE ENGINE POWER: 0 (ZERO) AND ANY VALUE BETWEEN\n");
                # 打印提示信息
                print("10 AND 100 PERCENT.\n");
                print("\n");
                # 打印提示信息
                print("NEGATIVE THRUST OR TIME IS PROHIBITED.\n");
                print("\n");
            }
            # 打印提示信息
            print("\n");
            # 打印提示信息
            print("INPUT: TIME INTERVAL IN SECONDS ------ (T)\n");
            # 打印提示信息
            print("       PERCENTAGE OF THRUST ---------- (P)\n");
            # 打印提示信息
            print("       ATTITUDE ANGLE IN DEGREES ----- (A)\n");
            print("\n");
            # 如果qs不等于"YES"，打印示例
            if (qs != "YES") {
                print("FOR EXAMPLE:\n");
                print("T,P,A? 10,65,-60\n");
                # 打印提示信息，告知用户可以随时中止任务
                print("TO ABORT THE MISSION AT ANY TIME, ENTER 0,0,0\n");
                # 打印空行
                print("\n");
            }
            # 打印输出信息
            print("OUTPUT: TOTAL TIME IN ELAPSED SECONDS\n");
            print("        HEIGHT IN " + ms + "\n");
            print("        DISTANCE FROM LANDING SITE IN " + ms + "\n");
            print("        VERTICAL VELOCITY IN " + ms + "/SECOND\n");
            print("        HORIZONTAL VELOCITY IN " + ms + "/SECOND\n");
            print("        FUEL UNITS REMAINING\n");
            print("\n");
        }
        # 进入无限循环
        while (1) {
            # 遍历n次
            for (i = 1; i <= n; i++) {
                # 如果m1不等于0
                if (m1 != 0) {
                    # m1减去m2
                    m1 -= m2;
                    # 如果m1小于等于0
                    if (m1 <= 0) {
                        # 更新f和m2的值
                        f = f * (1 + m1 / m2);
                        m2 = m1 + m2;
                        # 打印燃料用尽的提示信息
                        print("YOU ARE OUT OF FUEL.\n");
                        # 将m1设置为0
                        m1 = 0;
                } else {
                    f = 0;  # 如果条件不满足，则将 f 置为 0
                    m2 = 0;  # 如果条件不满足，则将 m2 置为 0
                }
                m = m - 0.5 * m2;  # 更新 m 的值为原值减去 m2 的一半
                r4 = r3;  # 将 r3 的值赋给 r4
                r3 = -0.5 * r0 * Math.pow(v0 / r, 2) + r * a1 * a1;  # 更新 r3 的值
                r2 = (3 * r3 - r4) / 2 + 0.00526 * f1 * f * c / m;  # 计算并更新 r2 的值
                a4 = a3;  # 将 a3 的值赋给 a4
                a3 = -2 * r1 * a1 / r;  # 更新 a3 的值
                a2 = (3 * a3 - a4) / 2 + 0.0056 * f1 * f * s / (m * r);  # 计算并更新 a2 的值
                x = r1 * t1 + 0.5 * r2 * t1 * t1;  # 计算 x 的值
                r = r + x;  # 更新 r 的值
                h0 = h0 + x;  # 更新 h0 的值
                r1 = r1 + r2 * t1;  # 更新 r1 的值
                a = a + a1 * t1 + 0.5 * a2 * t1 * t1;  # 更新 a 的值
                a1 = a1 + a2 * t1;  # 更新 a1 的值
                m = m - 0.5 * m2;  # 更新 m 的值
                t = t + t1;  # 更新 t 的值
                # 如果h0小于3.287828e-4，则跳出循环
                if (h0 < 3.287828e-4)
                    break;
            }
            # 计算h的值
            h = h0 * z;
            # 计算h1的值
            h1 = r1 * z;
            # 计算d的值
            d = r0 * a * z;
            # 计算d1的值
            d1 = r * a1 * z;
            # 计算t2的值
            t2 = m1 * b / m0;
            # 打印输出t、h、d、h1、d1、t2的值
            print(" " + t + "\t" + h + "\t" + d + "\t" + h1 + "\t" + d1 + "\t" + t2 + "\n");
            # 如果h0小于3.287828e-4
            if (h0 < 3.287828e-4) {
                # 如果r1小于-8.21957e-4或者|r * a1|大于4.93174e-4或者h0小于-3.287828e-4
                if (r1 < -8.21957e-4 || Math.abs(r * a1) > 4.93174e-4 || h0 < -3.287828e-4) {
                    # 打印输出提示信息
                    print("\n");
                    print("CRASH !!!!!!!!!!!!!!!!\n");
                    print("YOUR IMPACT CREATED A CRATER " + Math.abs(h) + " " + ms + " DEEP.\n");
                    # 计算x1的值
                    x1 = Math.sqrt(d1 * d1 + h1 * h1) * g3;
                    # 打印输出提示信息
                    print("AT CONTACT YOU WERE TRAVELING " + x1 + " " + ns + "/HR\n");
                    # 跳出循环
                    break;
                }
                # 如果|d|大于10 * z
                if (Math.abs(d) > 10 * z) {
                    # 打印输出提示信息
                    print("YOU ARE DOWN SAFELY - \n");
                    # 打印空行
                    print("\n");
                    # 打印信息，包括飞船偏离着陆点的距离和方向
                    print("BUT MISSED THE LANDING SITE BY " + Math.abs(d / g5) + " " + ns + ".\n");
                    # 跳出循环
                    break;
                }
                # 打印着陆成功的信息
                print("\n");
                print("TRANQUILITY BASE HERE -- THE EAGLE HAS LANDED.\n");
                print("CONGRATULATIONS -- THERE WAS NO SPACECRAFT DAMAGE.\n");
                print("YOU MAY NOW PROCEED WITH SURFACE EXPLORATION.\n");
                # 跳出循环
                break;
            }
            # 如果飞船偏离轨道太远，打印信息并跳出循环
            if (r0 * a > 164.474) {
                print("\n");
                print("YOU HAVE BEEN LOST IN SPACE WITH NO HOPE OF RECOVERY.\n");
                break;
            }
            # 如果燃料剩余大于0，进入循环
            if (m1 > 0) {
                # 循环直到条件不满足
                while (1) {
                    # 打印信息
                    print("T,P,A");
                    # 等待输入
                    str = await input();
                    # 将输入转换为浮点数
                    t1 = parseFloat(str);
                    // 从字符串中提取逗号后的数字并转换为浮点数赋值给变量f
                    f = parseFloat(str.substr(str.indexOf(",") + 1));
                    // 从字符串中提取最后一个逗号后的数字并转换为浮点数赋值给变量p
                    p = parseFloat(str.substr(str.lastIndexOf(",") + 1));
                    // 将f除以100
                    f = f / 100;
                    // 如果t1小于0，打印以下信息
                    if (t1 < 0) {
                        print("\n");
                        print("THIS SPACECRAFT IS NOT ABLE TO VIOLATE THE SPACE-");
                        print("TIME CONTINUUM.\n");
                        print("\n");
                    } 
                    // 如果t1等于0，跳出循环
                    else if (t1 == 0) {
                        break;
                    } 
                    // 如果f与0.05的差的绝对值大于1或者小于0.05，打印以下信息
                    else if (Math.abs(f - 0.05) > 1 || Math.abs(f - 0.05) < 0.05) {
                        print("IMPOSSIBLE THRUST VALUE ");
                        // 如果f小于0，打印NEGATIVE
                        if (f < 0) {
                            print("NEGATIVE\n");
                        } 
                        // 如果f-0.05小于0.05，打印TOO SMALL
                        else if (f - 0.05 < 0.05) {
                            print("TOO SMALL\n");
                        } 
                        // 否则打印TOO LARGE
                        else {
                            print("TOO LARGE\n");
                        }
                        print("\n");
                    } else if (Math.abs(p) > 180) {  # 如果绝对值大于180度
                        print("\n");  # 打印换行
                        print("IF YOU WANT TO SPIN AROUND, GO OUTSIDE THE MODULE\n");  # 打印提示信息
                        print("FOR AN E.V.A.\n");  # 打印提示信息
                        print("\n");  # 打印换行
                    } else {  # 否则
                        break;  # 跳出循环
                    }
                }
                if (t1 == 0) {  # 如果t1等于0
                    print("\n");  # 打印换行
                    print("MISSION ABENDED\n");  # 打印提示信息
                    break;  # 跳出循环
                }
            } else {  # 否则
                t1 = 20;  # t1赋值为20
                f = 0;  # f赋值为0
                p = 0;  # p赋值为0
            }
            n = 20;  # n赋值为20
            if (t1 >= 400)  # 如果 t1 大于等于 400
                n = t1 / 20;  # 则将 n 设为 t1 除以 20 的结果
            t1 = t1 / n;  # 将 t1 除以 n 的结果赋值给 t1
            p = p * 3.14159 / 180;  # 将 p 乘以 π/180 的结果赋值给 p
            s = Math.sin(p);  # 计算 p 的正弦值并赋值给 s
            c = Math.cos(p);  # 计算 p 的余弦值并赋值给 c
            m2 = m0 * t1 * f / b;  # 计算 m2 的值
            r3 = -0.5 * r0 * Math.pow(v0 / r, 2) + r * a1 * a1;  # 计算 r3 的值
            a3 = -2 * r1 * a1 / r;  # 计算 a3 的值
        }
        print("\n");  # 打印换行符
        while (1) {  # 进入无限循环
            print("DO YOU WANT TO TRY IT AGAIN (YES/NO)?\n");  # 打印提示信息
            zs = await input();  # 等待用户输入并赋值给 zs
            if (zs == "YES" || zs == "NO")  # 如果 zs 等于 "YES" 或者 zs 等于 "NO"
                break;  # 退出循环
        }
        if (zs != "YES")  # 如果 zs 不等于 "YES"
            break;  # 退出循环
    }
    print("\n");  # 打印空行
    print("TOO BAD, THE SPACE PROGRAM HATES TO LOSE EXPERIENCED\n");  # 打印提示信息
    print("ASTRONAUTS.\n");  # 打印提示信息
}

main();  # 调用主函数
```