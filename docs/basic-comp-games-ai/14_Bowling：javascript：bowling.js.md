# `d:/src/tocomm/basic-computer-games\14_Bowling\javascript\bowling.js`

```
// 创建一个名为print的函数，用于向页面输出指定的字符串
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 创建一个名为input的函数，用于获取用户输入的字符串
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为JavaScript代码添加了注释，解释了每个函数的作用和返回的Promise对象的用途。这样做可以帮助其他开发人员更容易地理解和使用这些函数。
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下的键是回车键时，获取输入的字符串，移除输入元素，打印输入的字符串并解析输入的字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器
});
}

# 定义一个函数 tab，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环减少 space 并在 str 后面添加一个空格
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// Main program
async function main()
{
    print(tab(34) + "BOWL\n");  // 在第34列打印字符串"BOWL"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在第15列打印字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    c = [];  // 初始化数组c为空数组
    a = [];  // 初始化数组a为空数组
    for (i = 0; i <= 15; i++)  // 循环15次
        c[i] = 0;  // 将数组c的第i个元素赋值为0
    print("WELCOME TO THE ALLEY\n");  // 打印"WELCOME TO THE ALLEY"
    print("BRING YOUR FRIENDS\n");  // 打印"BRING YOUR FRIENDS"
    print("OKAY LET'S FIRST GET ACQUAINTED\n");  // 打印"OKAY LET'S FIRST GET ACQUAINTED"
    print("\n");  // 打印空行
    # 打印提示信息，询问用户是否需要游戏说明
    print("THE INSTRUCTIONS (Y/N)\n");
    # 获取用户输入的字符串
    str = await input();
    # 如果用户输入的字符串以"Y"开头
    if (str.substr(0, 1) == "Y") {
        # 打印保龄球游戏的说明
        print("THE GAME OF BOWLING TAKES MIND AND SKILL.DURING THE GAME\n");
        print("THE COMPUTER WILL KEEP SCORE.YOU MAY COMPETE WITH\n");
        print("OTHER PLAYERS[UP TO FOUR].YOU WILL BE PLAYING TEN FRAMES\n");
        print("ON THE PIN DIAGRAM 'O' MEANS THE PIN IS DOWN...'+' MEANS THE\n");
        print("PIN IS STANDING.AFTER THE GAME THE COMPUTER WILL SHOW YOUR\n");
        print("SCORES .\n");
    }
    # 打印提示信息，询问玩家数量
    print("FIRST OF ALL...HOW MANY ARE PLAYING");
    # 获取用户输入的整数
    r = parseInt(await input());
    # 进入无限循环
    while (1) {
        # 打印换行符
        print("\n");
        # 打印提示信息
        print("VERY GOOD...\n");
        # 初始化二维数组a，表示保龄球的计分板
        for (i = 1; i <= 100; i++) {
            a[i] = [];
            for (j = 1; j <= 6; j++)
                a[i][j] = 0;
        }
        f = 1;  // 初始化变量 f 为 1
        do {
            for (p = 1; p <= r; p++) {  // 循环执行 r 次
                // m = 0; // 在原始代码中重复
                b = 1;  // 初始化变量 b 为 1
                m = 0;  // 初始化变量 m 为 0
                q = 0;  // 初始化变量 q 为 0
                for (i = 1; i <= 15; i++)  // 循环执行 15 次
                    c[i] = 0;  // 初始化数组 c 的每个元素为 0
                while (1) {  // 进入无限循环
                    // 使用模 '15' 系统生成球
                    print("TYPE ROLL TO GET THE BALL GOING.\n");  // 打印提示信息
                    ns = await input();  // 等待用户输入
                    k = 0;  // 初始化变量 k 为 0
                    d = 0;  // 初始化变量 d 为 0
                    for (i = 1; i <= 20; i++) {  // 循环执行 20 次
                        x = Math.floor(Math.random() * 100);  // 生成 0 到 99 之间的随机数
                        for (j = 1; j <= 10; j++)  // 循环执行 10 次
                            if (x < 15 * j)  // 判断条件
                                break;  // 跳出内层循环
// 设置数组 c 的值，根据球道上的击球情况标记
c[15 * j - x] = 1;
// 打印玩家、帧数和球数的信息
print("PLAYER: " + p + " FRAME: " + f + " BALL: " + b + "\n");
print("\n");
// 打印球道图示
for (i = 0; i <= 3; i++) {
    str = "";
    for (j = 1; j <= 4 - i; j++) {
        k++;
        while (str.length < i)
            str += " ";
        if (c[k] == 1)
            str += "O ";
        else
            str += "+ ";
    }
    print(str + "\n");
}
// 进行击球分析
for (i = 1; i <= 10; i++)
# 初始化变量d，用于存储当前帧的得分
d += c[i];
# 如果当前帧的得分减去上一帧的得分等于0，则打印“GUTTER!!\n”
if (d - m == 0)
    print("GUTTER!!\n");
# 如果当前帧是第一次投球且得分为10，则打印“STRIKE!!!!!\n”，并将变量q设为3
if (b == 1 && d == 10) {
    print("STRIKE!!!!!\n");
    q = 3;
}
# 如果当前帧是第二次投球且得分为10，则打印“SPARE!!!!\n”，并将变量q设为2
if (b == 2 && d == 10) {
    print("SPARE!!!!\n");
    q = 2;
}
# 如果当前帧是第二次投球且得分小于10，则打印“ERROR!!!\n”，并将变量q设为1
if (b == 2 && d < 10) {
    print("ERROR!!!\n");
    q = 1;
}
# 如果当前帧是第一次投球且得分小于10，则打印“ROLL YOUR 2ND BALL\n”
if (b == 1 && d < 10) {
    print("ROLL YOUR 2ND BALL\n");
}
# 打印换行符，用于存储得分
print("\n");
                    a[f * p][b] = d;  // 将变量d赋值给二维数组a的特定位置
                    if (b != 2) {  // 如果变量b不等于2
                        b = 2;  // 将变量b赋值为2
                        m = d;  // 将变量d赋值给变量m
                        if (q == 3) {  // 如果变量q等于3
                            a[f * p][b] = d;  // 将变量d赋值给二维数组a的特定位置
                        } else {  // 否则
                            a[f * p][b] = d - m;  // 将变量d减去变量m的值赋给二维数组a的特定位置
                            if (q == 0) // ROLL  // 如果变量q等于0，则执行ROLL操作
                                continue;  // 继续下一次循环
                        }
                    }
                    break;  // 跳出循环
                }
                a[f * p][3] = q;  // 将变量q赋值给二维数组a的特定位置
            }
        } while (++f < 11) ;  // 当f小于11时执行循环
        print("FRAMES\n");  // 打印字符串"FRAMES"
        for (i = 1; i <= 10; i++)  // 循环i从1到10
            print(" " + i + " ");  // 打印i的值
        print("\n");  # 打印空行
        for (p = 1; p <= r; p++) {  # 循环变量 p 从 1 到 r
            for (i = 1; i <= 3; i++) {  # 循环变量 i 从 1 到 3
                for (j = 1; j <= 10; j++) {  # 循环变量 j 从 1 到 10
                    print(" " + a[j * p][i] + " ");  # 打印数组 a 中的值
                }
                print("\n");  # 打印换行
            }
            print("\n");  # 打印空行
        }
        print("DO YOU WANT ANOTHER GAME");  # 打印提示信息
        str = await input();  # 获取用户输入
        if (str.substr(0, 1) != "Y")  # 如果用户输入的第一个字符不是 "Y"
            break;  # 跳出循环
        // Bug in original game, jumps to 2610, without restarting P variable  # 注释说明原始游戏中的错误，跳转到 2610 行，而没有重新启动 P 变量
    }
}

main();  # 调用主函数
```