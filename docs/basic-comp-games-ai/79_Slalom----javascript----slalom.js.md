# `79_Slalom\javascript\slalom.js`

```
// SLALOM
// 该程序是一个游戏，通过Javascript语言编写，由Oscar Toledo G. (nanochess)从BASIC转换而来

function print(str)
{
    // 将字符串输出到页面上
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    // 返回一个Promise对象，用于处理异步操作
    return new Promise(function (resolve) {
                       // 创建一个input元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置input元素的类型为文本
                       input_element.setAttribute("type", "text");
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
# 结束键盘按下事件监听器的定义
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
```
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var speed = [,14,18,26,29,18,  # 定义一个名为speed的数组，包含一系列数字

function show_instructions()  # 定义一个名为show_instructions的函数
{
    print("\n");  # 打印一个空行
    print("*** SLALOM: THIS IS THE 1976 WINTER OLYMPIC GIANT SLALOM.  YOU ARE\n");  # 打印一条消息
    print("            THE AMERICAN TEAM'S ONLY HOPE OF A GOLD MEDAL.\n");  # 打印一条消息
    print("\n");  # 打印一个空行
    print("     0 -- TYPE THIS IS YOU WANT TO SEE HOW LONG YOU'VE TAKEN.\n");  # 打印一条消息
    print("     1 -- TYPE THIS IF YOU WANT TO SPEED UP A LOT.\n");  # 打印一条消息
    print("     2 -- TYPE THIS IF YOU WANT TO SPEED UP A LITTLE.\n");  # 打印一条消息
    print("     3 -- TYPE THIS IF YOU WANT TO SPEED UP A TEENSY.\n");  # 打印一条消息
    # 打印提示信息，提示用户输入选项
    print("     4 -- TYPE THIS IF YOU WANT TO KEEP GOING THE SAME SPEED.\n");
    print("     5 -- TYPE THIS IF YOU WANT TO CHECK A TEENSY.\n");
    print("     6 -- TYPE THIS IF YOU WANT TO CHECK A LITTLE.\n");
    print("     7 -- TYPE THIS IF YOU WANT TO CHECK A LOT.\n");
    print("     8 -- TYPE THIS IF YOU WANT TO CHEAT AND TRY TO SKIP A GATE.\n");
    print("\n");
    print(" THE PLACE TO USE THESE OPTIONS IS WHEN THE COMPUTER ASKS:\n");
    print("\n");
    print("OPTION?\n");
    print("\n");
    print("                GOOD LUCK!\n");
    print("\n");
}

function show_speeds()
{
    # 打印提示信息，显示速度选项
    print("GATE MAX\n");
    print(" #  M.P.H.\n");
    print("----------\n");
    # 循环打印速度选项
    for (var b = 1; b <= v; b++) {
        console.log(" " + b + "  " + speed[b] + "\n"); // 打印每个速度的信息
    }
}

// 主程序
async function main()
{
    var gold = 0; // 初始化金牌数量
    var silver = 0; // 初始化银牌数量
    var bronze = 0; // 初始化铜牌数量

    console.log(tab(33) + "SLALOM\n"); // 打印比赛名称
    console.log(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n"); // 打印比赛地点
    console.log("\n");
    console.log("\n");
    console.log("\n");
    while (1) { // 进入循环
        console.log("HOW MANY GATES DOES THIS COURSE HAVE (1 TO 25)"); // 提示输入门数
        v = parseInt(await input()); // 获取输入的门数并转换为整数
        if (v >= 25) { // 如果门数大于等于25
            # 打印提示信息
            print("25 IS THE LIMIT\n");
            # 设置变量v的值为25
            v = 25;
        } else if (v < 1) {
            # 如果v小于1，打印提示信息
            print("TRY AGAIN.\n");
        } else {
            # 否则跳出循环
            break;
        }
    }
    # 打印换行
    print("\n");
    # 打印提示信息
    print("TYPE \"INS\" FOR INSTRUCTIONS\n");
    # 打印提示信息
    print("TYPE \"MAX\" FOR APPROXIMATE MAXIMUM SPEEDS\n");
    # 打印提示信息
    print("TYPE \"RUN\" FOR THE BEGINNING OF THE RACE\n");
    while (1) {
        # 打印提示信息
        print("COMMAND--");
        # 获取用户输入的字符串
        str = await input();
        if (str == "INS") {
            # 如果用户输入为"INS"，显示指令
            show_instructions();
        } else if (str == "MAX") {
            # 如果用户输入为"MAX"，显示速度
            show_speeds();
        } else if (str == "RUN") {
            break;  # 退出当前循环
        } else:
            print("\"" + str + "\" IS AN ILLEGAL COMMAND--RETRY")  # 打印错误信息
    }
    while (1):  # 进入无限循环
        print("RATE YOURSELF AS A SKIER, (1=WORST, 3=BEST)")  # 打印提示信息
        a = parseInt(await input())  # 获取输入并转换为整数赋值给变量a
        if (a < 1 or a > 3):  # 判断a的取值范围
            print("THE BOUNDS ARE 1-3\n")  # 打印错误信息
        else:
            break  # 退出当前循环
    }
    while (1):  # 进入无限循环
        print("THE STARTER COUNTS DOWN...5...4...3...2...1...GO!")  # 打印提示信息
        t = 0  # 变量t赋值为0
        s = Math.floor(Math.random(1) * (18 - 9) + 9)  # 生成随机数并赋值给变量s
        print("\n")  # 打印换行
        print("YOU'RE OFF!\n")  # 打印提示信息
        for (o = 1; o <= v; o++):  # 进入循环，o从1到v
            q = speed[o];  # 从列表 speed 中获取索引为 o 的值，赋给变量 q
            print("\n");  # 打印一个空行
            print("HERE COMES GATE #" + o + " :\n");  # 打印门的编号和换行符
            print(s + " M.P.H.\n");  # 打印当前速度和换行符
            s1 = s;  # 将当前速度赋给变量 s1
            while (1):  # 进入无限循环
                print("OPTION");  # 打印提示信息
                o1 = parseInt(await input());  # 从输入中获取一个整数，赋给变量 o1
                if (o1 < 0 || o1 > 8)  # 如果 o1 小于 0 或者大于 8
                    print("WHAT?\n");  # 打印提示信息
                else if (o1 == 0)  # 否则如果 o1 等于 0
                    print("YOU'VE TAKEN " + (t + Math.random()) + " SECONDS.\n");  # 打印花费的时间
                else  # 否则
                    break;  # 退出循环
            finish = false;  # 将变量 finish 设置为 false
            switch (o1):  # 根据 o1 的值进行不同的操作
                case 1:  # 如果 o1 的值为 1
                    s += Math.floor(Math.random() * (10 - 5) + 5);  # 将速度增加一个随机值
                    break;  # 退出 switch 语句
                case 2:  // 如果随机数为2
                    s += Math.floor(Math.random() * (5 - 3) + 3);  // 将s增加一个3到5之间的随机整数
                    break;  // 结束该case
                case 3:  // 如果随机数为3
                    s += Math.floor(Math.random() * (4 - 1) + 1);  // 将s增加一个1到4之间的随机整数
                    break;  // 结束该case
                case 4:  // 如果随机数为4
                    break;  // 结束该case
                case 5:  // 如果随机数为5
                    s -= Math.floor(Math.random() * (4 - 1) + 1);  // 将s减去一个1到4之间的随机整数
                    break;  // 结束该case
                case 6:  // 如果随机数为6
                    s -= Math.floor(Math.random() * (5 - 3) + 3);  // 将s减去一个3到5之间的随机整数
                    break;  // 结束该case
                case 7:  // 如果随机数为7
                    s -= Math.floor(Math.random() * (10 - 5) + 5);  // 将s减去一个5到10之间的随机整数
                    break;  // 结束该case
                case 8:  // 如果随机数为8
                    print("***CHEAT\n");  // 打印"***CHEAT\n"
                    if (Math.random() >= 0.7) {  // 如果随机数大于等于0.7
                        print("YOU MADE IT!\n");  # 打印消息表示成功通过
                        t += 1.5;  # 增加时间计数器
                    } else {
                        print("AN OFFICIAL CAUGHT YOU!\n");  # 打印消息表示被官员抓住
                        print("YOU TOOK " + (t + Math.random()) + " SECONDS.\n");  # 打印消息表示花费的时间
                        finish = true;  # 设置完成标志为真
                    }
                    break;  # 跳出循环
            }
            if (!finish) {  # 如果未完成
                if (o1 != 4)  # 如果o1不等于4
                    print(s + " M.P.H.\n");  # 打印速度
                if (s > q) {  # 如果速度大于最大速度
                    if (Math.random() < ((s - q) * 0.1) + 0.2) {  # 如果随机数小于计算得到的值
                        print("YOU WENT OVER THE MAXIMUM SPEED AND ");  # 打印消息表示超速
                        if (Math.random() < 0.5) {  # 如果随机数小于0.5
                            print("SNAGGED A FLAG!\n");  # 打印消息表示抓到了旗帜
                        } else {
                            print("WIPED OUT!\n");  # 打印消息表示摔倒了
                        }
# 如果满足条件，打印出完成游戏所花费的时间
print("YOU TOOK " + (t + Math.random()) + " SECONDS.\n");
# 将完成标志设置为真
finish = true;
# 如果未满足条件，打印出超速但还是成功的消息
print("YOU WENT OVER THE MAXIMUM SPEED AND MADE IT!\n");
# 如果速度接近最大速度，打印出接近成功的消息
print("CLOSE ONE!\n");
# 如果完成标志为真，跳出循环
if (finish)
    break;
# 如果速度小于7，打印出重新开始的提示，重置速度和尝试次数，继续循环
if (s < 7) {
    print("LET'S BE REALISTIC, OK?  LET'S GO BACK AND TRY AGAIN...\n");
    s = s1;
    o--;
    continue;
}
# 更新总时间，根据速度和最大速度的差值
t += q - s + 1;
# 如果速度超过最大速度，额外增加0.5秒的时间
if (s > q) {
    t += 0.5;
        }
        }
        if (!finish) {  # 如果比赛没有结束
            print("\n");  # 打印空行
            print("YOU TOOK " + (t + Math.random()) + " SECONDS.\n");  # 打印参赛者用时
            m = t;  # 将用时赋值给变量m
            m /= v;  # 用时除以速度，得到调整后的用时
            if (m < 1.5 - (a * 0.1)) {  # 如果调整后的用时小于1.5减去加速度的10%
                print("YOU WON A GOLD MEDAL!\n");  # 打印获得金牌的消息
                gold++;  # 金牌数量加一
            } else if (m < 2.9 - (a * 0.1)) {  # 如果调整后的用时小于2.9减去加速度的10%
                print("YOU WON A SILVER MEDAL\n");  # 打印获得银牌的消息
                silver++;  # 银牌数量加一
            } else if (m < 4.4 - (a * 0.1)) {  # 如果调整后的用时小于4.4减去加速度的10%
                print("YOU WON A BRONZE MEDAL\n");  # 打印获得铜牌的消息
                bronze++;  # 铜牌数量加一
            }
        }
        while (1) {  # 无限循环
            print("\n");  # 打印空行
# 打印提示信息，询问是否要再次参加比赛
print("DO YOU WANT TO RACE AGAIN");
# 等待用户输入，并将输入的字符串赋值给变量str
str = await input();
# 如果用户输入的字符串不是"YES"且不是"NO"，则打印提示信息
if (str != "YES" && str != "NO")
    print("PLEASE TYPE 'YES' OR 'NO'\n");
# 否则跳出循环
else
    break;
# 如果用户输入的字符串不是"YES"，则跳出循环
if (str != "YES")
    break;
# 打印感谢信息
print("THANKS FOR THE RACE\n");
# 如果金牌数大于等于1，则打印金牌数
if (gold >= 1)
    print("GOLD MEDALS: " + gold + "\n");
# 如果银牌数大于等于1，则打印银牌数
if (silver >= 1)
    print("SILVER MEDALS: " + silver + "\n");
# 如果铜牌数大于等于1，则打印铜牌数
if (bronze >= 1)
    print("BRONZE MEDALS: " + bronze + "\n");
# 调用主函数
main();
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```