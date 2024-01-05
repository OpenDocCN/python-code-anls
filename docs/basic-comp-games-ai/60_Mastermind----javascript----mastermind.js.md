# `60_Mastermind\javascript\mastermind.js`

```
// 创建一个名为print的函数，用于在页面上输出文本
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 创建一个名为input的函数，用于获取用户输入
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
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时，获取输入字符串，移除输入元素，打印输入字符串并解析为 Promise 对象
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 结束事件监听器的添加
});
}

# 定义一个函数 tab，用于生成指定数量的空格字符串
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环减少 space 并在 str 中添加一个空格
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串
}

var p9;  # 声明变量 p9
var c9;  # 声明变量 c9
var b;   # 声明变量 b
var w;   # 声明变量 w
var f;   # 声明变量 f
var m;   # 声明变量 m

var qa;  # 声明变量 qa
var sa;  # 声明变量 sa
var ss;  # 声明变量 ss
var as;  # 声明变量 as
var gs;  # 声明变量 gs
var hs;  # 声明变量 hs

function initialize_qa()  # 定义函数 initialize_qa
{
    for (s = 1; s <= p9; s++)
        qa[s] = 0;
}
```
这段代码是一个 for 循环，用于将数组 qa 的每个元素初始化为 0。

```
function increment_qa()
{
    if (qa[1] <= 0) {
        // If zero, this is our firt increment: make all ones
        for (s = 1; s <= p9; s++)
            qa[s] = 1;
    } else {
        q = 1;
        while (1) {
            qa[q] = qa[q] + 1;
            if (qa[q] <= c9)
                return;
            qa[q] = 1;
            q++;
        }
    }
}
```
这段代码定义了一个名为 increment_qa 的函数。首先它检查数组 qa 的第一个元素是否小于等于 0，如果是，则将数组 qa 的每个元素设置为 1。如果不是，则进入一个 while 循环，不断对数组 qa 的元素进行递增操作，直到某个元素小于等于 c9 为止。
}

# 将问答转换为数字
function convert_qa()
{
    # 遍历问题答案数组，根据索引从ls中获取对应位置的字符，存入as数组
    for (s = 1; s <= p9; s++) {
        as[s] = ls.substr(qa[s] - 1, 1);
    }
}

# 获取数字
function get_number()
{
    # 初始化变量b, w, f
    b = 0;
    w = 0;
    f = 0;
    # 遍历问题答案数组
    for (s = 1; s <= p9; s++) {
        # 如果猜测的数字和答案相同，b加1
        if (gs[s] == as[s]) {
            b++;
            # 将猜测的数字和答案对应位置的字符转换为特定的字符
            gs[s] = String.fromCharCode(f);
            as[s] = String.fromCharCode(f + 1);
            f += 2;
        } else {  # 如果条件不成立
            for (t = 1; t <= p9; t++) {  # 循环遍历 t 从 1 到 p9
                if (gs[s] == as[t] && gs[t] != as[t]) {  # 如果 gs[s] 等于 as[t] 并且 gs[t] 不等于 as[t]
                    w++;  # w 自增 1
                    as[t] = String.fromCharCode(f);  # 将 as[t] 转换为 Unicode 编码为 f 的字符
                    gs[s] = String.fromCharCode(f + 1);  # 将 gs[s] 转换为 Unicode 编码为 f+1 的字符
                    f += 2;  # f 自增 2
                    break;  # 跳出循环
                }
            }
        }
    }
}

function convert_qa_hs()  # 定义函数 convert_qa_hs
{
    for (s = 1; s <= p9; s++) {  # 循环遍历 s 从 1 到 p9
        hs[s] = ls.substr(qa[s] - 1, 1);  # 将 ls 中从索引为 qa[s]-1 的位置开始的 1 个字符赋值给 hs[s]
    }
}
# 复制数组 hs 的内容到数组 gs
function copy_hs()
{
    for (s = 1; s <= p9; s++) {  # 循环遍历数组 hs
        gs[s] = hs[s];  # 将数组 hs 的值复制到数组 gs
    }
}

# 打印游戏板的状态
function board_printout()
{
    print("\n");  # 打印空行
    print("BOARD\n");  # 打印标题 BOARD
    print("MOVE     GUESS          BLACK     WHITE\n");  # 打印表头
    for (z = 1; z <= m - 1; z++) {  # 循环遍历游戏板的状态
        str = " " + z + " ";  # 创建字符串并赋值
        while (str.length < 9)  # 当字符串长度小于 9 时
            str += " ";  # 在字符串末尾添加空格
        str += ss[z];  # 将游戏板状态添加到字符串末尾
        while (str.length < 25)  # 当字符串长度小于 25 时
            str += " ";  # 在字符串末尾添加空格
        str += sa[z][1];  # 将sa[z][1]的值添加到str字符串末尾
        while (str.length < 35)  # 当str字符串长度小于35时
            str += " ";  # 在str字符串末尾添加空格，直到长度达到35
        str += sa[z][2];  # 将sa[z][2]的值添加到str字符串末尾
        print(str + "\n");  # 打印str字符串并换行
    }
    print("\n");  # 打印空行
}

function quit()  # 定义名为quit的函数
{
    print("QUITTER!  MY COMBINATION WAS: ");  # 打印提示信息
    convert_qa();  # 调用convert_qa函数
    for (x = 1; x <= p9; x++) {  # 循环，x从1到p9
        print(as[x]);  # 打印as[x]的值
    }
    print("\n");  # 打印空行
    print("GOOD BYE\n");  # 打印结束语
}
# 显示分数的函数
function show_score()
{
    # 打印分数标题
    print("SCORE:\n");
    # 调用显示分数的函数
    show_points();
}

# 显示具体分数的函数
function show_points()
{
    # 打印计算机得分
    print("     COMPUTER " + c + "\n");
    # 打印玩家得分
    print("     HUMAN    " + h + "\n");
    # 打印空行
    print("\n");
}

# 颜色数组
var color = ["BLACK", "WHITE", "RED", "GREEN",
             "ORANGE", "YELLOW", "PURPLE", "TAN"];

# 主程序
async function main()
{
    # 打印游戏标题
    print(tab(30) + "MASTERMIND\n");
    # 打印标题和作者信息
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    #
    #  MASTERMIND II
    #  STEVE NORTH
    #  CREATIVE COMPUTING
    #  PO BOX 789-M MORRISTOWN NEW JERSEY 07960
    #
    #
    # 循环直到输入合法的颜色数量
    while (1):
        print("NUMBER OF COLORS");
        c9 = parseInt(await input());
        if (c9 <= 8):
            break;
        print("NO MORE THAN 8, PLEASE!\n");
    # 输入位置数量
    print("NUMBER OF POSITIONS");
    p9 = parseInt(await input());
    # 打印"NUMBER OF ROUNDS"字符串
    print("NUMBER OF ROUNDS");
    # 从输入中获取用户输入的整数值，赋给变量r9
    r9 = parseInt(await input());
    # 计算c9的p9次方，赋给变量p
    p = Math.pow(c9, p9);
    # 打印"TOTAL POSSIBILITIES = "和p的字符串形式
    print("TOTAL POSSIBILITIES = " + p + "\n");
    # 初始化变量h和c为0
    h = 0;
    c = 0;
    # 初始化空数组qa, sa, ss, as, gs, ia, hs
    qa = [];
    sa = [];
    ss = [];
    as = [];
    gs = [];
    ia = [];
    hs = [];
    # 初始化字符串ls为"BWRGOYPT"
    ls = "BWRGOYPT";
    # 打印两个空行
    print("\n");
    print("\n");
    # 打印"COLOR    LETTER"和"=====    ======"字符串
    print("COLOR    LETTER\n");
    print("=====    ======\n");
    # 循环遍历c9次
    for (x = 1; x <= c9; x++) {
        # 获取color数组中索引为x-1的元素，赋给变量str
        str = color[x - 1];
        while (str.length < 13)  # 当字符串长度小于13时
            str += " ";  # 在字符串末尾添加空格
        str += ls.substr(x - 1, 1);  # 将ls字符串中的第x-1个字符添加到str字符串末尾
        print(str + "\n");  # 打印str字符串并换行
    }
    print("\n");  # 打印空行
    for (r = 1; r <= r9; r++) {  # 循环r从1到r9
        print("\n");  # 打印空行
        print("ROUND NUMBER " + r + " ----\n");  # 打印ROUND NUMBER和r的值，并换行
        print("\n");  # 打印空行
        print("GUESS MY COMBINATION.\n");  # 打印GUESS MY COMBINATION并换行
        print("\n");  # 打印空行
        // Get a combination  # 获取一个组合
        a = Math.floor(p * Math.random() + 1);  # 生成一个随机数赋值给a
        initialize_qa();  # 调用initialize_qa函数
        for (x = 1; x <= a; x++) {  # 循环x从1到a
            increment_qa();  # 调用increment_qa函数
        }
        for (m = 1; m <= 10; m++) {  # 循环m从1到10
            while (1) {  # 进入无限循环
                # 打印 MOVE # 和猜测的提示
                print("MOVE # " + m + " GUESS ");
                # 等待用户输入
                str = await input();
                # 如果用户输入 BOARD，则打印游戏板
                if (str == "BOARD") {
                    board_printout();
                } 
                # 如果用户输入 QUIT，则退出游戏
                else if (str == "QUIT") {
                    quit();
                    return;
                } 
                # 如果用户输入的长度不等于 p9，则打印错误信息
                else if (str.length != p9) {
                    print("BAD NUMBER OF POSITIONS.\n");
                } 
                # 如果用户输入符合要求，则将其解析为 gs(1-p9)
                else {
                    # 将用户输入的字符串解析为 gs(1-p9)
                    for (x = 1; x <= p9; x++) {
                        y = ls.indexOf(str.substr(x - 1, 1));
                        # 如果字符不在 ls 中，则打印错误信息
                        if (y < 0) {
                            print("'" + str.substr(x - 1, 1) + "' IS UNRECOGNIZED.\n");
                            break;
                        }
                        gs[x] = str.substr(x - 1, 1);
                    }
                    # 如果解析成功，则进行下一步操作
                    if (x > p9)
            // 循环直到猜对或者超过10次
            while (true) {
                // 生成一个新的随机数序列
                generate_random_sequence();
                // 让用户猜测序列
                user_guess();
                // 如果猜对了，结束循环
                if (b == p9) {
                    break;
                }
            }
            // 猜对后输出信息并结束程序
            print("YOU GUESSED IT IN " + m + " MOVES!\n");
            break;
            // 输出猜测结果
            print("YOU HAVE " + b + " BLACKS AND " + w + " WHITES.")
            // 保存猜测结果用于后续打印
            ss[m] = str;
            sa[m] = [];
            sa[m][1] = b;
            sa[m][2] = w;
        }
        // 如果超过10次仍未猜对，结束程序
        if (m > 10) {
            print("YOU RAN OUT OF MOVES!  THAT'S ALL YOU GET!\n");
        }  // 打印消息，表示玩家已经用完了所有的移动次数
        h += m;  // 将 m 的值加到 h 上
        show_score();  // 显示分数

        //
        // Now computer guesses
        //
        for (x = 1; x <= p; x++)  // 初始化数组 ia
            ia[x] = 1;
        print("NOW I GUESS.  THINK OF A COMBINATION.\n");  // 打印消息，提示玩家电脑要猜测
        print("HIT RETURN WHEN READY:");  // 打印消息，提示玩家按回车键
        str = await input();  // 等待玩家输入
        for (m = 1; m <= 10; m++) {  // 循环 10 次
            initialize_qa();  // 初始化问题和答案
            // Find a guess
            g = Math.floor(p * Math.random() + 1);  // 生成一个随机数作为猜测
            if (ia[g] != 1) {  // 如果猜测已经被使用过
                for (x = g; x <= p; x++) {  // 从 g 开始往后遍历
                    if (ia[x] == 1)  // 如果当前位置还没有被使用
# 如果条件满足，则跳出循环
                        break;
                }
                # 如果 x 大于 p，则执行以下操作
                if (x > p) {
                    # 遍历数组 ia，直到找到值为 1 的元素
                    for (x = 1; x <= g; x++) {
                        if (ia[x] == 1)
                            break;
                    }
                    # 如果 x 大于 g，则输出提示信息并重置数组 ia 的值
                    if (x > g) {
                        print("YOU HAVE GIVEN ME INCONSISTENT INFORMATION.\n");
                        print("TRY AGAIN, AND THIS TIME PLEASE BE MORE CAREFUL.\n");
                        for (x = 1; x <= p; x++)
                            ia[x] = 1;
                        print("NOW I GUESS.  THINK OF A COMBINATION.\n");
                        print("HIT RETURN WHEN READY:");
                        str = await input();
                        m = 0;
                        continue;
                    }
                }
                # 将 g 的值设为 x
                g = x;
            } // 结束循环
            // 现在我们将猜测 #g 转换为 gs
            for (x = 1; x <= g; x++) { // 循环 g 次
                increment_qa(); // 调用增加 qa 的函数
            }
            convert_qa_hs(); // 调用转换 qa 到 hs 的函数
            print("MY GUESS IS: "); // 打印消息
            for (x = 1; x <= p9; x++) { // 循环 p9 次
                print(hs[x]); // 打印 hs 数组中的值
            }
            print("  BLACKS, WHITES "); // 打印消息
            str = await input(); // 等待用户输入
            b1 = parseInt(str); // 将输入的字符串转换为整数并赋值给 b1
            w1 = parseInt(str.substr(str.indexOf(",") + 1)); // 从逗号后的部分将字符串转换为整数并赋值给 w1
            if (b1 == p9) { // 如果 b1 等于 p9
                print("I GOT IT IN " + m + " MOVES!\n"); // 打印消息
                break; // 跳出循环
            }
            initialize_qa(); // 调用初始化 qa 的函数
            for (x = 1; x <= p; x++) { // 循环 p 次
                increment_qa();  # 调用函数increment_qa()，执行相应的操作
                if (ia[x] != 0) {  # 如果数组ia的第x个元素不等于0
                    copy_hs();  # 调用函数copy_hs()，执行相应的操作
                    convert_qa();  # 调用函数convert_qa()，执行相应的操作
                    get_number();  # 调用函数get_number()，执行相应的操作
                    if (b1 != b || w1 != w)  # 如果b1不等于b或者w1不等于w
                        ia[x] = 0;  # 将数组ia的第x个元素设为0
                }
            }
        }
        if (m > 10) {  # 如果m大于10
            print("I USED UP ALL MY MOVES!\n");  # 打印"I USED UP ALL MY MOVES!"
            print("I GUESS MY CPU I JUST HAVING AN OFF DAY.\n");  # 打印"I GUESS MY CPU I JUST HAVING AN OFF DAY."
        }
        c += m;  # 将c增加m的值
        show_score();  # 调用函数show_score()，显示相应的得分
    }
    print("GAME OVER\n");  # 打印"GAME OVER"
    print("FINAL SCORE:\n");  # 打印"FINAL SCORE:"
    show_points();  # 调用函数show_points()，显示最终得分
}

# 调用主函数
main();
```