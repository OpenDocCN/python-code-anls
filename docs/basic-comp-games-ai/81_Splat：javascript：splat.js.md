# `d:/src/tocomm/basic-computer-games\81_Splat\javascript\splat.js`

```
// 定义一个名为print的函数，用于在页面上输出字符串
function print(str) {
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个名为input的函数，用于获取用户输入
function input() {
    var input_element;
    var input_str;

    // 返回一个Promise对象，表示异步操作的最终完成或失败
    return new Promise(function (resolve) {
        // 创建一个input元素
        input_element = document.createElement("INPUT");

        // 在页面上输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为JavaScript代码添加了注释，解释了每个语句的作用。这样做可以帮助其他程序员更容易地理解代码的功能和逻辑。
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
        # 解析输入字符串
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
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

var aa = [];  // 创建一个空数组

// 主程序
async function main()
{
    print(tab(33) + "SPLAT\n");  // 打印带有制表符的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有制表符的字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    for (i = 0; i <= 42; i++)  // 循环，将数组aa的每个元素初始化为0
        aa[i] = 0;
    print("WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n");  // 打印欢迎消息
    print("JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE\n");  // 打印提示信息
    print("MOMENT WITHOUT GOING SPLAT.\n");  // 打印提示信息
    while (1) {  // 进入无限循环
        # 打印两个空行
        print("\n");
        print("\n");
        # 初始化变量
        d1 = 0;
        v = 0;
        a = 0;
        n = 0;
        m = 0;
        # 生成一个随机数并赋值给d1
        d1 = Math.floor(9001 * Math.random() + 1000);
        # 打印提示信息
        print("SELECT YOUR OWN TERMINAL VELOCITY (YES OR NO)");
        # 循环直到用户输入YES或NO
        while (1) {
            a1s = await input();
            if (a1s == "YES" || a1s == "NO")
                break;
            # 如果用户输入不是YES或NO，则提示用户重新输入
            print("YES OR NO");
        }
        # 如果用户输入YES
        if (a1s == "YES") {
            # 提示用户输入终端速度，并将其转换为每小时英里
            print("WHAT TERMINAL VELOCITY (MI/HR)");
            v1 = parseFloat(await input());
            v1 = v1 * (5280 / 3600);
        } else {
            v1 = Math.floor(1000 * Math.random());  # 生成一个随机数并向下取整，表示初始速度
            print("OK.  TERMINAL VELOCITY = " + v1 + " MI/HR\n");  # 打印初始速度

        }
        v = v1 + ((v1 * Math.random()) / 20) - ((v1 * Math.random()) / 20);  # 计算最终速度
        print("WANT TO SELECT ACCELERATION DUE TO GRAVITY (YES OR NO)");  # 打印提示信息，询问是否选择重力加速度
        while (1) {  # 进入循环，直到用户输入合法的选项
            b1s = await input();  # 获取用户输入
            if (b1s == "YES" || b1s == "NO")  # 判断用户输入是否为合法选项
                break;  # 如果是合法选项则跳出循环
            print("YES OR NO");  # 如果输入不合法，则提示用户重新输入
        }
        if (b1s == "YES") {  # 如果用户选择了是
            print("WHAT ACCELERATION (FT/SEC/SEC)");  # 打印提示信息，询问加速度
            a2 = parseFloat(await input());  # 获取用户输入的加速度值并转换为浮点数
        } else {  # 如果用户选择了否
            switch (Math.floor(1 + (10 * Math.random()))) {  # 随机生成一个1-10的整数
                case 1:  # 如果随机数为1
                    print("FINE. YOU'RE ON MERCURY. ACCELERATION=12.2 FT/SEC/SEC.\n");  # 打印提示信息，表示在水星上的加速度
                    a2 = 12.2;  # 设置加速度为12.2
                    break;  # 跳出switch语句
# 如果选择为2，则打印“ALL RIGHT. YOU'RE ON VENUS. ACCELERATION=28.3 FT/SEC/SEC.”并将a2赋值为28.3
case 2:
    print("ALL RIGHT. YOU'RE ON VENUS. ACCELERATION=28.3 FT/SEC/SEC.\n");
    a2 = 28.3;
    break;
# 如果选择为3，则打印“THEN YOU'RE ON EARTH. ACCELERATION=32.16 FT/SEC/SEC.”并将a2赋值为32.16
case 3:
    print("THEN YOU'RE ON EARTH. ACCELERATION=32.16 FT/SEC/SEC.\n");
    a2 = 32.16;
    break;
# 如果选择为4，则打印“FINE. YOU'RE ON THE MOON. ACCELERATION=5.15 FT/SEC/SEC.”并将a2赋值为5.15
case 4:
    print("FINE. YOU'RE ON THE MOON. ACCELERATION=5.15 FT/SEC/SEC.\n");
    a2 = 5.15;
    break;
# 如果选择为5，则打印“ALL RIGHT. YOU'RE ON MARS. ACCELERATION=12.5 FT/SEC/SEC.”并将a2赋值为12.5
case 5:
    print("ALL RIGHT. YOU'RE ON MARS. ACCELERATION=12.5 FT/SEC/SEC.\n");
    a2 = 12.5;
    break;
# 如果选择为6，则打印“THEN YOU'RE ON JUPITER. ACCELERATION=85.2 FT/SEC/SEC.”并将a2赋值为85.2
case 6:
    print("THEN YOU'RE ON JUPITER. ACCELERATION=85.2 FT/SEC/SEC.\n");
    a2 = 85.2;
    break;
                case 7:  # 如果条件为7
                    print("FINE. YOU'RE ON SATURN. ACCELERATION=37.6 FT/SEC/SEC.\n");  # 打印出在土星上的加速度
                    a2 = 37.6;  # 将加速度设置为37.6
                    break;  # 跳出switch语句
                case 8:  # 如果条件为8
                    print("ALL RIGHT. YOU'RE ON URANUS. ACCELERATION=33.8 FT/SEC/SEC.\n");  # 打印出在天王星上的加速度
                    a2 = 33.8;  # 将加速度设置为33.8
                    break;  # 跳出switch语句
                case 9:  # 如果条件为9
                    print("THEN YOU'RE ON NEPTUNE. ACCELERATION=39.6 FT/SEC/SEC.\n");  # 打印出在海王星上的加速度
                    a2 = 39.6;  # 将加速度设置为39.6
                    break;  # 跳出switch语句
                case 10:  # 如果条件为10
                    print("FINE. YOU'RE ON THE SUN. ACCELERATION=896 FT/SEC/SEC.\n");  # 打印出在太阳上的加速度
                    a2 = 896;  # 将加速度设置为896
                    break;  # 跳出switch语句
            }
        }
        a = a2 + ((a2 * Math.random()) / 20) - ((a2 * Math.random()) / 20);  # 根据设定的加速度计算出新的加速度
        print("\n");  # 打印一个空行
        # 打印高度
        print("    ALTITUDE         = " + d1 + " FT\n");
        # 打印终端速度
        print("    TERM. VELOCITY   = " + v1 + " FT/SEC +/-5%\n");
        # 打印加速度
        print("    ACCELERATION     = " + a2 + " FT/SEC/SEC +/-5%\n");
        # 设置倒计时器
        print("SET THE TIMER FOR YOUR FREEFALL.\n");
        # 获取用户输入的秒数
        print("HOW MANY SECONDS");
        t = parseFloat(await input());
        # 开始自由落体
        print("HERE WE GO.\n");
        print("\n");
        # 打印时间和下落距离的表头
        print("TIME (SEC)\tDIST TO FALL (FT)\n");
        print("==========\t=================\n");
        # 初始化终端速度和坠毁标志
        terminal = false;
        crash = false;
        # 循环计算下落过程中的距离
        for (i = 0; i <= t; i += t / 8) {
            # 判断是否达到终端速度
            if (i > v / a) {
                terminal = true;
                break;
            }
            # 计算下落距离
            d = d1 - ((a / 2) * Math.pow(i, 2));
            # 判断是否已经坠毁
            if (d <= 0) {
                print(Math.sqrt(2 * d1 / a) + "\tSPLAT\n");
                crash = true;  # 设置crash变量为true，表示发生了坠毁
                break;  # 跳出循环
            }
            print(i + "\t" + d + "\n");  # 打印i和d的值
        }
        if (terminal) {  # 如果达到了终端速度
            print("TERMINAL VELOCITY REACHED AT T PLUS " + v/a + " SECONDS.\n");  # 打印终端速度达到的时间
            for (; i <= t; i += t / 8) {  # 循环计算时间
                d = d1 - ((Math.pow(v, 2) / (2 * a)) + (v * (i - (v / a))));  # 计算距离
                if (d <= 0) {  # 如果距离小于等于0
                    print(((v / a) + ((d1 - (Math.pow(v, 2) / (2 * a))) / v)) + "\tSPLAT\n");  # 打印时间和"SPLAT"
                    crash = true;  # 设置crash变量为true，表示发生了坠毁
                    break;  # 跳出循环
                }
                print(i + "\t" + d + "\n");  # 打印i和d的值
            }
        }
        if (!crash) {  # 如果没有发生坠毁
            print("CHUTE OPEN\n");  # 打印"CHUTE OPEN"
            k = 0;  # 设置k的值为0
            k1 = 0;  // 初始化变量 k1 为 0
            for (j = 0; j <= 42; j++) {  // 循环遍历数组 aa 的元素
                if (aa[j] == 0)  // 如果数组 aa 的第 j 个元素为 0，则跳出循环
                    break;
                k++;  // 变量 k 自增
                if (d < aa[j])  // 如果变量 d 小于数组 aa 的第 j 个元素
                    k1++;  // 变量 k1 自增
            }
            // 在原始代码中，当表已满时跳转到第 540 行（未定义）
            aa[j] = d;  // 将变量 d 赋值给数组 aa 的第 j 个元素
            if (j <= 2) {  // 如果 j 小于等于 2
                print("AMAZING!!! NOT BAD FOR YOUR ");
                if (j == 0)
                    print("1ST ");  // 打印 "1ST "
                else if (j == 1)
                    print("2ND ");  // 打印 "2ND "
                else
                    print("3RD ");  // 打印 "3RD "
                print("SUCCESSFUL JUMP!!!\n");  // 打印 "SUCCESSFUL JUMP!!!\n"
            } else {
                if (k - k1 <= 0.1 * k) {  // 如果（当前跳伞高度 - 最低跳伞高度）小于等于总跳伞次数的10%
                    print("WOW!  THAT'S SOME JUMPING.  OF THE " + k + " SUCCESSFUL JUMPS\n");  // 打印成功跳伞次数
                    print("BEFORE YOURS, ONLY " + (k - k1) + " OPENED THEIR CHUTES LOWER THAN\n");  // 打印在你之前有多少人跳伞时打开降落伞的高度比你低
                    print("YOU DID.\n");  // 打印你的跳伞高度
                } else if (k - k1 <= 0.25 * k) {  // 如果（当前跳伞高度 - 最低跳伞高度）小于等于总跳伞次数的25%
                    print("PRETTY GOOD! " + k + " SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n");  // 打印成功跳伞次数
                    print((k - k1) + " OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n");  // 打印在你之前有多少人跳伞时打开降落伞的高度比你低
                    print("OPENED.\n");  // 打印他们的跳伞高度
                } else if (k - k1 <= 0.5 * k) {  // 如果（当前跳伞高度 - 最低跳伞高度）小于等于总跳伞次数的50%
                    print("NOT BAD.  THERE HAVE BEEN " + k + " SUCCESSFUL JUMPS BEFORE YOURS.\n");  // 打印成功跳伞次数
                    print("YOU WERE BEATEN OUT BY " + (k - k1) + " OF THEM.\n");  // 打印在你之前有多少人的跳伞高度比你低
                } else if (k - k1 <= 0.75 * k) {  // 如果（当前跳伞高度 - 最低跳伞高度）小于等于总跳伞次数的75%
                    print("CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY " + (k - k1) + " IN THE\n");  // 打印在你之前有多少人的跳伞高度比你低
                    print(k + " SUCCESSFUL JUMPS BEFORE YOURS.\n");  // 打印成功跳伞次数
                } else if (k - k1 <= 0.9 * k) {  // 如果（当前跳伞高度 - 最低跳伞高度）小于等于总跳伞次数的90%
                    print("HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE\n");  // 打印警告信息
                    print(k + " SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN " + k1 + "JUMPS\n");  // 打印成功跳伞次数和你的跳伞高度
                    print("BETTER THAN THE WORST.  SHAPE UP!!!\n");  // 打印警告信息
                } else {  // 如果以上条件都不满足
                    print("HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  " + k + " SUCCESSFUL\n");  // 打印警告信息
# 如果条件成立，执行以下代码块
if (k > k1) {
    # 打印特定消息
    print("JUMPS BEFORE YOURS AND YOU CAME IN NUMBER " + (k - k1) + "!  GET WITH IT!\n");
} else {
    # 如果条件不成立，执行以下代码块
    switch (Math.floor(1 + 10 * Math.random())) {
        # 如果随机数为1，执行以下代码块
        case 1:
            print("REQUIESCAT IN PACE.\n");
            break;
        # 如果随机数为2，执行以下代码块
        case 2:
            print("MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.\n");
            break;
        # 如果随机数为3，执行以下代码块
        case 3:
            print("REST IN PEACE.\n");
            break;
        # 如果随机数为4，执行以下代码块
        case 4:
            print("SON-OF-A-GUN.\n");
            break;
        # 如果随机数为5，执行以下代码块
        case 5:
            print("#%&&%!$\n");
            break;
# 如果变量的值为6，打印"A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT."，然后跳出循环
case 6:
    print("A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT.\n");
    break;
# 如果变量的值为7，打印"HMMM. SHOULD HAVE PICKED A SHORTER TIME."，然后跳出循环
case 7:
    print("HMMM. SHOULD HAVE PICKED A SHORTER TIME.\n");
    break;
# 如果变量的值为8，打印"MUTTER. MUTTER. MUTTER."，然后跳出循环
case 8:
    print("MUTTER. MUTTER. MUTTER.\n");
    break;
# 如果变量的值为9，打印"PUSHING UP DAISIES."，然后跳出循环
case 9:
    print("PUSHING UP DAISIES.\n");
    break;
# 如果变量的值为10，打印"EASY COME, EASY GO."，然后跳出循环
case 10:
    print("EASY COME, EASY GO.\n");
    break;
# 打印"I'LL GIVE YOU ANOTHER CHANCE."
print("I'LL GIVE YOU ANOTHER CHANCE.\n");
# 进入无限循环
while (1) {
    # 打印"DO YOU WANT TO PLAY AGAIN"
    print("DO YOU WANT TO PLAY AGAIN");
# 从输入中获取字符串
str = await input();
# 如果输入的字符串是"YES"或者"NO"，则跳出循环
if (str == "YES" || str == "NO")
    break;
# 打印提示信息
print("YES OR NO\n");
# 循环直到输入的字符串是"YES"为止
while (1) {
    str = await input();
    # 如果输入的字符串是"YES"或者"NO"，则跳出循环
    if (str == "YES" || str == "NO")
        break;
    # 打印提示信息
    print("YES OR NO");
}
# 如果输入的字符串是"YES"，则继续循环
if (str == "YES")
    continue;
# 打印提示信息
print("PLEASE");
# 循环直到输入的字符串是"YES"或者"NO"为止
while (1) {
    str = await input();
    # 如果输入的字符串是"YES"或者"NO"，则跳出循环
    if (str == "YES" || str == "NO")
        break;
    # 打印提示信息
    print("YES OR NO");
}
# 如果输入的字符串是"YES"，则继续循环
if (str == "YES")
    continue;
# 否则跳出循环
break;
# 打印提示信息
print("SSSSSSSSSS.\n");
# 打印空行
print("\n");
}

main();
```

这部分代码是一个函数的结束和一个主函数的调用。在这里，"}"表示函数的结束，"main();"是调用名为main的主函数。
```