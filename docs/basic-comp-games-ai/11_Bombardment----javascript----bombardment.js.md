# `11_Bombardment\javascript\bombardment.js`

```
// 定义一个名为print的函数，用于向页面输出文本
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

        // 在页面输出提示符
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```

在这个示例中，我们为JavaScript代码添加了注释，解释了每个语句的作用。这样做可以帮助其他程序员更容易地理解和使用这段代码。
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 为输入元素添加按键按下事件监听器
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入字符串设置为输入元素的值
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
# 结束事件监听器的添加
});
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

// 主程序
async function main()
{
    print(tab(33) + "BOMBARDMENT\n");  // 在指定位置打印字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在指定位置打印字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("YOU ARE ON A BATTLEFIELD WITH 4 PLATOONS AND YOU\n");  // 打印字符串
    print("HAVE 25 OUTPOSTS AVAILABLE WHERE THEY MAY BE PLACED.\n");  // 打印字符串
    print("YOU CAN ONLY PLACE ONE PLATOON AT ANY ONE OUTPOST.\n");  // 打印字符串
    print("THE COMPUTER DOES THE SAME WITH ITS FOUR PLATOONS.\n");  // 打印字符串
    print("\n");  // 打印空行
    print("THE OBJECT OF THE GAME IS TO FIRE MISSILES AT THE\n");  // 打印字符串
    print("OUTPOSTS OF THE COMPUTER.  IT WILL DO THE SAME TO YOU.\n");  // 打印字符串
    print("THE ONE WHO DESTROYS ALL FOUR OF THE ENEMY'S PLATOONS\n");  // 打印字符串
    # 打印字符串 "FIRST IS THE WINNER.\n"
    print("FIRST IS THE WINNER.\n");
    # 打印空行
    print("\n");
    # 打印字符串 "GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!\n"
    print("GOOD LUCK... AND TELL US WHERE YOU WANT THE BODIES SENT!\n");
    # 打印空行
    print("\n");
    # 打印字符串 "TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.\n"，注释说明这个字符串是用于打印在电传打字机上
    // "TEAR OFF" because it supposed this to be printed on a teletype
    print("TEAR OFF MATRIX AND USE IT TO CHECK OFF THE NUMBERS.\n");
    # 循环5次，每次打印一个空行
    for (r = 1; r <= 5; r++)
        print("\n");
    # 创建一个空数组 ma
    ma = [];
    # 循环100次，将数组 ma 的每个元素初始化为0
    for (r = 1; r <= 100; r++)
        ma[r] = 0;
    # 初始化变量 p, q, z 为0
    p = 0;
    q = 0;
    z = 0;
    # 循环5次，每次打印一行数字序列
    for (r = 1; r <= 5; r++) {
        i = (r - 1) * 5 + 1;
        print(i + "\t" + (i + 1) + "\t" + (i + 2) + "\t" + (i + 3) + "\t" + (i + 4) + "\n");
    }
    # 循环10次，每次打印一个空行
    for (r = 1; r <= 10; r++)
        print("\n");
    c = Math.floor(Math.random() * 25) + 1;  // 生成一个1到25之间的随机整数并赋值给变量c
    do {
        d = Math.floor(Math.random() * 25) + 1;  // 生成一个1到25之间的随机整数并赋值给变量d
        e = Math.floor(Math.random() * 25) + 1;  // 生成一个1到25之间的随机整数并赋值给变量e
        f = Math.floor(Math.random() * 25) + 1;  // 生成一个1到25之间的随机整数并赋值给变量f
    } while (c == d || c == e || c == f || d == e || d == f || e == f) ;  // 循环直到c、d、e、f四个变量的值都不相等
    print("WHAT ARE YOUR FOUR POSITIONS");  // 打印提示信息
    str = await input();  // 等待用户输入并将输入值赋给变量str
    g = parseInt(str);  // 将str转换为整数并赋值给变量g
    str = str.substr(str.indexOf(",") + 1);  // 从逗号后的位置开始截取str并重新赋值给str
    h = parseInt(str);  // 将str转换为整数并赋值给变量h
    str = str.substr(str.indexOf(",") + 1);  // 从逗号后的位置开始截取str并重新赋值给str
    k = parseInt(str);  // 将str转换为整数并赋值给变量k
    str = str.substr(str.indexOf(",") + 1);  // 从逗号后的位置开始截取str并重新赋值给str
    l = parseInt(str);  // 将str转换为整数并赋值给变量l
    print("\n");  // 打印换行符
    // Another "bug" your outpost can be in the same position as a computer outpost
    // Let us suppose both live in a different matrix.
    while (1) {  // 无限循环
        // The original game didn't limited the input to 1-25
        // 使用 do-while 循环，要求用户输入一个数字，直到输入的数字在 0 到 25 之间
        do {
            print("WHERE DO YOU WISH TO FIRE YOUR MISSLE");
            y = parseInt(await input());
        } while (y < 0 || y > 25) ;
        // 如果用户输入的数字等于 c、d、e 或 f，则执行以下代码
        if (y == c || y == d || y == e || y == f) {

            // 原始游戏存在一个 bug，允许多次攻击同一个据点，以下代码解决了这个问题
            if (y == c)
                c = 0;
            if (y == d)
                d = 0;
            if (y == e)
                e = 0;
            if (y == f)
                f = 0;
            // 每次攻击成功后，q 值加一
            q++;
            // 如果 q 值为 1，则打印 "ONE DOWN. THREE TO GO.\n"
            if (q == 1) {
                print("ONE DOWN. THREE TO GO.\n");
            } 
            // 如果 q 值为 2，则打印其他信息
            else if (q == 2) {
                # 打印提示信息
                print("TWO DOWN. TWO TO GO.\n");
            } else if (q == 3) {
                # 打印提示信息
                print("THREE DOWN. ONE TO GO.\n");
            } else {
                # 打印提示信息
                print("YOU GOT ME, I'M GOING FAST. BUT I'LL GET YOU WHEN\n");
                print("MY TRANSISTO&S RECUP%RA*E!\n");
                # 跳出循环
                break;
            }
        } else {
            # 打印提示信息
            print("HA, HA YOU MISSED. MY TURN NOW:\n");
        }
        # 打印空行
        print("\n");
        print("\n");
        # 循环生成随机数
        do {
            m = Math.floor(Math.random() * 25 + 1);
            p++;
            n = p - 1;
            # 遍历数组
            for (t = 1; t <= n; t++) {
                # 如果随机数等于数组中的某个值，则跳出循环
                if (m == ma[t])
                    break;
        } while (t <= n) ;  # 使用 do-while 循环，当 t 小于等于 n 时执行循环体
        x = m;  # 将变量 m 的值赋给变量 x
        ma[p] = m;  # 将变量 m 的值赋给数组 ma 的第 p 个元素
        if (x == g || x == h || x == l || x == k) {  # 如果 x 的值等于 g、h、l 或 k 中的任意一个
            z++;  # z 值加一
            if (z < 4)  # 如果 z 小于 4
                print("I GOT YOU. IT WON'T BE LONG NOW. POST " + x + " WAS HIT.\n");  # 打印消息
            if (z == 1) {  # 如果 z 等于 1
                print("YOU HAVE ONLY THREE OUTPOSTS LEFT.\n");  # 打印消息
            } else if (z == 2) {  # 如果 z 等于 2
                print("YOU HAVE ONLY TWO OUTPOSTS LEFT.\n");  # 打印消息
            } else if (z == 3) {  # 如果 z 等于 3
                print("YOU HAVE ONLY ONE OUTPOST LEFT.\n");  # 打印消息
            } else {  # 否则
                print("YOU'RE DEAD. YOUR LAST OUTPOST WAS AT " + x + ". HA, HA, HA.\n");  # 打印消息
                print("BETTER LUCK NEXT TIME.\n");  # 打印消息
            }
        } else {  # 否则
            print("I MISSED YOU, YOU DIRTY RAT. I PICKED " + m + ". YOUR TURN:\n");  # 打印消息
        }
        # 打印两个空行
        print("\n");
        print("\n");
    }
}

main();
```