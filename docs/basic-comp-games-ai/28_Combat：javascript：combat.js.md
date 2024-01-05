# `28_Combat\javascript\combat.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 input 元素，用于用户输入
// 在页面上打印问号提示
// 设置 input 元素的类型为文本输入
// ... (接下来的代码未提供，需要继续添加注释)
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
    # 如果按下的键是回车键（keyCode 为 13）
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
# 结束输入元素的添加
});
}

# 定义一个 tab 函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

}

// Main program
async function main()
{
    print(tab(33) + "COMBAT\n");  # 在指定位置打印字符串"COMBAT"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 在指定位置打印字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("I AM AT WAR WITH YOU.\n");  # 打印"I AM AT WAR WITH YOU."
    print("WE HAVE 72000 SOLDIERS APIECE.\n");  # 打印"WE HAVE 72000 SOLDIERS APIECE."
    do {
        print("\n");  # 打印空行
        print("DISTRIBUTE YOUR FORCES.\n");  # 打印"DISTRIBUTE YOUR FORCES."
        print("\tME\t  YOU\n");  # 打印制表符和"ME"、"YOU"
        print("ARMY\t30000\t");  # 打印"ARMY"和"30000"
        a = parseInt(await input());  # 从用户输入中获取整数并赋值给变量a
        # 打印字符串 "NAVY    20000    "
        print("NAVY\t20000\t");
        # 将用户输入的值转换为整数并赋给变量 b
        b = parseInt(await input());
        # 打印字符串 "A. F.    22000    "
        print("A. F.\t22000\t");
        # 将用户输入的值转换为整数并赋给变量 c
        c = parseInt(await input());
    } while (a + b + c > 72000) ;
    # 将变量 d 赋值为 30000
    d = 30000;
    # 将变量 e 赋值为 20000
    e = 20000;
    # 将变量 f 赋值为 22000
    f = 22000;
    # 打印提示信息 "YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;\n" 和 "AND (3) FOR AIR FORCE.\n"
    print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;\n");
    print("AND (3) FOR AIR FORCE.\n");
    # 将用户输入的值转换为整数并赋给变量 y
    y = parseInt(await input());
    do {
        # 打印提示信息 "HOW MANY MEN\n"
        print("HOW MANY MEN\n");
        # 将用户输入的值转换为整数并赋给变量 x
        x = parseInt(await input());
    } while ((y == 1 && x > a) || (y == 2 && x > b) || (y == 3 && x > c)) ;
    switch (y) {
        case 1:
            if (x < a / 3.0) {
                # 打印字符串 "YOU LOST " + x + " MEN FROM YOUR ARMY.\n"
                print("YOU LOST " + x + " MEN FROM YOUR ARMY.\n");
                # 从变量 a 中减去 x
                a -= x;
                break;  # 结束当前的循环或者 switch 语句的执行
            }
            if (x < 2 * a / 3) {  # 如果 x 小于 a 的 2/3
                print("YOU LOST " + Math.floor(x / 3.0) + " MEN, BUT I LOST " + Math.floor(2 * d / 3.0) + "\n");  # 打印信息，显示损失的人数
                a = Math.floor(a - x / 3.0);  # 更新变量 a 的值
                d = 0;  # 将变量 d 的值设为 0
                break;  # 结束当前的循环或者 switch 语句的执行
            }
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n");  # 打印信息，显示击沉敌方巡逻艇的情况
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.\n");  # 打印信息，显示摧毁敌方空军基地和三个陆军基地的情况
            a = Math.floor(a / 3.0);  # 更新变量 a 的值
            c = Math.floor(c / 3.0);  # 更新变量 c 的值
            e = Math.floor(2 * e / 3.0);  # 更新变量 e 的值
            break;  # 结束当前的循环或者 switch 语句的执行
        case 2:  # 如果 switch 表达式的值等于 2
            if (x < e / 3) {  # 如果 x 小于 e 的 1/3
                print("YOUR ATTACK WAS STOPPED!\n");  # 打印信息，显示攻击被阻止的情况
                b -= x;  # 更新变量 b 的值
                break;  # 结束当前的循环或者 switch 语句的执行
# 如果 x 小于 2/3 的 e，执行以下代码
if (x < 2 * e / 3) {
    # 打印信息，表示摧毁了敌方军队的一部分
    print("YOU DESTROYED " + Math.floor(2 * e / 3.0) + " OF MY ARMY.\n");
    # 更新敌方军队数量为原来的1/3
    e = Math.floor(e / 3.0);
    # 跳出循环
    break;
}
# 如果 x 大于等于 2/3 的 e，执行以下代码
print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n");
print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.\n");
# 更新敌方空军基地数量为原来的1/3
a = Math.floor(a / 3.0);
# 更新敌方陆军基地数量为原来的1/3
c = Math.floor(c / 3.0);
# 更新敌方军队数量为原来的2/3
e = Math.floor(2 * e / 3.0);
# 跳出循环
break;
# 如果情况为3，执行以下代码
case 3:
    # 如果 x 小于 c 的1/3，执行以下代码
    if (x < c / 3.0) {
        # 打印信息，表示敌方攻击被消灭
        print("YOUR ATTACK WAS WIPED OUT.\n");
        # 减少敌方陆军基地数量
        c -= x;
        # 跳出循环
        break;
    }
    # 如果 x 大于等于 c 的1/3 且小于 2/3，执行以下代码
    if (x < 2 * c / 3) {
        # 打印信息，表示进行了一场空战，玩家获胜并完成任务
        print("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.\n");
        # 更新敌方空军基地数量为原来的2/3
        d = Math.floor(2 * d / 3.0);
                e = Math.floor(e / 3.0);  # 将变量 e 除以 3 取整
                f = Math.floor(f / 3.0);  # 将变量 f 除以 3 取整
                break;  # 跳出循环
            }
            print("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED\n");  # 打印信息
            print("TWO NAVY BASES AND BOMBED THREE ARMY BASES.\n");  # 打印信息
            a = Math.floor(a / 4);  # 将变量 a 除以 4 取整
            b = Math.floor(b / 3.0);  # 将变量 b 除以 3 取整
            d = Math.floor(2 * d / 3.0);  # 将变量 d 乘以 2，再除以 3 取整
            break;  # 跳出循环
    }
    print("\n");  # 打印空行
    print("\tYOU\tME\n");  # 打印信息
    print("ARMY\t" + a + "\t" + d + "\n");  # 打印信息
    print("NAVY\t" + b + "\t" + e + "\n");  # 打印信息
    print("A. F.\t" + c + "\t" + f + "\n");  # 打印信息
    print("WHAT IS YOUR NEXT MOVE?\n");  # 打印信息
    print("ARMY=1  NAVY=2  AIR FORCE=3\n");  # 打印信息
    g = parseInt(await input());  # 将用户输入转换为整数并赋值给变量 g
    do {
        print("HOW MANY MEN\n");  // 打印提示信息，询问输入数量
        t = parseInt(await input());  // 从用户输入中获取数量并转换为整数赋值给变量t
    } while (t < 0 || (g == 1 && t > a) || (g == 2 && t > b) || (g == 3 && t > c)) ;  // 当t小于0或者(g为1且t大于a)或者(g为2且t大于b)或者(g为3且t大于c)时，继续循环
    crashed = false;  // 将crashed变量设置为false
    switch (g) {  // 根据变量g的值进行不同的操作
        case 1:  // 当g为1时
            if (t < d / 2) {  // 如果t小于d的一半
                print("I WIPED OUT YOUR ATTACK!\n");  // 打印提示信息
                a -= t;  // a减去t的值
            } else {  // 否则
                print("YOU DESTROYED MY ARMY!\n");  // 打印提示信息
                d = 0;  // 将d设置为0
            }
            break;  // 结束case 1
        case 2:  // 当g为2时
            if (t < e / 2) {  // 如果t小于e的一半
                print("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE\n");  // 打印提示信息
                print("WIPED OUT YOUR UNGUARDED CAPITOL.\n");  // 打印提示信息
                a /= 4.0;  // a除以4.0
                b /= 2.0;  // b除以2.0
                break;  # 结束当前的 case 分支，跳出 switch 语句
            }
            print("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES.\n");  # 打印消息
            print("AND SUNK THREE BATTLESHIPS.\n");  # 打印消息
            f = 2 * f / 3;  # 对变量 f 进行数学运算
            e /= 2;  # 对变量 e 进行数学运算
            break;  # 结束当前的 case 分支，跳出 switch 语句
        case 3:  # 开始一个新的 case 分支
            if (t > f / 2) {  # 如果条件成立
                print("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT\n");  # 打印消息
                print("YOUR COUNTRY IN SHAMBLES.\n");  # 打印消息
                a /= 3.0;  # 对变量 a 进行数学运算
                b /= 3.0;  # 对变量 b 进行数学运算
                c /= 3.0;  # 对变量 c 进行数学运算
                break;  # 结束当前的 case 分支，跳出 switch 语句
            }
            print("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.\n");  # 打印消息
            print("MY COUNTRY FELL APART.\n");  # 打印消息
            crashed = true;  # 修改变量 crashed 的值为 true
            won = 1;  # 修改变量 won 的值为 1
    break;  // 结束当前循环或者 switch 语句的执行

    if (!crashed) {  // 如果没有发生碰撞
        won = 0;  // 胜利者初始化为0
        print("\n");  // 输出换行
        print("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,\n");  // 输出信息
        if (a + b + c > 3.0 / 2.0 * (d + e + f))  // 如果a+b+c大于3/2*(d+e+f)
            won = 1;  // 胜利者为1
        if (a + b + c < 2.0 / 3.0 * (d + e + f))  // 如果a+b+c小于2/3*(d+e+f)
            won = 2;  // 胜利者为2
    }
    if (won == 0) {  // 如果胜利者为0
        print("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR\n");  // 输出信息
        print("RESPECTIVE COUNTRIES AND LIVE IN PEACE.\n");  // 输出信息
    } else if (won == 1) {  // 如果胜利者为1
        print("YOU WON, OH! SHUCKS!!!!\n");  // 输出信息
    } else if (won == 2) {  // 如果胜利者为2
        print("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU\n");  // 输出信息
        print("RIGHT FOR PLAYING THIS STUPID GAME!!!\n");  // 输出信息
    }
}

# 调用主函数
main();
```