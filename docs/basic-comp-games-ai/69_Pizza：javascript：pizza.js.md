# `d:/src/tocomm/basic-computer-games\69_Pizza\javascript\pizza.js`

```
// 定义一个名为print的函数，用于向页面输出字符串
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

        // 在页面输出问号提示
        print("? ");

        // 设置input元素的类型为文本
        input_element.setAttribute("type", "text");
```
在这个示例中，我们为JavaScript代码添加了注释，解释了每个语句的作用。这样做可以帮助其他程序员更容易地理解代码，并且在以后需要修改或维护代码时也能更快地找到需要的部分。
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

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

var sa = [, "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"];  // 定义包含字母的数组
var ma = [, "1","2","3","4"];  // 定义包含数字的数组
var a = [];  // 定义空数组

// Main program
async function main()
{
    print(tab(33) + "PIZZA\n");  // 打印带有制表符的字符串
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印带有制表符的字符串
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("PIZZA DELIVERY GAME\n");  // 打印游戏标题
    print("\n");  // 打印空行
    print("WHAT IS YOUR FIRST NAME");  // 打印提示信息
    ns = await input();  // 等待用户输入并将输入值存储在变量ns中
    # 打印空行
    print("\n");
    # 打印欢迎信息
    print("HI, " + ns + ". IN THIS GAME YOU ARE TO TAKE ORDERS\n");
    # 打印提示信息
    print("FOR PIZZAS.  THEN YOU ARE TO TELL A DELIVERY BOY\n");
    # 打印提示信息
    print("WHERE TO DELIVER THE ORDERED PIZZAS.\n");
    # 打印空行
    print("\n");
    # 打印空行
    print("\n");
    # 打印城市地图标题
    print("MAP OF THE CITY OF HYATTSVILLE\n");
    # 打印空行
    print("\n");
    # 打印城市地图的横线
    print(" -----1-----2-----3-----4-----\n");
    # 初始化变量 k
    k = 4;
    # 循环打印城市地图的竖线和地图内容
    for (i = 1; i <= 4; i++) {
        # 打印竖线
        print("-\n");
        print("-\n");
        print("-\n");
        print("-\n");
        # 打印地图内容
        print(ma[k]);
        # 计算 sa 数组的索引
        s1 = 16 - 4 * i + 1;
        # 打印地图内容
        print("     " + sa[s1] + "     " + sa[s1 + 1] + "     " + sa[s1 + 2] + "     ");
        print(sa[s1 + 3] + "     " + ma[k] + "\n");
        # 更新 k 变量
        k--;
    }  # 结束一个代码块
    print("-\n");  # 打印分隔线
    print("-\n");  # 打印分隔线
    print("-\n");  # 打印分隔线
    print("-\n");  # 打印分隔线
    print(" -----1-----2-----3-----4-----\n");  # 打印分隔线
    print("\n");  # 打印空行
    print("THE OUTPUT IS A MAP OF THE HOMES WHERE\n");  # 打印提示信息
    print("YOU ARE TO SEND PIZZAS.\n");  # 打印提示信息
    print("\n");  # 打印空行
    print("YOUR JOB IS TO GIVE A TRUCK DRIVER\n");  # 打印提示信息
    print("THE LOCATION OR COORDINATES OF THE\n");  # 打印提示信息
    print("HOME ORDERING THE PIZZA.\n");  # 打印提示信息
    print("\n");  # 打印空行
    while (1) {  # 进入无限循环
        print("DO YOU NEED MORE DIRECTIONS");  # 打印提示信息
        str = await input();  # 等待用户输入
        if (str == "YES" || str == "NO")  # 如果用户输入为"YES"或"NO"
            break;  # 退出循环
        print("'YES' OR 'NO' PLEASE, NOW THEN, ");  # 打印提示信息
    }
    if (str == "YES") {  # 如果输入的字符串为"YES"
        print("\n");  # 打印换行
        print("SOMEBODY WILL ASK FOR A PIZZA TO BE\n");  # 打印提示信息
        print("DELIVERED.  THEN A DELIVERY BOY WILL\n");  # 打印提示信息
        print("ASK YOU FOR THE LOCATION.\n");  # 打印提示信息
        print("     EXAMPLE:\n");  # 打印提示信息
        print("THIS IS J.  PLEASE SEND A PIZZA.\n");  # 打印提示信息
        print("DRIVER TO " + ns + ".  WHERE DOES J LIVE?\n");  # 打印提示信息
        print("YOUR ANSWER WOULD BE 2,3\n");  # 打印提示信息
        print("\n");  # 打印换行
        print("UNDERSTAND");  # 打印提示信息
        str = await input();  # 等待用户输入并将结果赋值给str
        if (str != "YES") {  # 如果输入的字符串不为"YES"
            print("THIS JOB IS DEFINITELY TOO DIFFICULT FOR YOU. THANKS ANYWAY");  # 打印提示信息
            return;  # 返回
        }
        print("GOOD.  YOU ARE NOW READY TO START TAKING ORDERS.\n");  # 打印提示信息
        print("\n");  # 打印换行
        print("GOOD LUCK!!\n");  # 打印提示信息
        print("\n");  # 打印一个空行
    }
    while (1) {  # 进入一个无限循环
        for (i = 1; i <= 5; i++) {  # 循环5次
            s = Math.floor(Math.random() * 16 + 1);  # 生成一个1到16之间的随机整数并赋值给s
            print("\n");  # 打印一个空行
            print("HELLO " + ns + "'S PIZZA.  THIS IS " + sa[s] + ".\n");  # 打印一条消息，包括变量ns和sa[s]的值
            print("  PLEASE SEND A PIZZA.\n");  # 打印一条消息
            while (1) {  # 进入一个内部的无限循环
                print("  DRIVER TO " + ns + ":  WHERE DOES " + sa[s] + " LIVE");  # 打印一条消息，包括变量ns和sa[s]的值
                str = await input();  # 等待用户输入并将输入的值赋给变量str
                a[1] = parseInt(str);  # 将str转换为整数并赋值给数组a的第一个元素
                a[2] = parseInt(str.substr(str.indexOf(",") + 1));  # 将str从逗号后的部分转换为整数并赋值给数组a的第二个元素
                t = a[1] + (a[2] - 1) * 4;  # 根据数组a的值计算出t的值
                if (t != s) {  # 如果t不等于s
                    print("THIS IS " + sa[t] + ". I DID NOT ORDER A PIZZA.\n");  # 打印一条消息，包括sa[t]的值
                    print("I LIVE AT " + a[1] + "," + a[2] + "\n");  # 打印一条消息，包括数组a的值
                } else {  # 否则
                    break;  # 退出内部循环
                }
            }
            # 打印欢迎信息和感谢信息
            print("HELLO " + ns + ".  THIS IS " + sa[s] + ", THANKS FOR THE PIZZA.\n");
        }
        # 打印空行
        print("\n");
        # 打印询问是否要继续送比萨
        print("DO YOU WANT TO DELIVER MORE PIZZAS");
        # 等待用户输入
        str = await input();
        # 如果用户输入不是"YES"，则跳出循环
        if (str != "YES")
            break;
    }
    # 打印空行
    print("\n");
    # 打印道别信息
    print("O.K. " + ns + ", SEE YOU LATER!\n");
    # 打印空行
    print("\n");
}

# 调用主函数
main();
```