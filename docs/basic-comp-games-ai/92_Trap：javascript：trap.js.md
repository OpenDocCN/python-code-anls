# `d:/src/tocomm/basic-computer-games\92_Trap\javascript\trap.js`

```
# TRAP
# 
# 由 Oscar Toledo G. (nanochess) 从 BASIC 转换为 Javascript
#

# 定义打印函数，将字符串添加到输出元素中
def print(str):
    document.getElementById("output").appendChild(document.createTextNode(str))

# 定义输入函数
def input():
    var input_element
    var input_str

    # 返回一个 Promise 对象
    return new Promise(function (resolve) {
                       # 创建一个输入元素
                       input_element = document.createElement("INPUT")

                       # 打印提示符
                       print("? ")
                       # 设置输入元素类型为文本
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

# 定义一个名为 tab 的函数，接受一个参数 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将字符串末尾添加一个空格
    return str;  // 返回处理后的字符串

}

// Main control section
async function main()
{
    print(tab(34) + "TRAP\n");  // 在控制台打印带有34个空格的字符串和"TRAP"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在控制台打印带有15个空格的字符串和"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 在控制台打印一个空行
    print("\n");  // 在控制台打印一个空行
    print("\n");  // 在控制台打印一个空行
    g = 6;  // 初始化变量g为6
    n = 100;  // 初始化变量n为100
    // Trap
    // Steve Ullman, Aug/01/1972
    print("INSTRUCTIONS");  // 在控制台打印"INSTRUCTIONS"
    str = await input();  // 等待用户输入并将输入值赋给变量str
    if (str.substr(0, 1) == "Y") {  // 如果输入字符串的第一个字符是"Y"
        print("I AM THINKING OF A NUMBER BETWEEN 1 AND " + n + "\n");  // 在控制台打印"I AM THINKING OF A NUMBER BETWEEN 1 AND "和变量n的值
        # 打印游戏规则提示信息
        print("TRY TO GUESS MY NUMBER. ON EACH GUESS,\n");
        print("YOU ARE TO ENTER 2 NUMBERS, TRYING TO TRAP\n");
        print("MY NUMBER BETWEEN THE TWO NUMBERS. I WILL\n");
        print("TELL YOU IF YOU HAVE TRAPPED MY NUMBER, IF MY\n");
        print("NUMBER IS LARGER THAN YOUR TWO NUMBERS, OR IF\n");
        print("MY NUMBER IS SMALLER THAN YOUR TWO NUMBERS.\n");
        print("IF YOU WANT TO GUESS ONE SINGLE NUMBER, TYPE\n");
        print("YOUR GUESS FOR BOTH YOUR TRAP NUMBERS.\n");
        print("YOU GET " + g + " GUESSES TO GET MY NUMBER.\n");
    }
    # 循环进行猜数字游戏
    while (1) {
        # 生成一个随机数作为被猜测的数字
        x = Math.floor(n * Math.random()) + 1;
        # 循环进行猜数字的次数
        for (q = 1; q <= g; q++) {
            # 提示用户进行猜测
            print("\n");
            print("GUESS #" + q + " ");
            # 获取用户输入的猜测值
            str = await input();
            # 将用户输入的字符串转换为整数
            a = parseInt(str);
            # 从用户输入的字符串中获取第二个数字并转换为整数
            b = parseInt(str.substr(str.indexOf(",") + 1));
            # 判断用户猜测的数字是否与随机数相等
            if (a == b && x == a) {
                # 如果猜测正确，打印提示信息
                print("YOU GOT IT!!!\n");
                break;  # 结束循环，跳出当前循环体
            }
            if (a > b) {  # 如果a大于b
                r = a;  # 将a的值赋给r
                a = b;  # 将b的值赋给a
                b = r;  # 将r的值赋给b
            }
            if (a <= x && x <= b) {  # 如果x在a和b之间
                print("YOU HAVE TRAPPED MY NUMBER.\n");  # 打印“你已经困住了我的数字。”
            } else if (x >= a) {  # 否则如果x大于等于a
                print("MY NUMBER IS LARGER THAN YOUR TRAP NUMBERS.\n");  # 打印“我的数字比你的陷阱数字大。”
            } else {  # 否则
                print("MY NUMBER IS SMALLER THAN YOUR TRAP NUMBERS.\n");  # 打印“我的数字比你的陷阱数字小。”
            }
        }
        print("\n");  # 打印空行
        print("TRY AGAIN.\n");  # 打印“再试一次。”
        print("\n");  # 打印空行
    }
}
# 调用名为main的函数，但是在给定的代码中并没有定义这个函数，所以这行代码会导致错误。
```