# `63_Name\javascript\name.js`

```
// 创建一个名为print的函数，用于在页面上输出文本
// 创建一个名为input的函数，用于获取用户输入的文本
// 创建一个Promise对象，用于处理异步操作
// 创建一个input元素，用于接收用户输入
// 在页面上输出提示符“? ”
// 设置input元素的类型为文本输入类型
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
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  # 将空格添加到字符串末尾
    return str;  # 返回修改后的字符串

var str;  # 声明变量str
var b;  # 声明变量b

// Main program  # 主程序开始
async function main()  # 异步函数main开始
{
    print(tab(34) + "NAME\n");  # 打印在第34列开始的字符串"NAME"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  # 打印在第15列开始的字符串"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("\n");  # 打印空行
    print("HELLO.\n");  # 打印"HELLO."
    print("MY NAME IS CREATIVE COMPUTER.\n");  # 打印"MY NAME IS CREATIVE COMPUTER."
    print("WHAT'S YOUR NAME (FIRST AND LAST)");  # 打印"WHAT'S YOUR NAME (FIRST AND LAST)"
    str = await input();  # 从用户输入获取字符串并赋值给变量str
    l = str.length;  # 获取字符串str的长度
    # 打印空行
    print("\n");
    # 打印感谢信息
    print("THANK YOU, ");
    # 倒序打印字符串中的字符
    for (i = l; i >= 1; i--)
        print(str[i - 1]);
    # 打印提示信息
    print(".\n");
    print("OOPS!  I GUESS I GOT IT BACKWARDS.  A SMART\n");
    print("COMPUTER LIKE ME SHOULDN'T MAKE A MISTAKE LIKE THAT!\n");
    # 打印空行
    print("\n");
    # 打印提示信息
    print("BUT I JUST NOTICED YOUR LETTERS ARE OUT OF ORDER.\n");
    # 打印提示信息
    print("LET'S PUT THEM IN ORDER LIKE THIS: ");
    # 创建空数组
    b = [];
    # 将字符串中每个字符的 ASCII 值存入数组 b
    for (i = 1; i <= l; i++)
        b[i - 1] = str.charCodeAt(i - 1);
    # 对数组 b 进行排序
    b.sort();
    # 将排序后的字符打印出来
    for (i = 1; i <= l; i++)
        print(String.fromCharCode(b[i - 1]));
    # 打印空行
    print("\n");
    # 打印提示信息
    print("DON'T YOU LIKE THAT BETTER");
    # 等待用户输入
    ds = await input();
    if (ds == "YES") {  # 如果变量 ds 的值为 "YES"，则执行下面的代码块
        print("\n");  # 打印空行
        print("I KNEW YOU'D AGREE!!\n");  # 打印消息 "I KNEW YOU'D AGREE!!" 并换行
    } else {  # 如果变量 ds 的值不为 "YES"，则执行下面的代码块
        print("\n");  # 打印空行
        print("I'M SORRY YOU DON'T LIKE IT THAT WAY.\n");  # 打印消息 "I'M SORRY YOU DON'T LIKE IT THAT WAY." 并换行
    }
    print("\n");  # 打印空行
    print("I REALLY ENJOYED MEETING YOU " + str + ".\n");  # 打印消息 "I REALLY ENJOYED MEETING YOU " 后跟变量 str 的值，并换行
    print("HAVE A NICE DAY!\n");  # 打印消息 "HAVE A NICE DAY!" 并换行
}

main();  # 调用函数 main()
```