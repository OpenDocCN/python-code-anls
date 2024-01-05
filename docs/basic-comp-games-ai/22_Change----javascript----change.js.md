# `22_Change\javascript\change.js`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时，获取输入字符串，移除输入元素，打印输入字符串，换行，解析输入字符串
input_element.addEventListener("keydown", function (event) {
    if (event.keyCode == 13) {
        input_str = input_element.value;
        document.getElementById("output").removeChild(input_element);
        print(input_str);
        print("\n");
        resolve(input_str);
    }
});
# 函数结束
});
# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时，执行循环
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串
}

// Main program
async function main()
{
    print(tab(33) + "CHANGE\n");  // 在第33列打印"CHANGE"
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 在第15列打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE\n");  // 打印"I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE"
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n");  // 打印"THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100."
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    while (1) {  // 进入循环
        print("COST OF ITEM");  // 打印"COST OF ITEM"
        a = parseFloat(await input());  // 从用户输入中获取浮点数并赋值给变量a
        print("AMOUNT OF PAYMENT");  // 打印"AMOUNT OF PAYMENT"
        p = parseFloat(await input()); // 从输入中获取用户支付的金额，并将其转换为浮点数赋值给变量p
        c = p - a; // 计算找零金额，将结果赋值给变量c
        m = c; // 将找零金额赋值给变量m
        if (c == 0) { // 如果找零金额等于0
            print("CORRECT AMOUNT, THANK YOU.\n"); // 打印正确的金额，谢谢您
        } else { // 否则
            print("YOUR CHANGE, $" + c + "\n"); // 打印找零金额
            d = Math.floor(c / 10); // 计算10美元纸币的数量，将结果赋值给变量d
            if (d) // 如果d不为0
                print(d + " TEN DOLLAR BILL(S)\n"); // 打印10美元纸币的数量
            c -= d * 10; // 减去10美元纸币的总额
            e = Math.floor(c / 5); // 计算5美元纸币的数量，将结果赋值给变量e
            if (e) // 如果e不为0
                print(e + " FIVE DOLLAR BILL(S)\n"); // 打印5美元纸币的数量
            c -= e * 5; // 减去5美元纸币的总额
            f = Math.floor(c); // 计算1美元纸币的数量，将结果赋值给变量f
            if (f) // 如果f不为0
                print(f + " ONE DOLLAR BILL(S)\n"); // 打印1美元纸币的数量
            c -= f; // 减去1美元纸币的总额
            c *= 100; // 将剩余的金额转换为以分为单位的整数
            g = Math.floor(c / 50);  # 计算 c 除以 50 的商，表示有多少个 50 美分的硬币
            if (g)  # 如果 g 不为 0，即有 50 美分的硬币
                print(g + " ONE HALF DOLLAR(S)\n");  # 打印出有多少个 50 美分的硬币
            c -= g * 50;  # 更新 c 的值，减去已经计算过的 50 美分的硬币价值
            h = Math.floor(c / 25);  # 计算 c 除以 25 的商，表示有多少个 25 美分的硬币
            if (h)  # 如果 h 不为 0，即有 25 美分的硬币
                print(h + " QUARTER(S)\n");  # 打印出有多少个 25 美分的硬币
            c -= h * 25;  # 更新 c 的值，减去已经计算过的 25 美分的硬币价值
            i = Math.floor(c / 10);  # 计算 c 除以 10 的商，表示有多少个 10 美分的硬币
            if (i)  # 如果 i 不为 0，即有 10 美分的硬币
                print(i + " DIME(S)\n");  # 打印出有多少个 10 美分的硬币
            c -= i * 10;  # 更新 c 的值，减去已经计算过的 10 美分的硬币价值
            j = Math.floor(c / 5);  # 计算 c 除以 5 的商，表示有多少个 5 美分的硬币
            if (j)  # 如果 j 不为 0，即有 5 美分的硬币
                print(j + " NICKEL(S)\n");  # 打印出有多少个 5 美分的硬币
            c -= j * 5;  # 更新 c 的值，减去已经计算过的 5 美分的硬币价值
            k = Math.floor(c + 0.5);  # 对 c 进行四舍五入，表示有多少个 1 美分的硬币
            if (k)  # 如果 k 不为 0，即有 1 美分的硬币
                print(k + " PENNY(S)\n");  # 打印出有多少个 1 美分的硬币
            print("THANK YOU, COME AGAIN.\n");  # 打印感谢信息
            print("\n");  # 打印两个换行符
            print("\n");  # 打印两个换行符
        }  # 结束 if 语句块
    }  # 结束 for 循环块
}  # 结束函数定义块

main();  # 调用 main 函数
```