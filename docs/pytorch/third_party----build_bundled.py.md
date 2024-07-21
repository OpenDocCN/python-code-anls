# `.\pytorch\third_party\build_bundled.py`

```py
#!/usr/bin/env python3
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统功能模块


mydir = os.path.dirname(__file__)  # 获取当前脚本文件所在目录的路径
licenses = {'LICENSE', 'LICENSE.txt', 'LICENSE.rst', 'COPYING.BSD'}  # 预定义的许可证文件名集合


def collect_license(current):
    """收集指定目录及其子目录中的许可证信息"""
    collected = {}  # 初始化收集到的许可证信息字典
    for root, dirs, files in os.walk(current):
        license = list(licenses & set(files))  # 在当前文件夹中查找匹配的许可证文件
        if license:
            name = root.split('/')[-1]  # 获取当前目录的名称
            license_file = os.path.join(root, license[0])  # 获取许可证文件的完整路径
            try:
                ident = identify_license(license_file)  # 尝试识别许可证类型
            except ValueError:
                raise ValueError('could not identify license file '
                                 f'for {root}') from None
            val = {
                'Name': name,
                'Files': [root],
                'License': ident,
                'License_file': [license_file],
            }
            if name in collected:
                # 只有在许可证类型不同的情况下才添加
                if collected[name]['License'] == ident:
                    collected[name]['Files'].append(root)
                    collected[name]['License_file'].append(license_file)
                else:
                    collected[name + f' ({root})'] = val
            else:
                collected[name] = val
    return collected  # 返回收集到的许可证信息字典


def create_bundled(d, outstream, include_files=False):
    """将信息写入到开放的输出流中"""
    collected = collect_license(d)  # 收集指定目录中的许可证信息
    sorted_keys = sorted(collected.keys())  # 对收集到的许可证信息按键进行排序
    outstream.write('The PyTorch repository and source distributions bundle '
                    'several libraries that are \n')
    outstream.write('compatibly licensed.  We list these here.')
    files_to_include = []
    for k in sorted_keys:
        c = collected[k]
        files = ',\n     '.join(c['Files'])  # 将文件列表转换为字符串形式
        license_file = ',\n     '.join(c['License_file'])  # 将许可证文件列表转换为字符串形式
        outstream.write('\n\n')
        outstream.write(f"Name: {c['Name']}\n")  # 输出项目名称
        outstream.write(f"License: {c['License']}\n")  # 输出许可证类型
        outstream.write(f"Files: {files}\n")  # 输出文件列表
        outstream.write('  For details, see')
        if include_files:
            outstream.write(' the files concatenated below: ')
            files_to_include += c['License_file']
        else:
            outstream.write(': ')
        outstream.write(license_file)  # 输出许可证文件列表
    for fname in files_to_include:
        outstream.write('\n\n')
        outstream.write(fname)
        outstream.write('\n' + '-' * len(fname) + '\n')
        with open(fname, 'r') as fid:
            outstream.write(fid.read())  # 将许可证文件内容写入输出流


def identify_license(f, exception=''):
    """
    读取许可证文件并尝试识别其许可证类型
    这个方法非常粗略，可能并不具备法律约束力，它是专门针对本仓库的。
    """
    def squeeze(t):
        """去除换行符和空格，并规范化引号"""
        t = t.replace('\n', '').replace(' ', '')
        t = t.replace('``', '"').replace("''", '"')
        return t
    # 使用 'with' 语句打开文件，并将文件对象赋给 fid
    with open(f) as fid:
        # 读取文件内容到 txt 变量中
        txt = fid.read()
        # 如果异常未被禁用且文件内容包含 'exception' 字符串
        if not exception and 'exception' in txt:
            # 调用 identify_license 函数识别出带有 exception 的许可证
            license = identify_license(f, 'exception')
            # 返回带有 exception 的许可证名称
            return license + ' with exception'
        # 对文件内容进行紧缩处理
        txt = squeeze(txt)
        
        # 根据文件内容判断可能的许可证类型
        if 'ApacheLicense' in txt:
            # 返回 Apache-2.0 许可证
            return 'Apache-2.0'
        elif 'MITLicense' in txt:
            # 返回 MIT 许可证
            return 'MIT'
        elif 'BSD-3-ClauseLicense' in txt:
            # 返回 BSD-3-Clause 许可证
            return 'BSD-3-Clause'
        elif 'BSD3-ClauseLicense' in txt:
            # 返回 BSD-3-Clause 许可证
            return 'BSD-3-Clause'
        elif 'BoostSoftwareLicense-Version1.0' in txt:
            # 返回 BSL-1.0 许可证
            return 'BSL-1.0'
        elif 'gettimeofday' in txt:
            # 对特定引用的许可证进行返回说明
            return 'Apache-2.0'
        elif 'libhungarian' in txt:
            # 对特定引用的许可证进行返回说明
            return 'Permissive (free to use)'
        elif 'PDCurses' in txt:
            # 对特定引用的许可证进行返回说明
            return 'Public Domain for core'
        elif 'Copyright1999UniversityofNorthCarolina' in txt:
            # 对特定引用的许可证进行返回说明
            return 'Apache-2.0'
        elif 'sigslot' in txt:
            # 对特定引用的许可证进行返回说明
            return 'Public Domain'
        elif squeeze("Clarified Artistic License") in txt:
            # 返回 Clarified Artistic License 许可证
            return 'Clarified Artistic License'
        elif all([squeeze(m) in txt.lower() for m in bsd3_txt]):
            # 如果所有 bsd3_txt 中的条目均在 txt 的小写版本中出现，则返回 BSD-3-Clause 许可证
            return 'BSD-3-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd3_v1_txt]):
            # 如果所有 bsd3_v1_txt 中的条目均在 txt 的小写版本中出现，则返回 BSD-3-Clause 许可证
            return 'BSD-3-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd2_txt]):
            # 如果所有 bsd2_txt 中的条目均在 txt 的小写版本中出现，则返回 BSD-2-Clause 许可证
            return 'BSD-2-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd3_src_txt]):
            # 如果所有 bsd3_src_txt 中的条目均在 txt 的小写版本中出现，则返回 BSD-Source-Code 许可证
            return 'BSD-Source-Code'
        elif any([squeeze(m) in txt.lower() for m in mit_txt]):
            # 如果 mit_txt 中的任一条目在 txt 的小写版本中出现，则返回 MIT 许可证
            return 'MIT'
        else:
            # 如果无法识别许可证，则引发 ValueError 异常
            raise ValueError('unknown license')
# 定义 MIT 许可证文本的列表，每一行代表一段文本内容
mit_txt = ['permission is hereby granted, free of charge, to any person ',
           'obtaining a copy of this software and associated documentation ',
           'files (the "software"), to deal in the software without ',
           'restriction, including without limitation the rights to use, copy, ',
           'modify, merge, publish, distribute, sublicense, and/or sell copies ',
           'of the software, and to permit persons to whom the software is ',
           'furnished to do so, subject to the following conditions:',

           'the above copyright notice and this permission notice shall be ',
           'included in all copies or substantial portions of the software.',

           'the software is provided "as is", without warranty of any kind, ',
           'express or implied, including but not limited to the warranties of ',
           'merchantability, fitness for a particular purpose and ',
           'noninfringement. in no event shall the authors or copyright holders ',
           'be liable for any claim, damages or other liability, whether in an ',
           'action of contract, tort or otherwise, arising from, out of or in ',
           'connection with the software or the use or other dealings in the ',
           'software.',
           ]

# 定义 BSD-3 许可证文本的列表，每一行代表一段文本内容
bsd3_txt = ['redistribution and use in source and binary forms, with or without '
            'modification, are permitted provided that the following conditions '
            'are met:',

            'redistributions of source code',

            'redistributions in binary form',

            'neither the name',

            'this software is provided by the copyright holders and '
            'contributors "as is" and any express or implied warranties, '
            'including, but not limited to, the implied warranties of '
            'merchantability and fitness for a particular purpose are disclaimed.',
            ]

# BSD-2 是 BSD-3 去除了最后一个条款 "neither the name..." 的版本
bsd2_txt = bsd3_txt[:3] + bsd3_txt[4:]

# 创建一个新的 BSD-3 变体文本，删除最后一个条款中的 "and contributors" 子句
v1 = bsd3_txt[4].replace('and contributors', '')
bsd3_v1_txt = bsd3_txt[:3] + [v1]

# 创建一个源码版本的 BSD-3 变体文本，删除了 "redistributions in binary form" 条款
bsd3_src_txt = bsd3_txt[:2] + bsd3_txt[4:]

# 检查当前脚本是否在主程序中执行
if __name__ == '__main__':
    # 计算相对路径以获取第三方库的目录
    third_party = os.path.relpath(mydir)
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(
        description="Generate bundled licenses file",
    )
    # 添加一个命令行参数，用于指定输出的捆绑许可证文件位置，默认为环境变量中的路径或默认路径
    parser.add_argument(
        "--out-file",
        type=str,
        default=os.environ.get(
            "PYTORCH_THIRD_PARTY_BUNDLED_LICENSE_FILE",
            str(os.path.join(third_party, 'LICENSES_BUNDLED.txt'))
        ),
        help="location to output new bundled licenses file",
    )
    # 添加一个命令行参数 "--include-files"
    parser.add_argument(
        "--include-files",
        action="store_true",  # 设置为 True 表示在命令行参数中包含此选项时，其值为 True
        default=False,        # 默认情况下，该选项的值为 False
        help="include actual license terms to the output",  # 帮助文本，解释此选项的作用
    )
    # 解析命令行参数并将结果存储在 args 变量中
    args = parser.parse_args()
    # 从命令行参数中获取输出文件名
    fname = args.out_file
    # 打印将要写入 bundled licenses 的输出文件名
    print(f"+ Writing bundled licenses to {args.out_file}")
    # 打开输出文件 fname，以写入模式，并使用 fid 作为文件对象
    with open(fname, 'w') as fid:
        # 调用 create_bundled 函数，将 third_party 和 args.include_files 作为参数传递进去
        create_bundled(third_party, fid, args.include_files)
```