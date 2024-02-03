# `bitsandbytes\benchmarking\switchback\make_plot_with_jsonl.py`

```py
# 导入必要的库
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

# 获取 'cool' 颜色映射
cmap=plt.get_cmap('cool')

# 主程序入口
if __name__ == '__main__':

    # 创建一个图形对象
    fig = plt.figure(tight_layout=True, figsize=(12,3.5))
    # 创建一个网格布局，1 行 2 列
    gs = gridspec.GridSpec(1, 2)

    # 定义需要考虑的维度
    dims_to_consider = [1024, 1280, 1408, 1664, 2048, 4096]
    # 绘制图1时使用的批处理大小
    batch_size_for_plot1 = 32768
    # 绘制图2时使用的批处理大小列表
    batch_sizes_for_plot2 = [2**14, 2**15, 2**16, 2**17]
    # X 轴刻度显示的维度
    dims_to_xtick = [1024, 2048, 4096]
    # 是否对图1使用对数刻度
    logscale_plot1 = True

    # 在网格布局中添加子图
    ax = fig.add_subplot(gs[0, 0])

    # 读取 JSON 文件中的数据
    rdf = pd.read_json('speed_benchmark/info_a100_py2.jsonl', lines=True)
    # 从数据中筛选出指定批处理大小的数据
    df = rdf[rdf.batch_size == batch_size_for_plot1]

    # 绘制不同操作所占用的时间
    for k, marker, ls, color, name in [
        ('standard_gx+standard_gw+standard_fwd', 's', '-', 'C2', 'Standard fp16 (sum of parts)'),
        ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'C4', 'SwitchBack int8 (sum of parts)'),

        ('standard_fwd', '^', '--', 'C2', 'Matmul XW (standard)'),
        ('standard_gw', '^', '-.', 'C2', 'Matmul GW (standard)'),
        ('standard_gx', '^', ':', 'gray', 'Matmul GX (both)'),

        ('global_fwd', '^', '--', 'C4', 'Int8 Matmul XW (switchback)'),
        ('global_bwd', '^', '-.', 'C4', 'Int8 Matmul GW (switchback)'),

        ('x_quantize_rowwise', 'P', '--', 'C4', 'Quantize rowwise X (switchback)'),
        ('g_quantize_rowwise', 'P', '-.', 'C4', 'Quantize rowwise G (switchback)'),
        ('w_quantize_global', '.', '--', 'C4', 'Quatnize global W (switchback)'),
        ('w_quantize_global_transpose', '.', '-.', 'C4', 'Quantize gloabl and\ntranspose W (switchback)'),
    ]:
        # 初始化空列表用于存储 x 值和 y 值
        xs = []
        ys = []
        # 遍历要考虑的维度
        for embed_dim in dims_to_consider:
            # 从数据框中筛选出输入维度为 embed_dim，输出维度为 embed_dim * 4 的数据
            df_ = df[df.dim_in == embed_dim]
            df_ = df_[df_.dim_out == embed_dim * 4]
            # 将 embed_dim 添加到 x 值列表中
            xs.append(embed_dim)
            y_ = 0
            # 遍历 k 中的每个元素，将对应的值相加
            for k_ in k.split('+'):
                y_ += df_[k_].values[0]
            # 从数据框中筛选出输入维度为 embed_dim * 4，输出维度为 embed_dim 的数据
            df_ = df[df.dim_in == embed_dim * 4]
            df_ = df_[df_.dim_out == embed_dim]
            # 再次遍历 k 中的每个元素，将对应的值相加
            for k_ in k.split('+'):
                y_ += df_[k_].values[0]
            # 将计算得到的 y_ 值乘以 0.5，并添加到 y 值列表中
            ys.append(y_ * 0.5)

        # 在图中绘制 x 值和 y 值的折线图，设置颜色、标签、标记、线型等属性
        ax.plot(xs, ys, color=color, label=name, marker=marker, markersize=5 if marker=='s' else 5, linestyle=ls, linewidth=2 if '+' in k else 1.)

    # 设置 x 轴标签和 y 轴标签的字体大小
    ax.set_xlabel('dim', fontsize=13)
    ax.set_ylabel('time (ms)', fontsize=13)

    # 添加网格线
    ax.grid()

    # 设置 x 轴为对数坐标轴
    ax.set_xscale('log')
    # 如果 logscale_plot1 为真，则设置 y 轴为对数坐标轴
    if logscale_plot1:
        ax.set_yscale('log')

    # 设置 x 轴和 y 轴的刻度标签字体大小
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # 设置 x 轴的刻度值和标签
    ax.set_xticks(dims_to_xtick)
    ax.set_xticklabels(dims_to_xtick)
    ax.set_xticks([], minor=True)

    # 添加图例，并设置位置、字体大小等属性
    leg = ax.legend(loc='upper center', bbox_to_anchor=(-0.64,  1.), ncol=1, fontsize=10)
    leg.get_texts()[0].set_fontweight('bold')
    leg.get_texts()[1].set_fontweight('bold')
    plt.subplots_adjust(left=0.1)
    # 设置图表标题
    ax.set_title('  Linear layer, batch * sequence length = 32k', fontsize=10, loc='left', y=1.05, pad=-20)

    # 在图中添加一个子图
    ax = fig.add_subplot(gs[0, 1])

    # 现在绘制不同批次大小的速度提升百分比
    # 遍历批量大小列表，获取索引和批量大小
    for j, batch_size in enumerate(batch_sizes_for_plot2):
        # 初始化存储所有 x 和 y 值的列表
        all_xs, all_ys = [], []
        # 遍历包含不同参数的元组列表
        for k, marker, ls, color, name in [
            ('standard_gx+standard_gw+standard_fwd', 's', '-', 'C2', 'Standard fp16 (total time)'),
            ('x_quantize_rowwise+g_quantize_rowwise+w_quantize_global+w_quantize_global_transpose+standard_gw+global_fwd+global_bwd', 'o', '-', 'C4', 'SwitchBack int8 (total time)'),
        ]:
            # 初始化存储 x 和 y 值的列表
            xs, ys = [], []
            # 从数据框中筛选出指定批量大小的数据
            df = rdf[rdf.batch_size == batch_size]
            # 遍历要考虑的维度
            for embed_dim in dims_to_consider:
                # 从数据框中筛选出指定输入维度和输出维度的数据
                df_ = df[df.dim_in == embed_dim]
                df_ = df_[df_.dim_out == embed_dim * 4]
                xs.append(embed_dim)
                y_ = 0
                # 遍历参数列表，计算总时间
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                df_ = df[df.dim_in == embed_dim * 4]
                df_ = df_[df_.dim_out == embed_dim]
                for k_ in k.split('+'):
                    y_ += df_[k_].values[0]
                ys.append(y_ * 0.5)
            # 将 x 和 y 值添加到总列表中
            all_xs.append(xs)
            all_ys.append(ys)

        # 计算颜色
        color = cmap(j * 0.25)
        # 计算真实的速度提升百分比
        real_ys = [-((all_ys[1][i] - all_ys[0][i]) / all_ys[0][i]) * 100 for i in range(len(all_ys[0]))]
        # 标记列表
        markers = ['^', 'v', 'P', 'o']
        # 绘制图形
        ax.plot(all_xs[0], real_ys, color=color, label=f'batch * sequence length = {batch_size}', marker=markers[j], markersize=5 if marker=='s' else 5)

    # 添加图例
    ax.legend()
    # 设置 x 轴标签
    ax.set_xlabel('dim', fontsize=13)
    # 设置 x 轴为对数刻度
    ax.set_xscale('log')
    # 添加网格线
    ax.grid()
    # 设置 y 轴标签
    ax.set_ylabel(r'% speedup', fontsize=13)

    # 设置 x 轴标签大小
    ax.tick_params(axis='x', labelsize=11)
    # 设置 y 轴标签大小
    ax.tick_params(axis='y', labelsize=11)

    # 设置 x 轴刻度
    ax.set_xticks(dims_to_xtick)
    ax.set_xticklabels(dims_to_xtick)
    ax.set_xticks([], minor=True)

    # 设置标题
    ax.set_title('  Linear layer summary, varying dimensions', fontsize=10, loc='left', y=1.05, pad=-20)

    # 保存图形为 PDF 文件
    plt.savefig('speed_benchmark/plot_with_info.pdf', bbox_inches='tight')
```