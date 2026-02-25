
# `bitsandbytes\scripts\stale.py` 详细设计文档

一个自动化脚本，用于检测并关闭GitHub仓库中长期无活动的过期Issue，通过检查Issue的更新时间、创建时间和评论状态，自动标记为stale或关闭。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[获取GitHub Token并认证]
    B --> C[获取指定仓库的开放Issue]
    C --> D[遍历每个开放Issue]
    D --> E[获取Issue的所有评论并按时间倒序排序]
    F{是否有评论?}
    F -- 否 --> G{Issue创建时间 >= 30天?}
    F -- 是 --> H[获取最后一条评论]
    H --> I{最后评论者是 github-actions[bot]?}
    I -- 是 --> J{Issue更新时间 > 7天?}
    I -- 否 --> K{Issue更新时间 > 23天?}
    J -- 是 --> L{Issue创建时间 >= 30天?}
    J -- 否 --> M[跳过当前Issue]
    L -- 是 --> N{是否有豁免标签?}
    N -- 否 --> O[关闭Issue]
    N -- 是 --> P[跳过当前Issue]
    K -- 是 --> Q{是否有豁免标签?}
    Q -- 否 --> R[创建stale评论]
    Q -- 是 --> S[跳过当前Issue]
    M --> T[检查下一个Issue]
    O --> T
    P --> T
    R --> T
    S --> T
    G -- 是 --> K
    G -- 否 --> T
    T --> U{还有更多Issue?}
    U -- 是 --> D
    U -- 否 --> V[结束]
```

## 类结构

```
Script (主脚本模块)
└── main() (主函数)
```

## 全局变量及字段


### `LABELS_TO_EXEMPT`
    
存储不需要处理的GitHub issue标签列表，用于排除某些标签的issue不被自动关闭或标记为过时

类型：`List[str]`
    


    

## 全局函数及方法



### `main`

该函数是脚本的入口点，用于自动关闭过期的GitHub Issues。它连接到指定的GitHub仓库，遍历所有开放的Issues，根据最后评论时间、Issue创建时间和标签条件，判断是否需要自动关闭Issue或在其上添加"stale"标记评论。

参数：

- 该函数无参数

返回值：`None`，无返回值

#### 流程图

```mermaid
flowchart TD
    A([开始 main]) --> B[使用GITHUB_TOKEN创建Github客户端]
    B --> C[获取指定仓库 TimDettmers/bitsandbytes]
    C --> D[获取所有状态为open的issues]
    D --> E{遍历每个issue}
    E --> F[获取该issue的所有评论并按时间倒序排列]
    F --> G[获取最后一条评论<br/>last_comment]
    H{判断条件1}
    H -->|是| I[关闭issue]
    H -->|否| J{判断条件2}
    J -->|是| K[创建stale评论]
    J -->|否| E
    I --> E
    K --> E
    
    subgraph 条件判断逻辑
    L[条件1: 最后评论来自github-actions[bot]<br/>且更新时间距今>7天<br/>且创建时间>=30天<br/>且没有exempt标签]
    M[条件2: 更新时间距今>23天<br/>且创建时间>=30天<br/>且没有exempt标签]
    end
    
    G --> H
    H -.-> L
    J -.-> M
    
    E --> N{是否遍历完所有issues}
    N -->|是| O([结束])
```

#### 带注释源码

```python
def main():
    """
    主函数：自动检测并关闭过期的GitHub Issues
    
    逻辑说明：
    1. 使用环境变量中的GITHUB_TOKEN进行认证
    2. 连接到指定的仓库（TimDettmers/bitsandbytes）
    3. 遍历所有开放的issues
    4. 根据条件判断是关闭issue还是添加stale评论
    """
    
    # 使用GitHub Token创建GitHub客户端实例
    # Token存储在环境变量GITHUB_TOKEN中
    g = Github(os.environ["GITHUB_TOKEN"])
    
    # 获取目标仓库对象
    repo = g.get_repo("TimDettmers/bitsandbytes")
    
    # 获取仓库中所有状态为open的issues
    open_issues = repo.get_issues(state="open")

    # 遍历每一个开放的issue
    for issue in open_issues:
        # 获取该issue的所有评论，并按创建时间倒序排列（最新的在前）
        # 这里使用列表推导式收集所有评论
        comments = sorted(
            [comment for comment in issue.get_comments()],
            key=lambda i: i.created_at,
            reverse=True
        )
        
        # 获取最后一条评论，如果没有任何评论则为None
        last_comment = comments[0] if len(comments) > 0 else None
        
        # 条件1：自动关闭issue的条件
        if (
            last_comment is not None  # 存在最后评论
            and last_comment.user.login == "github-actions[bot]"  # 最后评论来自机器人
            and (dt.now(timezone.utc) - issue.updated_at).days > 7  # 更新时间距今超过7天
            and (dt.now(timezone.utc) - issue.created_at).days >= 30  # 创建时间距今至少30天
            # 检查issue标签，不包含豁免标签
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # 满足条件，关闭该issue
            issue.edit(state="closed")
            
        # 条件2：标记为stale并添加评论的条件
        elif (
            (dt.now(timezone.utc) - issue.updated_at).days > 23  # 更新时间距今超过23天
            and (dt.now(timezone.utc) - issue.created_at).days >= 30  # 创建时间距今至少30天
            # 检查issue标签，不包含豁免标签
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # 创建评论提醒维护者该issue可能被忽略
            issue.create_comment(
                "This issue has been automatically marked as stale because it has not had "
                "recent activity. If you think this still needs to be addressed "
                "please comment on this thread.\n\n",
            )
```

---

## 全局变量和全局函数信息

### 全局变量

| 名称 | 类型 | 描述 |
|------|------|------|
| `LABELS_TO_EXEMPT` | `List[str]` | 需要豁免的标签列表，包含"feature-request"，带有这些标签的issues不会被自动处理 |
| `dt` | `datetime` | 从datetime模块导入的datetime类，用于处理时间日期 |
| `timezone` | `timezone` | 从datetime模块导入的timezone类，用于处理时区（UTC） |
| `os` | `module` | Python标准库os模块，用于访问环境变量 |
| `Github` | `class` | PyGithub库的Github类，用于与GitHub API交互 |

### 全局函数

| 名称 | 参数 | 描述 |
|------|------|------|
| `main` | 无 | 脚本主函数，自动化处理过期issues的逻辑入口 |

---

## 关键组件信息

| 组件名称 | 描述 |
|----------|------|
| `Github` 客户端 | 使用PyGithub库创建的API客户端，用于与GitHub仓库交互 |
| `repo.get_issues()` | 获取仓库中issues的迭代器，支持过滤状态 |
| `issue.get_comments()` | 获取某个issue的所有评论 |
| `issue.edit(state="closed")` | 将issue的状态修改为关闭 |
| `issue.create_comment()` | 在issue下创建新评论 |

---

## 潜在技术债务与优化空间

1. **硬编码的仓库名称**：仓库名称"TimDettmers/bitsandbytes"直接写在代码中，建议改为配置项或命令行参数
2. **缺少错误处理**：网络请求、API调用等没有任何try-except保护
3. **魔法数字**：时间阈值（7天、23天、30天）散落在代码各处，建议提取为常量或配置
4. **环境变量依赖**：没有检查`GITHUB_TOKEN`是否存在，脚本可能在无token时给出不明确的错误
5. **API调用效率**：遍历所有open issues时会逐个获取评论，可能产生大量API调用（GitHub API有速率限制）
6. **评论排序性能**：对每个issue的评论进行完整排序，可能在评论数量多时影响性能

---

## 其它项目说明

### 设计目标与约束
- **目标**：自动化维护仓库健康度，关闭长期无活动的过期issues
- **约束**：不处理带有"feature-request"标签的issues

### 错误处理与异常设计
- 当前代码缺乏异常处理机制
- 建议增加：Token有效性检查、API速率限制处理、网络异常重试等

### 数据流与状态机
```
Open Issue → 获取评论 → 检查时间条件 → 检查标签条件 
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
               关闭issue            添加stale评论
```

### 外部依赖
- `PyGithub` 库（github包）
- `github-token` 环境变量
- GitHub API（受速率限制约束）

### 接口契约
- 依赖环境变量：`GITHUB_TOKEN` 必须设置
- 目标仓库：`TimDettmers/bitsandbytes`

## 关键组件




### GitHub Issue 陈旧检测与自动处理

该脚本通过 GitHub API 自动检测并处理仓库中的陈旧问题：对于超过30天创建但7天内未更新且最后评论来自机器人的问题直接关闭；对于超过30天创建且23天未更新的问题则自动添加提醒评论。

### 标签豁免机制

定义了 `LABELS_TO_EXEMPT` 列表，用于排除特定类型的问题不被自动处理，避免误关闭功能请求等重要issue。

### 时间条件判断逻辑

使用 `datetime` 模块计算当前时间与issue创建/更新时间的时间差，用于判断issue是否满足陈旧条件（创建≥30天，更新>7天或>23天）。

### GitHub API 交互层

通过 `PyGithub` 库与GitHub API进行交互，获取仓库、issue列表、评论信息，并执行issue状态编辑和评论创建操作。


## 问题及建议




### 已知问题

- **缺乏错误处理机制**：脚本未对GitHub API调用进行任何异常捕获，网络中断、API限流或认证失败都可能导致脚本直接崩溃
- **N+1查询问题**：在循环中对每个issue调用`issue.get_comments()`和`issue.get_labels()`，导致API请求次数过多，性能低下
- **硬编码配置**：仓库名"TimDettmers/bitsandbytes"、时间阈值(7、23、30天)均硬编码在代码中，缺乏灵活配置
- **标签比较不一致**：`label.name.lower()`与未转小写的`LABELS_TO_EXEMPT`列表直接比较，可能导致匹配失败
- **无日志记录**：脚本执行过程中没有任何日志输出，无法追踪哪些issue被关闭或标记，无法进行问题排查
- **注释排序效率低**：先获取所有评论再排序获取最后一条，当评论数量多时浪费内存和时间
- **缺少环境变量校验**：未检查`GITHUB_TOKEN`是否存在，脚本会在缺少环境变量时抛出不友好的KeyError异常
- **无试运行模式**：无法在不实际执行操作的情况下测试脚本逻辑，可能导致生产环境误操作

### 优化建议

- 添加`try-except`块捕获`GithubException`等异常，实现重试机制和优雅的错误处理
- 使用GitHub GraphQL API或预先获取comments和labels信息，减少API调用次数
- 将配置项提取至配置文件或环境变量，支持通过命令行参数自定义
- 将`LABELS_TO_EXEMPT`中的标签转为小写：`[label.lower() for label in LABELS_TO_EXEMPT]`
- 引入`logging`模块记录脚本执行过程、关闭的issue编号、操作时间戳等信息
- 直接获取最新评论而非全量排序，可使用`issue.get_comments().get_page(0)`或限制返回数量
- 脚本启动时检查`GITHUB_TOKEN`是否存在，不存在时给出明确提示
- 添加`--dry-run`参数支持模拟运行，仅输出将要执行的操作而不实际修改issue状态


## 其它




### 设计目标与约束

**设计目标**：自动检测并关闭GitHub仓库中长时间无活动的陈旧issues，同时对即将过期的issues添加stale标记提醒，维护项目issues的活跃度和整洁度。

**约束条件**：
- 仅针对"TimDettmers/bitsandbytes"仓库
- 仅处理开放状态的issues
- 豁免标签列表固定为["feature-request"]
- 时间阈值硬编码（7天、23天、30天）

### 错误处理与异常设计

**异常场景**：
1. **环境变量缺失**：`os.environ["GITHUB_TOKEN"]` 不存在时，`Github()` 初始化会抛出异常
2. **API调用失败**：网络问题或GitHub API限流时，`repo.get_repo()`、`get_issues()`、`issue.get_comments()` 等可能抛出异常
3. **空迭代器处理**：`open_issues` 可能为空，`issue.get_comments()` 可能返回空列表
4. **索引访问**：`comments[0]` 访问空列表时会抛出 IndexError

**当前处理方式**：无try-catch保护，脚本遇错即终止

### 外部依赖与接口契约

**依赖项**：
- `github` 库（PyGithub）：GitHub API Python客户端
- `datetime`：时间计算
- `os`：环境变量读取

**环境变量**：
- `GITHUB_TOKEN`：GitHub个人访问令牌，需repo级别权限

**外部API契约**：
- `Github(token).get_repo(full_name)`：获取仓库对象
- `repo.get_issues(state)`：获取issues迭代器
- `issue.get_comments()`：获取评论迭代器
- `issue.edit(state)`：修改issue状态
- `issue.create_comment(body)`：创建评论

### 安全性考虑

1. **凭据安全**：GitHub Token通过环境变量传入，避免硬编码
2. **Token权限**：应使用最小权限token（仅需repo的issues读写权限）
3. **API限流**：未实现指数退避或限流处理，可能触发GitHub API限流
4. **日志记录**：无操作日志，无法追溯执行历史

### 配置管理

**当前硬编码配置**：
- 仓库名："TimDettmers/bitsandbytes"
- 豁免标签：["feature-request"]
- 陈旧关闭阈值：7天未更新
- stale标记阈值：23天未更新
- 最低issue年龄：30天

**建议改进**：将配置项提取为环境变量或配置文件

### 性能与资源考虑

1. **API调用量**：遍历所有开放issues，每个issue调用`get_comments()`和`get_labels()`，可能导致大量API调用
2. **迭代器内存**：使用迭代器而非列表，避免一次性加载所有issues
3. **排序开销**：`sorted()` 对所有评论排序，时间复杂度O(n log n)
4. **建议优化**：使用分页、添加速率限制、仅处理最近更新的issues

### 可维护性与扩展性

1. **代码复用性低**：main()函数包含所有逻辑，难以复用
2. **可测试性差**：无单元测试，无接口抽象
3. **扩展建议**：
   - 提取配置到独立模块
   - 将issue判断逻辑封装为函数
   - 添加日志记录
   - 支持命令行参数配置仓库和阈值

### 部署与运维

**部署方式**：定时任务（如GitHub Actions、cron）
**执行频率建议**：每日执行一次
**监控指标**：成功执行的issues数量、API调用次数、异常次数
**回滚方案**：手动重新打开被误关闭的issues

    