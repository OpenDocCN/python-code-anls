# `.\DB-GPT-src\web\components\agent\market-plugins.tsx`

```py
import { apiInterceptors, postAgentHubUpdate, postAgentInstall, postAgentQuery, postAgentUninstall } from '@/client/api';
import { IAgentPlugin, PostAgentQueryParams } from '@/types/agent';
import { useRequest } from 'ahooks';
import { Button, Card, Form, Input, Spin, Tag, Tooltip, message } from 'antd';
import { useCallback, useMemo, useState } from 'react';
import MyEmpty from '../common/MyEmpty';
import { ClearOutlined, DownloadOutlined, GithubOutlined, LoadingOutlined, SearchOutlined, SyncOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

function MarketPlugins() {
  const { t } = useTranslation();

  // 状态钩子：上传状态
  const [uploading, setUploading] = useState(false);
  // 状态钩子：是否出错
  const [isError, setIsError] = useState(false);
  // 状态钩子：当前操作的插件索引
  const [actionIndex, setActionIndex] = useState<number | undefined>();

  // 表单实例
  const [form] = Form.useForm<PostAgentQueryParams['filter']>();

  // 分页信息
  const pagination = useMemo<{ pageNo: number; pageSize: number }>(
    () => ({
      pageNo: 1,
      pageSize: 20,
    }),
    [],
  );

  // 请求钩子：获取代理列表数据
  const {
    data: agents = [], // 代理列表数据，默认为空数组
    loading, // 加载状态
    refresh, // 刷新函数
  } = useRequest(async () => {
    // 构建查询参数
    const queryParams: PostAgentQueryParams = {
      page_index: pagination.pageNo, // 当前页码
      page_size: pagination.pageSize, // 每页数量
      filter: form.getFieldsValue(), // 表单过滤条件
    };
    // 发起带拦截器的 API 请求
    const [err, res] = await apiInterceptors(postAgentQuery(queryParams));
    setIsError(!!err); // 根据错误情况设置错误状态
    return res?.datas ?? []; // 返回代理数据列表
  });

  // 更新来自 GitHub 的插件
  const updateFromGithub = async () => {
    try {
      setUploading(true); // 设置上传状态为 true
      const [err] = await apiInterceptors(postAgentHubUpdate()); // 发起带拦截器的更新请求
      if (err) return; // 如果出现错误，直接返回
      message.success('success'); // 成功消息提示
      refresh(); // 刷新数据
    } finally {
      setUploading(false); // 无论成功与否，重置上传状态为 false
    }
  };

  // 插件操作函数
  const pluginAction = useCallback(
    async (name: string, index: number, isInstall: boolean) => {
      if (actionIndex) return; // 如果正在进行其他操作，则直接返回
      setActionIndex(index); // 设置当前操作的索引
      const [err] = await apiInterceptors((isInstall ? postAgentInstall : postAgentUninstall)(name)); // 根据操作类型调用安装或卸载 API
      if (!err) {
        message.success('success'); // 成功消息提示
        refresh(); // 刷新数据
      }
      setActionIndex(undefined); // 操作完成后，重置操作索引
    },
    [actionIndex, refresh],
  );

  // 渲染操作按钮
  const renderAction = useCallback(
    (agent: IAgentPlugin, index: number) => {
      if (index === actionIndex) {
        return <LoadingOutlined />; // 如果当前索引与操作索引匹配，显示加载图标
      }
      return agent.installed ? ( // 如果插件已安装
        <Tooltip title="Uninstall"> {/* 提示卸载 */}
          <div
            className="w-full h-full"
            onClick={() => {
              pluginAction(agent.name, index, false); // 点击执行卸载操作
            }}
          >
            <ClearOutlined /> {/* 卸载图标 */}
          </div>
        </Tooltip>
      ) : ( // 如果插件未安装
        <Tooltip title="Install"> {/* 提示安装 */}
          <div
            className="w-full h-full"
            onClick={() => {
              pluginAction(agent.name, index, true); // 点击执行安装操作
            }}
          >
            <DownloadOutlined /> {/* 安装图标 */}
          </div>
        </Tooltip>
      );
    },
    [actionIndex, pluginAction],
  );

  return (
    # 根据 loading 状态显示加载中动画
    <Spin spinning={loading}>
      # 创建一个内联表单，设置布局为内联，提交时触发 refresh 函数
      <Form form={form} layout="inline" onFinish={refresh} className="mb-2">
        # 表单项，用于输入名称
        <Form.Item className="!mb-2" name="name" label={'Name'}>
          # 输入框，可清除内容，宽度为 48
          <Input allowClear className="w-48" />
        </Form.Item>
        <Form.Item>
          # 搜索按钮，点击时提交表单
          <Button className="mr-2" type="primary" htmlType="submit" icon={<SearchOutlined />}>
            {t('Search')}
          </Button>
          # 从 Github 更新按钮，点击时触发 updateFromGithub 函数
          <Button loading={uploading} type="primary" icon={<SyncOutlined />} onClick={updateFromGithub}>
            {t('Update_From_Github')}
          </Button>
        </Form.Item>
      </Form>
      # 如果代理列表为空且不在加载状态，则显示自定义空组件
      {!agents.length && !loading && <MyEmpty error={isError} refresh={refresh} />}
      # 代理卡片列表，根据 agents 数组渲染
      <div className="flex flex-wrap gap-2 md:gap-4">
        {agents.map((agent, index) => (
          # 代理卡片，设置宽度和 key，包含操作按钮和 Github 链接
          <Card
            className="w-full md:w-1/2 lg:w-1/3 xl:w-1/4"
            key={agent.id}
            actions={[
              renderAction(agent, index),  # 渲染操作按钮
              <Tooltip key="github" title="Github">  # Github 提示框
                <div
                  className="w-full h-full"
                  onClick={() => {
                    window.open(agent.storage_url, '_blank');  # 点击打开 Github 链接
                  }}
                >
                  <GithubOutlined />  # Github 图标
                </div>
              </Tooltip>,
            ]}
          >
            <Tooltip title={agent.name}>  # 代理名称提示
              <h2 className="mb-2 text-base font-semibold line-clamp-1">{agent.name}</h2>
            </Tooltip>
            {agent.author && <Tag>{agent.author}</Tag>}  # 作者标签
            {agent.version && <Tag>v{agent.version}</Tag>}  # 版本标签
            {agent.type && <Tag>Type {agent.type}</Tag>}  # 类型标签
            {agent.storage_channel && <Tag>{agent.storage_channel}</Tag>}  # 存储渠道标签
            <Tooltip title={agent.description}>  # 代理描述提示
              <p className="mt-2 line-clamp-2 text-gray-400 text-sm">{agent.description}</p>
            </Tooltip>
          </Card>
        ))}
      </div>
    </Spin>
  );
}

export default MarketPlugins;
```