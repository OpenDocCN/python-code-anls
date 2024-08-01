# `.\DB-GPT-src\web\client\api\request.ts`

```py
/** Axios 配置信息 */
import { AxiosRequestConfig } from 'axios';
/** HTTP 请求方法 */
import { DELETE, GET, POST, PUT } from '.';
/** 数据库相关响应类型 */
import { DbListResponse, DbSupportTypeResponse, PostDbParams, ChatFeedBackSchema, PostDbRefreshParams } from '@/types/db';
/** 对话相关响应类型 */
import { DialogueListResponse, IChatDialogueSchema, NewDialogueParam, SceneResponse, ChatHistoryResponse, FeedBack, IDB } from '@/types/chat';
/** 模型相关数据类型 */
import { IModelData, StartModelParams, BaseModelParams, SupportModel } from '@/types/model';
/** 编辑器相关请求类型 */
import {
  GetEditorSQLRoundRequest,
  GetEditorySqlParams,
  PostEditorChartRunParams,
  PostEditorChartRunResponse,
  PostEditorSQLRunParams,
  PostSQLEditorSubmitParams,
} from '@/types/editor';
/** Agent 相关请求参数和响应类型 */
import {
  PostAgentHubUpdateParams,
  PostAgentQueryParams,
  PostAgentPluginResponse,
  PostAgentMyPluginResponse,
  GetDBGPTsListResponse,
} from '@/types/agent';
/** 知识管理相关参数和响应类型 */
import {
  AddKnowledgeParams,
  ArgumentsParams,
  ChunkListParams,
  DocumentParams,
  IArguments,
  IChunkList,
  GraphVisResult,
  IChunkStrategyResponse,
  IDocumentResponse,
  ISpace,
  ISyncBatchParameter,
  ISyncBatchResponse,
  SpaceConfig,
} from '@/types/knowledge';
/** 提示相关参数类型 */
import { UpdatePromptParams, IPrompt, PromptParams } from '@/types/prompt';
/** 流程相关数据类型 */
import { IFlow, IFlowNode, IFlowResponse, IFlowUpdateParam } from '@/types/flow';
/** 应用相关数据类型 */
import { IAgent, IApp, IAppData, ITeamModal } from '@/types/app';

/** 应用相关 API：获取场景信息 */
export const postScenes = () => {
  return POST<null, Array<SceneResponse>>('/api/v1/chat/dialogue/scenes');
};

/** 应用相关 API：创建新对话 */
export const newDialogue = (data: NewDialogueParam) => {
  return POST<NewDialogueParam, IChatDialogueSchema>('/api/v1/chat/dialogue/new', data);
};

/** 数据库页面相关 API：获取数据库列表 */
export const getDbList = () => {
  return GET<null, DbListResponse>('/api/v1/chat/db/list');
};

/** 数据库页面相关 API：获取数据库支持类型 */
export const getDbSupportType = () => {
  return GET<null, DbSupportTypeResponse>('/api/v1/chat/db/support/type');
};

/** 数据库页面相关 API：删除数据库 */
export const postDbDelete = (dbName: string) => {
  return POST(`/api/v1/chat/db/delete?db_name=${dbName}`);
};

/** 数据库页面相关 API：编辑数据库信息 */
export const postDbEdit = (data: PostDbParams) => {
  return POST<PostDbParams, null>('/api/v1/chat/db/edit', data);
};

/** 数据库页面相关 API：添加新数据库 */
export const postDbAdd = (data: PostDbParams) => {
  return POST<PostDbParams, null>('/api/v1/chat/db/add', data);
};

/** 数据库页面相关 API：测试数据库连接 */
export const postDbTestConnect = (data: PostDbParams) => {
  return POST<PostDbParams, null>('/api/v1/chat/db/test/connect', data);
};

/** 数据库页面相关 API：刷新数据库 */
export const postDbRefresh = (data: PostDbRefreshParams) => {
  return POST<PostDbRefreshParams, boolean>('/api/v1/chat/db/refresh', data);
};

/** 聊天页面相关 API：获取对话列表 */
export const getDialogueList = () => {
  return GET<null, DialogueListResponse>('/api/v1/chat/dialogue/list');
};

/** 聊天页面相关 API：获取可用模型列表 */
export const getUsableModels = () => {
  return GET<null, Array<string>>('/api/v1/model/types');
};

/** 聊天页面相关 API：获取指定聊天模式的参数列表 */
export const postChatModeParamsList = (chatMode: string) => {
  return POST<null, IDB[]>(`/api/v1/chat/mode/params/list?chat_mode=${chatMode}`);
};

/** 聊天页面相关 API：获取指定聊天模式参数信息 */
export const postChatModeParamsInfoList = (chatMode: string) => {
  return POST<null, Record<string, string>>(`/api/v1/chat/mode/params/info?chat_mode=${chatMode}`);
};
/** knowledge */

// 获取特定知识库的参数列表
export const getArguments = (knowledgeName: string) => {
  return POST<any, IArguments>(`/knowledge/${knowledgeName}/arguments`, {});
};

// 保存特定知识库的参数
export const saveArguments = (knowledgeName: string, data: ArgumentsParams) => {
  return POST<ArgumentsParams, IArguments>(`/knowledge/${knowledgeName}/argument/save`, data);
};

// 获取知识空间列表
export const getSpaceList = () => {
  return POST<any, Array<ISpace>>('/knowledge/space/list', {});
};

// 获取指定知识空间的文档列表
export const getDocumentList = (spaceName: string, data: Record<string, number | Array<number>>) => {
  return POST<Record<string, number | Array<number>>, IDocumentResponse>(`/knowledge/${spaceName}/document/list`, data);
};

// 获取指定知识空间的图形可视化结果
export const getGraphVis = (spaceName: string, data: { limit: number }) => {
  return POST<Record<string, number>, GraphVisResult>(`/knowledge/${spaceName}/graphvis`, data);
};

// 添加指定知识库的文档
export const addDocument = (knowledgeName: string, data: DocumentParams) => {
  return POST<DocumentParams, number>(`/knowledge/${knowledgeName}/document/add`, data);
};

// 添加知识空间
export const addSpace = (data: AddKnowledgeParams) => {
  return POST<AddKnowledgeParams, Array<any>>(`/knowledge/space/add`, data);
};

// 获取文档块策略列表
export const getChunkStrategies = () => {
  return GET<null, Array<IChunkStrategyResponse>>('/knowledge/document/chunkstrategies');
};

/** Menu */

// 删除对话（指定会话 ID）
export const delDialogue = (conv_uid: string) => {
  return POST(`/api/v1/chat/dialogue/delete?con_uid=${conv_uid}`);
};

/** Editor */

// 获取编辑器中 SQL 回合信息
export const getEditorSqlRounds = (id: string) => {
  return GET<null, GetEditorSQLRoundRequest>(`/api/v1/editor/sql/rounds?con_uid=${id}`);
};

// 发送 SQL 运行请求
export const postEditorSqlRun = (data: PostEditorSQLRunParams) => {
  return POST<PostEditorSQLRunParams>(`/api/v1/editor/sql/run`, data);
};

// 发送图表运行请求
export const postEditorChartRun = (data: PostEditorChartRunParams) => {
  return POST<PostEditorChartRunParams, PostEditorChartRunResponse>(`/api/v1/editor/chart/run`, data);
};

// 提交 SQL 编辑器内容
export const postSqlEditorSubmit = (data: PostSQLEditorSubmitParams) => {
  return POST<PostSQLEditorSubmitParams>(`/api/v1/sql/editor/submit`, data);
};

// 获取编辑器中 SQL 数据
export const getEditorSql = (id: string, round: string | number) => {
  return POST<GetEditorySqlParams, string | Array<any>>('/api/v1/editor/sql', { con_uid: id, round });
};

/** Chat */

// 获取会话历史记录
export const getChatHistory = (convId: string) => {
  return GET<null, ChatHistoryResponse>(`/api/v1/chat/dialogue/messages/history?con_uid=${convId}`);
};

// 上传聊天模式参数文件
export const postChatModeParamsFileLoad = ({
  convUid,
  chatMode,
  data,
  config,
  model,
}: {
  convUid: string;
  chatMode: string;
  data: FormData;
  model: string;
  config?: Omit<AxiosRequestConfig, 'headers'>;
}) => {
  return POST<FormData, ChatHistoryResponse>(
    `/api/v1/chat/mode/params/file/load?conv_uid=${convUid}&chat_mode=${chatMode}&model_name=${model}`,
    data,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      ...config,
    },
  );
};
/** 同步单个文档到指定空间 */
export const syncDocument = (spaceName: string, data: Record<string, Array<number>>) => {
  return POST<Record<string, Array<number>>, string | null>(`/knowledge/${spaceName}/document/sync`, data);
};

/** 批量同步文档到指定空间 */
export const syncBatchDocument = (spaceName: string, data: Array<ISyncBatchParameter>) => {
  return POST<Array<ISyncBatchParameter>, ISyncBatchResponse>(`/knowledge/${spaceName}/document/sync_batch`, data);
};

/** 上传文档到指定知识库 */
export const uploadDocument = (knowLedgeName: string, data: FormData) => {
  return POST<FormData, number>(`/knowledge/${knowLedgeName}/document/upload`, data);
};

/** 获取指定空间的分块列表 */
export const getChunkList = (spaceName: string, data: ChunkListParams) => {
  return POST<ChunkListParams, IChunkList>(`/knowledge/${spaceName}/chunk/list`, data);
};

/** 删除指定空间中的文档 */
export const delDocument = (spaceName: string, data: Record<string, number>) => {
  return POST<Record<string, number>>(`/knowledge/${spaceName}/document/delete`, data);
};

/** 删除指定空间 */
export const delSpace = (data: Record<string, string>) => {
  return POST<Record<string, string>, null>(`/knowledge/space/delete`, data);
};

/** 获取模型列表 */
export const getModelList = () => {
  return GET<null, Array<IModelData>>('/api/v1/worker/model/list');
};

/** 停止指定模型 */
export const stopModel = (data: BaseModelParams) => {
  return POST<BaseModelParams, boolean>('/api/v1/worker/model/stop', data);
};

/** 启动指定模型 */
export const startModel = (data: StartModelParams) => {
  return POST<StartModelParams, boolean>('/api/v1/worker/model/start', data);
};

/** 获取支持的模型列表 */
export const getSupportModels = () => {
  return GET<null, Array<SupportModel>>('/api/v1/worker/model/params');
};

/** 发送代理查询请求 */
export const postAgentQuery = (data: PostAgentQueryParams) => {
  return POST<PostAgentQueryParams, PostAgentPluginResponse>('/api/v1/agent/query', data);
};

/** 发送代理中心更新请求 */
export const postAgentHubUpdate = (data?: PostAgentHubUpdateParams) => {
  return POST<PostAgentHubUpdateParams>('/api/v1/agent/hub/update', data ?? { channel: '', url: '', branch: '', authorization: '' });
};

/** 发送代理自定义请求 */
export const postAgentMy = (user?: string) => {
  return POST<undefined, PostAgentMyPluginResponse>('/api/v1/agent/my', undefined, { params: { user } });
};

/** 安装指定代理插件 */
export const postAgentInstall = (pluginName: string, user?: string) => {
  return POST('/api/v1/agent/install', undefined, { params: { plugin_name: pluginName, user }, timeout: 60000 });
};

/** 卸载指定代理插件 */
export const postAgentUninstall = (pluginName: string, user?: string) => {
  return POST('/api/v1/agent/uninstall', undefined, { params: { plugin_name: pluginName, user }, timeout: 60000 });
};

/** 上传个人代理文件 */
export const postAgentUpload = (user = '', data: FormData, config?: Omit<AxiosRequestConfig, 'headers'>) => {
  return POST<FormData>('/api/v1/personal/agent/upload', data, {
    params: { user },
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    ...config,
  });
};

/** 获取调试点列表 */
export const getDbgptsList = () => {
  return GET<undefined, GetDBGPTsListResponse>('/api/v1/dbgpts/list');
};
/** Chat Feedback */

/** 获取聊天反馈选择列表 */
export const getChatFeedBackSelect = () => {
  return GET<null, FeedBack>(`/api/v1/feedback/select`, undefined);
};

/** 获取特定聊天反馈条目 */
export const getChatFeedBackItme = (conv_uid: string, conv_index: number) => {
  return GET<null, Record<string, string>>(`/api/v1/feedback/find?conv_uid=${conv_uid}&conv_index=${conv_index}`, undefined);
};

/** 提交聊天反馈表单 */
export const postChatFeedBackForm = ({ data, config }: { data: ChatFeedBackSchema; config?: Omit<AxiosRequestConfig, 'headers'> }) => {
  return POST<ChatFeedBackSchema, any>(`/api/v1/feedback/commit`, data, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...config,
  });
};

/** Prompt */

/** 获取提示列表 */
export const getPromptList = (data: PromptParams) => {
  return POST<PromptParams, Array<IPrompt>>('/prompt/list', data);
};

/** 更新提示 */
export const updatePrompt = (data: UpdatePromptParams) => {
  return POST<UpdatePromptParams, []>('/prompt/update', data);
};

/** 添加提示 */
export const addPrompt = (data: UpdatePromptParams) => {
  return POST<UpdatePromptParams, []>('/prompt/add', data);
};

/** AWEL Flow */

/** 添加 AWEL 流程 */
export const addFlow = (data: IFlowUpdateParam) => {
  return POST<IFlowUpdateParam, IFlow>('/api/v1/serve/awel/flows', data);
};

/** 获取所有 AWEL 流程 */
export const getFlows = () => {
  return GET<null, IFlowResponse>('/api/v1/serve/awel/flows');
};

/** 根据 ID 获取 AWEL 流程 */
export const getFlowById = (id: string) => {
  return GET<null, IFlow>(`/api/v1/serve/awel/flows/${id}`);
};

/** 根据 ID 更新 AWEL 流程 */
export const updateFlowById = (id: string, data: IFlowUpdateParam) => {
  return PUT<IFlowUpdateParam, IFlow>(`/api/v1/serve/awel/flows/${id}`, data);
};

/** 根据 ID 删除 AWEL 流程 */
export const deleteFlowById = (id: string) => {
  return DELETE<null, null>(`/api/v1/serve/awel/flows/${id}`);
};

/** 获取所有 AWEL 流程节点 */
export const getFlowNodes = () => {
  return GET<null, Array<IFlowNode>>(`/api/v1/serve/awel/nodes`);
};

/** App */

/** 创建应用 */
export const addApp = (data: IApp) => {
  return POST<IApp, []>('/api/v1/app/create', data);
};

/** 获取应用列表 */
export const getAppList = (data: Record<string, string>) => {
  return POST<Record<string, string>, IAppData>('/api/v1/app/list', data);
};

/** 收藏应用 */
export const collectApp = (data: Record<string, string>) => {
  return POST<Record<string, string>, []>('/api/v1/app/collect', data);
};

/** 取消收藏应用 */
export const unCollectApp = (data: Record<string, string>) => {
  return POST<Record<string, string>, []>('/api/v1/app/uncollect', data);
};

/** 删除应用 */
export const delApp = (data: Record<string, string>) => {
  return POST<Record<string, string>, []>('/api/v1/app/remove', data);
};

/** 获取代理人列表 */
export const getAgents = () => {
  return GET<object, IAgent[]>('/api/v1/agents/list', {});
};

/** 获取团队模式列表 */
export const getTeamMode = () => {
  return GET<null, string[]>('/api/v1/team-mode/list');
};

/** 获取资源类型列表 */
export const getResourceType = () => {
  return GET<null, string[]>('/api/v1/resource-type/list');
};

/** 根据资源类型获取资源列表 */
export const getResource = (data: Record<string, string>) => {
  return GET<Record<string, string>, []>(`/api/v1/app/resources/list?type=${data.type}`);
};

/** 更新应用 */
export const updateApp = (data: IApp) => {
  return POST<IApp, []>('/api/v1/app/edit', data);
};
# 定义一个名为 getAppStrategy 的常量，其值为一个箭头函数，返回调用 GET 函数的结果
# GET 函数调用 '/api/v1/llm-strategy/list' 接口，接收 null 类型参数，返回一个空数组
export const getAppStrategy = () => {
  return GET<null, []>(`/api/v1/llm-strategy/list`);
};

# 定义一个名为 getAppStrategyValues 的常量，其值为一个箭头函数，接收一个类型为 string 的参数 'type'
# 返回调用 GET 函数的结果，GET 函数调用 '/api/v1/llm-strategy/value/list?type=${type}' 接口
# 该接口返回一个字符串数组，根据传入的 type 参数获取相关策略值列表
export const getAppStrategyValues = (type: string) => {
  return GET<string, []>(`/api/v1/llm-strategy/value/list?type=${type}`);
};

# 定义一个名为 getSpaceConfig 的常量，其值为一个箭头函数，返回调用 GET 函数的结果
# GET 函数调用 '/knowledge/space/config' 接口，接收一个字符串类型参数，返回 SpaceConfig 类型的数据
# 获取空间配置信息的接口调用
export const getSpaceConfig = () => {
  return GET<string, SpaceConfig>(`/knowledge/space/config`);
};
```