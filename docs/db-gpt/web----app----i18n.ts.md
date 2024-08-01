# `.\DB-GPT-src\web\app\i18n.ts`

```py
# 导入 'i18next' 模块
import i18n from 'i18next';
# 从 'react-i18next' 模块中导入 'initReactI18next' 函数
import { initReactI18next } from 'react-i18next';

# 定义英文语言包对象 'en'
const en = {
  Knowledge_Space: 'Knowledge',
  space: 'space',
  Vector: 'Vector',
  Owner: 'Owner',
  Count: 'Count',
  File_type_Invalid: 'The file type is invalid',
  Knowledge_Space_Config: 'Space Config',
  Choose_a_Datasource_type: 'Datasource type',
  Segmentation: 'Segmentation',
  No_parameter: `No segementation parameter required.`,
  Knowledge_Space_Name: 'Knowledge Space Name',
  Please_input_the_name: 'Please input the name',
  Please_input_the_owner: 'Please input the owner',
  Please_select_file: 'Please select one file',
  Description: 'Description',
  Storage: 'Storage',
  Please_input_the_description: 'Please input the description',
  Please_select_the_storage: 'Please select the storage',
  Please_select_the_domain_type: 'Please select the domain type',
  Next: 'Next',
  the_name_can_only_contain: 'the name can only contain numbers, letters, Chinese characters, "-" and "_"',
  Text: 'Text',
  'Fill your raw text': 'Fill your raw text',
  URL: 'URL',
  Fetch_the_content_of_a_URL: 'Fetch the content of a URL',
  Document: 'Document',
  Upload_a_document: 'Upload a document, document type can be PDF, CSV, Text, PowerPoint, Word, Markdown',
  Name: 'Name',
  Text_Source: 'Text Source(Optional)',
  Please_input_the_text_source: 'Please input the text source',
  Sync: 'Sync',
  Back: 'Back',
  Finish: 'Finish',
  Web_Page_URL: 'Web Page URL',
  Please_input_the_Web_Page_URL: 'Please input the Web Page URL',
  Select_or_Drop_file: 'Select or Drop file',
  Documents: 'Documents',
  Chat: 'Chat',
  Add_Datasource: 'Add Datasource',
  View_Graph: 'View Graph',
  Arguments: 'Arguments',
  Type: 'Type',
  Size: 'Size',
  Last_Sync: 'Last Sync',
  Status: 'Status',
  Result: 'Result',
  Details: 'Details',
  Delete: 'Delete',
  Operation: 'Operation',
  Submit: 'Submit',
  Chunks: 'Chunks',
  Content: 'Content',
  Meta_Data: 'Meta Data',
  Please_select_a_file: 'Please select a file',
  Please_input_the_text: 'Please input the text',
  Embedding: 'Embedding',
  topk: 'topk',
  the_top_k_vectors: 'the top k vectors based on similarity score',
  recall_score: 'recall_score',
  Set_a_threshold_score: 'Set a threshold score for the retrieval of similar vectors',
  recall_type: 'recall_type',
  model: 'model',
  A_model_used: 'A model used to create vector representations of text or other data',
  Automatic: 'Automatic',
  Process: 'Process',
  Automatic_desc: 'Automatically set segmentation and preprocessing rules.',
  chunk_size: 'chunk_size',
  The_size_of_the_data_chunks: 'The size of the data chunks used in processing',
  chunk_overlap: 'chunk_overlap',
  The_amount_of_overlap: 'The amount of overlap between adjacent data chunks',
  Prompt: 'Prompt',
  scene: 'scene',
  A_contextual_parameter: 'A contextual parameter used to define the setting or environment in which the prompt is being used',
  template: 'template',
  structure_or_format: # 此行无法完全显示，可能需要进一步查看代码补全注释
    # 预定义的用于提示的结构或格式，可确保AI系统生成与所需风格或语调一致的响应。
    'A pre-defined structure or format for the prompt, which can help ensure that the AI system generates responses that are consistent with the desired style or tone.',
    
    # max_token: 'max_token'
    max_token: 'max_token',
    
    # max_iteration: 'max_iteration'
    max_iteration: 'max_iteration',
    
    # concurrency_limit: 'concurrency_limit'
    concurrency_limit: 'concurrency_limit',
    
    # The_maximum_number_of_tokens: 'The maximum number of tokens or words allowed in a prompt'
    The_maximum_number_of_tokens: 'The maximum number of tokens or words allowed in a prompt',
    
    # Theme: 'Theme'
    Theme: 'Theme',
    
    # Port: 'Port'
    Port: 'Port',
    
    # Username: 'Username'
    Username: 'Username',
    
    # Password: 'Password'
    Password: 'Password',
    
    # Remark: 'Remark'
    Remark: 'Remark',
    
    # Edit: 'Edit'
    Edit: 'Edit',
    
    # Database: 'Database'
    Database: 'Database',
    
    # Data_Source: 'Data Center'
    Data_Source: 'Data Center',
    
    # Close_Sidebar: 'Fold'
    Close_Sidebar: 'Fold',
    
    # Show_Sidebar: 'UnFold'
    Show_Sidebar: 'UnFold',
    
    # language: 'Language'
    language: 'Language',
    
    # choose_model: 'Please choose a model'
    choose_model: 'Please choose a model',
    
    # data_center_desc: 'DB-GPT also offers a user-friendly data center management interface for efficient data maintenance.'
    data_center_desc: 'DB-GPT also offers a user-friendly data center management interface for efficient data maintenance.',
    
    # create_database: 'Create Database'
    create_database: 'Create Database',
    
    # create_knowledge: 'Create Knowledge'
    create_knowledge: 'Create Knowledge',
    
    # path: 'Path'
    path: 'Path',
    
    # model_manage: 'Models'
    model_manage: 'Models',
    
    # stop_model_success: 'Stop model success'
    stop_model_success: 'Stop model success',
    
    # create_model: 'Create Model'
    create_model: 'Create Model',
    
    # model_select_tips: 'Please select a model'
    model_select_tips: 'Please select a model',
    
    # language_select_tips: 'Please select a language'
    language_select_tips: 'Please select a language',
    
    # submit: 'Submit'
    submit: 'Submit',
    
    # close: 'Close'
    close: 'Close',
    
    # start_model_success: 'Start model success'
    start_model_success: 'Start model success',
    
    # download_model_tip: 'Please download model first.'
    download_model_tip: 'Please download model first.',
    
    # Plugins: 'Plugins'
    Plugins: 'Plugins',
    
    # try_again: 'Try again'
    try_again: 'Try again',
    
    # no_data: 'No data'
    no_data: 'No data',
    
    # Open_Sidebar: 'Unfold'
    Open_Sidebar: 'Unfold',
    
    # cancel: 'Cancel'
    cancel: 'Cancel',
    
    # Edit_Success: 'Edit Success'
    Edit_Success: 'Edit Success',
    
    # Add: 'Add'
    Add: 'Add',
    
    # Add_Success: 'Add Success'
    Add_Success: 'Add Success',
    
    # Error_Message: 'Something Error'
    Error_Message: 'Something Error',
    
    # Please_Input: 'Please Input'
    Please_Input: 'Please Input',
    
    # Prompt_Info_Scene: 'Scene'
    Prompt_Info_Scene: 'Scene',
    
    # Prompt_Info_Sub_Scene: 'Sub Scene'
    Prompt_Info_Sub_Scene: 'Sub Scene',
    
    # Prompt_Info_Name: 'Name'
    Prompt_Info_Name: 'Name',
    
    # Prompt_Info_Content: 'Content'
    Prompt_Info_Content: 'Content',
    
    # Public: 'Public'
    Public: 'Public',
    
    # Private: 'Private'
    Private: 'Private',
    
    # Lowest: 'Lowest'
    Lowest: 'Lowest',
    
    # Missed: 'Missed'
    Missed: 'Missed',
    
    # Lost: 'Lost'
    Lost: 'Lost',
    
    # Incorrect: 'Incorrect'
    Incorrect: 'Incorrect',
    
    # Verbose: 'Verbose'
    Verbose: 'Verbose',
    
    # Best: 'Best'
    Best: 'Best',
    
    # Rating: 'Rating'
    Rating: 'Rating',
    
    # Q_A_Category: 'Q&A Category'
    Q_A_Category: 'Q&A Category',
    
    # Q_A_Rating: 'Q&A Rating'
    Q_A_Rating: 'Q&A Rating',
    
    # feed_back_desc:
    # '0: No results\n' +
    # '1: Results exist, but they are irrelevant, the question is not understood\n' +
    # '2: Results exist, the question is understood, but it indicates that the question cannot be answered\n' +
    # '3: Results exist, the question is understood, and an answer is given, but the answer is incorrect\n' +
    # '4: Results exist, the question is understood, the answer is correct, but it is verbose and lacks a summary\n' +
    feed_back_desc:
      '0: No results\n' +
      '1: Results exist, but they are irrelevant, the question is not understood\n' +
      '2: Results exist, the question is understood, but it indicates that the question cannot be answered\n' +
      '3: Results exist, the question is understood, and an answer is given, but the answer is incorrect\n' +
      '4: Results exist, the question is understood, the answer is correct, but it is verbose and lacks a summary\n' +
} as const;  // 使用 TypeScript 的 const assertion，确保对象属性不可修改

export type I18nKeys = keyof typeof en;  // 定义 I18nKeys 类型为 en 对象的键集合

export interface Resources {
  translation: Record<I18nKeys, string>;  // 定义 Resources 接口，包含一个翻译记录对象，键为 I18nKeys，值为 string
}

    '0: 无结果\n' +  // 提供了关于不同评分的文本说明，每个评分对应不同的反馈
    '1: 有结果，但是在文不对题，没有理解问题\n' +
    '2: 有结果，理解了问题，但是提示回答不了这个问题\n' +
    '3: 有结果，理解了问题，并做出回答，但是回答的结果错误\n' +
    '4: 有结果，理解了问题，回答结果正确，但是比较啰嗦，缺乏总结\n' +
    '5: 有结果，理解了问题，回答结果正确，推理正确，并给出了总结，言简意赅\n',  // 调查评分说明，每行对应不同的评分

  input_count: '共计输入',  // 输入计数的标签文本
  input_unit: '字',  // 输入单位的标签文本
  Copy: '复制',  // 复制按钮的文本
  Copy_success: '内容复制成功',  // 复制成功的提示文本
  Copy_nothing: '内容复制为空',  // 复制内容为空的提示文本
  Copry_error: '复制失败',  // 复制失败的提示文本
  Click_Select: '点击选择',  // 点击选择的提示文本
  Quick_Start: '快速开始',  // 快速开始的提示文本
  Select_Plugins: '选择插件',  // 选择插件的提示文本
  Search: '搜索',  // 搜索的提示文本
  Reset: '重置',  // 重置按钮的文本
  Update_From_Github: '更新Github插件',  // 更新 Github 插件的提示文本
  Upload: '上传',  // 上传按钮的文本
  Market_Plugins: '插件市场',  // 插件市场的提示文本
  My_Plugins: '我的插件',  // 我的插件的提示文本
  Del_Knowledge_Tips: '你确定删除该知识库吗',  // 删除知识库的确认提示文本
  Del_Document_Tips: '你确定删除该文档吗',  // 删除文档的确认提示文本
  Tips: '提示',  // 提示的文本
  Limit_Upload_File_Count_Tips: '一次只能上传一个文件',  // 上传文件数量限制的提示文本
  To_Plugin_Market: '前往插件市场',  // 前往插件市场的提示文本
  Summary: '总结',  // 总结的文本
  stacked_column_chart: '堆叠柱状图',  // 堆叠柱状图的文本
  column_chart: '柱状图',  // 柱状图的文本
  percent_stacked_column_chart: '百分比堆叠柱状图',  // 百分比堆叠柱状图的文本
  grouped_column_chart: '簇形柱状图',  // 簇形柱状图的文本
  time_column: '簇形柱状图',  // 簇形柱状图（重复定义，可能是错误）
  pie_chart: '饼图',  // 饼图的文本
  line_chart: '折线图',  // 折线图的文本
  area_chart: '面积图',  // 面积图的文本
  stacked_area_chart: '堆叠面积图',  // 堆叠面积图的文本
  scatter_plot: '散点图',  // 散点图的文本
  bubble_chart: '气泡图',  // 气泡图的文本
  stacked_bar_chart: '堆叠条形图',  // 堆叠条形图的文本
  bar_chart: '条形图',  // 条形图的文本
  percent_stacked_bar_chart: '百分比堆叠条形图',  // 百分比堆叠条形图的文本
  grouped_bar_chart: '簇形条形图',  // 簇形条形图的文本
  water_fall_chart: '瀑布图',  // 瀑布图的文本
  table: '表格',  // 表格的文本
  multi_line_chart: '多折线图',  // 多折线图的文本
  multi_measure_column_chart: '多指标柱形图',  // 多指标柱形图的文本
  multi_measure_line_chart: '多指标折线图',  // 多指标折线图的文本
  Advices: '自动推荐',  // 自动推荐的文本
  Retry: '重试',  // 重试按钮的文本
  Load_more: '加载更多',  // 加载更多按钮的文本
  new_chat: '创建会话',  // 创建会话的文本
  choice_agent_tip: '请选择代理',  // 选择代理的提示文本
  no_context_tip: '请输入你的问题',  // 输入问题为空的提示文本
  Terminal: '终端',  // 终端的文本
  awel_flow: 'AWEL 工作流',  // AWEL 工作流的文本
  save: '保存',  // 保存按钮的文本
  add_node: '添加节点',  // 添加节点的文本
  no_node: '没有可编排节点',  // 没有可编排节点的提示文本
  connect_warning: '节点无法连接',  // 节点无法连接的警告文本
  flow_modal_title: '保存工作流',  // 保存工作流模态框标题的文本
  flow_name: '工作流名称',  // 工作流名称的文本
  flow_description: '工作流描述',  // 工作流描述的文本
  flow_name_required: '请输入工作流名称',  // 工作流名称必填的提示文本
  flow_description_required: '请输入工作流描述',  // 工作流描述必填的提示文本
  save_flow_success: '保存工作流成功',  // 保存工作流成功的提示文本
  delete_flow_confirm: '确定删除该工作流吗？',  // 确认删除工作流的提示文本
  related_nodes: '关联节点',  // 关联节点的文本
  language_select_tips: '请选择语言',  // 选择语言的提示文本
  add_resource: '添加资源',  // 添加资源的文本
  team_modal: '工作模式',  // 工作模式的文本
  App: '应用程序',  // 应用程序的文本
  resource: '资源',  // 资源的文本
  resource_name: '资源名',  // 资源名的文本
  resource_type: '资源类型',  // 资源类型的文本
  resource_value: '参数',  // 参数的文本
  resource_dynamic: '动态',  // 动态的文本
  Please_input_the_work_modal: '请选择工作模式',  // 请选择工作模式的提示文本
  available_resources: '可用资源',  // 可用资源的文本
  edit_new_applications: '编辑新的应用',  // 编辑新的应用的文本
  collect: '收藏',  // 收藏按钮的文本
  collected: '已收藏',  // 已收藏的文本
  create: '创建',  // 创建按钮的文本
  Agents: '智能体',  // 智能体的文本
```