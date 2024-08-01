# `.\DB-GPT-src\web\types\app.ts`

```py
// app
/**
 * 应用数据结构接口定义
 */
export type IApp = {
  app_code: string;
  /**
   * 应用名
   */
  app_name: string;
  /**
   * 应用描述信息/简介
   */
  app_describe: string;
  /**
   * 语言/prompt关联
   */
  language: 'en' | 'zh';
  /**
   * 组织模式（AutoPlan/LayOut）
   */
  team_mode: string;
  /**
   * 组织上下文/ None
   */
  team_context: string;
  /**
   * 应用节点信息
   */
  details?: IDetail[];
  /**
   * 是否已收藏
   */
  is_collected: string;
};

/**
 * 应用数据
```