# `.\DB-GPT-src\web\global.d.ts`

```py
declare namespace JSX {
  // 声明 JSX 的命名空间
  interface IntrinsicElements {
    // 内置元素接口，定义了可用的元素及其属性类型
    summary: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>;
    // 自定义元素 'summary'，继承自 React 的 HTML 属性，并指定其类型为 HTMLElement
    'custom-view': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>;
    // 自定义元素 'custom-view'，同样继承自 React 的 HTML 属性，并指定其类型为 HTMLElement
    references: React.DetailedHTMLProps<
      React.HTMLAttributes<HTMLElement> & {
        title: string;
        references: any;
      },
      HTMLElement
    >;
    // 自定义元素 'references'，继承自 React 的 HTML 属性，并添加了额外的 'title' 和 'references' 属性
    'chart-view': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement>;
    // 自定义元素 'chart-view'，同样继承自 React 的 HTML 属性，并指定其类型为 HTMLElement
  }
}

declare module 'cytoscape-euler';
// 声明模块 'cytoscape-euler'，该模块可能包含特定的类型或功能扩展
```