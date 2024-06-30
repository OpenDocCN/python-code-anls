# `D:\src\scipysrc\scikit-learn\doc\js\scripts\api-search.js`

```
/**
 * This script is for initializing the search table on the API index page. See
 * DataTables documentation for more information: https://datatables.net/
 */

document.addEventListener("DOMContentLoaded", function () {
  // 创建一个新的 DataTable 实例，用于指定的表格元素
  new DataTable("table.apisearch-table", {
    order: [], // 保持原始顺序
    lengthMenu: [10, 25, 50, 100, { label: "All", value: -1 }], // 指定每页显示条目数的选项，包括显示所有条目
    pageLength: -1, // 默认显示所有条目
  });
});
```