# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\1647-5c6bd87432337e74.js`

```py
    &${t}-mini ${t}-prev ${t}-item-link,
    &${t}-mini ${t}-next ${t}-item-link
    `]:{
        backgroundColor:"transparent",
        borderColor:"transparent",
        "&::after":{
            height:e.itemSizeSM,
            lineHeight:`${e.itemSizeSM}px`
        }
    },
    
    [`&${t}-mini ${t}-jump-prev, &${t}-mini ${t}-jump-next`]:{
        height:e.itemSizeSM,
        marginInlineEnd:0,
        lineHeight:`${e.itemSizeSM}px`
    },
    
    [`&${t}-mini ${t}-options`]:{
        marginInlineStart:e.paginationMiniOptionsMarginInlineStart,
        "&-size-changer":{
            top:e.miniOptionsSizeChangerTop
        },
        "&-quick-jumper":{
            height:e.itemSizeSM,
            lineHeight:`${e.itemSizeSM}px`,
            input:Object.assign(Object.assign({},(0,Z.x0)(e)),{
                width:e.paginationMiniQuickJumperInputWidth,
                height:e.controlHeightSM
            })
        }
    }
    
    
    Explanation:
    - The first block applies styles to previous and next pagination buttons in a mini pagination component (`&${t}-mini ${t}-prev ${t}-item-link` and `&${t}-mini ${t}-next ${t}-item-link`). It sets `backgroundColor` and `borderColor` to transparent and defines styles for the `::after` pseudo-element.
    - The second block styles jump previous and next buttons (`&${t}-mini ${t}-jump-prev, &${t}-mini ${t}-jump-next`) in the mini pagination component, setting their height, `marginInlineEnd`, and `lineHeight`.
    - The third block styles options in the mini pagination component (`&${t}-mini ${t}-options`), setting `marginInlineStart`. It further styles size changer and quick jumper elements within this component, adjusting their appearance based on various properties (`e.paginationMiniOptionsMarginInlineStart`, `e.miniOptionsSizeChangerTop`, etc.).
```