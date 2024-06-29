# `D:\src\scipysrc\matplotlib\lib\matplotlib\_color_data.py`

```py
# 基础颜色字典，包含常见颜色的 RGB 值
BASE_COLORS = {
    'b': (0, 0, 1),        # 蓝色
    'g': (0, 0.5, 0),      # 绿色
    'r': (1, 0, 0),        # 红色
    'c': (0, 0.75, 0.75),  # 青色
    'm': (0.75, 0, 0.75),  # 品红色
    'y': (0.75, 0.75, 0),  # 黄色
    'k': (0, 0, 0),        # 黑色
    'w': (1, 1, 1),        # 白色
}


# 这些颜色来自于 Tableau 的颜色表
TABLEAU_COLORS = {
    'tab:blue': '#1f77b4',
    'tab:orange': '#ff7f0e',
    'tab:green': '#2ca02c',
    'tab:red': '#d62728',
    'tab:purple': '#9467bd',
    'tab:brown': '#8c564b',
    'tab:pink': '#e377c2',
    'tab:gray': '#7f7f7f',
    'tab:olive': '#bcbd22',
    'tab:cyan': '#17becf',
}


# 这个颜色名称到十六进制值的映射来自于 Randall Munroe 进行的调查
# 详细信息请参见：https://blog.xkcd.com/2010/05/03/color-survey-results/
# 调查结果托管在 https://xkcd.com/color/rgb/
# 也可以作为文本文件在 https://xkcd.com/color/rgb.txt 中找到
#
# 许可证：https://creativecommons.org/publicdomain/zero/1.0/
XKCD_COLORS = {
    'cloudy blue': '#acc2d9',
    'dark pastel green': '#56ae57',
    'dust': '#b2996e',
    'electric lime': '#a8ff04',
    'fresh green': '#69d84f',
    'light eggplant': '#894585',
    'nasty green': '#70b23f',
    'really light blue': '#d4ffff',
    'tea': '#65ab7c',
    'warm purple': '#952e8f',
    'yellowish tan': '#fcfc81',
    'cement': '#a5a391',
    'dark grass green': '#388004',
    'dusty teal': '#4c9085',
    'grey teal': '#5e9b8a',
    'macaroni and cheese': '#efb435',
    'pinkish tan': '#d99b82',
    'spruce': '#0a5f38',
    'strong blue': '#0c06f7',
    'toxic green': '#61de2a',
    'windows blue': '#3778bf',
    'blue blue': '#2242c7',
    'blue with a hint of purple': '#533cc6',
    'booger': '#9bb53c',
    'bright sea green': '#05ffa6',
    'dark green blue': '#1f6357',
    'deep turquoise': '#017374',
    'green teal': '#0cb577',
    'strong pink': '#ff0789',
    'bland': '#afa88b',
    'deep aqua': '#08787f',
    'lavender pink': '#dd85d7',
    'light moss green': '#a6c875',
    'light seafoam green': '#a7ffb5',
    'olive yellow': '#c2b709',
    'pig pink': '#e78ea5',
    'deep lilac': '#966ebd',
    'desert': '#ccad60',
    'dusty lavender': '#ac86a8',
    'purpley grey': '#947e94',
    'purply': '#983fb2',
    'candy pink': '#ff63e9',
    'light pastel green': '#b2fba5',
    'boring green': '#63b365',
    'kiwi green': '#8ee53f',
    'light grey green': '#b7e1a1',
    'orange pink': '#ff6f52',
    'tea green': '#bdf8a3',
    'very light brown': '#d3b683',
    'egg shell': '#fffcc4',
    'eggplant purple': '#430541',
    'powder pink': '#ffb2d0',
    'reddish grey': '#997570',
    'baby shit brown': '#ad900d',
    'liliac': '#c48efd',
    'stormy blue': '#507b9c',
    'ugly brown': '#7d7103',
    'custard': '#fffd78',
    'darkish pink': '#da467d',
    'deep brown': '#410200',
    'greenish beige': '#c9d179',
    'manilla': '#fffa86',
    'off blue': '#5684ae',
    'battleship grey': '#6b7c85',
    'browny green': '#6f6c0a',
}
    'bruise': '#7e4071',
    'kelley green': '#009337',
    'sickly yellow': '#d0e429',
    'sunny yellow': '#fff917',
    'azul': '#1d5dec',
    'darkgreen': '#054907',
    'green/yellow': '#b5ce08',
    'lichen': '#8fb67b',
    'light light green': '#c8ffb0',
    'pale gold': '#fdde6c',
    'sun yellow': '#ffdf22',
    'tan green': '#a9be70',
    'burple': '#6832e3',
    'butterscotch': '#fdb147',
    'toupe': '#c7ac7d',
    'dark cream': '#fff39a',
    'indian red': '#850e04',
    'light lavendar': '#efc0fe',
    'poison green': '#40fd14',
    'baby puke green': '#b6c406',
    'bright yellow green': '#9dff00',
    'charcoal grey': '#3c4142',
    'squash': '#f2ab15',
    'cinnamon': '#ac4f06',
    'light pea green': '#c4fe82',
    'radioactive green': '#2cfa1f',
    'raw sienna': '#9a6200',
    'baby purple': '#ca9bf7',
    'cocoa': '#875f42',
    'light royal blue': '#3a2efe',
    'orangeish': '#fd8d49',
    'rust brown': '#8b3103',
    'sand brown': '#cba560',
    'swamp': '#698339',
    'tealish green': '#0cdc73',
    'burnt siena': '#b75203',
    'camo': '#7f8f4e',
    'dusk blue': '#26538d',
    'fern': '#63a950',
    'old rose': '#c87f89',
    'pale light green': '#b1fc99',
    'peachy pink': '#ff9a8a',
    'rosy pink': '#f6688e',
    'light bluish green': '#76fda8',
    'light bright green': '#53fe5c',
    'light neon green': '#4efd54',
    'light seafoam': '#a0febf',
    'tiffany blue': '#7bf2da',
    'washed out green': '#bcf5a6',
    'browny orange': '#ca6b02',
    'nice blue': '#107ab0',
    'sapphire': '#2138ab',
    'greyish teal': '#719f91',
    'orangey yellow': '#fdb915',
    'parchment': '#fefcaf',
    'straw': '#fcf679',
    'very dark brown': '#1d0200',
    'terracota': '#cb6843',
    'ugly blue': '#31668a',
    'clear blue': '#247afd',
    'creme': '#ffffb6',
    'foam green': '#90fda9',
    'grey/green': '#86a17d',
    'light gold': '#fddc5c',
    'seafoam blue': '#78d1b6',
    'topaz': '#13bbaf',
    'violet pink': '#fb5ffc',
    'wintergreen': '#20f986',
    'yellow tan': '#ffe36e',
    'dark fuchsia': '#9d0759',
    'indigo blue': '#3a18b1',
    'light yellowish green': '#c2ff89',
    'pale magenta': '#d767ad',
    'rich purple': '#720058',
    'sunflower yellow': '#ffda03',
    'green/blue': '#01c08d',
    'leather': '#ac7434',
    'racing green': '#014600',
    'vivid purple': '#9900fa',
    'dark royal blue': '#02066f',
    'hazel': '#8e7618',
    'muted pink': '#d1768f',
    'booger green': '#96b403',
    'canary': '#fdff63',
    'cool grey': '#95a3a6',
    'dark taupe': '#7f684e',
    'darkish purple': '#751973',
    'true green': '#089404',
    'coral pink': '#ff6163',
    'dark sage': '#598556',
    'dark slate blue': '#214761',
    'flat blue': '#3c73a8',
    'mushroom': '#ba9e88',
    'rich blue': '#021bf9',
    'dirty purple': '#734a65',
    'greenblue': '#23c48b',
    'icky green': '#8fae22',
    'light khaki': '#e6f2a2',
    'warm blue': '#4b57db',
    'dark hot pink': '#d90166',


注释：
    # 颜色名称 'deep sea blue' 对应的十六进制颜色值
    'deep sea blue': '#015482',
    # 颜色名称 'carmine' 对应的十六进制颜色值
    'carmine': '#9d0216',
    # 颜色名称 'dark yellow green' 对应的十六进制颜色值
    'dark yellow green': '#728f02',
    # 颜色名称 'pale peach' 对应的十六进制颜色值
    'pale peach': '#ffe5ad',
    # 颜色名称 'plum purple' 对应的十六进制颜色值
    'plum purple': '#4e0550',
    # 颜色名称 'golden rod' 对应的十六进制颜色值
    'golden rod': '#f9bc08',
    # 颜色名称 'neon red' 对应的十六进制颜色值
    'neon red': '#ff073a',
    # 颜色名称 'old pink' 对应的十六进制颜色值
    'old pink': '#c77986',
    # 颜色名称 'very pale blue' 对应的十六进制颜色值
    'very pale blue': '#d6fffe',
    # 颜色名称 'blood orange' 对应的十六进制颜色值
    'blood orange': '#fe4b03',
    # 颜色名称 'grapefruit' 对应的十六进制颜色值
    'grapefruit': '#fd5956',
    # 颜色名称 'sand yellow' 对应的十六进制颜色值
    'sand yellow': '#fce166',
    # 颜色名称 'clay brown' 对应的十六进制颜色值
    'clay brown': '#b2713d',
    # 颜色名称 'dark blue grey' 对应的十六进制颜色值
    'dark blue grey': '#1f3b4d',
    # 颜色名称 'flat green' 对应的十六进制颜色值
    'flat green': '#699d4c',
    # 颜色名称 'light green blue' 对应的十六进制颜色值
    'light green blue': '#56fca2',
    # 颜色名称 'warm pink' 对应的十六进制颜色值
    'warm pink': '#fb5581',
    # 颜色名称 'dodger blue' 对应的十六进制颜色值
    'dodger blue': '#3e82fc',
    # 颜色名称 'gross green' 对应的十六进制颜色值
    'gross green': '#a0bf16',
    # 颜色名称 'ice' 对应的十六进制颜色值
    'ice': '#d6fffa',
    # 颜色名称 'metallic blue' 对应的十六进制颜色值
    'metallic blue': '#4f738e',
    # 颜色名称 'pale salmon' 对应的十六进制颜色值
    'pale salmon': '#ffb19a',
    # 颜色名称 'sap green' 对应的十六进制颜色值
    'sap green': '#5c8b15',
    # 颜色名称 'algae' 对应的十六进制颜色值
    'algae': '#54ac68',
    # 颜色名称 'bluey grey' 对应的十六进制颜色值
    'bluey grey': '#89a0b0',
    # 颜色名称 'greeny grey' 对应的十六进制颜色值
    'greeny grey': '#7ea07a',
    # 颜色名称 'highlighter green' 对应的十六进制颜色值
    'highlighter green': '#1bfc06',
    # 颜色名称 'light light blue' 对应的十六进制颜色值
    'light light blue': '#cafffb',
    # 颜色名称 'light mint' 对应的十六进制颜色值
    'light mint': '#b6ffbb',
    # 颜色名称 'raw umber' 对应的十六进制颜色值
    'raw umber': '#a75e09',
    # 颜色名称 'vivid blue' 对应的十六进制颜色值
    'vivid blue': '#152eff',
    # 颜色名称 'deep lavender' 对应的十六进制颜色值
    'deep lavender': '#8d5eb7',
    # 颜色名称 'dull teal' 对应的十六进制颜色值
    'dull teal': '#5f9e8f',
    # 颜色名称 'light greenish blue' 对应的十六进制颜色值
    'light greenish blue': '#63f7b4',
    # 颜色名称 'mud green' 对应的十六进制颜色值
    'mud green': '#606602',
    # 颜色名称 'pinky' 对应的十六进制颜色值
    'pinky': '#fc86aa',
    # 颜色名称 'red wine' 对应的十六进制颜色值
    'red wine': '#8c0034',
    # 颜色名称 'shit green' 对应的十六进制颜色值
    'shit green': '#758000',
    # 颜色名称 'tan brown' 对应的十六进制颜色值
    'tan brown': '#ab7e4c',
    # 颜色名称 'darkblue' 对应的十六进制颜色值
    'darkblue': '#030764',
    # 颜色名称 'rosa' 对应的十六进制颜色值
    'rosa': '#fe86a4',
    # 颜色名称 'lipstick' 对应的十六进制颜色值
    'lipstick': '#d5174e',
    # 颜色名称 'pale mauve' 对应的十六进制颜色值
    'pale mauve': '#fed0fc',
    # 颜色名称 'claret' 对应的十六进制颜色值
    'claret': '#680018',
    # 颜色名称 'dandelion' 对应的十六进制颜色值
    'dandelion': '#fedf08',
    # 颜色名称 'orangered' 对应的十六进制颜色值
    'orangered': '#fe420f',
    # 颜色名称 'poop green' 对应的十六进制颜色值
    'poop green': '#6f7c00',
    # 颜色名称 'ruby' 对应的十六进制颜色值
    'ruby': '#ca0147',
    # 颜色名称 'dark' 对应的十六进制颜色值
    'dark': '#1b2431',
    # 颜色名称 'greenish turquoise' 对应的十六进制颜色值
    'greenish turquoise': '#00fbb0',
    # 颜色名称 'pastel red' 对应的十六进制颜色值
    'pastel red': '#db5856',
    # 颜色名称 'piss yellow' 对应的十六进制颜色值
    'piss yellow': '#ddd618',
    # 颜色名称 'bright cyan' 对应的十六进制颜色值
    'bright cyan': '#41fdfe',
    # 颜色名称 'dark coral' 对应的十六进制颜色值
    'dark coral': '#cf524e',
    # 颜色名称 'algae green' 对应的十六进制颜色值
    'algae green': '#21c36f',
    # 颜色名称 'darkish red' 对应的十六进制颜色值
    'darkish red': '#a90308',
    # 颜色名称 'reddy brown' 对应的十六进制颜色值
    'reddy brown': '#6e1005',
    # 颜色名称 'blush pink' 对应的十六进制颜色值
    'blush pink': '#fe828c',
    # 颜色名称 'camouflage green' 对应的十六进制颜色值
    'camouflage green': '#4b6113',
    # 颜色名称 'lawn green' 对应的十六进制颜色值
    'lawn green': '#4da409',
    # 颜色名称 'putty' 对应的十六进制颜色值
    'putty
    'purple/pink': '#d725de',
    'brown yellow': '#b29705',
    'purple brown': '#673a3f',
    'wisteria': '#a87dc2',
    'banana yellow': '#fafe4b',
    'lipstick red': '#c0022f',
    'water blue': '#0e87cc',
    'brown grey': '#8d8468',
    'vibrant purple': '#ad03de',
    'baby green': '#8cff9e',
    'barf green': '#94ac02',
    'eggshell blue': '#c4fff7',
    'sandy yellow': '#fdee73',
    'cool green': '#33b864',
    'pale': '#fff9d0',
    'blue/grey': '#758da3',
    'hot magenta': '#f504c9',
    'greyblue': '#77a1b5',
    'purpley': '#8756e4',
    'baby shit green': '#889717',
    'brownish pink': '#c27e79',
    'dark aquamarine': '#017371',
    'diarrhea': '#9f8303',
    'light mustard': '#f7d560',
    'pale sky blue': '#bdf6fe',
    'turtle green': '#75b84f',
    'bright olive': '#9cbb04',
    'dark grey blue': '#29465b',
    'greeny brown': '#696006',
    'lemon green': '#adf802',
    'light periwinkle': '#c1c6fc',
    'seaweed green': '#35ad6b',
    'sunshine yellow': '#fffd37',
    'ugly purple': '#a442a0',
    'medium pink': '#f36196',
    'puke brown': '#947706',
    'very light pink': '#fff4f2',
    'viridian': '#1e9167',
    'bile': '#b5c306',
    'faded yellow': '#feff7f',
    'very pale green': '#cffdbc',
    'vibrant green': '#0add08',
    'bright lime': '#87fd05',
    'spearmint': '#1ef876',
    'light aquamarine': '#7bfdc7',
    'light sage': '#bcecac',
    'yellowgreen': '#bbf90f',
    'baby poo': '#ab9004',
    'dark seafoam': '#1fb57a',
    'deep teal': '#00555a',
    'heather': '#a484ac',
    'rust orange': '#c45508',
    'dirty blue': '#3f829d',
    'fern green': '#548d44',
    'bright lilac': '#c95efb',
    'weird green': '#3ae57f',
    'peacock blue': '#016795',
    'avocado green': '#87a922',
    'faded orange': '#f0944d',
    'grape purple': '#5d1451',
    'hot green': '#25ff29',
    'lime yellow': '#d0fe1d',
    'mango': '#ffa62b',
    'shamrock': '#01b44c',
    'bubblegum': '#ff6cb5',
    'purplish brown': '#6b4247',
    'vomit yellow': '#c7c10c',
    'pale cyan': '#b7fffa',
    'key lime': '#aeff6e',
    'tomato red': '#ec2d01',
    'lightgreen': '#76ff7b',
    'merlot': '#730039',
    'night blue': '#040348',
    'purpleish pink': '#df4ec8',
    'apple': '#6ecb3c',
    'baby poop green': '#8f9805',
    'green apple': '#5edc1f',
    'heliotrope': '#d94ff5',
    'yellow/green': '#c8fd3d',
    'almost black': '#070d0d',
    'cool blue': '#4984b8',
    'leafy green': '#51b73b',
    'mustard brown': '#ac7e04',
    'dusk': '#4e5481',
    'dull brown': '#876e4b',
    'frog green': '#58bc08',
    'vivid green': '#2fef10',
    'bright light green': '#2dfe54',
    'fluro green': '#0aff02',
    'kiwi': '#9cef43',
    'seaweed': '#18d17b',
    'navy green': '#35530a',
    'ultramarine blue': '#1805db',
    'iris': '#6258c4',
    'pastel orange': '#ff964f',
    'yellowish orange': '#ffab0f',
    'perrywinkle': '#8f8ce7',
    'tealish': '#24bca8',
    'dark plum': '#3f012c',
    'pear': '#cbf85f',


注释：


'purple/pink': '#d725de',
'brown yellow': '#b29705',
'purple brown': '#673a3f',
'wisteria': '#a87dc2',
'banana yellow': '#fafe4b',
'lipstick red': '#c0022f',
'water blue': '#0e87cc',
'brown grey': '#8d8468',
'vibrant purple': '#ad03de',
'baby green': '#8cff9e',
'barf green': '#94ac02',
'eggshell blue': '#c4fff7',
'sandy yellow': '#fdee73',
'cool green': '#33b864',
'pale': '#fff9d0',
'blue/grey': '#758da3',
'hot magenta': '#f504c9',
'greyblue': '#77a1b5',
'purpley': '#8756e4',
'baby shit green': '#889717',
'brownish pink': '#c27e79',
'dark aquamarine': '#017371',
'diarrhea': '#9f8303',
'light mustard': '#f7d560',
'pale sky blue': '#bdf6fe',
'turtle green': '#75b84f',
'bright olive': '#9cbb04',
'dark grey blue': '#29465b',
'greeny brown': '#696006',
'lemon green': '#adf802',
'light periwinkle': '#c1c6fc',
'seaweed green': '#35ad6b',
'sunshine yellow': '#fffd37',
'ugly purple': '#a442a0',
'medium pink': '#f36196',
'puke brown': '#947706',
'very light pink': '#fff4f2',
'viridian': '#1e9167',
'bile': '#b5c306',
'faded yellow': '#feff7f',
'very pale green': '#cffdbc',
'vibrant green': '#0add08',
'bright lime': '#87fd05',
'spearmint': '#1ef876',
'light aquamarine': '#7bfdc7',
'light sage': '#bcecac',
'yellowgreen': '#bbf90f',
'baby poo': '#ab9004',
'dark seafoam': '#1fb57a',
'deep teal': '#00555a',
'heather': '#a484ac',
'rust orange': '#c45508',
'dirty blue': '#3f829d',
'fern green': '#548d44',
'bright lilac': '#c95efb',
'weird green': '#3ae57f',
'peacock blue': '#016795',
'avocado green': '#87a922',
'faded orange': '#f0944d',
'grape purple': '#5d1451',
'hot green': '#25ff29',
'lime yellow': '#d0fe1d',
'mango': '#ffa62b',
'shamrock': '#01b44c',
'bubblegum': '#ff6cb5',
'purplish brown': '#6b4247',
'vomit yellow': '#c7c10c',
'pale cyan': '#b7fffa',
'key lime': '#aeff6e',
'tomato red': '#ec2d01',
'lightgreen': '#76ff7b',
'merlot': '#730039',
'night blue': '#040348',
'purpleish pink': '#df4ec8',
'apple': '#6ecb3c',
'baby poop green': '#8f9805',
'green apple': '#5edc1f',
'heliotrope': '#d94ff5',
'yellow/green': '#c8fd3d',
'almost black': '#070d0d',
'cool blue': '#4984b8',
'leafy green': '#51b73b',
'mustard brown': '#ac7e04',
'dusk': '#4e5481',
'dull brown': '#876e4b',
'frog green': '#58bc08',
'vivid green': '#2fef10',
'bright light green': '#2dfe54',
'fluro green': '#0aff02',
'
    # 定义一个字典，将颜色名映射到十六进制颜色代码
    colors = {
        'pinkish orange': '#ff724c',
        'midnight purple': '#280137',
        'light urple': '#b36ff6',
        'dark mint': '#48c072',
        'greenish tan': '#bccb7a',
        'light burgundy': '#a8415b',
        'turquoise blue': '#06b1c4',
        'ugly pink': '#cd7584',
        'sandy': '#f1da7a',
        'electric pink': '#ff0490',
        'muted purple': '#805b87',
        'mid green': '#50a747',
        'greyish': '#a8a495',
        'neon yellow': '#cfff04',
        'banana': '#ffff7e',
        'carnation pink': '#ff7fa7',
        'tomato': '#ef4026',
        'sea': '#3c9992',
        'muddy brown': '#886806',
        'turquoise green': '#04f489',
        'buff': '#fef69e',
        'fawn': '#cfaf7b',
        'muted blue': '#3b719f',
        'pale rose': '#fdc1c5',
        'dark mint green': '#20c073',
        'amethyst': '#9b5fc0',
        'blue/green': '#0f9b8e',
        'chestnut': '#742802',
        'sick green': '#9db92c',
        'pea': '#a4bf20',
        'rusty orange': '#cd5909',
        'stone': '#ada587',
        'rose red': '#be013c',
        'pale aqua': '#b8ffeb',
        'deep orange': '#dc4d01',
        'earth': '#a2653e',
        'mossy green': '#638b27',
        'grassy green': '#419c03',
        'pale lime green': '#b1ff65',
        'light grey blue': '#9dbcd4',
        'pale grey': '#fdfdfe',
        'asparagus': '#77ab56',
        'blueberry': '#464196',
        'purple red': '#990147',
        'pale lime': '#befd73',
        'greenish teal': '#32bf84',
        'caramel': '#af6f09',
        'deep magenta': '#a0025c',
        'light peach': '#ffd8b1',
        'milk chocolate': '#7f4e1e',
        'ocher': '#bf9b0c',
        'off green': '#6ba353',
        'purply pink': '#f075e6',
        'lightblue': '#7bc8f6',
        'dusky blue': '#475f94',
        'golden': '#f5bf03',
        'light beige': '#fffeb6',
        'butter yellow': '#fffd74',
        'dusky purple': '#895b7b',
        'french blue': '#436bad',
        'ugly yellow': '#d0c101',
        'greeny yellow': '#c6f808',
        'orangish red': '#f43605',
        'shamrock green': '#02c14d',
        'orangish brown': '#b25f03',
        'tree green': '#2a7e19',
        'deep violet': '#490648',
        'gunmetal': '#536267',
        'blue/purple': '#5a06ef',
        'cherry': '#cf0234',
        'sandy brown': '#c4a661',
        'warm grey': '#978a84',
        'dark indigo': '#1f0954',
        'midnight': '#03012d',
        'bluey green': '#2bb179',
        'grey pink': '#c3909b',
        'soft purple': '#a66fb5',
        'blood': '#770001',
        'brown red': '#922b05',
        'medium grey': '#7d7f7c',
        'berry': '#990f4b',
        'poo': '#8f7303',
        'purpley pink': '#c83cb9',
        'light salmon': '#fea993',
        'snot': '#acbb0d',
        'easter purple': '#c071fe',
        'light yellow green': '#ccfd7f',
        'dark navy blue': '#00022e',
        'drab': '#828344',
        'light rose': '#ffc5cb',
        'rouge': '#ab1239',
        'purplish red': '#b0054b',
        'slime green': '#99cc04',
        'baby poop': '#937c00',
        'irish green': '#019529',
        'pink/purple': '#ef1de7',
        'dark navy': '#000435',
        'greeny blue': '#42b395',
        'light plum': '#9d5783',
        'pinkish grey': '#c8aca9',
        'dirty orange': '#c87606',
        'rust red': '#aa2704',
        'pale lilac': '#e4cbff',
    }
    'orangey red': '#fa4224',
    # 定义颜色名称 'orangey red' 对应的十六进制颜色码
    'primary blue': '#0804f9',
    # 定义颜色名称 'primary blue' 对应的十六进制颜色码
    'kermit green': '#5cb200',
    # 定义颜色名称 'kermit green' 对应的十六进制颜色码
    'brownish purple': '#76424e',
    # 定义颜色名称 'brownish purple' 对应的十六进制颜色码
    'murky green': '#6c7a0e',
    # 定义颜色名称 'murky green' 对应的十六进制颜色码
    'wheat': '#fbdd7e',
    # 定义颜色名称 'wheat' 对应的十六进制颜色码
    'very dark purple': '#2a0134',
    # 定义颜色名称 'very dark purple' 对应的十六进制颜色码
    'bottle green': '#044a05',
    # 定义颜色名称 'bottle green' 对应的十六进制颜色码
    'watermelon': '#fd4659',
    # 定义颜色名称 'watermelon' 对应的十六进制颜色码
    'deep sky blue': '#0d75f8',
    # 定义颜色名称 'deep sky blue' 对应的十六进制颜色码
    'fire engine red': '#fe0002',
    # 定义颜色名称 'fire engine red' 对应的十六进制颜色码
    'yellow ochre': '#cb9d06',
    # 定义颜色名称 'yellow ochre' 对应的十六进制颜色码
    'pumpkin orange': '#fb7d07',
    # 定义颜色名称 'pumpkin orange' 对应的十六进制颜色码
    'pale olive': '#b9cc81',
    # 定义颜色名称 'pale olive' 对应的十六进制颜色码
    'light lilac': '#edc8ff',
    # 定义颜色名称 'light lilac' 对应的十六进制颜色码
    'lightish green': '#61e160',
    # 定义颜色名称 'lightish green' 对应的十六进制颜色码
    'carolina blue': '#8ab8fe',
    # 定义颜色名称 'carolina blue' 对应的十六进制颜色码
    'mulberry': '#920a4e',
    # 定义颜色名称 'mulberry' 对应的十六进制颜色码
    'shocking pink': '#fe02a2',
    # 定义颜色名称 'shocking pink' 对应的十六进制颜色码
    'auburn': '#9a3001',
    # 定义颜色名称 'auburn' 对应的十六进制颜色码
    'bright lime green': '#65fe08',
    # 定义颜色名称 'bright lime green' 对应的十六进制颜色码
    'celadon': '#befdb7',
    # 定义颜色名称 'celadon' 对应的十六进制颜色码
    'pinkish brown': '#b17261',
    # 定义颜色名称 'pinkish brown' 对应的十六进制颜色码
    'poo brown': '#885f01',
    # 定义颜色名称 'poo brown' 对应的十六进制颜色码
    'bright sky blue': '#02ccfe',
    # 定义颜色名称 'bright sky blue' 对应的十六进制颜色码
    'celery': '#c1fd95',
    # 定义颜色名称 'celery' 对应的十六进制颜色码
    'dirt brown': '#836539',
    # 定义颜色名称 'dirt brown' 对应的十六进制颜色码
    'strawberry': '#fb2943',
    # 定义颜色名称 'strawberry' 对应的十六进制颜色码
    'dark lime': '#84b701',
    # 定义颜色名称 'dark lime' 对应的十六进制颜色码
    'copper': '#b66325',
    # 定义颜色名称 'copper' 对应的十六进制颜色码
    'medium brown': '#7f5112',
    # 定义颜色名称 'medium brown' 对应的十六进制颜色码
    'muted green': '#5fa052',
    # 定义颜色名称 'muted green' 对应的十六进制颜色码
    "robin's egg": '#6dedfd',
    # 定义颜色名称 "robin's egg" 对应的十六进制颜色码
    'bright aqua': '#0bf9ea',
    # 定义颜色名称 'bright aqua' 对应的十六进制颜色码
    'bright lavender': '#c760ff',
    # 定义颜色名称 'bright lavender' 对应的十六进制颜色码
    'ivory': '#ffffcb',
    # 定义颜色名称 'ivory' 对应的十六进制颜色码
    'very light purple': '#f6cefc',
    # 定义颜色名称 'very light purple' 对应的十六进制颜色码
    'light navy': '#155084',
    # 定义颜色名称 'light navy' 对应的十六进制颜色码
    'pink red': '#f5054f',
    # 定义颜色名称 'pink red' 对应的十六进制颜色码
    'olive brown': '#645403',
    # 定义颜色名称 'olive brown' 对应的十六进制颜色码
    'poop brown': '#7a5901',
    # 定义颜色名称 'poop brown' 对应的十六进制颜色码
    'mustard green': '#a8b504',
    # 定义颜色名称 'mustard green' 对应的十六进制颜色码
    'ocean green': '#3d9973',
    # 定义颜色名称 'ocean green' 对应的十六进制颜色码
    'very dark blue': '#000133',
    # 定义颜色名称 'very dark blue' 对应的十六进制颜色码
    'dusty green': '#76a973',
    # 定义颜色名称 'dusty green' 对应的十六进制颜色码
    'light navy blue': '#2e5a88',
    # 定义颜色名称 'light navy blue' 对应的十六进制颜色码
    'minty green': '#0bf77d',
    # 定义颜色名称 'minty green' 对应的十六进制颜色码
    'adobe': '#bd6c48',
    # 定义颜色名称 'adobe' 对应的十六进制颜色码
    'barney': '#ac1db8',
    # 定义颜色名称 'barney' 对应的十六进制颜色码
    'jade green': '#2baf6a',
    # 定义颜色名称 'jade green' 对应的十六进制颜色码
    'bright light blue': '#26f7fd',
    # 定义颜色名称 'bright light blue' 对应的十六进制颜色码
    'light lime': '#aefd6c',
    # 定义颜色名称 'light lime' 对应的十六进制颜色码
    'dark khaki': '#9b8f55',
    # 定义颜色名称 'dark khaki' 对应的十六进制颜色码
    'orange yellow': '#ffad01',
    # 定义颜色名称 'orange yellow' 对应的十六
    'golden brown': '#b27a01',
    # 颜色名称 'golden brown' 对应的十六进制颜色码
    'bright turquoise': '#0ffef9',
    # 颜色名称 'bright turquoise' 对应的十六进制颜色码
    'red pink': '#fa2a55',
    # 颜色名称 'red pink' 对应的十六进制颜色码
    'red purple': '#820747',
    # 颜色名称 'red purple' 对应的十六进制颜色码
    'greyish brown': '#7a6a4f',
    # 颜色名称 'greyish brown' 对应的十六进制颜色码
    'vermillion': '#f4320c',
    # 颜色名称 'vermillion' 对应的十六进制颜色码
    'russet': '#a13905',
    # 颜色名称 'russet' 对应的十六进制颜色码
    'steel grey': '#6f828a',
    # 颜色名称 'steel grey' 对应的十六进制颜色码
    'lighter purple': '#a55af4',
    # 颜色名称 'lighter purple' 对应的十六进制颜色码
    'bright violet': '#ad0afd',
    # 颜色名称 'bright violet' 对应的十六进制颜色码
    'prussian blue': '#004577',
    # 颜色名称 'prussian blue' 对应的十六进制颜色码
    'slate green': '#658d6d',
    # 颜色名称 'slate green' 对应的十六进制颜色码
    'dirty pink': '#ca7b80',
    # 颜色名称 'dirty pink' 对应的十六进制颜色码
    'dark blue green': '#005249',
    # 颜色名称 'dark blue green' 对应的十六进制颜色码
    'pine': '#2b5d34',
    # 颜色名称 'pine' 对应的十六进制颜色码
    'yellowy green': '#bff128',
    # 颜色名称 'yellowy green' 对应的十六进制颜色码
    'dark gold': '#b59410',
    # 颜色名称 'dark gold' 对应的十六进制颜色码
    'bluish': '#2976bb',
    # 颜色名称 'bluish' 对应的十六进制颜色码
    'darkish blue': '#014182',
    # 颜色名称 'darkish blue' 对应的十六进制颜色码
    'dull red': '#bb3f3f',
    # 颜色名称 'dull red' 对应的十六进制颜色码
    'pinky red': '#fc2647',
    # 颜色名称 'pinky red' 对应的十六进制颜色码
    'bronze': '#a87900',
    # 颜色名称 'bronze' 对应的十六进制颜色码
    'pale teal': '#82cbb2',
    # 颜色名称 'pale teal' 对应的十六进制颜色码
    'military green': '#667c3e',
    # 颜色名称 'military green' 对应的十六进制颜色码
    'barbie pink': '#fe46a5',
    # 颜色名称 'barbie pink' 对应的十六进制颜色码
    'bubblegum pink': '#fe83cc',
    # 颜色名称 'bubblegum pink' 对应的十六进制颜色码
    'pea soup green': '#94a617',
    # 颜色名称 'pea soup green' 对应的十六进制颜色码
    'dark mustard': '#a88905',
    # 颜色名称 'dark mustard' 对应的十六进制颜色码
    'shit': '#7f5f00',
    # 颜色名称 'shit' 对应的十六进制颜色码
    'medium purple': '#9e43a2',
    # 颜色名称 'medium purple' 对应的十六进制颜色码
    'very dark green': '#062e03',
    # 颜色名称 'very dark green' 对应的十六进制颜色码
    'dirt': '#8a6e45',
    # 颜色名称 'dirt' 对应的十六进制颜色码
    'dusky pink': '#cc7a8b',
    # 颜色名称 'dusky pink' 对应的十六进制颜色码
    'red violet': '#9e0168',
    # 颜色名称 'red violet' 对应的十六进制颜色码
    'lemon yellow': '#fdff38',
    # 颜色名称 'lemon yellow' 对应的十六进制颜色码
    'pistachio': '#c0fa8b',
    # 颜色名称 'pistachio' 对应的十六进制颜色码
    'dull yellow': '#eedc5b',
    # 颜色名称 'dull yellow' 对应的十六进制颜色码
    'dark lime green': '#7ebd01',
    # 颜色名称 'dark lime green' 对应的十六进制颜色码
    'denim blue': '#3b5b92',
    # 颜色名称 'denim blue' 对应的十六进制颜色码
    'teal blue': '#01889f',
    # 颜色名称 'teal blue' 对应的十六进制颜色码
    'lightish blue': '#3d7afd',
    # 颜色名称 'lightish blue' 对应的十六进制颜色码
    'purpley blue': '#5f34e7',
    # 颜色名称 'purpley blue' 对应的十六进制颜色码
    'light indigo': '#6d5acf',
    # 颜色名称 'light indigo' 对应的十六进制颜色码
    'swamp green': '#748500',
    # 颜色名称 'swamp green' 对应的十六进制颜色码
    'brown green': '#706c11',
    # 颜色名称 'brown green' 对应的十六进制颜色码
    'dark maroon': '#3c0008',
    # 颜色名称 'dark maroon' 对应的十六进制颜色码
    'hot purple': '#cb00f5',
    # 颜色名称 'hot purple' 对应的十六进制颜色码
    'dark forest green': '#002d04',
    # 颜色名称 'dark forest green' 对应的十六进制颜色码
    'faded blue': '#658cbb',
    # 颜色名称 'faded blue' 对应的十六进制颜色码
    'drab green': '#749551',
    # 颜色名称 'drab green' 对应的十六进制颜色码
    'light lime green': '#b9ff66',
    # 颜色名称 'light lime green' 对应的十六进制颜色码
    'snot green': '#9dc100',
    # 颜色名称 'snot green' 对应的十六进制颜色码
    'yellowish': '#faee66',
    # 颜色名称 'yellowish' 对应的十六进制颜色码
    'light blue green': '#7efbb3',
    # 颜色名称 'light blue green' 对应的十六进制颜色码
    'bordeaux': '#7b002c',
    # 颜色名称 'bordeaux' 对应的十六进制颜色码
    'light mauve': '#c292a1',
    # 颜色名称 'light mauve' 对应的十六进制颜色码
    'ocean': '#017b92',
    # 颜色名称 'ocean' 对应的十六进制颜色码
    'light maroon': '#a24857',  # 定义颜色 'light maroon' 的十六进制表示
    'dusty purple': '#825f87',  # 定义颜色 'dusty purple' 的十六进制表示
    'terra cotta': '#c9643b',   # 定义颜色 'terra cotta' 的十六进制表示
    'avocado': '#90b134',       # 定义颜色 'avocado' 的十六进制表示
    'marine blue': '#01386a',   # 定义颜色 'marine blue' 的十六进制表示
    'teal green': '#25a36f',    # 定义颜色 'teal green' 的十六进制表示
    'slate grey': '#59656d',    # 定义颜色 'slate grey' 的十六进制表示
    'lighter green': '#75fd63', # 定义颜色 'lighter green' 的十六进制表示
    'electric green': '#21fc0d',# 定义颜色 'electric green' 的十六进制表示
    'dusty blue': '#5a86ad',    # 定义颜色 'dusty blue' 的十六进制表示
    'golden yellow': '#fec615', # 定义颜色 'golden yellow' 的十六进制表示
    'bright yellow': '#fffd01', # 定义颜色 'bright yellow' 的十六进制表示
    'light lavender': '#dfc5fe',# 定义颜色 'light lavender' 的十六进制表示
    'umber': '#b26400',         # 定义颜色 'umber' 的十六进制表示
    'poop': '#7f5e00',          # 定义颜色 'poop' 的十六进制表示
    'dark peach': '#de7e5d',    # 定义颜色 'dark peach' 的十六进制表示
    'jungle green': '#048243',  # 定义颜色 'jungle green' 的十六进制表示
    'eggshell': '#ffffd4',      # 定义颜色 'eggshell' 的十六进制表示
    'denim': '#3b638c',         # 定义颜色 'denim' 的十六进制表示
    'yellow brown': '#b79400',  # 定义颜色 'yellow brown' 的十六进制表示
    'dull purple': '#84597e',   # 定义颜色 'dull purple' 的十六进制表示
    'chocolate brown': '#411900',# 定义颜色 'chocolate brown' 的十六进制表示
    'wine red': '#7b0323',      # 定义颜色 'wine red' 的十六进制表示
    'neon blue': '#04d9ff',     # 定义颜色 'neon blue' 的十六进制表示
    'dirty green': '#667e2c',   # 定义颜色 'dirty green' 的十六进制表示
    'light tan': '#fbeeac',     # 定义颜色 'light tan' 的十六进制表示
    'ice blue': '#d7fffe',      # 定义颜色 'ice blue' 的十六进制表示
    'cadet blue': '#4e7496',    # 定义颜色 'cadet blue' 的十六进制表示
    'dark mauve': '#874c62',    # 定义颜色 'dark mauve' 的十六进制表示
    'very light blue': '#d5ffff',# 定义颜色 'very light blue' 的十六进制表示
    'grey purple': '#826d8c',   # 定义颜色 'grey purple' 的十六进制表示
    'pastel pink': '#ffbacd',   # 定义颜色 'pastel pink' 的十六进制表示
    'very light green': '#d1ffbd',# 定义颜色 'very light green' 的十六进制表示
    'dark sky blue': '#448ee4', # 定义颜色 'dark sky blue' 的十六进制表示
    'evergreen': '#05472a',     # 定义颜色 'evergreen' 的十六进制表示
    'dull pink': '#d5869d',     # 定义颜色 'dull pink' 的十六进制表示
    'aubergine': '#3d0734',     # 定义颜色 'aubergine' 的十六进制表示
    'mahogany': '#4a0100',      # 定义颜色 'mahogany' 的十六进制表示
    'reddish orange': '#f8481c',# 定义颜色 'reddish orange' 的十六进制表示
    'deep green': '#02590f',    # 定义颜色 'deep green' 的十六进制表示
    'vomit green': '#89a203',   # 定义颜色 'vomit green' 的十六进制表示
    'purple pink': '#e03fd8',   # 定义颜色 'purple pink' 的十六进制表示
    'dusty pink': '#d58a94',    # 定义颜色 'dusty pink' 的十六进制表示
    'faded green': '#7bb274',   # 定义颜色 'faded green' 的十六进制表示
    'camo green': '#526525',    # 定义颜色 'camo green' 的十六进制表示
    'pinky purple': '#c94cbe',  # 定义颜色 'pinky purple' 的十六进制表示
    'pink purple': '#db4bda',   # 定义颜色 'pink purple' 的十六进制表示
    'brownish red': '#9e3623',  # 定义颜色 'brownish red' 的十六进制表示
    'dark rose': '#b5485d',     # 定义颜色 'dark rose' 的十六进制表示
    'mud': '#735c12',           # 定义颜色 'mud' 的十六进制表示
    'brownish': '#9c6d57',      # 定义颜色 'brownish' 的十六进制表示
    'emerald green': '#028f1e', # 定义颜色 'emerald green' 的十六进制表示
    'pale brown': '#b1916e',    # 定义颜色 'pale brown' 的十六进制表示
    'dull blue': '#49759c',     # 定义颜色 'dull blue' 的十六进制表示
    'burnt umber': '#a0450e',   # 定义颜色 'burnt umber' 的十六进制表示
    'medium green': '#39ad48',   # 定义颜色 'medium green' 的十六进制表示
    'clay': '#b66a50',          # 定义颜色 'clay' 的十六进制表示
    'light aqua': '#8cffdb',    # 定义颜色 'light aqua' 的十六进制表示
    'light olive green': '#a4be5c',# 定义颜色 'light olive green' 的十六进制表示
    'brownish orange': '#cb7723',# 定义颜色 'brownish orange' 的十六进制表示
    'dark aqua': '#05696b',     # 定义颜色 'dark aqua' 的十六进制表示
    'purplish pink': '#ce5dae', # 定义颜色 'purplish pink' 的十六进制表示
    'dark salmon': '#c85a53',   # 定义颜色 'dark salmon' 的十六进制表示
    'greenish grey': '#96ae8d', # 定义颜色 'greenish grey' 的十六进制表示
    'jade': '#1fa774',          # 定义颜色 'jade' 的十六进制表示
    'ug
    'sienna': '#a9561e',
    'pastel purple': '#caa0ff',
    'terracotta': '#ca6641',
    'aqua blue': '#02d8e9',
    'sage green': '#88b378',
    'blood red': '#980002',
    'deep pink': '#cb0162',
    'grass': '#5cac2d',
    'moss': '#769958',
    'pastel blue': '#a2bffe',
    'bluish green': '#10a674',
    'green blue': '#06b48b',
    'dark tan': '#af884a',
    'greenish blue': '#0b8b87',
    'pale orange': '#ffa756',
    'vomit': '#a2a415',
    'forrest green': '#154406',
    'dark lavender': '#856798',
    'dark violet': '#34013f',
    'purple blue': '#632de9',
    'dark cyan': '#0a888a',
    'olive drab': '#6f7632',
    'pinkish': '#d46a7e',
    'cobalt': '#1e488f',
    'neon purple': '#bc13fe',
    'light turquoise': '#7ef4cc',
    'apple green': '#76cd26',
    'dull green': '#74a662',
    'wine': '#80013f',
    'powder blue': '#b1d1fc',
    'off white': '#ffffe4',
    'electric blue': '#0652ff',
    'dark turquoise': '#045c5a',
    'blue purple': '#5729ce',
    'azure': '#069af3',
    'bright red': '#ff000d',
    'pinkish red': '#f10c45',
    'cornflower blue': '#5170d7',
    'light olive': '#acbf69',
    'grape': '#6c3461',
    'greyish blue': '#5e819d',
    'purplish blue': '#601ef9',
    'yellowish green': '#b0dd16',
    'greenish yellow': '#cdfd02',
    'medium blue': '#2c6fbb',
    'dusty rose': '#c0737a',
    'light violet': '#d6b4fc',
    'midnight blue': '#020035',
    'bluish purple': '#703be7',
    'red orange': '#fd3c06',
    'dark magenta': '#960056',
    'greenish': '#40a368',
    'ocean blue': '#03719c',
    'coral': '#fc5a50',
    'cream': '#ffffc2',
    'reddish brown': '#7f2b0a',
    'burnt sienna': '#b04e0f',
    'brick': '#a03623',
    'sage': '#87ae73',
    'grey green': '#789b73',
    'white': '#ffffff',
    "robin's egg blue": '#98eff9',
    'moss green': '#658b38',
    'steel blue': '#5a7d9a',
    'eggplant': '#380835',
    'light yellow': '#fffe7a',
    'leaf green': '#5ca904',
    'light grey': '#d8dcd6',
    'puke': '#a5a502',
    'pinkish purple': '#d648d7',
    'sea blue': '#047495',
    'pale purple': '#b790d4',
    'slate blue': '#5b7c99',
    'blue grey': '#607c8e',
    'hunter green': '#0b4008',
    'fuchsia': '#ed0dd9',
    'crimson': '#8c000f',
    'pale yellow': '#ffff84',
    'ochre': '#bf9005',
    'mustard yellow': '#d2bd0a',
    'light red': '#ff474c',
    'cerulean': '#0485d1',
    'pale pink': '#ffcfdc',
    'deep blue': '#040273',
    'rust': '#a83c09',
    'light teal': '#90e4c1',
    'slate': '#516572',
    'goldenrod': '#fac205',
    'dark yellow': '#d5b60a',
    'dark grey': '#363737',
    'army green': '#4b5d16',
    'grey blue': '#6b8ba4',
    'seafoam': '#80f9ad',
    'puce': '#a57e52',
    'spring green': '#a9f971',
    'dark orange': '#c65102',
    'sand': '#e2ca76',
    'pastel green': '#b0ff9d',
    'mint': '#9ffeb0',
    'light orange': '#fdaa48',
    'bright pink': '#fe01b1',
    'chartreuse': '#c1f80a',
    'deep purple': '#36013f',
    'dark brown': '#341c02',


注释：


    'sienna': '#a9561e',               # 赭色
    'pastel purple': '#caa0ff',        # 淡紫色
    'terracotta': '#ca6641',           # 赤陶土色
    'aqua blue': '#02d8e9',            # 水蓝色
    'sage green': '#88b378',           # 鼠尾草绿色
    'blood red': '#980002',            # 鲜血红色
    'deep pink': '#cb0162',            # 深粉色
    'grass': '#5cac2d',                # 青草色
    'moss': '#769958',                 # 苔藓绿色
    'pastel blue': '#a2bffe',          # 淡蓝色
    'bluish green': '#10a674',         # 蓝绿色
    'green blue': '#06b48b',           # 绿蓝色
    'dark tan': '#af884a',             # 深棕黄色
    'greenish blue': '#0b8b87',        # 绿蓝色
    'pale orange': '#ffa756',          # 浅橙色
    'vomit': '#a2a415',                # 呕吐色
    'forrest green': '#154406',        # 森林绿色
    'dark lavender': '#856798',        # 深紫色
    'dark violet': '#34013f',          # 深紫色
    'purple blue': '#632de9',          # 紫蓝色
    'dark cyan': '#0a888a',            # 深青色
    'olive drab': '#6f7632',           # 橄榄褐色
    'pinkish': '#d46a7e',              # 粉红色
    'cobalt': '#1e488f',               # 钴色
    'neon purple': '#bc13fe',          # 霓虹紫色
    'light turquoise': '#7ef4cc',      # 浅宝石绿色
    'apple green': '#76cd26',          # 苹果绿色
    'dull green': '#74a662',           # 暗绿色
    'wine': '#80013f',                 # 红酒色
    'powder blue': '#b1d1fc',          # 粉蓝色
    'off white': '#ffffe4',            # 象牙白色
    'electric blue': '#0652ff',        # 电光蓝色
    'dark turquoise': '#045c5a',       # 深青色
    'blue purple': '#5729ce',          # 蓝紫色
    'azure': '#069af3',                # 天蓝色
    'bright red': '#ff000d',           # 亮红色
    'pinkish red': '#f10c45',          # 粉红红色
    'cornflower blue': '#5170d7',      # 矢车菊蓝色
    'light olive': '#acbf69',          # 浅橄榄色
    'grape': '#6c3461',                # 葡萄色
    'greyish blue': '#5e819d',         # 灰蓝色
    'purplish blue': '#601ef9',        # 紫蓝色
    'yellowish green': '#b0dd16',      # 黄绿色
    'greenish yellow': '#cdfd02',      # 绿黄色
    # 定义一个包含颜色名称和十六进制表示的字典
    {
        'taupe': '#b9a281',            # 淡褐色
        'pea green': '#8eab12',        # 豌豆绿
        'puke green': '#9aae07',       # 呕吐绿
        'kelly green': '#02ab2e',      # 艾利绿
        'seafoam green': '#7af9ab',    # 海泡绿
        'blue green': '#137e6d',       # 蓝绿色
        'khaki': '#aaa662',            # 卡其色
        'burgundy': '#610023',         # 酒红色
        'dark teal': '#014d4e',        # 深青色
        'brick red': '#8f1402',        # 砖红色
        'royal purple': '#4b006e',     # 皇家紫
        'plum': '#580f41',             # 洋李色
        'mint green': '#8fff9f',       # 薄荷绿
        'gold': '#dbb40c',             # 金色
        'baby blue': '#a2cffe',        # 淡蓝色
        'yellow green': '#c0fb2d',     # 黄绿色
        'bright purple': '#be03fd',    # 明亮紫色
        'dark red': '#840000',         # 深红色
        'pale blue': '#d0fefe',        # 苍白蓝色
        'grass green': '#3f9b0b',      # 草绿色
        'navy': '#01153e',             # 海军蓝
        'aquamarine': '#04d8b2',       # 宝石绿
        'burnt orange': '#c04e01',     # 烧橙色
        'neon green': '#0cff0c',       # 霓虹绿
        'bright blue': '#0165fc',      # 明亮蓝色
        'rose': '#cf6275',             # 玫瑰色
        'light pink': '#ffd1df',       # 浅粉色
        'mustard': '#ceb301',          # 芥末色
        'indigo': '#380282',           # 靛青色
        'lime': '#aaff32',             # 酸橙色
        'sea green': '#53fca1',        # 海绿色
        'periwinkle': '#8e82fe',       # 浅紫色
        'dark pink': '#cb416b',        # 深粉色
        'olive green': '#677a04',      # 橄榄绿
        'peach': '#ffb07c',            # 桃色
        'pale green': '#c7fdb5',       # 苍白绿
        'light brown': '#ad8150',      # 浅棕色
        'hot pink': '#ff028d',         # 亮粉红色
        'black': '#000000',            # 黑色
        'lilac': '#cea2fd',            # 紫丁香色
        'navy blue': '#001146',        # 海军蓝
        'royal blue': '#0504aa',       # 皇家蓝
        'beige': '#e6daa6',            # 米色
        'salmon': '#ff796c',           # 鲑鱼色
        'olive': '#6e750e',            # 橄榄色
        'maroon': '#650021',           # 栗色
        'bright green': '#01ff07',     # 明亮绿色
        'dark purple': '#35063e',      # 深紫色
        'mauve': '#ae7181',            # 粉紫色
        'forest green': '#06470c',     # 深绿色
        'aqua': '#13eac9',             # 水绿色
        'cyan': '#00ffff',             # 青色
        'tan': '#d1b26f',              # 淡棕色
        'dark blue': '#00035b',        # 深蓝色
        'lavender': '#c79fef',         # 薰衣草色
        'turquoise': '#06c2ac',        # 绿松石色
        'dark green': '#033500',       # 深绿色
        'violet': '#9a0eea',           # 紫罗兰色
        'light purple': '#bf77f6',     # 浅紫色
        'lime green': '#89fe05',       # 酸橙绿
        'grey': '#929591',             # 灰色
        'sky blue': '#75bbfd',         # 天蓝色
        'yellow': '#ffff14',           # 黄色
        'magenta': '#c20078',          # 洋红色
        'light green': '#96f97b',      # 浅绿色
        'orange': '#f97306',           # 橙色
        'teal': '#029386',             # 青色
        'light blue': '#95d0fc',       # 浅蓝色
        'red': '#e50000',              # 红色
        'brown': '#653700',            # 棕色
        'pink': '#ff81c0',             # 粉色
        'blue': '#0343df',             # 蓝色
        'green': '#15b01a',            # 绿色
        'purple': '#7e1e9c'            # 紫色
    }
# 将 XKCD_COLORS 字典中的每个颜色名称前加上 "xkcd:"，以避免名称冲突。
XKCD_COLORS = {'xkcd:' + name: value for name, value in XKCD_COLORS.items()}


# CSS4_COLORS 字典包含了 CSS Color Module Level 4 中定义的命名颜色及其十六进制表示。
# 参考链接：https://drafts.csswg.org/css-color-4/#named-colors
CSS4_COLORS = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkgrey':             '#A9A9A9',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkslategrey':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dimgrey':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'grey':                 '#808080',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    # 此处省略了部分颜色，因为其数量较多，继续按字典格式列出。
}
    # 定义了各种颜色的十六进制表示的字典，键是颜色名，值是颜色的十六进制表示
    
    'lightgray':            '#D3D3D3',       # 浅灰色
    'lightgreen':           '#90EE90',       # 浅绿色
    'lightgrey':            '#D3D3D3',       # 同 lightgray，浅灰色
    'lightpink':            '#FFB6C1',       # 浅粉色
    'lightsalmon':          '#FFA07A',       # 浅鲑色
    'lightseagreen':        '#20B2AA',       # 浅海洋绿色
    'lightskyblue':         '#87CEFA',       # 浅天蓝色
    'lightslategray':       '#778899',       # 浅蓝灰色
    'lightslategrey':       '#778899',       # 同 lightslategray，浅蓝灰色
    'lightsteelblue':       '#B0C4DE',       # 亮钢蓝色
    'lightyellow':          '#FFFFE0',       # 浅黄色
    'lime':                 '#00FF00',       # 酸橙色
    'limegreen':            '#32CD32',       # 酸橙绿色
    'linen':                '#FAF0E6',       # 亚麻色
    'magenta':              '#FF00FF',       # 洋红色
    'maroon':               '#800000',       # 栗色
    'mediumaquamarine':     '#66CDAA',       # 中碧绿色
    'mediumblue':           '#0000CD',       # 中蓝色
    'mediumorchid':         '#BA55D3',       # 中兰花紫色
    'mediumpurple':         '#9370DB',       # 中紫色
    'mediumseagreen':       '#3CB371',       # 中海洋绿色
    'mediumslateblue':      '#7B68EE',       # 中蓝灰色
    'mediumspringgreen':    '#00FA9A',       # 中春绿色
    'mediumturquoise':      '#48D1CC',       # 中绿宝石色
    'mediumvioletred':      '#C71585',       # 中紫罗兰色
    'midnightblue':         '#191970',       # 午夜蓝色
    'mintcream':            '#F5FFFA',       # 薄荷奶油色
    'mistyrose':            '#FFE4E1',       # 浅玫瑰色
    'moccasin':             '#FFE4B5',       # 鹿皮色
    'navajowhite':          '#FFDEAD',       # 纳瓦白色
    'navy':                 '#000080',       # 海军蓝色
    'oldlace':              '#FDF5E6',       # 旧布色
    'olive':                '#808000',       # 橄榄色
    'olivedrab':            '#6B8E23',       # 橄榄褐色
    'orange':               '#FFA500',       # 橙色
    'orangered':            '#FF4500',       # 橙红色
    'orchid':               '#DA70D6',       # 兰花紫色
    'palegoldenrod':        '#EEE8AA',       # 苍麒麟色
    'palegreen':            '#98FB98',       # 苍绿色
    'paleturquoise':        '#AFEEEE',       # 苍宝石绿色
    'palevioletred':        '#DB7093',       # 苍紫罗兰色
    'papayawhip':           '#FFEFD5',       # 番木瓜色
    'peachpuff':            '#FFDAB9',       # 桃肉色
    'peru':                 '#CD853F',       # 秘鲁色
    'pink':                 '#FFC0CB',       # 粉红色
    'plum':                 '#DDA0DD',       # 洋李色
    'powderblue':           '#B0E0E6',       # 粉蓝色
    'purple':               '#800080',       # 紫色
    'rebeccapurple':        '#663399',       # 丽贝卡紫色
    'red':                  '#FF0000',       # 红色
    'rosybrown':            '#BC8F8F',       # 玫瑰棕色
    'royalblue':            '#4169E1',       # 皇家蓝色
    'saddlebrown':          '#8B4513',       # 鞍褐色
    'salmon':               '#FA8072',       # 鲑鱼色
    'sandybrown':           '#F4A460',       # 沙褐色
    'seagreen':             '#2E8B57',       # 海绿色
    'seashell':             '#FFF5EE',       # 海贝色
    'sienna':               '#A0522D',       # 黄土赭色
    'silver':               '#C0C0C0',       # 银色
    'skyblue':              '#87CEEB',       # 天蓝色
    'slateblue':            '#6A5ACD',       # 石蓝色
    'slategray':            '#708090',       # 石灰色
    'slategrey':            '#708090',       # 同 slategray，石灰色
    'snow':                 '#FFFAFA',       # 雪白色
    'springgreen':          '#00FF7F',       # 春绿色
    'steelblue':            '#4682B4',       # 钢蓝色
    'tan':                  '#D2B48C',       # 茶色
    'teal':                 '#008080',       # 水鸭色
    'thistle':              '#D8BFD8',       # 蓟色
    'tomato':               '#FF6347',       # 番茄色
    'turquoise':            '#40E0D0',       # 绿宝石色
    'violet':               '#EE82EE',       # 紫罗兰色
    'wheat':                '#F5DEB3',       # 小麦色
    'white':                '#FFFFFF',       # 白色
    'whitesmoke':           '#F5F5F5',       # 烟白色
    'yellow':               '#FFFF00',       # 黄色
    'yellowgreen':          '#9ACD32'}


# 定义一个字典，将颜色名称映射到十六进制表示的颜色码
```