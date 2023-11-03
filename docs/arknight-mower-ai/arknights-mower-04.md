# ArknightMower源码解析 4

# `arknights_mower/ocr/decode.py`

This appears to be a function that performs a spatial voting operation on an image. The function takes an image bitmap, a bounding box and a score array. The function returns the updated bounding box and an updated score array.

The function first copies the bounding box and the original score array and then applies a mask to the bitmap using OpenCV. The mask is created by using the minimum and maximum values of the ymin and xmin, and the cv2.fillPoly function. This mask is then passed to the cv2.mean function along with the original bitmap. The cv2.mean function returns the average value of the pixel values within the bounding box and the mask.

The function then updates the score array by keeping only the pixels that fall within the bounding box and are flagged by the mask. This gives the function a fresh look at the score array, as the original score array is copied, but the mask ensures that only the positive ( flagged) pixels are included.

Finally, the function returns the updated bounding box and the updated score array.


```py
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class SegDetectorRepresenter:
    def __init__(self, thresh=0.3, box_thresh=0.5, max_candidates=1000, unclip_ratio=2.0):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, height, width):
        """
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        """

        pred = pred[0, :, :]
        segmentation = self.binarize(pred)

        boxes, scores = self.boxes_from_bitmap(
            pred, segmentation, width, height)

        return boxes, scores

    def binarize(self, pred):
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """
        bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        """

        assert len(bitmap.shape) == 2
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue
            box = self.unclip(
                points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)

        distance = poly.area * unclip_ratio / (poly.length)
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

```

# `arknights_mower/ocr/keys.py`

这是一段文字，可能是某种文章或诗歌中的一部分。其中描述了一些或不完整的词语，可能是为了逗乐或神秘而服务的。由于语言和拼写都不太清楚，我无法对其进行详细的解释或翻译。


```py
alphabetChinese = u'\'疗绚诚娇溜题贿者廖更纳加奉公一就汴计与路房原妇208-7其>:],，骑刈全消昏傈安久钟嗅不影处驽蜿资关椤地瘸专问忖票嫉炎韵要月田节陂鄙捌备拳伺眼网盎大傍心东愉汇蹿科每业里航晏字平录先13彤鲶产稍督腴有象岳注绍在泺文定核名水过理让偷率等这发”为含肥酉相鄱七编猥锛日镀蒂掰倒辆栾栗综涩州雌滑馀了机块司宰甙兴矽抚保用沧秩如收息滥页疑埠!！姥异橹钇向下跄的椴沫国绥獠报开民蜇何分凇长讥藏掏施羽中讲派嘟人提浼间世而古多倪唇饯控庚首赛蜓味断制觉技替艰溢潮夕钺外摘枋动双单啮户枇确锦曜杜或能效霜盒然侗电晁放步鹃新杖蜂吒濂瞬评总隍对独合也是府青天诲墙组滴级邀帘示已时骸仄泅和遨店雇疫持巍踮境只亨目鉴崤闲体泄杂作般轰化解迂诿蛭璀腾告版服省师小规程线海办引二桧牌砺洄裴修图痫胡许犊事郛基柴呼食研奶律蛋因葆察戏褒戒再李骁工貂油鹅章啄休场给睡纷豆器捎说敏学会浒设诊格廓查来霓室溆￠诡寥焕舜柒狐回戟砾厄实翩尿五入径惭喹股宇篝|;美期云九祺扮靠锝槌系企酰阊暂蚕忻豁本羹执条钦H獒限进季楦于芘玖铋茯未答粘括样精欠矢甥帷嵩扣令仔风皈行支部蓉刮站蜡救钊汗松嫌成可.鹤院从交政怕活调球局验髌第韫谗串到圆年米/*友忿检区看自敢刃个兹弄流留同没齿星聆轼湖什三建蛔儿椋汕震颧鲤跟力情璺铨陪务指族训滦鄣濮扒商箱十召慷辗所莞管护臭横硒嗓接侦六露党馋驾剖高侬妪幂猗绺骐央酐孝筝课徇缰门男西项句谙瞒秃篇教碲罚声呐景前富嘴鳌稀免朋啬睐去赈鱼住肩愕速旁波厅健茼厥鲟谅投攸炔数方击呋谈绩别愫僚躬鹧胪炳招喇膨泵蹦毛结54谱识陕粽婚拟构且搜任潘比郢妨醪陀桔碘扎选哈骷楷亿明缆脯监睫逻婵共赴淝凡惦及达揖谩澹减焰蛹番祁柏员禄怡峤龙白叽生闯起细装谕竟聚钙上导渊按艾辘挡耒盹饪臀记邮蕙受各医搂普滇朗茸带翻酚(光堤墟蔷万幻〓瑙辈昧盏亘蛀吉铰请子假闻税井诩哨嫂好面琐校馊鬣缂营访炖占农缀否经钚棵趟张亟吏茶谨捻论迸堂玉信吧瞠乡姬寺咬溏苄皿意赉宝尔钰艺特唳踉都荣倚登荐丧奇涵批炭近符傩感道着菊虹仲众懈濯颞眺南释北缝标既茗整撼迤贲挎耱拒某妍卫哇英矶藩治他元领膜遮穗蛾飞荒棺劫么市火温拈棚洼转果奕卸迪伸泳斗邡侄涨屯萋胭氡崮枞惧冒彩斜手豚随旭淑妞形菌吲沱争驯歹挟兆柱传至包内响临红功弩衡寂禁老棍耆渍织害氵渑布载靥嗬虽苹咨娄库雉榜帜嘲套瑚亲簸欧边6腿旮抛吹瞳得镓梗厨继漾愣憨士策窑抑躯襟脏参贸言干绸鳄穷藜音折详)举悍甸癌黎谴死罩迁寒驷袖媒蒋掘模纠恣观祖蛆碍位稿主澧跌筏京锏帝贴证糠才黄鲸略炯饱四出园犀牧容汉杆浈汰瑷造虫瘩怪驴济应花沣谔夙旅价矿以考su呦晒巡茅准肟瓴詹仟褂译桌混宁怦郑抿些余鄂饴攒珑群阖岔琨藓预环洮岌宀杲瀵最常囡周踊女鼓袭喉简范薯遐疏粱黜禧法箔斤遥汝奥直贞撑置绱集她馅逗钧橱魉[恙躁唤9旺膘待脾惫购吗依盲度瘿蠖俾之镗拇鲵厝簧续款展啃表剔品钻腭损清锶统涌寸滨贪链吠冈伎迥咏吁览防迅失汾阔逵绀蔑列川凭努熨揪利俱绉抢鸨我即责膦易毓鹊刹玷岿空嘞绊排术估锷违们苟铜播肘件烫审鲂广像铌惰铟巳胍鲍康憧色恢想拷尤疳知SYFDA峄裕帮握搔氐氘难墒沮雨叁缥悴藐湫娟苑稠颛簇后阕闭蕤缚怎佞码嘤蔡痊舱螯帕赫昵升烬岫、疵蜻髁蕨隶烛械丑盂梁强鲛由拘揉劭龟撤钩呕孛费妻漂求阑崖秤甘通深补赃坎床啪承吼量暇钼烨阂擎脱逮称P神属矗华届狍葑汹育患窒蛰佼静槎运鳗庆逝曼疱克代官此麸耧蚌晟例础榛副测唰缢迹灬霁身岁赭扛又菡乜雾板读陷徉贯郁虑变钓菜圾现琢式乐维渔浜左吾脑钡警T啵拴偌漱湿硕止骼魄积燥联踢玛|则窿见振畿送班钽您赵刨印讨踝籍谡舌崧汽蔽沪酥绒怖财帖肱私莎勋羔霸励哼帐将帅渠纪婴娩岭厘滕吻伤坝冠戊隆瘁介涧物黍并姗奢蹑掣垸锴命箍捉病辖琰眭迩艘绌繁寅若毋思诉类诈燮轲酮狂重反职筱县委磕绣奖晋濉志徽肠呈獐坻口片碰几村柿劳料获亩惕晕厌号罢池正鏖煨家棕复尝懋蜥锅岛扰队坠瘾钬@卧疣镇譬冰彷频黯据垄采八缪瘫型熹砰楠襁箐但嘶绳啤拍盥穆傲洗盯塘怔筛丿台恒喂葛永￥烟酒桦书砂蚝缉态瀚袄圳轻蛛超榧遛姒奘铮右荽望偻卡丶氰附做革索戚坨桷唁垅榻岐偎坛莨山殊微骇陈爨推嗝驹澡藁呤卤嘻糅逛侵郓酌德摇※鬃被慨殡羸昌泡戛鞋河宪沿玲鲨翅哽源铅语照邯址荃佬顺鸳町霭睾瓢夸椁晓酿痈咔侏券噎湍签嚷离午尚社锤背孟使浪缦潍鞅军姹驶笑鳟鲁》孽钜绿洱礴焯椰颖囔乌孔巴互性椽哞聘昨早暮胶炀隧低彗昝铁呓氽藉喔癖瑗姨权胱韦堑蜜酋楝砝毁靓歙锲究屋喳骨辨碑武鸠宫辜烊适坡殃培佩供走蜈迟翼况姣凛浔吃飘债犟金促苛崇坂莳畔绂兵蠕斋根砍亢欢恬崔剁餐榫快扶‖濒缠鳜当彭驭浦篮昀锆秸钳弋娣瞑夷龛苫拱致%嵊障隐弑初娓抉汩累蓖"唬助苓昙押毙破城郧逢嚏獭瞻溱婿赊跨恼璧萃姻貉灵炉密氛陶砸谬衔点琛沛枳层岱诺脍榈埂征冷裁打蹴素瘘逞蛐聊激腱萘踵飒蓟吆取咙簋涓矩曝挺揣座你史舵焱尘苏笈脚溉榨诵樊邓焊义庶儋蟋蒲赦呷杞诠豪还试颓茉太除紫逃痴草充鳕珉祗墨渭烩蘸慕璇镶穴嵘恶骂险绋幕碉肺戳刘潞秣纾潜銮洛须罘销瘪汞兮屉r林厕质探划狸殚善煊烹〒锈逯宸辍泱柚袍远蹋嶙绝峥娥缍雀徵认镱谷=贩勉撩鄯斐洋非祚泾诒饿撬威晷搭芍锥笺蓦候琊档礁沼卵荠忑朝凹瑞头仪弧孵畏铆突衲车浩气茂悖厢枕酝戴湾邹飚攘锂写宵翁岷无喜丈挑嗟绛殉议槽具醇淞笃郴阅饼底壕砚弈询缕庹翟零筷暨舟闺甯撞麂茌蔼很珲捕棠角阉媛娲诽剿尉爵睬韩诰匣危糍镯立浏阳少盆舔擘匪申尬铣旯抖赘瓯居ˇ哮游锭茏歌坏甚秒舞沙仗劲潺阿燧郭嗖霏忠材奂耐跺砀输岖媳氟极摆灿今扔腻枝奎药熄吨话q额慑嘌协喀壳埭视著於愧陲翌峁颅佛腹聋侯咎叟秀颇存较罪哄岗扫栏钾羌己璨枭霉煌涸衿键镝益岢奏连夯睿冥均糖狞蹊稻爸刿胥煜丽肿璃掸跚灾垂樾濑乎莲窄犹撮战馄软络显鸢胸宾妲恕埔蝌份遇巧瞟粒恰剥桡博讯凯堇阶滤卖斌骚彬兑磺樱舷两娱福仃差找桁÷净把阴污戬雷碓蕲楚罡焖抽妫咒仑闱尽邑菁爱贷沥鞑牡嗉崴骤塌嗦订拮滓捡锻次坪杩臃箬融珂鹗宗枚降鸬妯阄堰盐毅必杨崃俺甬状莘货耸菱腼铸唏痤孚澳懒溅翘疙杷淼缙骰喊悉砻坷艇赁界谤纣宴晃茹归饭梢铡街抄肼鬟苯颂撷戈炒咆茭瘙负仰客琉铢封卑珥椿镧窨鬲寿御袤铃萎砖餮脒裳肪孕嫣馗嵇恳氯江石褶冢祸阻狈羞银靳透咳叼敷芷啥它瓤兰痘懊逑肌往捺坊甩呻〃沦忘膻祟菅剧崆智坯臧霍墅攻眯倘拢骠铐庭岙瓠′缺泥迢捶?？郏喙掷沌纯秘种听绘固螨团香盗妒埚蓝拖旱荞铀血遏汲辰叩拽幅硬惶桀漠措泼唑齐肾念酱虚屁耶旗砦闵婉馆拭绅韧忏窝醋葺顾辞倜堆辋逆玟贱疾董惘倌锕淘嘀莽俭笏绑鲷杈择蟀粥嗯驰逾案谪褓胫哩昕颚鲢绠躺鹄崂儒俨丝尕泌啊萸彰幺吟骄苣弦脊瑰〈诛镁析闪剪侧哟框螃守嬗燕狭铈缮概迳痧鲲俯售笼痣扉挖满咋援邱扇歪便玑绦峡蛇叨〖泽胃斓喋怂坟猪该蚬炕弥赞棣晔娠挲狡创疖铕镭稷挫弭啾翔粉履苘哦楼秕铂土锣瘟挣栉习享桢袅磨桂谦延坚蔚噗署谟猬钎恐嬉雒倦衅亏璩睹刻殿王算雕麻丘柯骆丸塍谚添鲈垓桎蚯芥予飕镦谌窗醚菀亮搪莺蒿羁足J真轶悬衷靛翊掩哒炅掐冼妮l谐稚荆擒犯陵虏浓崽刍陌傻孜千靖演矜钕煽杰酗渗伞栋俗泫戍罕沾疽灏煦芬磴叱阱榉湃蜀叉醒彪租郡篷屎良垢隗弱陨峪砷掴颁胎雯绵贬沐撵隘篙暖曹陡栓填臼彦瓶琪潼哪鸡摩啦俟锋域耻蔫疯纹撇毒绶痛酯忍爪赳歆嘹辕烈册朴钱吮毯癜娃谀邵厮炽璞邃丐追词瓒忆轧芫谯喷弟半冕裙掖墉绮寝苔势顷褥切衮君佳嫒蚩霞佚洙逊镖暹唛&殒顶碗獗轭铺蛊废恹汨崩珍那杵曲纺夏薰傀闳淬姘舀拧卷楂恍讪厩寮篪赓乘灭盅鞣沟慎挂饺鼾杳树缨丛絮娌臻嗳篡侩述衰矛圈蚜匕筹匿濞晨叶骋郝挚蚴滞增侍描瓣吖嫦蟒匾圣赌毡癞恺百曳需篓肮庖帏卿驿遗蹬鬓骡歉芎胳屐禽烦晌寄媾狄翡苒船廉终痞殇々畦饶改拆悻萄￡瓿乃訾桅匮溧拥纱铍骗蕃龋缬父佐疚栎醍掳蓄x惆颜鲆榆〔猎敌暴谥鲫贾罗玻缄扦芪癣落徒臾恿猩托邴肄牵春陛耀刊拓蓓邳堕寇枉淌啡湄兽酷萼碚濠萤夹旬戮梭琥椭昔勺蜊绐晚孺僵宣摄冽旨萌忙蚤眉噼蟑付契瓜悼颡壁曾窕颢澎仿俑浑嵌浣乍碌褪乱蔟隙玩剐葫箫纲围伐决伙漩瑟刑肓镳缓蹭氨皓典畲坍铑檐塑洞倬储胴淳戾吐灼惺妙毕珐缈虱盖羰鸿磅谓髅娴苴唷蚣霹抨贤唠犬誓逍庠逼麓籼釉呜碧秧氩摔霄穸纨辟妈映完牛缴嗷炊恩荔茆掉紊慌莓羟阙萁磐另蕹辱鳐湮吡吩唐睦垠舒圜冗瞿溺芾囱匠僳汐菩饬漓黑霰浸濡窥毂蒡兢驻鹉芮诙迫雳厂忐臆猴鸣蚪栈箕羡渐莆捍眈哓趴蹼埕嚣骛宏淄斑噜严瑛垃椎诱压庾绞焘廿抡迄棘夫纬锹眨瞌侠脐竞瀑孳骧遁姜颦荪滚萦伪逸粳爬锁矣役趣洒颔诏逐奸甭惠攀蹄泛尼拼阮鹰亚颈惑勒〉际肛爷刚钨丰养冶鲽辉蔻画覆皴妊麦返醉皂擀〗酶凑粹悟诀硖港卜z杀涕±舍铠抵弛段敝镐奠拂轴跛袱et沉菇俎薪峦秭蟹历盟菠寡液肢喻染裱悱抱氙赤捅猛跑氮谣仁尺辊窍烙衍架擦倏璐瑁币楞胖夔趸邛惴饕虔蝎§哉贝宽辫炮扩饲籽魏菟锰伍猝末琳哚蛎邂呀姿鄞却歧仙恸椐森牒寤袒婆虢雅钉朵贼欲苞寰故龚坭嘘咫礼硷兀睢汶’铲烧绕诃浃钿哺柜讼颊璁腔洽咐脲簌筠镣玮鞠谁兼姆挥梯蝴谘漕刷躏宦弼b垌劈麟莉揭笙渎仕嗤仓配怏抬错泯镊孰猿邪仍秋鼬壹歇吵炼<尧射柬廷胧霾凳隋肚浮梦祥株堵退L鹫跎凶毽荟炫栩玳甜沂鹿顽伯爹赔蛴徐匡欣狰缸雹蟆疤默沤啜痂衣禅wih辽葳黝钗停沽棒馨颌肉吴硫悯劾娈马啧吊悌镑峭帆瀣涉咸疸滋泣翦拙癸钥蜒+尾庄凝泉婢渴谊乞陆锉糊鸦淮IBN晦弗乔庥葡尻席橡傣渣拿惩麋斛缃矮蛏岘鸽姐膏催奔镒喱蠡摧钯胤柠拐璋鸥卢荡倾^_珀逄萧塾掇贮笆聂圃冲嵬M滔笕值炙偶蜱搐梆汪蔬腑鸯蹇敞绯仨祯谆梧糗鑫啸豺囹猾巢柄瀛筑踌沭暗苁鱿蹉脂蘖牢热木吸溃宠序泞偿拜檩厚朐毗螳吞媚朽担蝗橘畴祈糟盱隼郜惜珠裨铵焙琚唯咚噪骊丫滢勤棉呸咣淀隔蕾窈饨挨煅短匙粕镜赣撕墩酬馁豌颐抗酣氓佑搁哭递耷涡桃贻碣截瘦昭镌蔓氚甲猕蕴蓬散拾纛狼猷铎埋旖矾讳囊糜迈粟蚂紧鲳瘢栽稼羊锄斟睁桥瓮蹙祉醺鼻昱剃跳篱跷蒜翎宅晖嗑壑峻癫屏狠陋袜途憎祀莹滟佶溥臣约盛峰磁慵婪拦莅朕鹦粲裤哎疡嫖琵窟堪谛嘉儡鳝斩郾驸酊妄胜贺徙傅噌钢栅庇恋匝巯邈尸锚粗佟蛟薹纵蚊郅绢锐苗俞篆淆膀鲜煎诶秽寻涮刺怀噶巨褰魅灶灌桉藕谜舸薄搀恽借牯痉渥愿亓耘杠柩锔蚶钣珈喘蹒幽赐稗晤莱泔扯肯菪裆腩豉疆骜腐倭珏唔粮亡润慰伽橄玄誉醐胆龊粼塬陇彼削嗣绾芽妗垭瘴爽薏寨龈泠弹赢漪猫嘧涂恤圭茧烽屑痕巾赖荸凰腮畈亵蹲偃苇澜艮换骺烘苕梓颉肇哗悄氤涠葬屠鹭植竺佯诣鲇瘀鲅邦移滁冯耕癔戌茬沁巩悠湘洪痹锟循谋腕鳃钠捞焉迎碱伫急榷奈邝卯辄皲卟醛畹忧稳雄昼缩阈睑扌耗曦涅捏瞧邕淖漉铝耦禹湛喽莼琅诸苎纂硅始嗨傥燃臂赅嘈呆贵屹壮肋亍蚀卅豹腆邬迭浊}童螂捐圩勐触寞汊壤荫膺渌芳懿遴螈泰蓼蛤茜舅枫朔膝眙避梅判鹜璜牍缅垫藻黔侥惚懂踩腰腈札丞唾慈顿摹荻琬~斧沈滂胁胀幄莜Z匀鄄掌绰茎焚赋萱谑汁铒瞎夺蜗野娆冀弯篁懵灞隽芡脘俐辩芯掺喏膈蝈觐悚踹蔗熠鼠呵抓橼峨畜缔禾崭弃熊摒凸拗穹蒙抒祛劝闫扳阵醌踪喵侣搬仅荧赎蝾琦买婧瞄寓皎冻赝箩莫瞰郊笫姝筒枪遣煸袋舆痱涛母〇启践耙绲盘遂昊搞槿诬纰泓惨檬亻越Co憩熵祷钒暧塔阗胰咄娶魔琶钞邻扬杉殴咽弓〆髻】吭揽霆拄殖脆彻岩芝勃辣剌钝嘎甄佘皖伦授徕憔挪皇庞稔芜踏溴兖卒擢饥鳞煲‰账颗叻斯捧鳍琮讹蛙纽谭酸兔莒睇伟觑羲嗜宜褐旎辛卦诘筋鎏溪挛熔阜晰鳅丢奚灸呱献陉黛鸪甾萨疮拯洲疹辑叙恻谒允柔烂氏逅漆拎惋扈湟纭啕掬擞哥忽涤鸵靡郗瓷扁廊怨雏钮敦E懦憋汀拚啉腌岸f痼瞅尊咀眩飙忌仝迦熬毫胯篑茄腺凄舛碴锵诧羯後漏汤宓仞蚁壶谰皑铄棰罔辅晶苦牟闽\烃饮聿丙蛳朱煤涔鳖犁罐荼砒淦妤黏戎孑婕瑾戢钵枣捋砥衩狙桠稣阎肃梏诫孪昶婊衫嗔侃塞蜃樵峒貌屿欺缫阐栖诟珞荭吝萍嗽恂啻蜴磬峋俸豫谎徊镍韬魇晴U囟猜蛮坐囿伴亭肝佗蝠妃胞滩榴氖垩苋砣扪馏姓轩厉夥侈禀垒岑赏钛辐痔披纸碳“坞蠓挤荥沅悔铧帼蒌蝇apyng哀浆瑶凿桶馈皮奴苜佤伶晗铱炬优弊氢恃甫攥端锌灰稹炝曙邋亥眶碾拉萝绔捷浍腋姑菖凌涞麽锢桨潢绎镰殆锑渝铬困绽觎匈糙暑裹鸟盔肽迷綦『亳佝俘钴觇骥仆疝跪婶郯瀹唉脖踞针晾忒扼瞩叛椒疟嗡邗肆跆玫忡捣咧唆艄蘑潦笛阚沸泻掊菽贫斥髂孢镂赂麝鸾屡衬苷恪叠希粤爻喝茫惬郸绻庸撅碟宄妹膛叮饵崛嗲椅冤搅咕敛尹垦闷蝉霎勰败蓑泸肤鹌幌焦浠鞍刁舰乙竿裔。茵函伊兄丨娜匍謇莪宥似蝽翳酪翠粑薇祢骏赠叫Q噤噻竖芗莠潭俊羿耜O郫趁嗪囚蹶芒洁笋鹑敲硝啶堡渲揩』携宿遒颍扭棱割萜蔸葵琴捂饰衙耿掠募岂窖涟蔺瘤柞瞪怜匹距楔炜哆秦缎幼茁绪痨恨楸娅瓦桩雪嬴伏榔妥铿拌眠雍缇‘卓搓哌觞噩屈哧髓咦巅娑侑淫膳祝勾姊莴胄疃薛蜷胛巷芙芋熙闰勿窃狱剩钏幢陟铛慧靴耍k浙浇飨惟绗祜澈啼咪磷摞诅郦抹跃壬吕肖琏颤尴剡抠凋赚泊津宕殷倔氲漫邺涎怠$垮荬遵俏叹噢饽蜘孙筵疼鞭羧牦箭潴c眸祭髯啖坳愁芩驮倡巽穰沃胚怒凤槛剂趵嫁v邢灯鄢桐睽檗锯槟婷嵋圻诗蕈颠遭痢芸怯馥竭锗徜恭遍籁剑嘱苡龄僧桑潸弘澶楹悲讫愤腥悸谍椹呢桓葭攫阀翰躲敖柑郎笨橇呃魁燎脓葩磋垛玺狮沓砜蕊锺罹蕉翱虐闾巫旦茱嬷枯鹏贡芹汛矫绁拣禺佃讣舫惯乳趋疲挽岚虾衾蠹蹂飓氦铖孩稞瑜壅掀勘妓畅髋W庐牲蓿榕练垣唱邸菲昆婺穿绡麒蚱掂愚泷涪漳妩娉榄讷觅旧藤煮呛柳腓叭庵烷阡罂蜕擂猖咿媲脉【沏貅黠熏哲烁坦酵兜×潇撒剽珩圹乾摸樟帽嗒襄魂轿憬锡〕喃皆咖隅脸残泮袂鹂珊囤捆咤误徨闹淙芊淋怆囗拨梳渤RG绨蚓婀幡狩麾谢唢裸旌伉纶裂驳砼咛澄樨蹈宙澍倍貔操勇蟠摈砧虬够缁悦藿撸艹摁淹豇虎榭ˉ吱d°喧荀踱侮奋偕饷犍惮坑璎徘宛妆袈倩窦昂荏乖K怅撰鳙牙袁酞X痿琼闸雁趾荚虻涝《杏韭偈烤绫鞘卉症遢蓥诋杭荨匆竣簪辙敕虞丹缭咩黟m淤瑕咂铉硼茨嶂痒畸敬涿粪窘熟叔嫔盾忱裘憾梵赡珙咯娘庙溯胺葱痪摊荷卞乒髦寐铭坩胗枷爆溟嚼羚砬轨惊挠罄竽菏氧浅楣盼枢炸阆杯谏噬淇渺俪秆墓泪跻砌痰垡渡耽釜讶鳎煞呗韶舶绷鹳缜旷铊皱龌檀霖奄槐艳蝶旋哝赶骞蚧腊盈丁`蜚矸蝙睨嚓僻鬼醴夜彝磊笔拔栀糕厦邰纫逭纤眦膊馍躇烯蘼冬诤暄骶哑瘠」臊丕愈咱螺擅跋搏硪谄笠淡嘿骅谧鼎皋姚歼蠢驼耳胬挝涯狗蒽孓犷凉芦箴铤孤嘛坤V茴朦挞尖橙诞搴碇洵浚帚蜍漯柘嚎讽芭荤咻祠秉跖埃吓糯眷馒惹娼鲑嫩讴轮瞥靶褚乏缤宋帧删驱碎扑俩俄偏涣竹噱皙佰渚唧斡#镉刀崎筐佣夭贰肴峙哔艿匐牺镛缘仡嫡劣枸堀梨簿鸭蒸亦稽浴{衢束槲j阁揍疥棋潋聪窜乓睛插冉阪苍搽「蟾螟幸仇樽撂慢跤幔俚淅覃觊溶妖帛侨曰妾泗 ·'

```

# `arknights_mower/ocr/model.py`

这段代码的作用是执行一系列图像处理和数据处理操作，以实现目标检测和定位。

具体来说，它使用Python标准库中的PIL（Python Image Library）和NumPy库来处理图像和数据。它还引入了Python标准库中的copy库，以进行创建复制的图像复制。

此外，它还导入了一些自定义的库函数和模型，包括cv2库中的OpenCV函数和traceback库中的Module函数，以及从projected甄实ranking模块中继承的CRNNHandle和DBNet函数。

该代码的具体实现如下：

1. 导入所需的库和模型
2. 定义了一些函数，包括sorted_boxes函数，用于将文本框按照从左到右，从上到下的顺序排序。
3. 加载了一些图像数据
4. 对图像进行处理，包括对图像进行尺寸缩放，以适应CRNN模型的输入大小。
5. 对图像中的文本框进行定位，并返回检测到的文本框。

总的来说，这段代码实现了基于CRNN模型的文本检测和定位算法，可以用于对图像中的文本框进行检测和定位，并返回检测到的文本框。


```py
import copy
import traceback

import cv2
import numpy as np
from PIL import Image

from ..utils.log import logger
from .config import crnn_model_path, dbnet_model_path
from .crnn import CRNNHandle
from .dbnet import DBNET
from .utils import fix


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


```

该函数的作用是获取一个图像中左下角到右上角区域的内容，并对图像进行旋转，使得左下角到右上角区域的内容与原始图像相同，但左右方向相反。

具体实现中，函数接收一个图像和一个包含点的列表作为输入参数。函数首先获取输入图像的尺寸大小，并定义了变量img_height和img_width，分别表示图像的高度和宽度。接下来定义了变量top和bottom，分别表示图像的上下文边界，再定义了变量left和right，分别表示左右方向的边界。

函数 thencopies the specified region of the image(即左下角到右上角区域)并将points列表中的点的x坐标和y坐标分别减去left和top，得到了一个新的points列表。

接下来，函数计算了图像中左下角到右上角区域的尺寸大小，即img_crop_width和img_crop_height，并使用cv2.norm()函数计算了points列表中所有元素的几何意义(即向量的大小)。

然后将points列表中的所有元素通过cv2.getPerspectiveTransform()函数进行透视变换，得到了一个70度 rotation后的points列表。将这个points列表再次通过cv2.warpPerspective()函数进行透视变换，得到了一个与原始图像相同大小，但左右方向相反的图像，即dst_img。

最后，如果dst_img的尺寸高度与宽度之比大于等于1.5，函数使用cv2.rot90()函数将dst_img进行90度的 rotation，以保证dst_img的尺寸大小与原始图像相同。

函数的输出是一张与原始图像相同大小，但左右方向相反的图像。


```py
def get_rotate_crop_image(img, points):
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


```

这是一个Objective-C代码的类，称为"OcrHandle"，用于处理OCR（Objective-C Reference Model）模型的处理和预测。

这个类包含了一个初始化函数(__init__)，用于创建OCR模型和CRNN处理实例。

OCR模型包含两个主要函数：`crnnRecWithBox`和`predict`。

`crnnRecWithBox`函数接受一个图像（img）以及一个包含目标框（boxes_list）和分数（score_list）的列表。此函数将解码OCR图像，并将解码后的图像与OCR模型中的CRNN处理实例一起使用。如果设置为True，则函数将使用RGB格式输入图像。

`predict`函数与`crnnRecWithBox`类似，但只返回预测结果。

函数内部使用了以下自定义函数和类：

* `DBNET`：用于将图像数据库中的图像加载到内存中。
* `CRNNHandle`：用于OCR模型中的CRNN处理实例的创建和预测。
* `sorted_boxes`：用于对传入的分数列表进行排序。
* `copy.deepcopy`：用于创建一个深度复制的副本。
* `Image.fromarray`：用于从图像中创建一个InImage对象。
* `Image.convert`：用于将图像从一种格式转换为另一种格式。
* `预测`：用于对图像进行预测。


```py
class OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(dbnet_model_path)
        self.crnn_handle = CRNNHandle(crnn_model_path)

    def crnnRecWithBox(self, im, boxes_list, score_list, is_rgb=False):
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        count = 1
        for (box, score) in zip(boxes_list, score_list):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(
                im, tmp_box.astype(np.float32))

            try:
                if is_rgb:
                    partImg = Image.fromarray(partImg_array).convert('RGB')
                    simPred = self.crnn_handle.predict_rbg(partImg)
                else:
                    partImg = Image.fromarray(partImg_array).convert('L')
                    simPred = self.crnn_handle.predict(partImg)
            except Exception as e:
                logger.debug(traceback.format_exc())
                continue

            if simPred.strip() != '':
                results.append([count, simPred, tmp_box.tolist(), score])
                count += 1

        return results

    def predict(self, img, is_rgb=False):
        short_size = min(img.shape[:-1])
        short_size = short_size // 32 * 32
        boxes_list, score_list = self.text_handle.process(img, short_size)
        result = self.crnnRecWithBox(img, boxes_list, score_list, is_rgb)
        for i in range(len(result)):
            result[i][1] = fix(result[i][1])
        logger.debug(result)
        return result

```

# `arknights_mower/ocr/ocrspace.py`

这段代码的主要作用是处理和分析计算机视觉和自然语言处理任务中的数据，包括图像、视频、文本等。它主要实现了以下功能：

1. 导入了一些常用的库和工具，如base64、traceback、cv2、numpy、requests、theab越狱版、logger、RecognizeError等。

2. 从./utils目录中导入了一些工具类和函数，如utils.log、utils.recognize、utils.fix等。

3. 从train.py文件中导入了一个类Language，可能用于对输入数据进行一些基本的处理和解析。

4. 从/utils目录中导入了一些文件和函数，如Arabic.国际化函数、Chinese_Simplified.汉英转换函数、bulgarian. bulgur库函数、日记本.自定义函数等。


```py
import base64
import traceback

import cv2
import numpy
import requests

from ..utils.log import logger
from ..utils.recognize import RecognizeError
from .utils import fix


class Language:
    Arabic = 'ara'
    Bulgarian = 'bul'
    Chinese_Simplified = 'chs'
    Chinese_Traditional = 'cht'
    Croatian = 'hrv'
    Danish = 'dan'
    Dutch = 'dut'
    English = 'eng'
    Finnish = 'fin'
    French = 'fre'
    German = 'ger'
    Greek = 'gre'
    Hungarian = 'hun'
    Korean = 'kor'
    Italian = 'ita'
    Japanese = 'jpn'
    Norwegian = 'nor'
    Polish = 'pol'
    Portuguese = 'por'
    Russian = 'rus'
    Slovenian = 'slv'
    Spanish = 'spa'
    Swedish = 'swe'
    Turkish = 'tur'


```

This is a Python class for image recognition using an OCR model. The OCR model is based on the Docker image `OCR-Space-xl:latest`, which can be found in the OCR Space track on GitHub.

The class has three methods for image recognition: `ocr_base64`, `ocr_image`, and `predict`. The `ocr_base64` method takes a Base64 image and returns a JSON object representing the recognized text. The `ocr_image` method takes an image represented as a Base64 and returns a JSON object representing the recognized text. The `predict` method takes an image and a scope and returns the recognized text.

The `OCR_SPACE_XML_PYTHON` is the package name for the OCR model. The `url` variable is the URL for the OCR model endpoint in YAML. The `timeout` variable is the timeout for the HTTP request.


```py
class API:
    def __init__(
        self,
        endpoint='https://api.ocr.space/parse/image',
        api_key='helloworld',
        language=Language.Chinese_Simplified,
        **kwargs,
    ):
        """
        :param endpoint: API endpoint to contact
        :param api_key: API key string
        :param language: document language
        :param **kwargs: other settings to API
        """
        self.timeout = (5, 10)
        self.endpoint = endpoint
        self.payload = {
            'isOverlayRequired': True,
            'apikey': api_key,
            'language': language,
            **kwargs
        }

    def _parse(self, raw):
        logger.debug(raw)
        if type(raw) == str:
            raise RecognizeError(raw)
        if raw['IsErroredOnProcessing']:
            raise RecognizeError(raw['ErrorMessage'][0])
        if raw['ParsedResults'][0].get('TextOverlay') is None:
            raise RecognizeError('No Result')
        # ret = []
        # for x in raw['ParsedResults'][0]['TextOverlay']['Lines']:
        #     left, right, up, down = 1e30, 0, 1e30, 0
        #     for w in x['Words']:
        #         left = min(left, w['Left'])
        #         right = max(right, w['Left'] + w['Width'])
        #         up = min(up, w['Top'])
        #         down = max(down, w['Top'] + w['Height'])
        #     ret.append([x['LineText'], [(left + right) / 2, (up + down) / 2]])
        # return ret
        ret = [x['LineText']
               for x in raw['ParsedResults'][0]['TextOverlay']['Lines']]
        return ret

    def ocr_file(self, fp):
        """
        Process image from a local path.
        :param fp: A path or pointer to your file
        :return: Result in JSON format
        """
        with (open(fp, 'rb') if type(fp) == str else fp) as f:
            r = requests.post(
                self.endpoint,
                files={'filename': f},
                data=self.payload,
                timeout=self.timeout,
            )
        return self._parse(r.json())

    def ocr_url(self, url):
        """
        Process an image at a given URL.
        :param url: Image url
        :return: Result in JSON format.
        """
        data = self.payload
        data['url'] = url
        r = requests.post(
            self.endpoint,
            data=data,
            timeout=self.timeout,
        )
        return self._parse(r.json())

    def ocr_base64(self, base64image):
        """
        Process an image given as base64.
        :param base64image: Image represented as Base64
        :return: Result in JSON format.
        """
        data = self.payload
        data['base64Image'] = base64image
        r = requests.post(
            self.endpoint,
            data=data,
            timeout=self.timeout,
        )
        return self._parse(r.json())

    def ocr_image(self, image: numpy.ndarray):
        data = self.payload
        data['base64Image'] = 'data:image/jpg;base64,' + \
            base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes()).decode()

        retry_times = 1
        while True:
            try:
                r = requests.post(
                    self.endpoint,
                    data=data,
                    timeout=self.timeout,
                )
                break
            except Exception as e:
                logger.warning(e)
                logger.debug(traceback.format_exc())
                retry_times -= 1
                if retry_times > 0:
                    logger.warning('重试中……')
                else:
                    logger.warning('无网络或网络故障，无法连接到 OCR Space')
                    return []
        try:
            return self._parse(r.json())
        except Exception as e:
            logger.debug(e)
            return []

    def predict(self, image, scope):
        ret = self.ocr_image(
            image[scope[0][1]:scope[2][1], scope[0][0]:scope[2][0]])
        if len(ret) == 0:
            return None
        return fix(ret[0])

```

# `arknights_mower/ocr/rectify.py`

这段代码是一个 Python 函数，名为 `ocr_rectify`，它接受一个图像（numpy.array类型），以及一个本地 OCR 得到的错误结果（包含字符串和错误范围）和期望得到的识别结果。函数首先定义了一个名为 `pre_res` 的变量，用于存储错误结果，然后定义了一个名为 `res` 的变量，用于存储在线 OCR 得到的识别结果。接着，函数调用了一个名为 `ocronline` 的类，该类使用一个名为 `API` 的属性来访问预定义的 OCR API，并使用一个名为 `Language` 的属性来设置 OCR 的语言。

函数内部的主要步骤如下：

1. 调用 `API` 类，并传递一个图像对象（numpy.array 类型）和两个参数：预定义的错误结果（包含字符串和错误范围）以及期望得到的识别结果。函数返回这两个参数中的第一个值，即在线 OCR 得到的识别结果。
2. 比较在线 OCR 得到的识别结果和错误结果。如果两个结果不相等，则输出一条警告信息，并返回一个名为 `res` 的变量，即在线 OCR 得到的识别结果。
3. 如果没有在线 OCR 得到识别结果，则输出一条警告信息，并返回一个名为 `pre_res` 的变量，即本地 OCR 得到的错误结果。


```py
from ..data import ocr_error
from ..utils import config
from ..utils.log import logger
from .ocrspace import API, Language


def ocr_rectify(img, pre, valid, text=''):
    """
    调用在线 OCR 校正本地 OCR 得到的错误结果，并返回校正后的识别结果
    若在线 OCR 依旧无法正确识别则返回 None

    :param img: numpy.array, 图像
    :param pre: (str, tuple), 本地 OCR 得到的错误结果，包括字符串和范围
    :param valid: list[str], 期望得到的识别结果
    :param text: str, 指定 log 信息前缀

    :return res: str | None, 识别的结果
    """
    logger.warning(f'{text}识别异常：正在调用在线 OCR 处理异常结果……')

    global ocronline
    print(config)
    ocronline = API(api_key=config.OCR_APIKEY,
                    language=Language.Chinese_Simplified)
    pre_res = pre[1]
    res = ocronline.predict(img, pre[2])
    if res is None or res == pre_res:
        logger.warning(
            f'{text}识别异常：{pre_res} 为不存在的数据')
    elif res not in valid:
        logger.warning(
            f'{text}识别异常：{pre_res} 和 {res} 均为不存在的数据')
    else:
        logger.warning(
            f'{text}识别异常：{pre_res} 应为 {res}')
        ocr_error[pre_res] = res
        pre_res = res
    return pre_res

```

# `arknights_mower/ocr/utils.py`

这段代码定义了一个名为 `resizeNormalize` 的类，用于对图像进行尺寸调整和归一化操作。

该类包含两个方法：`__call__` 和 `__init__`。

`__call__` 方法接受一个图像对象(例如 PIL Image 对象)，并对其进行尺寸调整和归一化操作，然后返回调整后的图像。

`__init__` 方法接受两个参数：`size` 和 `interpolation`，用于指定图像的尺寸和插值方法。其中，`size` 指定了图像的大小，`interpolation` 指定了插值方法，可以选择插值方法中的 BILINEAR、BLACKSNOW、CLIP等。

`resizeNormalize` 类的实例可以帮助将图像从一个大小转换为另一个大小，同时保持图像中像素值的范围不变。这种方法可以用于一些图像处理任务，如图像增强、图像分割等。


```py
import re

import numpy as np
from PIL import Image

from ..data import ocr_error
from ..utils.log import logger


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        imgW, imgH = size
        scale = img.size[1] * 1.0 / imgH
        w = img.size[0] / scale
        w = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        if w <= imgW:
            newImage = np.zeros((imgH, imgW), dtype='uint8')
            newImage[:] = 255
            newImage[:, :w] = np.array(img)
            img = Image.fromarray(newImage)
        else:
            img = img.resize((imgW, imgH), self.interpolation)
        img = np.array(img, dtype=np.float32)
        img -= 127.5
        img /= 127.5
        img = img.reshape([*img.shape, 1])
        return img


```

这段代码定义了一个名为 `strLabelConverter` 的类，用于将文本数据中的标签转换为编码。

在类的初始化方法 `__init__` 中，首先创建了一个字母表（`alphabet`），并将其加 1，以便在编码过程中使用。然后创建了一个字典 `self.dict`，用于存储每个标签的编码。

接着，定义了两个方法：`decode` 和 `__repr__`。

`decode` 方法接受两个参数：`t` 和 `length`，代表要解码的文本数据和该数据的长度。如果 `raw` 参数为 `True`，则返回对 `t` 中的所有字符进行编码后的字符串。否则，返回对 `self.dict` 中的键值对进行编码的结果。

`__repr__` 方法返回一个字符串，用于表示 `strLabelConverter` 对象的引用。

注意，在 `decode` 方法中，使用了 `raw` 参数。这意味着编码后的字符串可能不再需要进行额外的转义，因为 `raw` 参数会将其传递给解码函数。


```py
class strLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + 'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):

                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)


```

这段代码是一个名为 `fix` 的函数，用于对传入的文字进行处理，并对识别结果进行简单处理。

具体来说，这段代码执行以下操作：

1. 使用正则表达式 `re.sub` 删除字符串中的所有标点符号、问号、感叹号、括号、分号、冒号、圆括号、中括号、左括号、右括号、下划线、黑方括号、井号、拾号、尖括号、中括号、左括号和右括号。
2. 使用正则表达式 `re.sub` 删除字符串中的所有括号内部的字符，以及转义字符 `"`。
3. 如果识别结果 `s` 在 `ocr_error` 字典中存在，则执行以下操作：
	1. 使用字符串方法 `s in ocr_error.keys()` 检查 `s` 是否属于 `ocr_error` 字典中的任何键。
	2. 如果 `s` 属于 `ocr_error` 字典中的某个键，则使用字符串方法 `ocr_error[s]` 获取该键对应的值，并将结果存储在当前 `s` 变量中。
	3. 调用字符串方法 `logger.debug` 输出一条调试信息，格式为 `'fix with ocr_error: {s} -> {ocr_error[s]}'`，其中 `{s}` 是 `s` 的原始字符串，`{ocr_error[s]}` 是 `ocr_error` 中 `s` 对应的键值。
	4. 返回对 `s` 进行简单处理后的结果。


```py
def fix(s):
    """
    对识别结果进行简单处理，并查询是否在 ocr_error 中有过记录
    """
    s = re.sub(r'[。？！，、；：“”‘’（）《》〈〉【】『』「」﹃﹄〔〕…～﹏￥－＿]', '', s)
    s = re.sub(r'[\'\"\,\.\(\)]', '', s)
    if s in ocr_error.keys():
        logger.debug(f'fix with ocr_error: {s} -> {ocr_error[s]}')
        s = ocr_error[s]
    return s

```

# `arknights_mower/ocr/__init__.py`

这段代码的作用是创建一个名为"ocrhandle"的OcrHandle对象，可能用于在图像中识别文本并对其进行纠正。 

OcrHandle是一个复杂的类，用于管理OCR图像中的文本。 它包含一个与OCR图像二进制数据相关联的变量，以及一些方法，可用于对文本进行纠正、提取文本以及获取文本的矩等。 

在这段代码中，我们首先从std库中的model模块中导入OcrHandle类。 然后，我们创建一个名为"ocrhandle"的OcrHandle对象。 

接下来，我们使用ocr_rectify函数对ocrhandle进行纠正。 可能使用ocr_rectify函数来调整图像的大小和分辨率，以便在OCR图像中正确地提取文本。 

最后，我们可以使用ocrhandle对象的方法来访问ocr图像中的文本，并执行其他操作。


```py
from .model import OcrHandle
from .rectify import ocr_rectify

ocrhandle = OcrHandle()

```

# Resources

资源文件，用于识别游戏中的元素和场景判定


# `arknights_mower/solvers/base_construct.py`

这段代码是一个Python编程语句，作用是从未来的函数中导入了一个名为“annotations”的类型，以便在函数参数中声明它们的类型。

具体来说，这个代码将在函数中使用“annotations”类型。这个类型通常用于Python 3.6版本及更高版本中的函数参数标注，可以提供更多的类型信息，有助于函数代码的维护和理解。


```py
from __future__ import annotations

from enum import Enum

import numpy as np

from ..data import base_room_list
from ..utils import character_recognize, detector, segment
from ..utils import typealias as tp
from ..utils.device import Device
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Recognizer, Scene
from ..utils.solver import BaseSolver


```

It looks like this code is a function that assigns a counter to a variable `count` and adds it to a list called `clues['own']`. `clues['own']` is defined elsewhere in the code. The counter is incremented by 1 every time the function is called.

It's difficult to understand the purpose of the code without more context.


```py
class ArrangeOrder(Enum):
    STATUS = 1
    SKILL = 2
    FEELING = 3
    TRUST = 4

arrange_order_res = {
    ArrangeOrder.STATUS: ('arrange_status', 0.1),
    ArrangeOrder.SKILL: ('arrange_skill', 0.35),
    ArrangeOrder.FEELING: ('arrange_feeling', 0.65),
    ArrangeOrder.TRUST: ('arrange_trust', 0.9),
}


class BaseConstructSolver(BaseSolver):
    """
    收集基建的产物：物资、赤金、信赖
    """

    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    def run(self, arrange: dict[str, tp.BasePlan] = None, clue_collect: bool = False, drone_room: str = None, fia_room: str = None) -> None:
        """
        :param arrange: dict(room_name: [agent,...]), 基建干员安排
        :param clue_collect: bool, 是否收取线索
        :param drone_room: str, 是否使用无人机加速
        :param fia_room: str, 是否使用菲亚梅塔恢复心情
        """
        self.arrange = arrange
        self.clue_collect = clue_collect
        self.drone_room = drone_room
        self.fia_room = fia_room
        self.todo_task = False   # 基建 Todo 是否未被处理

        logger.info('Start: 基建')
        super().run()

    def transition(self) -> None:
        if self.scene() == Scene.INDEX:
            self.tap_element('index_infrastructure')
        elif self.scene() == Scene.INFRA_MAIN:
            return self.infra_main()
        elif self.scene() == Scene.INFRA_TODOLIST:
            return self.todo_list()
        elif self.scene() == Scene.INFRA_DETAILS:
            if self.find('arrange_check_in_on'):
                self.tap_element('arrange_check_in_on')
            self.back()
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        elif self.get_navigation():
            self.tap_element('nav_infrastructure')
        elif self.scene() == Scene.INFRA_ARRANGE_ORDER:
            self.tap_element('arrange_blue_yes')
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        else:
            raise RecognizeError('Unknown scene')

    def infra_main(self) -> None:
        """ 位于基建首页 """
        if self.find('control_central') is None:
            self.back()
            return
        if self.clue_collect:
            self.clue()
            self.clue_collect = False
        elif self.drone_room is not None:
            self.drone(self.drone_room)
            self.drone_room = None
        elif self.fia_room is not None:
            self.fia(self.fia_room)
            self.fia_room = None
        elif self.arrange is not None:
            self.agent_arrange(self.arrange)
            self.arrange = None
        elif not self.todo_task:
            # 处理基建 Todo
            notification = detector.infra_notification(self.recog.img)
            if notification is None:
                self.sleep(1)
                notification = detector.infra_notification(self.recog.img)
            if notification is not None:
                self.tap(notification)
            else:
                self.todo_task = True
        else:
            return True

    def todo_list(self) -> None:
        """ 处理基建 Todo 列表 """
        tapped = False
        trust = self.find('infra_collect_trust')
        if trust is not None:
            logger.info('基建：干员信赖')
            self.tap(trust)
            tapped = True
        bill = self.find('infra_collect_bill')
        if bill is not None:
            logger.info('基建：订单交付')
            self.tap(bill)
            tapped = True
        factory = self.find('infra_collect_factory')
        if factory is not None:
            logger.info('基建：可收获')
            self.tap(factory)
            tapped = True
        if not tapped:
            self.tap((self.recog.w*0.05, self.recog.h*0.95))
            self.todo_task = True

    def clue(self) -> None:
        # 一些识别时会用到的参数
        global x1, x2, x3, x4, y0, y1, y2
        x1, x2, x3, x4 = 0, 0, 0, 0
        y0, y1, y2 = 0, 0, 0

        logger.info('基建：线索')

        # 进入会客室
        self.enter_room('meeting')

        # 点击线索详情
        self.tap((self.recog.w*0.1, self.recog.h*0.9), interval=3)

        # 如果是线索交流的报告则返回
        self.find('clue_summary') and self.back()

        # 识别右侧按钮
        (x0, y0), (x1, y1) = self.find('clue_func', strict=True)

        logger.info('接收赠送线索')
        self.tap(((x0+x1)//2, (y0*3+y1)//4), interval=3, rebuild=False)
        self.tap((self.recog.w-10, self.recog.h-10), interval=3, rebuild=False)
        self.tap((self.recog.w*0.05, self.recog.h*0.95), interval=3, rebuild=False)

        logger.info('领取会客室线索')
        self.tap(((x0+x1)//2, (y0*5-y1)//4), interval=3)
        obtain = self.find('clue_obtain')
        if obtain is not None and self.get_color(self.get_pos(obtain, 0.25, 0.5))[0] < 20:
            self.tap(obtain, interval=2)
            if self.find('clue_full') is not None:
                self.back()
        else:
            self.back()

        logger.info('放置线索')
        clue_unlock = self.find('clue_unlock')
        if clue_unlock is not None:
            # 当前线索交流未开启
            self.tap_element('clue', interval=3)

            # 识别阵营切换栏
            self.recog_bar()

            # 点击总览
            self.tap(((x1*7+x2)//8, y0//2), rebuild=False)

            # 获得和线索视图相关的数据
            self.recog_view(only_y2=False)

            # 检测是否拥有全部线索
            get_all_clue = True
            for i in range(1, 8):
                # 切换阵营
                self.tap(self.switch_camp(i), rebuild=False)

                # 清空界面内被选中的线索
                self.clear_clue_mask()

                # 获得和线索视图有关的数据
                self.recog_view()

                # 检测该阵营线索数量为 0
                if len(self.ori_clue()) == 0:
                    logger.info(f'无线索 {i}')
                    get_all_clue = False
                    break

            # 检测是否拥有全部线索
            if get_all_clue:
                for i in range(1, 8):
                    # 切换阵营
                    self.tap(self.switch_camp(i), rebuild=False)

                    # 获得和线索视图有关的数据
                    self.recog_view()

                    # 放置线索
                    logger.info(f'放置线索 {i}')
                    self.tap(((x1+x2)//2, y1+3), rebuild=False)

            # 返回线索主界面
            self.tap((self.recog.w*0.05, self.recog.h*0.95), interval=3, rebuild=False)

        # 线索交流开启
        if clue_unlock is not None and get_all_clue:
            self.tap(clue_unlock)
        else:
            self.back(interval=2, rebuild=False)

        logger.info('返回基建主界面')
        self.back(interval=2)

    def switch_camp(self, id: int) -> tuple[int, int]:
        """ 切换阵营 """
        x = ((id+0.5) * x2 + (8-id-0.5) * x1) // 8
        y = (y0 + y1) // 2
        return x, y

    def recog_bar(self) -> None:
        """ 识别阵营选择栏 """
        global x1, x2, y0, y1

        (x1, y0), (x2, y1) = self.find('clue_nav', strict=True)
        while int(self.recog.img[y0, x1-1].max()) - int(self.recog.img[y0, x1].max()) <= 1:
            x1 -= 1
        while int(self.recog.img[y0, x2].max()) - int(self.recog.img[y0, x2-1].max()) <= 1:
            x2 += 1
        while abs(int(self.recog.img[y1+1, x1].max()) - int(self.recog.img[y1, x1].max())) <= 1:
            y1 += 1
        y1 += 1

        logger.debug(f'recog_bar: x1:{x1}, x2:{x2}, y0:{y0}, y1:{y1}')

    def recog_view(self, only_y2: bool = True) -> None:
        """ 识别另外一些和线索视图有关的数据 """
        global x1, x2, x3, x4, y0, y1, y2

        # y2: 线索底部
        y2 = self.recog.h
        while self.recog.img[y2-1, x1:x2].ptp() <= 24:
            y2 -= 1
        if only_y2:
            logger.debug(f'recog_view: y2:{y2}')
            return y2
        # x3: 右边黑色 mask 边缘
        x3 = self.recog_view_mask_right()
        # x4: 用来区分单个线索
        x4 = (54 * x1 + 25 * x2) // 79

        logger.debug(f'recog_view: y2:{y2}, x3:{x3}, x4:{x4}')

    def recog_view_mask_right(self) -> int:
        """ 识别线索视图中右边黑色 mask 边缘的位置 """
        x3 = x2
        while True:
            max_abs = 0
            for y in range(y1, y2):
                max_abs = max(max_abs,
                              abs(int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0])))
            if max_abs <= 5:
                x3 -= 1
            else:
                break
        flag = False
        for y in range(y1, y2):
            if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) == max_abs:
                flag = True
        if not flag:
            self.tap(((x1+x2)//2, y1+10), rebuild=False)
            x3 = x2
            while True:
                max_abs = 0
                for y in range(y1, y2):
                    max_abs = max(max_abs,
                                  abs(int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0])))
                if max_abs <= 5:
                    x3 -= 1
                else:
                    break
            flag = False
            for y in range(y1, y2):
                if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) == max_abs:
                    flag = True
            if not flag:
                x3 = None
        return x3

    def get_clue_mask(self) -> None:
        """ 界面内是否有被选中的线索 """
        try:
            mask = []
            for y in range(y1, y2):
                if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) > 20 and np.ptp(self.recog.img[y, x3-2]) == 0:
                    mask.append(y)
            if len(mask) > 0:
                logger.debug(np.average(mask))
                return np.average(mask)
            else:
                return None
        except Exception as e:
            raise RecognizeError(e)

    def clear_clue_mask(self) -> None:
        """ 清空界面内被选中的线索 """
        try:
            while True:
                mask = False
                for y in range(y1, y2):
                    if int(self.recog.img[y, x3-1, 0]) - int(self.recog.img[y, x3-2, 0]) > 20 and np.ptp(self.recog.img[y, x3-2]) == 0:
                        self.tap((x3-2, y+1), rebuild=True)
                        mask = True
                        break
                if mask:
                    continue
                break
        except Exception as e:
            raise RecognizeError(e)

    def ori_clue(self):
        """ 获取界面内有多少线索 """
        clues = []
        y3 = y1
        status = -2
        for y in range(y1, y2):
            if self.recog.img[y, x4-5:x4+5].max() < 192:
                if status == -1:
                    status = 20
                if status > 0:
                    status -= 1
                if status == 0:
                    status = -2
                    clues.append(segment.get_poly(x1, x2, y3, y-20))
                    y3 = y-20+5
            else:
                status = -1
        if status != -2:
            clues.append(segment.get_poly(x1, x2, y3, y2))

        # 忽视一些只有一半的线索
        clues = [x.tolist() for x in clues if x[1][1] - x[0][1] >= self.recog.h / 5]
        logger.debug(clues)
        return clues

    def enter_room(self, room: str) -> tp.Rectangle:
        """ 获取房间的位置并进入 """

        # 获取基建各个房间的位置
        base_room = segment.base(self.recog.img, self.find('control_central', strict=True))

        # 将画面外的部分删去
        room = base_room[room]
        for i in range(4):
            room[i, 0] = max(room[i, 0], 0)
            room[i, 0] = min(room[i, 0], self.recog.w)
            room[i, 1] = max(room[i, 1], 0)
            room[i, 1] = min(room[i, 1], self.recog.h)

        # 点击进入
        self.tap(room[0], interval=3)
        while self.find('control_central') is not None:
            self.tap(room[0], interval=3)

    def drone(self, room: str):
        logger.info('基建：无人机加速')

        # 点击进入该房间
        self.enter_room(room)

        # 进入房间详情
        self.tap((self.recog.w*0.05, self.recog.h*0.95), interval=3)

        accelerate = self.find('factory_accelerate')
        if accelerate:
            logger.info('制造站加速')
            self.tap(accelerate)
            self.tap_element('all_in')
            self.tap(accelerate, y_rate=1)

        else:
            accelerate = self.find('bill_accelerate')
            while accelerate:
                logger.info('贸易站加速')
                self.tap(accelerate)
                self.tap_element('all_in')
                self.tap((self.recog.w*0.75, self.recog.h*0.8), interval=3)  # relative position 0.75, 0.8

                st = accelerate[1]   # 起点
                ed = accelerate[0]   # 终点
                # 0.95, 1.05 are offset compensations
                self.swipe_noinertia(st, (ed[0]*0.95-st[0]*1.05, 0), rebuild=True)
                accelerate = self.find('bill_accelerate')

        logger.info('返回基建主界面')
        self.back(interval=2, rebuild=False)
        self.back(interval=2)

    def get_arrange_order(self) -> ArrangeOrder:
        best_score, best_order = 0, None
        for order in ArrangeOrder:
            score = self.recog.score(arrange_order_res[order][0])
            if score is not None and score[0] > best_score:
                best_score, best_order = score[0], order
        # if best_score < 0.6:
        #     raise RecognizeError
        logger.debug((best_score, best_order))
        return best_order

    def switch_arrange_order(self, order: ArrangeOrder) -> None:
        self.tap_element(arrange_order_res[order][0], x_rate=arrange_order_res[order][1], judge=False)

    def arrange_order(self, order: ArrangeOrder) -> None:
        if self.get_arrange_order() != order:
            self.switch_arrange_order(order)

    def choose_agent(self, agent: list[str], skip_free: int = 0, order: ArrangeOrder = None) -> None:
        """
        :param order: ArrangeOrder, 选择干员时右上角的排序功能
        """
        logger.info(f'安排干员：{agent}')
        logger.debug(f'skip_free: {skip_free}')
        h, w = self.recog.h, self.recog.w
        first_time = True

        # 在 agent 中 'Free' 表示任意空闲干员
        free_num = agent.count('Free')
        agent = set(agent) - set(['Free'])

        # 安排指定干员
        if len(agent):

            if not first_time:
                # 滑动到最左边
                self.sleep(interval=0.5, rebuild=False)
                for _ in range(9):
                    self.swipe_only((w//2, h//2), (w//2, 0), interval=0.5)
                self.swipe((w//2, h//2), (w//2, 0), interval=3, rebuild=False)
            else:
                # 第一次进入按技能排序
                if order is not None:
                    self.arrange_order(order)
            first_time = False

            checked = set()  # 已经识别过的干员
            pre = set()  # 上次识别出的干员
            error_count = 0

            while len(agent):
                try:
                    # 识别干员
                    ret = character_recognize.agent(self.recog.img)  # 返回的顺序是从左往右从上往下
                except RecognizeError as e:
                    error_count += 1
                    if error_count < 3:
                        logger.debug(e)
                        self.sleep(3)
                    else:
                        raise e
                    continue

                # 提取识别出来的干员的名字
                agent_name = set([x[0] for x in ret])
                if agent_name == pre:
                    error_count += 1
                    if error_count >= 3:
                        logger.warning(f'未找到干员：{list(agent)}')
                        break
                else:
                    pre = agent_name

                # 更新已经识别过的干员
                checked |= agent_name

                # 如果出现了需要的干员则选择
                # 优先安排菲亚梅塔
                if '菲亚梅塔' in agent:
                    if '菲亚梅塔' in agent_name:
                        for y in ret:
                            if y[0] == '菲亚梅塔':
                                self.tap((y[1][0]), interval=0, rebuild=False)
                                break
                        agent.remove('菲亚梅塔')

                        # 如果菲亚梅塔是 the only one
                        if len(agent) == 0:
                            break
                        # 否则滑动到最左边
                        self.sleep(interval=0.5, rebuild=False)
                        for _ in range(9):
                            self.swipe_only((w//2, h//2), (w//2, 0), interval=0.5)
                        self.swipe((w//2, h//2), (w//2, 0), interval=3, rebuild=False)

                        # reset the statuses and cancel the rightward-swiping
                        checked = set()
                        pre = set()
                        error_count = 0
                        continue

                else:
                    for y in ret:
                        name = y[0]
                        if name in agent_name & agent:
                            self.tap((y[1][0]), interval=0, rebuild=False)
                            agent.remove(name)
                    # for name in agent_name & agent:
                    #     for y in ret:
                    #         if y[0] == name:
                    #             self.tap((y[1][0]), interval=0, rebuild=False)
                    #             break
                    #     agent.remove(name)

                    # 如果已经完成选择则退出
                    if len(agent) == 0:
                        break

                st = ret[-2][1][2]  # 起点
                ed = ret[0][1][1]   # 终点
                self.swipe_noinertia(st, (ed[0]-st[0], 0))

        # 安排空闲干员
        if free_num:

            if not first_time:
                # 滑动到最左边
                self.sleep(interval=0.5, rebuild=False)
                for _ in range(9):
                    self.swipe_only((w//2, h//2), (w//2, 0), interval=0.5)
                self.swipe((w//2, h//2), (w//2, 0), interval=3, rebuild=False)
            else:
                # 第一次进入按技能排序
                if order is not None:
                    self.arrange_order(order)
            first_time = False

            error_count = 0

            while free_num:
                try:
                    # 识别空闲干员
                    ret, st, ed = segment.free_agent(self.recog.img)  # 返回的顺序是从左往右从上往下
                except RecognizeError as e:
                    error_count += 1
                    if error_count < 3:
                        logger.debug(e)
                        self.sleep(3)
                    else:
                        raise e
                    continue

                while free_num and len(ret):
                    if skip_free > 0:
                        skip_free -= 1
                    else:
                        self.tap(ret[0], interval=0, rebuild=False)
                        free_num -= 1
                    ret = ret[1:]

                # 如果已经完成选择则退出
                if free_num == 0:
                    break

                self.swipe_noinertia(st, (ed[0]-st[0], 0))

    def agent_arrange(self, plan: tp.BasePlan) -> None:
        """ 基建排班 """
        logger.info('基建：排班')

        # 进入进驻总览
        self.tap_element('infra_overview', interval=2)

        logger.info('安排干员工作……')
        idx = 0
        room_total = len(base_room_list)
        need_empty = set(list(plan.keys()))
        while idx < room_total:
            ret, switch, mode = segment.worker(self.recog.img)
            if len(ret) == 0:
                raise RecognizeError('未识别到进驻总览中的房间列表')

            # 关闭撤下干员按钮
            if mode:
                self.tap((switch[0][0]+5, switch[0][1]+5), rebuild=False)
                continue

            if room_total-idx < len(ret):
                # 已经滑动到底部
                ret = ret[-(room_total-idx):]

            for block in ret:
                if base_room_list[idx] in need_empty:
                    need_empty.remove(base_room_list[idx])
                    # 对这个房间进行换班
                    finished = len(plan[base_room_list[idx]]) == 0
                    skip_free = 0
                    error_count = 0
                    while not finished:
                        x = (7*block[0][0]+3*block[2][0])//10
                        y = (block[0][1]+block[2][1])//2
                        self.tap((x, y))

                        # 若不是空房间，则清空工作中的干员
                        if self.find('arrange_empty_room') is None:
                            if self.find('arrange_clean') is not None:
                                self.tap_element('arrange_clean')
                            else:
                                # 对于只有一个干员的房间，没有清空按钮，需要点击干员清空
                                self.tap((self.recog.w*0.38, self.recog.h*0.3), interval=0)

                        try:
                            if base_room_list[idx].startswith('dormitory'):
                                default_order = ArrangeOrder.FEELING
                            else:
                                default_order = ArrangeOrder.SKILL
                            self.choose_agent(
                                plan[base_room_list[idx]], skip_free, default_order)
                        except RecognizeError as e:
                            error_count += 1
                            if error_count >= 3:
                                raise e
                            # 返回基建干员进驻总览
                            self.recog.update()
                            while self.scene() not in [Scene.INFRA_ARRANGE, Scene.INFRA_MAIN] and self.scene() // 100 != 1:
                                pre_scene = self.scene()
                                self.back(interval=3)
                                if self.scene() == pre_scene:
                                    break
                            if self.scene() != Scene.INFRA_ARRANGE:
                                raise e
                            continue
                        self.recog.update()
                        self.tap_element(
                            'confirm_blue', detected=True, judge=False, interval=3)
                        if self.scene() == Scene.INFRA_ARRANGE_CONFIRM:
                            x = self.recog.w // 3 * 2  # double confirm
                            y = self.recog.h - 10
                            self.tap((x, y), rebuild=True)
                        finished = True
                        while self.scene() == Scene.CONNECTING:
                            self.sleep(3)
                idx += 1

            # 换班结束
            if idx == room_total or len(need_empty) == 0:
                break
            block = ret[-1]
            top = switch[2][1]
            self.swipe_noinertia(tuple(block[1]), (0, top-block[1][1]))

        logger.info('返回基建主界面')
        self.back()

    def choose_agent_in_order(self, agent: list[str], exclude: list[str] = None, exclude_checked_in: bool = False, dormitory: bool = False):
        """
        按照顺序选择干员，只选择未在工作、未注意力涣散、未在休息的空闲干员

        :param agent: 指定入驻干员列表
        :param exclude: 排除干员列表，不选择这些干员
        :param exclude_checked_in: 是否仅选择未进驻干员
        :param dormitory: 是否是入驻宿舍，如果不是，则不选择注意力涣散的干员
        """
        if exclude is None:
            exclude = []
        if exclude_checked_in:
            self.tap_element('arrange_order_options')
            self.tap_element('arrange_non_check_in')
            self.tap_element('arrange_blue_yes')
        self.tap_element('arrange_clean')

        h, w = self.recog.h, self.recog.w
        first_time = True
        far_left = True
        _free = None
        idx = 0
        while idx < len(agent):
            logger.info('寻找干员: %s', agent[idx])
            found = 0
            while found == 0:
                ret = character_recognize.agent(self.recog.img)
                # 'Free'代表占位符，选择空闲干员
                if agent[idx] == 'Free':
                    for x in ret:
                        status_coord = x[1].copy()
                        status_coord[0, 1] -= 0.147*self.recog.h
                        status_coord[2, 1] -= 0.135*self.recog.h
                        
                        room_coord = x[1].copy()
                        room_coord[0, 1] -= 0.340*self.recog.h
                        room_coord[2, 1] -= 0.340*self.recog.h

                        if x[0] not in agent and x[0] not in exclude:
                            # 不选择已进驻的干员，如果非宿舍则进一步不选择精神涣散的干员
                            if not (self.find('agent_on_shift', scope=(status_coord[0], status_coord[2]))
                                    or self.find('agent_resting', scope=(status_coord[0], status_coord[2]))
                                    or self.find('agent_in_dormitory', scope=(room_coord[0], room_coord[2]))
                                    or (not dormitory and self.find('agent_distracted', scope=(status_coord[0], status_coord[2])))):
                                self.tap(x[1], x_rate=0.5, y_rate=0.5, interval=0)
                                agent[idx] = x[0]
                                _free = x[0]
                                found = 1
                                break

                elif agent[idx] != 'Free':
                    for x in ret:
                        if x[0] == agent[idx]:
                            self.tap(x[1], x_rate=0.5, y_rate=0.5, interval=0)
                            found = 1
                            break

                if found == 1:
                    idx += 1
                    first_time = True
                    break

                if first_time and not far_left and agent[idx] != 'Free':
                    # 如果是寻找这位干员目前为止的第一次滑动, 且目前不是最左端，则滑动到最左端
                    self.sleep(interval=0.5, rebuild=False)
                    for _ in range(9):
                        self.swipe_only((w//2, h//2), (w//2, 0), interval=0.5)
                    self.swipe((w//2, h//2), (w//2, 0), interval=3, rebuild=True)
                    far_left = True
                    first_time = False
                else:
                    st = ret[-2][1][2]  # 起点
                    ed = ret[0][1][1]   # 终点
                    self.swipe_noinertia(st, (ed[0]-st[0], 0), rebuild=True)
                    far_left = False
                    first_time = False
        self.recog.update()
        return _free

    def fia(self, room: str):
        """
        使用菲亚梅塔恢复指定房间心情最差的干员的心情，同时保持工位顺序不变
        目前仅支持制造站、贸易站、发电站 （因为其他房间在输入命令时较为繁琐，且需求不大）
        使用前需要菲亚梅塔位于任意一间宿舍
        """
        # 基建干员选择界面，导航栏四个排序选项的相对位置
        BY_STATUS = [0.622, 0.05]   # 按工作状态排序
        BY_SKILL = [0.680, 0.05]    # 按技能排序
        BY_EMO = [0.755, 0.05]      # 按心情排序
        BY_TRUST = [0.821, 0.05]    # 按信赖排序

        logger.info('基建：使用菲亚梅塔恢复心情')
        self.tap_element('infra_overview', interval=2)

        logger.info('查询菲亚梅塔状态')
        idx = 0
        room_total = len(base_room_list)
        fia_resting = fia_full = None
        while idx < room_total:
            ret, switch, _ = segment.worker(self.recog.img)
            if room_total-idx < len(ret):
                # 已经滑动到底部
                ret = ret[-(room_total-idx):]

            for block in ret:
                if 'dormitory' in base_room_list[idx]:
                    fia_resting = self.find('fia_resting', scope=(block[0], block[2])) \
                            or self.find('fia_resting_elite2', scope=(block[0], block[2]))
                    if fia_resting:
                        logger.info('菲亚梅塔还在休息')
                        break
                    
                    fia_full = self.find('fia_full', scope=(block[0], block[2])) \
                            or self.find('fia_full_elite2', scope=(block[0], block[2]))
                    if fia_full:
                        fia_full = base_room_list[idx]
                        break
                idx += 1

            if fia_full or fia_resting:
                break

            block = ret[-1]
            top = switch[2][1]
            self.swipe_noinertia(tuple(block[1]), (0, top-block[1][1]), rebuild=True)

        if not fia_resting and not fia_full:
            logger.warning('未找到菲亚梅塔，使用本功能前请将菲亚梅塔置于宿舍！')
            
        elif fia_full:
            logger.info('菲亚梅塔心情已满，位于%s', fia_full)
            logger.info('查询指定房间状态')
            self.back(interval=2)
            self.enter_room(room)
            # 进入进驻详情
            if not self.find('arrange_check_in_on'):
                self.tap_element('arrange_check_in', interval=2, rebuild=False)
            self.tap((self.recog.w*0.82, self.recog.h*0.25), interval=2)
            # 确保按工作状态排序 防止出错
            self.tap((self.recog.w*BY_TRUST[0], self.recog.h*BY_TRUST[1]), interval=0)
            self.tap((self.recog.w*BY_STATUS[0], self.recog.h*BY_STATUS[1]), interval=0.1)
            # 记录房间中的干员及其工位顺序
            ret = character_recognize.agent(self.recog.img)
            on_shift_agents = []
            for x in ret:
                x[1][0, 1] -= 0.147*self.recog.h
                x[1][2, 1] -= 0.135*self.recog.h
                if self.find('agent_on_shift', scope=(x[1][0], x[1][2])) \
                        or self.find('agent_distracted', scope=(x[1][0], x[1][2])):
                    self.tap(x[1], x_rate=0.5, y_rate=0.5, interval=0)
                    on_shift_agents.append(x[0])
            if len(on_shift_agents) == 0:
                logger.warning('该房间没有干员在工作')
                return
            logger.info('房间内的干员顺序为: %s', on_shift_agents)

            # 确保按心情升序排列
            self.tap((self.recog.w*BY_TRUST[0], self.recog.h*BY_TRUST[1]), interval=0)
            self.tap((self.recog.w*BY_EMO[0], self.recog.h*BY_EMO[1]), interval=0)
            self.tap((self.recog.w*BY_EMO[0], self.recog.h*BY_EMO[1]), interval=0.1)
            # 寻找这个房间里心情最低的干员,
            _temp_on_shift_agents = on_shift_agents.copy()
            while 'Free' not in _temp_on_shift_agents:
                ret = character_recognize.agent(self.recog.img)
                for x in ret:
                    if x[0] in _temp_on_shift_agents:
                        # 用占位符替代on_shift_agents中这个agent
                        _temp_on_shift_agents[_temp_on_shift_agents.index(x[0])] = 'Free'
                        logger.info('房间内心情最差的干员为: %s', x[0])
                        _recover = x[0]
                        break
                if 'Free' in _temp_on_shift_agents:
                    break

                st = ret[-2][1][2]  # 起点
                ed = ret[0][1][1]   # 终点
                self.swipe_noinertia(st, (ed[0]-st[0], 0), rebuild=True)
            self.back(interval=2)        
            self.back(interval=2)
            
            logger.info('进入菲亚梅塔所在宿舍，为%s恢复心情', _recover)
            self.enter_room(fia_full)
            # 进入进驻详情
            if not self.find('arrange_check_in_on'):
                self.tap_element('arrange_check_in', interval=2, rebuild=False)
            self.tap((self.recog.w*0.82, self.recog.h*0.25), interval=2)
            # 选择待恢复干员和菲亚梅塔
            rest_agents = [_recover, '菲亚梅塔']
            self.choose_agent_in_order(rest_agents, exclude_checked_in=False)
            self.tap_element('confirm_blue', detected=True, judge=False, interval=1)
            # double confirm
            if self.scene() == Scene.INFRA_ARRANGE_CONFIRM:
                x = self.recog.w // 3 * 2  
                y = self.recog.h - 10
                self.tap((x, y), rebuild=True)
            while self.scene() == Scene.CONNECTING:
                self.sleep(3)
                
            logger.info('恢复完毕，填满宿舍')
            rest_agents = '菲亚梅塔 Free Free Free Free'.split()
            self.tap((self.recog.w*0.82, self.recog.h*0.25), interval=2)
            self.choose_agent_in_order(rest_agents, exclude=[_recover], dormitory=True)
            self.tap_element('confirm_blue', detected=True, judge=False, interval=3)
            while self.scene() == Scene.CONNECTING:
                self.sleep(3)

            logger.info('恢复原职')
            self.back(interval=2)
            self.enter_room(room)
            if not self.find('arrange_check_in_on'):
                self.tap_element('arrange_check_in', interval=2, rebuild=False)
            self.tap((self.recog.w*0.82, self.recog.h*0.25), interval=2)
            self.choose_agent_in_order(on_shift_agents)
            self.tap_element('confirm_blue', detected=True, judge=False, interval=3)
            while self.scene() == Scene.CONNECTING:
                self.sleep(3)
            self.back(interval=2)

    # def clue_statis(self):

    #     clues = {'all': {}, 'own': {}}

    #     self.recog_bar()
    #     self.tap(((x1*7+x2)//8, y0//2), rebuild=False)
    #     self.tap(((x1*7.5+x2*0.5)//8, (y0+y1)//2), rebuild=False)
    #     self.recog_view(only_y2=False)

    #     if x3 is None:
    #         return clues

    #     for i in range(1, 8):

    #         self.tap((((i+0.5)*x2+(8-i-0.5)*x1)//8, (y0+y1)//2), rebuild=False)
    #         self.clear_clue_mask()
    #         self.recog_view()

    #         count = 0
    #         if y2 < self.recog.h - 20:
    #             count = len(self.ori_clue())
    #         else:
    #             while True:
    #                 restart = False
    #                 count = 0
    #                 ret = self.ori_clue()
    #                 while True:

    #                     y4 = 0
    #                     for poly in ret:
    #                         count += 1
    #                         y4 = poly[0, 1]

    #                     self.tap((x4, y4+10), rebuild=False)
    #                     self.device.swipe([(x4, y4), (x4, y1+10), (0, y1+10)], duration=(y4-y1-10)*3)
    #                     self.sleep(1, rebuild=False)

    #                     mask = self.get_clue_mask()
    #                     if mask is not None:
    #                         self.clear_clue_mask()
    #                     ret = self.ori_clue()

    #                     if mask is None or not (ret[0][0, 1] <= mask <= ret[-1][1, 1]):
    #                         # 漂移了的话
    #                         restart = True
    #                         break

    #                     if ret[0][0, 1] <= mask <= ret[0][1, 1]:
    #                         count -= 1
    #                         continue
    #                     else:
    #                         for poly in ret:
    #                             if mask < poly[0, 1]:
    #                                 count += 1
    #                         break

    #                 if restart:
    #                     self.swipe((x4, y1+10), (0, 1000),
    #                                duration=500, interval=3, rebuild=False)
    #                     continue
    #                 break

    #         clues['all'][i] = count

    #     self.tap(((x1+x2)//2, y0//2), rebuild=False)

    #     for i in range(1, 8):
    #         self.tap((((i+0.5)*x2+(8-i-0.5)*x1)//8, (y0+y1)//2), rebuild=False)

    #         self.clear_clue_mask()
    #         self.recog_view()

    #         count = 0
    #         if y2 < self.recog.h - 20:
    #             count = len(self.ori_clue())
    #         else:
    #             while True:
    #                 restart = False
    #                 count = 0
    #                 ret = self.ori_clue()
    #                 while True:

    #                     y4 = 0
    #                     for poly in ret:
    #                         count += 1
    #                         y4 = poly[0, 1]

    #                     self.tap((x4, y4+10), rebuild=False)
    #                     self.device.swipe([(x4, y4), (x4, y1+10), (0, y1+10)], duration=(y4-y1-10)*3)
    #                     self.sleep(1, rebuild=False)

    #                     mask = self.get_clue_mask()
    #                     if mask is not None:
    #                         self.clear_clue_mask()
    #                     ret = self.ori_clue()

    #                     if mask is None or not (ret[0][0, 1] <= mask <= ret[-1][1, 1]):
    #                         # 漂移了的话
    #                         restart = True
    #                         break

    #                     if ret[0][0, 1] <= mask <= ret[0][1, 1]:
    #                         count -= 1
    #                         continue
    #                     else:
    #                         for poly in ret:
    #                             if mask < poly[0, 1]:
    #                                 count += 1
    #                         break

    #                 if restart:
    #                     self.swipe((x4, y1+10), (0, 1000),
    #                                duration=500, interval=3, rebuild=False)
    #                     continue
    #                 break

    #         clues['own'][i] = count

    #     return clues

```