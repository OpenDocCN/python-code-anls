# `.\Chat-Haruhi-Suzumiya\research\personality\code\prompts.py`

```py
# 提供关闭问题模板的消息提示，包含一个占位符 '{}'，用于填充具体问题内容
close_prompt_template = "嗯，那对于'{}'这个问题，请你从['完全同意', '基本同意', '部分同意', '既不同意也不否认', '不太同意', '基本不同意', '完全不同意']中选择一个适合你的选项。请务必用中文回答，并用单引号强调你的选项。"

# 提供一个大五人格量表的消息模板，用于生成心理评估报告
bigfive_scale_prompt_template = """
You will read a psychological assessment report. This psychological assessment report assesses whether the subject has a high {} personality.Based on this report, output a jsoncontaining two fields: score and reasonscore is between -5 to 5 pointsIf the subject shows high {} personality in many factors, the score is 5 pointsIf the subject shows high {} personality in a single factor, the score is 2 pointsIf the report is unable to determine the subject's personality, the score is 0 pointsIf the subject shows low {} personality in a single factor, the score is -2 pointsIf the subject shows low {} personality in many factors, the score is -5 points. 
Reason is a brief summary of the reportOnly output the json, do not output any additional information, Expecting property name enclosed in double quotesReport:
"""

# 提供一个 MBTI 测试的消息模板，用于根据交谈内容评估被测者的 MBTI 类型
mbti_assess_prompt_template_wo_percent = '''You are an expert in MBTI. I am conducting an MBTI test on someone. My goal is to gauge their position on the {} spectrum of the MBTI through a series of open-ended questions. For clarity, here's some background on differentiating this particular dimension:
===
{}
===

I've invited a participant, {}, and had the following conversations in Chinese:
===
{}
===

Please help me distinguish whether {} leans more towards the {} or {} category within the MBTI's {} dimension. Please output in the following json format:
===
{{
    "analysis": <your analysis in Chinese, based on the conversations>,
    "result": <your result, either "{}" or "{}">
}}
'''

# 提供一个大五人格评估的消息模板，用于根据交谈内容评估被测者在某一人格维度上的得分
bigfive_assess_prompt_template = '''You are a psychologist with expertise in personality theories. I'm conducting an experiment to evaluate participants' scores in the Big Five personality traits, especially on the {} dimension. For clarity, here's some background on differentiating this particular dimension and its factors:
===
{}
===

I've invited a participant, {}, and had the following conversations in Chinese:
===
{}
===

Please help me evaluates whether {} possesses a high {} personality or a low {} personality, and provide an integer score ranging from -5 to 5. 

Below are some scoring references. If the subject demonstrates a high {} personality in many factors, the score is 5 points. If the subject exhibits a high {} personality in a single factor, the score is 2 points. If the subject's personality cannot be determined, the score is 0 points. If the subject shows a low {} personality in one factor, the score is -2 points. If the subject indicates a low {} personality across multiple factors, the score is -5 points. 

Please output in the following json format:
===
{{
    "analysis": <your analysis in Chinese, based on the conversations>,
    "result": <the person's score on {}, ranging from -5 to 5>
}}
===
'''
# MBTI 测试提示模板，用于根据回答的开放性问题评估参与者在 MBTI 维度上的位置
mbti_assess_prompt_template = '''You are an expert in MBTI. I am conducting an MBTI test on someone. My goal is to gauge their position on the {} spectrum of the MBTI through a series of open-ended questions. For clarity, here's some background on differentiating this particular dimension:
===
{}
===

I've invited a participant, {}, and had the following conversations in Chinese:
===
{}
===

Please help me distinguish whether {} leans more towards the {} or {} category within the MBTI's {} dimension. You should provide the person's percentage of each category, which sums to 100%, e.g., 30% A and 70% B. 
Please output in the following json format:
===
{{
    "analysis": <your analysis in Chinese, based on the conversations>,
    "result": {{ "{}": <percentage 1>, "{}": <percentage 2> }} (The sum of percentage 1 and percentage 2 should be 100%. Output without percent sign.) 
}}
'''

# MBTI 各维度的提示信息，包括 E/I、S/N、T/F 维度
mbti_dimension_prompt = {
    'E/I': '''E/I Dimension: Extraversion (E) vs Introversion (I)

E (Extraversion): Extraverts draw energy from interacting with others. They feel comfortable in social settings and tend to express their thoughts. Extraverts are often more active, seek social stimulation, and enjoy participating in group activities. For them, connecting with people, sharing, and exchanging ideas is often a need. They might be more focused on external world stimuli, such as sounds, colors, and social dynamics.

I (Introversion): Introverts feel more comfortable when alone. They derive energy from inner reflection and personal time. Contrary to extraverts, prolonged social interaction might tire them. Introverts might be more introspective, enjoy deep thinking, and tend to have meaningful personal relationships. They are more concerned with the inner world, such as thoughts, emotions, and imaginations.''',

    'S/N': '''S/N Dimension: Sensing (S) vs Intuition (N)

S (Sensing): Sensing individuals value the concrete, practical, and present situations. They rely on their five senses to process information and often focus on details. For them, past experiences and tangible evidence play a significant role in decision-making. They are typically pragmatic and tend to deal with what they "see" and "hear".

N (Intuition): Intuitive individuals tend to focus on potential possibilities and future opportunities. They like to think about "what could be", rather than just "what is". They lean more towards abstract thinking and can capture concepts and patterns effectively. Intuitives are often more innovative, preferring new ideas and approaches.''',

    'T/F': '''T/F Dimension: Thinking (T) vs Feeling (F)

T (Thinking): Thinking individuals rely primarily on logic and analysis when making decisions. They pursue fairness and objectivity and might be more direct and frank. For them, finding the most efficient method or the most logical solution is crucial, even if it might hurt some people's feelings.
'''
}
# 以下是一个包含 MBTI 类型信息的字典，用于描述个人特征和倾向
MBTI_description = {
    'E/I': '''E/I Dimension: Extraversion (E) vs Introversion (I)

E (Extraversion): Extraverts are energized by social interactions and tend to be outgoing, talkative, and enthusiastic. They enjoy being around others, seek stimulation, and often prefer to work in groups. Extraverts typically enjoy a wide range of activities and may feel bored when alone for extended periods.

I (Introversion): Introverts are energized by time alone and tend to be inwardly focused, reflective, and calm. They often prefer solitary activities or small group interactions over large gatherings. Introverts value depth of knowledge and may find social interactions draining, needing time alone to recharge.'''
    ,

    'S/N': '''S/N Dimension: Sensing (S) vs Intuition (N)

S (Sensing): Sensing individuals rely on information gained through their senses and are grounded in the present reality. They pay attention to details and practicalities, focusing on what is actual and observable. Sensing types often prefer concrete facts and experiences over abstract concepts.

N (Intuition): Intuitive individuals focus on patterns, connections, and possibilities. They are more interested in the big picture than in specific details, and they enjoy exploring new ideas and imagining future possibilities. Intuitives are drawn to abstract theories and may find routine tasks uninteresting.'''
    ,

    'T/F': '''T/F Dimension: Thinking (T) vs Feeling (F)

T (Thinking): Thinkers prioritize objectivity and logic in decision-making. They analyze situations impersonally, focusing on consistency and fairness. Thinkers value truth and tend to make decisions based on rationality and principles rather than emotions.

F (Feeling): Feeling individuals consider people's emotions and needs more when making decisions. They strive for harmony, tend to build relationships, and avoid conflicts. They are often more empathetic, valuing personal values and emotions, rather than just facts or logic.'''
    ,

    'P/J': '''P/J Dimension: Perceiving (P) vs Judging (J)

P (Perceiving): Perceivers are more open and flexible. They tend to "go with the flow" rather than overly planning or organizing things. Perceivers like to explore various possibilities and prefer to leave options open to address unforeseen circumstances. They lean towards postponing decisions to gather more information and better understanding. For them, life is a continuous process of change, not an event with fixed goals or plans. They often focus more on the experience itself rather than just the outcome.

J (Judging): Judging individuals are more structured and planned in their lives. They prefer clear expectations and structures and often set goals and pursue them. Judgers are usually more organized and tend to make decisions in advance. They like to act according to plans and feel comfortable in an orderly environment. For them, achieving goals and completing tasks are often priorities. They might focus more on efficiency and structure rather than openness or spontaneity.'''
}

# 以下是一个字符串模板，用于提示用户如何进行 MBTI 测试
to_option_prompt_template = '''You are an expert in MBTI. I am conducting an MBTI test on someone. I've invited a participant, {}, and asked a question in Chinese. Please help me classify the participant's response to this question into one the the following options: ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree'] 

Please output in the json format as follows:
===
{{
"analysis": <your analysis in Chinese, based on the conversations>,
"result": <your result from ['fully agree', 'generally agree', 'partially agree', 'neither agree nor disagree', 'partially disagree', 'generally disagree', 'fully disagree']>
}}
===
The question and response is as follows, where {} is my name:
'''

# 以下是一个空字典，用于存储 Big Five 人格特质的信息
bigfive_dimension_prompt = {}
# 将 'conscientiousness' 维度添加到 bigfive_dimension_prompt 字典中，并赋予其详细的定义文本
bigfive_dimension_prompt['conscientiousness'] = """Conscientiousness refers to the way we control, regulate, and direct our impulses. It assesses organization, persistence, and motivation in goal-directed behavior. It contrasts dependable, disciplined individuals with those who are lackadaisical and disorganized. Conscientiousness also reflects the level of self-control and the ability to delay gratification. Impulsiveness is not necessarily bad, sometimes the environment requires quick decision-making. Impulsive individuals are often seen as fun, interesting companions. However, impulsive behavior often gets people into trouble, providing momentary gratification at the expense of long-term negative consequences, such as aggression or substance abuse. Impulsive people generally do not accomplish major achievements. Conscientious people more easily avoid trouble and achieve greater success. They are generally seen as intelligent and reliable, although highly conscientious people may be perfectionists or workaholics. Extremely prudent individuals can seem monotonous, dull, and lifeless.

Conscientiousness can be divided into six facets:

C1 COMPETENCE

Refers to the sense that one is capable, sensible, prudent, and effective. High scorers feel well-prepared to deal with life. Low scorers have a lower opinion of their abilities, admitting that they are often unprepared and inept.

High scorers: Confident in own abilities. Efficient, thorough, confident, intelligent.

Low scorers: Lack confidence in own abilities, do not feel in control of work and life. Confused, forgetful, foolish.

C2 ORDER

High scorers are neat, tidy, well-organized, they put things in their proper places. Low scorers cannot organize things well, describe themselves as unmethodical.

High scorers: Well-organized, like making plans and following them. Precise, efficient, methodical.

Low scorers: Lack planning and orderliness, appear haphazard. Disorderly, impulsive, careless.

C3 DUTIFULNESS

To some extent, dutifulness refers to adherence to one's conscience, assessed by this facet. High scorers strictly follow their moral principles and scrupulously fulfill their moral obligations. Low scorers are more casual about such matters, somewhat unreliable or undependable.

High scorers: Dutiful, follow the rules. Reliable, polite, organized, thorough.

Low scorers: Feel restricted by rules and regulations. Often seen by others as unreliable, irresponsible. Careless, thoughtless, distracted.

C4 ACHIEVEMENT STRIVING

High scorers have high aspiration levels and work hard to achieve their goals. They are industrious, purposeful, and have a sense of direction. Low scorers are lackadaisical, even lazy, lacking motivation to succeed, having no ambitions and appearing to drift aimlessly. But they are often quite satisfied with their modest level of accomplishment.
"""
# 定义一个字典，用于存储关于“开放性”这一人格维度的信息及其相关内容
bigfive_dimension_prompt['openness'] = """
Openness describes a person's cognitive style. Openness to experience is defined as: the proactive seeking and appreciation of experience for its own sake, and tolerance for and exploration of the unfamiliar. This dimension contrasts intellectually curious, creative people open to novelty with traditional, down-to-earth, closed-minded individuals lacking artistic interests. Open people prefer abstract thinking, have wide interests. Closed people emphasize the concrete, conventional, are more traditional and conservative. Open people are suited to professions like teaching, closed people to occupations like police, sales, service.

Openness can be divided into six facets:

O1 FANTASY

Open people have vivid imaginations and active fantasy lives. Their daydreams are not just escapes, but ways to create interesting inner worlds. They elaborate and flesh out their fantasies, and believe imagination is essential for a rich, creative life. Low scorers are more prosaic, keeping their minds on the task at hand.

High scorers: Find the real world too plain and ordinary. Enjoy imagining, creating a more interesting, enriching world. Imaginative, daydreaming.

Low scorers: Matter-of-fact, prefers real-world thinking. Practical, prefer concrete thought.

O2 AESTHETICS

High scorers have deep appreciation for art and beauty. They are moved by poetry, absorbed in music, and touched by art. They may not have artistic talent or refined taste, but most have strong interests that enrich their experience. Low scorers are relatively insensitive and indifferent to art and beauty.

High scorers: Appreciate beauty in nature and the arts. Value aesthetic experiences, touched by art and beauty.

Low scorers: Insensitive to beauty, disinterested in the arts. Insensitive to art, cannot understand it.

O3 FEELINGS
"""
# Agreeableness assesses the degree to which an individual is likable,
# while examining an individual's attitudes toward others,
# encompassing both compassion and antagonism.
# This facet represents the broad interpersonal orientation.
# Representing "love", it values cooperation and social harmony.
bigfive_dimension_prompt['agreeableness'] = """Agreeableness assesses the degree to which an individual is likable, while Agreeableness examines an individual's attitudes toward others, encompassing both a compassionate, sympathetic orientation along with antagonism, distrust, indifference. This facet represents the broad interpersonal orientation. Agreeableness represents "love", how much value is placed on cooperation and social harmony.
"""
# 这段文本描述了关于人格特质中的“宜人性”的概念及其六个方面。
# 宜人性是指个体对他人的信任和善意的态度，以及他们在行为上是否愿意帮助他人。
# 这六个方面分别为TRUST（信任）、STRAIGHTFORWARDNESS（直率）、ALTRUISM（利他主义）、COMPLIANCE（顺从性）、MODESTY（谦逊）、TENDER-MINDEDNESS（温柔善良）。
# 每个方面都有高分者和低分者的行为特征对比，以及相应的描述。
# 定义描述同情和关注他人态度的大五人格维度：同情心
# 高分者被他人需求感动，倡导人道主义社会政策；低分者以客观逻辑为基础，自豪于冷静的评估。
Measures attitudes of sympathy and concern for others. High scorers are moved by others' needs and advocate humane social policies. Low scorers are hardheaded, unmoved by appeals to pity. They pride themselves on making objective appraisals based on cool logic.

# 描述同情心高分者和低分者的特点
High scorers: Sympathetic, moved by others' suffering, express pity. Friendly, warm-hearted, gentle, soft-hearted.
Low scorers: Do not strongly feel others' pain, pride themselves on objectivity, more concerned with truth and fairness than mercy. Callous, hardhearted, opinionated, ungenerous.

"""

# 定义大五人格维度的一部分：外向性
bigfive_dimension_prompt['extraversion'] = """
# 外向性代表人际互动的数量和强度，以及对刺激的需求和欢乐的能力。此维度对比社交、外向、行动导向的个体与内向、沉着、害羞、沉默类型的个体。外向性可以通过两个方面来衡量：人际互动程度和活动水平。前者评估个体喜欢与他人在一起的程度，后者反映个体的个人步调和活力。

# 描述外向性的两个方面衡量标准
Extraverted people enjoy interacting with others, are full of energy, and often experience positive emotions. They are enthusiastic, enjoy physical activities, and like excitement and adventure. In a group, they are very talkative, confident, and enjoy being the center of attention.

Introverted people are quieter, more cautious, and do not enjoy too much interaction with the outside world. Their lack of desire for interaction should not be confused with shyness or depression, it is simply because compared to extraverts, they do not need as much stimulation and prefer being alone. An introvert's tendencies are sometimes wrongly viewed as arrogance or unfriendliness, but they are often very kind people once you get to know them.

# 外向性可分为六个方面：
Extraversion can be divided into the following six facets:

# E1: 热情
E1 WARMTH
Most relevant to interpersonal intimacy. Warm people are affectionate and friendly. They genuinely like others and easily form close relationships. Low scorers are not necessarily hostile or lacking in compassion, but are more formal, reserved, and detached in their behavior.

High scorers: Warm people enjoy those around them and often express positive, friendly emotions towards others. They are good at making friends and forming intimate relationships. Sociable, talkative, affectionate.

Low scorers: Although not necessarily cold or unfriendly, they are often seen as distant by others.

# E2: 社交性
E2 GREGARIOUSNESS
Refers to a preference for other people's company. Gregarious people enjoy the company of others and the more people the merrier. Low scorers tend to be loners, they do not seek out and even actively avoid social stimulation.

High scorers: Enjoy being with people, prefer lively, crowded settings. Outgoing, having many friends, seek social affiliations.
# 将大五人格维度'神经质'的提示文本添加到字典中
bigfive_dimension_prompt['neuroticism'] = """Neuroticism or Emotional Stability: Having tendencies of anxiety, hostility, depression, self-consciousness, impulsiveness, vulnerability.

N1 ANXIETY

Anxious individuals tend to worry, fear, be easily concerned, tense, and oversensitive. Those who score high are more likely to have free-floating anxiety and apprehension. Those with low scores tend to be calm, relaxed. They do not constantly worry about things that might go wrong.

High scorers: Anxiety, easily feel danger and threats, tend to be tense, fearful, worried, uneasy.

Low scorers: Calm state of mind, relaxed, not easily scared, won't always worry about things that could go wrong, emotions are calm, relaxed, stable.

N2 ANGRY HOSTILITY

Reflects the tendency to experience anger and related states (e.g. frustration, bitterness). Measures the ease with which an individual experiences anger.
"""
# N3 DEPRESSION

# 测量个体倾向于经历抑郁情绪的差异。高分者倾向于感到内疚、悲伤、绝望和孤独。他们容易感到气馁，经常感到沮丧。低分者很少经历这些情绪。
High scorers: Despairing, guilty, gloomy, dejected. Prone to feeling sorrow, abandonment, discouraged. Prone to feelings of guilt, sadness, disappointment, and loneliness. Easily discouraged, often feeling down.

Low scorers: Not prone to feeling sad, rarely feels abandoned.

# N4 SELF-CONSCIOUSNESS

# 核心是羞怯和容易尴尬。这类个体在群体中感到不舒服，对嘲笑敏感，容易产生自卑感。自我意识类似于羞怯和社交焦虑。低分者不一定在社交场合表现良好或社交技能高超，只是在尴尬的社交情境中不太受影响。
High scorers: Too concerned with what others think, afraid of being laughed at, tend to feel shy, anxious, inferior, awkward in social situations.

Low scorers: Composed, confident in social situations, not easily made tense or shy.

# N5 IMPULSIVENESS

# 指对冲动和欲望的控制能力。个体容易屈服于冲动和诱惑，不考虑长期后果（例如食物、香烟、物品），尽管他们后悔自己的行为。低分者能更好地抵制诱惑，具有较高的挫折容忍力。
High scorers: Cannot resist cravings when experiencing strong urges, tend to pursue short-term satisfaction without considering long-term consequences. Cannot resist temptations, rash, spiteful, self-centered.

Low scorers: Self-controlled, can resist temptation.

# N6 VULNERABILITY

# 指对压力的易感性。高分者在应对压力方面有困难，面对紧急情况时会感到恐慌、无助和绝望。低分者认为自己能够适当处理困难情况。
High scorers: Under stress, easily feel panic, confusion, helpless, cannot cope with stress.

Low scorers: Under stress, feel calm, confident. Resilient, clear-headed, brave.
```