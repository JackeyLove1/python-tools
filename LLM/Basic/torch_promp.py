from langchain import PromptTemplate
from dataclasses import dataclass


@dataclass
class PromptParam:
    actor = "一位人工智能和python方面的专家"
    language = "pytorch"
    algorithm = "自然语言处理NLP的TextRNN"
    domain = "自然语言处理NLP"


prompt_template = PromptTemplate.from_template(
    "我希望你是{actor}，请你使用{language},以简单通俗的语言一步一步地实现{algorithm}，对于每一步我都希望你对给出详细的解释和说明，"
    "最好能给出可以运行的例子.最后请进一步给出一些这个算法在{domain}的相关知识。"
)

param = PromptParam

template = prompt_template.format(actor=param.actor, language=param.language, algorithm=param.algorithm, domain=param.domain)
print(template)

