"""各处 prompt 共用的小工具：token 估算与截断、不可信内容的包裹。

论文、评论、用户可编辑的文件（兴趣描述、学到的偏好、屏蔽主题）进入 prompt 时都是不可信数据：
用标签包裹，并在 system prompt 中声明其中出现的指令一律忽略。为了不让内容伪造标签「逃出」包裹，
写入前把其中与包裹标签同名的 "<" 换成形近字符。

放在 llm 包里而不是 reader 包里：profile、ranking 也要用，而 reader 依赖 ranking，
放在 reader 里会循环导入。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

# 没有分词器，按字符数估算 token：英文平均约 4 个字符一个 token，论文里的公式、数字和
# 表格切得更碎，这里保守地按 3.5 个字符计；中日韩字符按每字 1 个 token 计（实际通常更少）
CHARS_PER_TOKEN = 3.5
_CJK_RE = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
TRUNCATION_MARK = " …（以下截断）"
_LANGUAGE_NAMES = {"zh": "简体中文", "en": "English"}


def language_name(code: str) -> str:
    """配置中的输出语言（zh、en）在 prompt 里的写法。"""
    return _LANGUAGE_NAMES.get(code, code)


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数（估算方式见 CHARS_PER_TOKEN）。"""
    cjk = len(_CJK_RE.findall(text))
    return cjk + math.ceil((len(text) - cjk) / CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """截断到约 max_tokens 个 token，尽量断在空白处，并在末尾注明截断。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget = max(max_tokens - estimate_tokens(TRUNCATION_MARK), 0)
    if not _CJK_RE.search(text):
        cut = int(budget * CHARS_PER_TOKEN)
    else:
        cost, cut = 0.0, len(text)
        for i, char in enumerate(text):
            cost += 1 if _CJK_RE.match(char) else 1 / CHARS_PER_TOKEN
            if cost > budget:
                cut = i
                break
    head = text[:cut]
    space = max(head.rfind(" "), head.rfind("\n"))
    if space > len(head) * 0.8:  # 不截断单词，但也不为此丢掉太多内容
        head = head[:space]
    return head.rstrip() + TRUNCATION_MARK


def escape_tags(text: str, tags: Iterable[str]) -> str:
    """把 text 中与 tags 同名的开、闭标签的 "<" 换成形近字符，避免提前闭合包裹标签。"""
    names = "|".join(re.escape(tag) for tag in tags)
    return re.sub(rf"<(\s*/?\s*(?:{names})\b)", r"‹\1", text, flags=re.IGNORECASE)


def wrap(tag: str, text: str, tags: Iterable[str] = ()) -> str:
    """用 <tag> 包裹不可信内容。

    内容中与 tag 及 tags（同一 prompt 里的其它包裹标签）同名的标签会被转义。
    """
    return f"<{tag}>\n{escape_tags(text, (tag, *tags))}\n</{tag}>"
