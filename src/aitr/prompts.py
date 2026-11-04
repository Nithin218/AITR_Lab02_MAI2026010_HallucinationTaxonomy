"""Prompts and answer parsing for E2.

Agrawal et al. prompts are copied verbatim from microsoft/hallucinated-references@bf3ef5f3
(code/src/prompts.py and the consistency prompt in code/src/llama_run.py), MIT licence.
VERBAL_P (detector D4) is ours (protocol §3).
"""
import re

prompt_Q = lambda x : [{"role" : "user", "content" : f'List 5 existing references related to "{x}". Just output the titles. Do not mention the authors. Output format should be - <num.>. "<TITLE>".'}]  # noqa: E501,E731

prompt_IQ = lambda x : [{"role" : "user", "content" : f"""Who were the authors of the reference, "{x}"? Please list only the author names, formatted as - AUTHORS: <firstname> <lastname>, separated by commas. Do not mention the reference in the answer."""}]  # noqa: E501,E731

prompt_DQ1_sample = lambda x : [{"role" : "user" , "content" : f"""Does the reference "{x}" exist? Just output yes/no."""}]  # noqa: E731

prompt_DQ2_sample = lambda x : [{"role": "user", "content" : 'Give a famous reference for reading.'},{"role": "assistant", "content" : f'"{x}"'},{"role": "user", "content" : """Does the above reference exist? Just output yes/no."""}]  # noqa: E501,E731

prompt_DQ3_sample = lambda x,y :  [{"role": "user", "content" : f"""A language model generated references related to a research topic with the following titles:
{x}
Does the reference with title #{y} exist? Just output yes/no."""}]  # noqa: E731

CONSISTENCY_PROMPT = """Below are what should be two lists of authors. On a scale of 0-100%, how much overlap is there in the author names (ignore minor variations such as middle initials or accents)? Answer with a number between 0 and 100. Also, provide a justification. Note: if either of them is not a list of authors, output 0. Output format should be ANS: <ans> JUSTIFICATION: <justification>.

    1. <NAME_LIST1>
    2. <NAME_LIST2>"""  # noqa: E501, W293


def prompt_consistency(list1: str, list2: str):
    """As consistency_check_pair() in llama_run.py."""
    l1 = str(list1).strip().replace("\n", " ")
    l2 = str(list2).strip().replace("\n", " ")
    return [{"role": "user", "content": CONSISTENCY_PROMPT.replace("<NAME_LIST1>", l1).replace("<NAME_LIST2>", l2)}]


def get_agreement_frac(s: str) -> float:
    """Verbatim logic of llama_run.get_agreement_frac."""
    x = s.strip().upper()
    if x.startswith("ANSWER"):
        x = x[6:].strip()
    if x.startswith("ANS"):
        x = x[3:].strip()
    x = x.strip(": ")
    m = re.match(r"^[0-9]+\.?[0-9]*", x)
    if not m:
        return 0.0
    return min(float(m.group(0)) / 100.0, 1.0)


def dq_yes(answer: str) -> bool:
    """As DQ_query_sample(): an answer counts as 'exists' if it contains 'yes'."""
    return "yes" in answer.lower()


VERBAL_P = lambda x: [{"role": "user", "content": f'On a scale from 0 to 100, how confident are you that the reference "{x}" exists? Answer with a single number only.'}]  # noqa: E501,E731


def parse_verbal(answer: str):
    m = re.search(r"\d+(?:\.\d+)?", answer)
    return min(float(m.group(0)), 100.0) if m else None


def extract_titles(answer: str, k: int = 5) -> tuple[list[str], str]:
    """Agrawal's extract_answers() regex first (numbered + quoted); else numbered lines, quotes/markdown stripped.

    Returns (titles, method) with method in {"agrawal", "fallback"}.
    """
    quoted, numbered = [], []
    for line in answer.split("\n"):
        if not re.search(r"\d+\.", line):
            continue
        m = re.search(r'(\d+)\.\s*(?:“|")(.+)(?:"|”)', line)
        if m:
            quoted.append(m.group(2).strip())
        m2 = re.match(r"^\s*[-*]?\s*\d+\.\s*(.+)$", line)
        if m2:
            numbered.append(m2.group(1).replace("**", "").strip().strip('"“”*').strip().rstrip("."))
    if quoted:
        return quoted[:k], "agrawal"
    return [t for t in numbered if t][:k], "fallback"
