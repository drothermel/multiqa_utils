import urllib
import unicodedata
import textwrap
import re

import utils.file_utils as fu

WORDS_TO_IGNORE_PATH = (
    "/scratch/ddr8143/repos/multiqa_utils/data_files/words_to_ignore.json"
)
WORDS_TO_IGNORE = set(fu.load_file(WORDS_TO_IGNORE_PATH))

GREEN_START = "\x1b[32m"
RED_START = "\x1b[31m"
COLOR_END = "\x1b[0m"


def parse_question_to_words(question):
    qbase = question.strip("?")
    qwords = [w for w in qbase.split() if w not in WORDS_TO_IGNORE]
    return qwords


def remove_punc(instr):
    return re.sub(r"[^\w\s]", "", instr)


def color_text(text, color, match_list):
    start = GREEN_START if color == "green" else RED_START
    for w in match_list:
        text = re.sub(w, start + w + COLOR_END, text, flags=re.IGNORECASE)
    return text


def print_wrapped(text, width):
    wrapped = textwrap.wrap(text, width=width)
    for i, w in enumerate(wrapped):
        if i == 0:
            print(f"    >> {w}")
        else:
            print(f"       {w}")


# {'text', 'title', 'score'}
def print_ctx(
    ctx,
    answers=None,
    question=None,
    width=150,
):
    print_ctx = ctx["text"]
    if question is not None:
        qwords = parse_question_to_words(question)
        print_ctx = color_text(print_ctx, "red", qwords)

    if answers is not None:
        print_ctx = color_text(print_ctx, "green", answers)

    colored_title = color_text(ctx["title"], "green", answers)
    print(f"{ctx['score']:3.4f} | {colored_title}")
    print_wrapped(print_ctx, width)
    for i, w in enumerate(wrapped):
        if i == 0:
            print(f"    >> {w}")
        else:
            print(f"       {w}")


def get_answer_str(answers):
    colored_answers = [color_text(a, "green", [a]) for a in answers]
    astr = ", ".join(colored_answers)
    return astr


def get_question_keyword_str(question):
    colored_keywords = [
        color_text(w, "red", [w]) for w in parse_question_to_words(question)
    ]
    qwstr = ", ".join(colored_keywords)
    return qwstr


def print_ctx_list(ctx_list, sort_score=True, answers=None, question=None):
    if sort_score:
        ctx_list = sorted(ctx_list, key=lambda x: x["score"], reverse=True)

    print("----------------------------------")
    for ctx in ctx_list:
        print_ctx(ctx, answers=answers, question=question)
        print()
    print("----------------------------------\n")
