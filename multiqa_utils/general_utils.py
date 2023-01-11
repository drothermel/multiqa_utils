import os
import json
import jsonlines
import urllib
import unicodedata
import shutil
import argparse
import textwrap
import re

WORDS_TO_IGNORE = set(
    [
        "a",
        "about",
        "all",
        "also",
        "and",
        "as",
        "at",
        "be",
        "because",
        "but",
        "by",
        "can",
        "come",
        "could",
        "day",
        "do",
        "even",
        "find",
        "first",
        "for",
        "from",
        "get",
        "give",
        "go",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "how",
        "I",
        "if",
        "in",
        "into",
        "it",
        "its",
        "is",
        "just",
        "know",
        "like",
        "look",
        "make",
        "man",
        "many",
        "me",
        "more",
        "my",
        "new",
        "no",
        "not",
        "now",
        "of",
        "on",
        "one",
        "only",
        "or",
        "other",
        "our",
        "out",
        "people",
        "say",
        "see",
        "she",
        "so",
        "some",
        "take",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "think",
        "this",
        "those",
        "time",
        "to",
        "two",
        "up",
        "use",
        "very",
        "was",
        "want",
        "way",
        "we",
        "well",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "year",
        "you",
        "your",
    ]
)

GREEN_START = "\x1b[32m"
RED_START = "\x1b[31m"
COLOR_END = "\x1b[0m"


def current_default_path_args():
    path_config = {
        # Input Data from previous steps
        "gpt_ans_raw_path": "/scratch/ddr8143/multiqa/qampari_data/qmp_simple_gpt3_answers.json",
        "gpt_ans_path": "/scratch/ddr8143/multiqa/qampari_data/qmp_simple_gpt3_answers_structured.json",
        "elq_ans_path": "/scratch/ddr8143/multiqa/qampari_data/eql_default_tagging_v0_qmp_dev.jsonl",
        # Key Directories
        "gt_wiki_dir": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/",
        "gt_wiki_postp_dir": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/",
        # Paths for setting up ent_str to wiki_page cache
        "gt_title_set_path": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/gt_title_set.json",
        "wikitags_path_regexp": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/title2linktagmestrs_*.json",
        "strs_for_cache_path": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/strs_to_add_to_cache_v0.json",
        "cache_path": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/str2wikipage_cache.json",
        "disambig_cache_path": "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/str2wikipage_disambig_cache.json",
    }
    path_args = argparse.Namespace(**path_config)
    return path_args


def loadjsonl(filename):
    all_lines = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            all_lines.append(obj)
    return all_lines


def normalize(text, unquote=False):
    if unquote:
        text = urllib.parse.unquote(text)
    text = unicodedata.normalize("NFD", text).lower().replace(" ", "_")
    return text


def unnormalize(text):
    return text.replace("_", " ")


def get_backup_path(path):
    return f"{path}__old"


def checkpoint_json(
    data,
    path,
    suffix="",
    backup=True,
):
    cp_base = f"{path}{suffix}"
    if backup:
        cp_backup = get_backup_path(cp_base)
        if os.path.exists(cp_base):
            shutil.move(cp_base, cp_backup)
    json.dump(data, open(cp_base, "w+"))
    print(">> Dumped:", cp_base, flush=True)


################################################
##                Viz Utils                   ##
################################################


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

    wrapped = textwrap.wrap(print_ctx, width=width)

    colored_title = color_text(ctx["title"], "green", answers)
    print(f"{ctx['score']:3.4f} | {colored_title}")
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
