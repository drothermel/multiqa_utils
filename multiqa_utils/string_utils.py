import re
import string
import regex
import urllib
import unicodedata

from functools import lru_cache
from operator import itemgetter

from sacremoses import MosesDetokenizer


# From qampari, models/evaluation/retriever_metric.py
def longest_common_substring(x: str, y: str) -> (int, int, int):
    # function to find the longest common substring

    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1)
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:

        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0

    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():

        # upper right triangle of the 2D array
        for k in range(len(x)):
            yield from (
                (longest_common_prefix(i, j), i, j)
                for i, j in zip(range(k, -1, -1), range(len(y) - 1, -1, -1))
            )

        # lower left triangle of the 2D array
        for k in range(len(y)):
            yield from (
                (longest_common_prefix(i, j), i, j)
                for i, j in zip(range(k, -1, -1), range(len(x) - 1, -1, -1))
            )

    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))


# ---------- Normalization Utils ----------- #


def get_detokenizer():
    return MosesDetokenizer(lang="en")


# Normalization used by qmp when loading in wiki title and text
# So the chunks already have this and we should apply it to the
# candidate strs before returning them as answers.
def normalize(detokenizer, el):
    el = fix_qu(el.replace("'", "'"))
    tokens = el.split(" ")
    return detokenizer.detokenize(tokens).replace("'", "'")


def fix_qu(string):
    pat = re.compile('"(.*?)"')
    pat2 = re.compile('" (.*?) "')
    pat3 = re.compile("'(.*?)'")
    pat4 = re.compile("' (.*?) '")
    for x in pat.finditer(string):
        to_replace = x.group(0)
        res = pat2.match(to_replace)
        if res:
            replace_with = f'"{res.group(1)}"'
            string = string.replace(to_replace, replace_with)
    for x in pat3.finditer(string):
        to_replace = x.group(0)
        res = pat4.match(to_replace)
        if res:
            replace_with = f'"{res.group(1)}"'
            string = string.replace(to_replace, replace_with)
    return string


# The version of norm that qampari provides in
#     models/evaluation/reader_metrics.py
def qmp_norm(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s is None or s == "":
        return s

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# My current best guess at the full normalization applied by qampari
def qnn_norm(detokenizer, s):
    return qmp_norm(normalize(detokenizer, s))


def unorm(text):
    return unicodedata.normalize("NFD", text)

def lnorm(text):
    return text.lower()

def lunorm(text):
    return unorm(text.lower())

def prep_norm(text):
    return text.split("(")[0]
    


# Used to link redirects
def old_norm(text, link=False):
    if link:
        text = urllib.parse.unquote(text)
    text = unicodedata.normalize("NFD", text).lower().replace(" ", "_")
    return text


# Used to extract linked entity from extracted links
def norm_links(text):
    if text is None or text == "":
        return ""
    text = urllib.parse.unquote(text)
    if text is None or ("http" in text and "://" in text):
        return ""
    elif "#" in text:
        text = text.split("#")[0]
    return text
