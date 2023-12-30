import re
import string
from pygtrie import CharTrie
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


def find_span_indices_in_passage(passage, spans):
    trie = CharTrie()
    for span in spans:
        trie[span] = []

    for i in range(len(passage)):
        node, j = trie.longest_prefix(passage[i:])
        if node is not None and node.key in trie:
            trie[node.key].append(i)

    # Convert the trie values back to a regular dictionary for output
    span_indices = {span: trie[span] for span in spans}
    return span_indices


# ---------- Normalization Utils ----------- #


# Apply a sequence of norm_fxns to a single string
def apply_norms(ori_str, norm_fxns):
    normed_str = ori_str
    for nf in norm_fxns:
        normed_str = nf(normed_str)
    return normed_str


def get_all_norm_fxns():
    dtk = get_detokenizer()
    # matches types in cfg.wiki_processing.norm_types
    norm_fxns = {
        'l': lnorm,
        'qnn': qmp_norm,
        'prep': prep_norm,
        'qnn_l': lambda st: apply_norm(st, [lnorm, qnn]),
        'prep_l': lambda st: apply_norms(st, [prep_norm, lnorm]),
        'prep_qnn': lambda st: apply_norms(st, [prep_norm, apply_qnn]),
        'prep_qnn_l': lambda st: apply_norms(st, [prep_norm, apply_qnn, lnorm]),
    }
    return norm_fxns


def get_detokenizer():
    return MosesDetokenizer(lang="en")


# From qampari github
# Normalization used by qmp when loading in wiki title and text
# So the chunks already have this and we should apply it to the
# candidate strs before returning them as answers.
def normalize(detokenizer, el):
    el = fix_qu(el.replace("'", "'"))
    tokens = el.split(" ")
    return detokenizer.detokenize(tokens).replace("'", "'")


# From qampari github
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
def qnn_norm(
    detokenizer,
    st,
    all_str_key=None,
    sid_normer=None,
    qnn_str_key=None,
):
    if st is None or st == "":
        return st

    # Lazy check for existance and nothing going wrong
    try:
        sid = all_str_key.get_str2sid(st)
        nsid = sid_normer.get_sid2nsid(sid)
        nstr = qnn_str_key.get_sid2str(nsid)
        return nstr
    except:  # noqa: E722
        return qmp_norm(normalize(detokenizer, st))


def unorm(text):
    if text is None or text == "":
        return text
    return unicodedata.normalize("NFD", text)


def lnorm(text):
    if text is None or text == "":
        return text
    return text.lower()


def lunorm(text):
    if text is None or text == "":
        return text
    return unorm(text.lower())


def prep_norm(text):
    if text is None or text == "":
        return text
    return text.split("(")[0].strip()


# Quest proof text specific norm
def quest_norm(text):
    if text is None or text == "":
        return text
    return text.replace("'''''", "'")


# Used to link redirects
def old_norm(text, link=False):
    if text is None or text == "":
        return text
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
