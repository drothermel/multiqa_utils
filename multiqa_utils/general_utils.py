import jsonlines
import urllib
import unicodedata

def readjsonl(filename):
    all_lines = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            all_lines.append(obj)
    return all_lines


def normalize(text, unquote=False):
    if unquote:
        text = urllib.parse.unquote(text)
    text = unicodedata.normalize('NFD', text).lower().replace(' ', '_')
    return text


def unnormalize(text):
    return text.replace('_', ' ')