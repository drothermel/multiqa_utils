import html
import re

import utils.file_utils as fu


###################################
##    Process New Wikidump       ##
###################################

# TODO: pull in the utils from QAMPARI repo here too

# After wikiextractor has already processed the wikidump then we can
# use this to create base files for a page index.
#
# HTML Escaping from: https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
# This will:
#    1) Remove any html tags remaining from the text
#    2) Append the keywords "Title:" and "Article:" along with the title to the text
#    3) Format the final output file into a .jsonl in the format expected by pyserini index builder
def postprocess_wikipedia_segment_to_page_index(infile, outfile, verbose=True):
    clean = re.compile("<.*?>")
    orig_file = fu.load_file(infile, ending='.jsonl')

    postprocess_pages = []
    for obj in orig_file:
        if obj["text"]:
            cleaned_text = re.sub(clean, "", html.unescape(obj["text"]))
            new_text = f"Title: {obj['title']}\nArticle: {cleaned_text}"
            postprocess_pages.append(
                {
                    "id": obj["id"],
                    "contents": new_text,
                }
            )

    fu.dumpjsonl(postprocess_pages, outfile, verbose=verbose)


