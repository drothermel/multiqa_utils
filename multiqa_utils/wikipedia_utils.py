# Utils for wikipedia processing

import os
import glob
import json
import jsonlines
import html
import re
from tqdm import tqdm


METADATA_KEYS = ["titles_with_text", "all_titles"]  # might not be updated


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

    postprocess_pages = []
    with jsonlines.open(infile) as reader:
        for obj in reader:
            if obj["text"]:
                cleaned_text = re.sub(clean, "", html.unescape(obj["text"]))
                new_text = f"Title: {obj['title']}\nArticle: {cleaned_text}"
                postprocess_pages.append(
                    {
                        "id": obj["id"],
                        "contents": new_text,
                    }
                )

    with jsonlines.open(outfile, mode="w") as writer:
        writer.write_all(postprocess_pages)
    if verbose:
        print(f">> Wrote: {outfile}")


# Pass in the input wikipath (to the directory that contains AA-GN) and an
# output directory and this writes pre-index files in the format "ED_wiki_18.jsonl"
# ready to be used by pyserini to create a full page index.
def postprocess_wikipedia_to_page_index(
    input_wikipath, output_dir, verbose=True, force=False
):
    for alpha_seg in sorted(os.listdir(input_wikipath)):
        wiki_segment = get_wikiseg_path(input_wikipath, alpha_seg)
        for input_path in sorted(glob.glob(f"{wiki_segment}/wiki_[0-9][0-9]")):
            subseg = input_path.split("/")[-1]
            output_path = f"{output_dir}/{alpha_seg}_{subseg}.jsonl"
            if os.path.exists(output_path) and not force:
                continue
            postprocess_wikipedia_segment_to_page_index(
                input_path, output_path, verbose=verbose
            )
    print(">> Finished Postprocessing Wikipedia to Page Index")


# ==== Utils for extracting metadata from wikipedia segments ==== #


def get_wikiseg_path(wikipath, segment):
    return f"{wikipath}/{segment}"


def get_metadata_path(wikipath, segment):
    return f"{get_wikiseg_path(wikipath, segment)}/metadata.json"


# Read in the segment, extract all the titles that have text and store them in a
# metadata.json file in the same segment directory.
#
# Example:
# wikipath = '/scratch/ddr8143/wikipedia/enwiki_20220701/'
# for i, segment in enumerate(sorted(os.listdir(wikipath))):
#    get_segment_metadata(wikipath, segment, force=False)
#
# >> Metadata exists: /scratch/ddr8143/wikipedia/enwiki_20220701/AA/metadata.json
# ...
#
# json.load(open('/scratch/ddr8143/wikipedia/enwiki_20220701/AE/metadata.json')).keys()
# >> dict_keys(['all_titles', 'titles_with_text'])
def get_segment_metadata(wikipath, segment, force=False, verbose=False):
    mdpath = get_metadata_path(wikipath, segment)
    if not force and os.path.exists(mdpath):
        if verbose:
            print(f">> Metadata exists: {mdpath}")
        return

    wiki_segment = get_wikiseg_path(wikipath, segment)
    seg_title_to_info = defaultdict(list)
    seg_title_to_info_wtext = defaultdict(list)
    for subseg in sorted(os.listdir(wiki_segment)):
        if "metadata" in subseg:
            continue
        subseg_path = f"{wiki_segment}/{subseg}"
        # print(f">>     Processing {subseg_path}")
        with open(f"{wiki_segment}/{subseg}") as f:
            for i, jl in enumerate(f):
                l = json.loads(jl)
                try:
                    ltitle = l["title"]
                    ldata = {
                        "id": l["id"],
                        "has_text": l["text"] != "",
                        "url": l["url"],
                    }
                    seg_title_to_info[ltitle].append(ldata)
                    if ldata["has_text"]:
                        seg_title_to_info_wtext[ltitle].append(
                            {k: v for k, v in ldata.items() if k != "has_text"}
                        )
                except:
                    print("Exception!!!")
                    print(l)

    # Validate results
    duplicate_titles = {k: v for k, v in seg_title_to_info.items() if len(v) > 1}
    assert (
        len(duplicate_titles) == 0
    ), f"Number duplicate titles: {len(duplicate_titles)} for wiki segment: {wiki_segment}"

    # Write metadata
    metadata = {
        "all_titles": dict(seg_title_to_info),
        "titles_with_text": dict(seg_title_to_info_wtext),
    }
    with open(mdpath, "w+") as mdf:
        json.dump(metadata, mdf)
    if verbose:
        num_titles = len(seg_title_to_info)
        num_wtext = len(seg_title_to_info_wtext)
        print(
            f">> Wrote metadata for {num_titles:6} titles ({num_wtext:6} with text) to {mdpath}"
        )


# Load all of the titles according to the key ("all_titles" or "titles_with_text") from a
# full wikipedia dump.
def aggregate_wikipedia_metadata_key(wikipath, key, use_tqdm=False):
    assert key in METADATA_KEYS
    wrapper = tqdm if use_tqdm else (lambda fxn: fxn)

    all_metadata = {}
    all_segs = sorted(os.listdir(wikipath))
    for segment in wrapper(all_segs):
        mdpath = get_metadata_path(wikipath, segment)
        md = json.load(open(mdpath))
        all_metadata.update(md[key])
        del md

    return all_metadata
