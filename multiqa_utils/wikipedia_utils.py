# Utils for wikipedia processing

import os
import json
from tqdm import tqdm

WIKIPATH = "/scratch/ddr8143/wikipedia/enwiki_20220701"
METADATA_KEYS = ["titles_with_text", "all_titles"]  # might not be updated

dummy_wrapper = lambda fxn: fxn


def get_wikiseg_path(wikipath, segment):
    return f"{wikipath}/{segment}"


def get_metadata_path(wikipath, segment):
    return f"{get_wikiseg_path(wikipath, segment)}/metadata.json"


# Note that this writes to the wikipath
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


def aggregate_wikipedia_metadata_key(wikipath, key, use_tqdm=False):
    assert key in METADATA_KEYS
    wrapper = tqdm if use_tqdm else dummy_wrapper

    all_metadata = {}
    all_segs = sorted(os.listdir(wikipath))
    for segment in wrapper(all_segs):
        mdpath = get_metadata_path(wikipath, segment)
        md = json.load(open(mdpath))
        all_metadata.update(md[key])
        del md

    return all_metadata
