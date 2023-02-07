# Utils for wikipedia processing

import os
import shutil
import glob
import json
import jsonlines
import html
import re
from tqdm import tqdm

import urllib
import unicodedata

import wikipedia

import multiqa_utils.general_utils as gu



###################################
##         Global Utils          ##
###################################

def process_all_wikipath_subsegs(
    input_wikipath,
    fxn,
    output_dir=None,
    output_name=None,
    verbose=False,
    force=False,
    start=None
):
    all_outputs = []
    if output_dir is not None and not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(">> Creating output dir:", output_dir)
    
    # alpha_path ~ wikipath/AA
    alpha_files = sorted(glob.glob(f'{input_wikipath}[A-Z][A-Z]'))
    start_i = 0
    if start is not None:
        for i, af in enumerate(alpha_files):
            if start in af:
                start_i = i
                break
    for alpha_path in alpha_files[start_i:]:
        alpha = alpha_path.split('/')[-1]
        subseg_files = sorted(glob.glob(f'{alpha_path}/wiki_[0-9][0-9]'))
        if verbose:
            print("Alpha Seg:", alpha, len(subseg_files))
    
        # subseg_path ~ wikipath/AA/wiki_00
        for subseg_path in subseg_files:
            subseg = subseg_path.split('/')[-1]
            if output_dir is not None:
                on = output_name + '_' if output_name is not None else ''
                out_path = f"{output_dir}{on}{alpha}_{subseg}"
                if os.path.exists(out_path) and not force:
                    continue
                fxn(subseg_path, out_path, verbose=verbose)
                all_outputs.append(out_path)
            else:
                all_outputs.append(fxn(subseg_path, verbose=verbose))
    return all_outputs


###################################
##    Entity Strings to Pages    ##
###################################

def string_to_wikipages(ent_str, disambig_cache={}, wikipage_cache=None, max_level=2, force_contains=True):
    # Return from cache if exists
    norm_e = gu.normalize(ent_str)
    if wikipage_cache is not None and norm_e in wikipage_cache:
        return wikipage_cache[norm_e]
    
    # Otherwise, use wikipedia api to search
    try:
        answers = wikipedia.search(ent_str)
        first_answer = answers[0]
        return [gu.normalize(wikipedia.page(first_answer).title)], disambig_cache
    except wikipedia.DisambiguationError as e:
        poss_ans_set, disambig_cache = wikipage_disambig_contains(
            ent_str, 
            e.options,
            max_level=max_level,
            force_contains=force_contains,
            disambig_cache=disambig_cache,
        )
        return list(poss_ans_set), disambig_cache
    except:
        return [], disambig_cache


def wikipage_disambig_contains(ent_str, options, max_level=2, force_contains=True, disambig_cache={}):
    norm_e = gu.normalize(ent_str)
    checked = set()
    possible_answers = set()
    
    level = 0
    next_to_check = []
    if force_contains:
        to_check = [o for o in options if norm_e in gu.normalize(o)]
    else:
        to_check = [o for o in options]
        
    while len(to_check) > 0 and level < max_level:
        for tc in to_check:
            if tc in checked:
                continue
            checked.add(tc)
            
            # Setup for next level of bfs            
            # Cache disambig since queries take forever
            if tc in disambig_cache:
                for o in disambig_cache[tc]:
                    norm_o = gu.normalize(o)
                    contains_check = norm_e in norm_o if force_contains else True
                    if o not in checked and contains_check:
                        next_to_check.append(o)
            else:
                try:
                    page = wikipedia.page(tc).title
                    possible_answers.add(gu.normalize(page))
                except wikipedia.DisambiguationError as e:
                    disambig_cache[tc] = e.options
                    for o in e.options:
                        norm_o = gu.normalize(o)
                        contains_check = norm_e in norm_o if force_contains else True
                        if o not in checked and contains_check:
                            next_to_check.append(o)
                except:
                    pass
        to_check = next_to_check
        next_to_check = []
        level += 1
    return possible_answers, disambig_cache

def checkpoint_caches(cache, disambig_cache, cache_path, disambig_cache_path):
    cache_backup_path = cache_path + "__old"
    disambig_cache_backup_path = disambig_cache_path + "__old"
    if os.path.exists(cache_path):
        shutil.move(cache_path, cache_backup_path)
    if os.path.exists(disambig_cache_path):
        shutil.move(disambig_cache_path, disambig_cache_backup_path)
    json.dump(cache, open(cache_path, 'w+'))
    json.dump(disambig_cache, open(disambig_cache_path, 'w+'))
    print(">> Dumped cache to:", cache_path)
    print(">> Dumped disambig_cache to:", disambig_cache_path)


# Note, all elements are normalized in the cache
def build_str2wikipage_cache(
    strs_to_add=[],
    disambig_cache_path='/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/str2wikipage_disambig_cache.json',
    cache_path='/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/str2wikipage_cache.json',
    wikiversion_title_set_path='/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia/postprocessed/wikiversion_title_set.json',
    force=False,
    use_tqdm=False,
    write_every=None,
):
    cache = {}
    if os.path.exists(disambig_cache_path):
        disambig_cache = json.load(open(disambig_cache_path))
    else:
        disambig_cache = {}
    
    ## All true titles should map to themselves
    wikiversion_title_set = set(json.load(open(wikiversion_title_set_path)))
    # Assume if the cache exists it already has the title set in it (unless force)
    if not os.path.exists(cache_path) or force:
        print(">> Adding true titles to cache")
        for t in tqdm(wikiversion_title_set, disable=(not use_tqdm)):
            cache[t] = [t]
    else:
        print(">> Loading the cache")
        cache = json.load(open(cache_path))
    print(">> Initial cache size:", len(cache))
    added_something = False
        
    ## Then, for all new strings, if not in cache, do wikipedia seach + validate that the result is in GT
    print(">> Adding new strings to cache:", len(strs_to_add))
    num_s = -1
    for s in tqdm(strs_to_add, disable=(not use_tqdm)):
        s_norm = gu.normalize(s)
        if s_norm in cache:
            continue
        
        num_s += 1
        s_unnorm = gu.unnormalize(s_norm)
        if s_unnorm.strip() == '':
            print(f"Empty String: |{s}|{s_norm}|{s_unnorm}")
            continue
        
        s_pages = []
        possible_pages, disambig_cache = string_to_wikipages(
            s_unnorm, 
            disambig_cache=disambig_cache, 
            wikipage_cache=cache,
        )

        for pp in possible_pages:
            norm_pp = gu.normalize(pp)
            if norm_pp in wikiversion_title_set:
                s_pages.append(norm_pp)
                
        if len(s_pages) == 0:
            # TODO: fuzzy wuzzy
            # print("Whoops, no real pages for string:", s)
            continue
            
        if write_every is not None and num_s != 0 and num_s % write_every == 0:
            print(f">> Dumping intermediate cache after processing {num_s} words")
            checkpoint_caches(cache, disambig_cache, cache_path, disambig_cache_path)
        
        cache[s_norm] = s_pages
        added_something = True
    
    print(">> Final cache size:", len(cache))
    if added_something:
        checkpoint_caches(cache, disambig_cache, cache_path, disambig_cache_path)
    else:
        print(">> No changes, cache at:", cache_path)


###################################
##       Entity Extraction       ##
###################################
    

def extract_page_set_from_extracted_links(wikipedia_link_list):
    wk_pages = set()
    for wl in wikipedia_link_list:
        wk_pages.add(gu.normalize(wl['linked_et'], unquote=True))
    return wk_pages


def extract_entity_set_from_tagme_list(tagme_list):
    return set([gu.normalize(t[0]) for t in tagme_list if t[0] is not None])


def write_title_to_links_tagmes_subseg(
    input_path,
    output_path,
    verbose=False,
):
    subseg_ind = {}
    
    wiki_pages = gu.readjsonl(input_path)
    for wiki_page in wiki_pages:
        if len(wiki_page['clean_text']) == 0:
            continue

        title = gu.normalize(wiki_page['title'])
        links = extract_page_set_from_extracted_links(wiki_page['links'])
        tagmes = extract_entity_set_from_tagme_list(wiki_page['tagme_links'])
        if title in subseg_ind:
            links.update(subseg_ind[title]['links'])
            tagmes.update(subseg_ind[title]['tagmes'])
            subseg_ind[title]['data_paths'].append(input_path)
            subseg_ind[title]['links'] = list(links)
            subseg_ind[title]['tagmes'] = list(tagmes)
        else:
            subseg_ind[title] = {
                "data_paths": [input_path],
                "title": title,
                "links": list(links),
                "tagmes": list(tagmes),
            }
    output_path_full = f"{output_path}.json"
    json.dump(subseg_ind, open(output_path_full, 'w+'))


def wikipedia_title_to_links_tagmes_strs(
    input_wikipath,
    output_dir,
    output_name,
    verbose=False,
    force=False,
    start=None,
):
    all_paths = process_all_wikipath_subsegs(
        input_wikipath=input_wikipath,
        fxn=write_title_to_links_tagmes_subseg,
        output_dir=output_dir,
        output_name=output_name,
        verbose=verbose,
        force=force,
        start=start,
    )
    print(">> Finished processing all segments.")
    return all_paths
    
    
def get_title_set_subseg(
    input_path,
    verbose=False,
):
    title_set = set()
    wikipages = gu.readjsonl(input_path)
    for wiki_page in wikipages:
        if len(wiki_page['clean_text']) != 0:
            title_set.add(gu.normalize(wiki_page['title']))
    return title_set

def wikipedia_title_set(
    input_wikipath,
    verbose=False,
):
    all_segs = process_all_wikipath_subsegs(
        input_wikipath=input_wikipath,
        fxn=get_title_set_subseg,
        verbose=verbose,
    )
    all_titles = set()
    for s in all_segs:
        all_titles.update(s)
    return all_titles



###################################
##    Process New Wikidump       ##
###################################

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
#
# TODO: Test if you use this again, slight refactor
def postprocess_wikipedia_to_page_index(
    input_wikipath, output_dir, force=False, verbose=False,
):
    process_all_wikipath_subsegs(
        input_wikipath=input_wikipath,
        fxn=postprocess_wikipedia_segment_to_page_index,
        output_dir=output_dir,
        verbose=verbose,
        force=force,
    )


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
