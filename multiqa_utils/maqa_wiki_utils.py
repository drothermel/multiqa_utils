import jsonlines
import glob
import os
from collections import defaultdict, namedtuple
import regex
import string
import logging
import wikipedia
import re
import multiprocessing
import numpy as np

import utils.file_utils as fu
import utils.run_utils as ru
import multiqa_utils.dpr_tokenizer_utils as tu
from multiqa_utils.helper_classes import StringKey, PassageData, SidNormer, Ori2Ent

## ---- Path Builder Utils ---- ##

def get_graph_data_path(cfg, graph_type, data_type):
    assert graph_type in cfg.graph_types
    assert data_type in cfg.graphs
    return f"{cfg.postp_dir}{graph_type}__{data_type}.pkl"


def get_set_data_path(cfg, data_type):
    assert data_type in cfg.sets
    return f"{cfg.postp_dir}{data_type}_set.pkl"


def get_ori2entdetailed_path(cfg, graph_type, str_ent_type):
    assert graph_type in cfg.graph_types
    assert str_ent_type in cfg.str_ent_types
    return f"{cfg.postp_dir}{graph_type}__ori2entdetailed_{str_ent_type}.pkl"

def get_mapped_file(mapped_dir, input_name):
    return f'{mapped_dir}{input_name}_mapped.pkl'

def get_all_mapped_files(mapped_dir):
    return sorted(glob.glob(get_mapped_file(mapped_dir, '*')))
    
## ---- Key Parsing ---- ##

def get_norm_bool_from_ent_str_type(ent_str_type):
    ent_s, str_s = ent_str_type.split()
    norm_bools = (ent_s[0] == 'q', str_s[0] == 'q')
    return norm_bools
        

## --------- Postprocess Wikipedia v2 ----------##

## --------- v2: Reducing Functions ----------##
def mapped_wiki_into_passage_data(passage_data, mapped_passage_data_input):
    for pid, pdata in mapped_passage_data_input.items():
        passage_data.add_page(
            page_id=pid,
            title=pdata["title"],
            prep_title=pdata["prep_title"],
            on_title=pdata["on_title"],
            qn_title=pdata["qn_title"],
        )
        for i, passage in enumerate(pdata["passage_list"]):
            chunk_num = pdata["chunk_num_list"][i]
            passage_data.add_passage(
                page_id=pid,
                chunk_num=chunk_num,
                content=passage,
            )


def mapped_wiki_into_qnnstrkey(qnn_str_key, mapped_str2qnn):
    for qnn_str in mapped_str2qnn.values():
        qnn_str_key.add_str(qnn_str)


def mapped_wiki_into_strkey_and_sidsets(str_key, sid_sets, mapped_sets):
    # Note we're using tags and links only because it is a superset of the
    # strings that appear in each graph
    tnl_sets = mapped_sets["graph_data_tags_and_links"]
    for set_name, str_set in tnl_sets.items():
        if set_name not in sid_sets:
            sid_sets[set_name] = set()
        for s in str_set:
            sid = str_key.add_str(s)
            sid_sets[set_name].add(sid)


def mapped_wiki_into_sidnormer(str_key, qnn_key, sid_normer, mapped_str2qnn):
    for st, qnn_str in mapped_str2qnn.items():
        sid = str_key.get_str2sid(st)
        nsid = qnn_key.get_str2sid(qnn_str)
        sid_normer.add_sid_to_nsid(sid, nsid)


def reducing__pagedata_strkeys_sidsets_sidnormer(cfg, test=False):
    files = get_all_mapped_files(get_mapped_dir(cfg))
    files = files[:3] if test else files

    # Initialize to build these from scratch
    all_str_key = StringKey(cfg.string_key.dir, cfg.string_key.all_key_name, reset=True)
    qnn_str_key = StringKey(cfg.string_key.dir, cfg.string_key.qnn_key_name, reset=True)
    passage_data = PassageData(
        cfg.passage_data.dir,
        cfg.passage_data.name,
        prep2title=True,
        extra_norms=True,
        reset=True,
    )
    sid_normer = SidNormer(cfg.sid_normer.dir, cfg.sid_normer.name, reset=True)
    sid_sets = {}

    # Iterate through mapped files adding to data classes
    for i, f in enumerate(files):
        ru.processed_log(i, len(files))
        in_data = fu.load_file(f)
        # Note order matters, make keys then use them for str -> sid only
        mapped_wiki_into_passage_data(passage_data, in_data["passage_data"])
        mapped_wiki_into_qnnstrkey(qnn_str_key, in_data["str2qnn"])
        mapped_wiki_into_strkey_and_sidsets(all_str_key, sid_sets, in_data["sets"])
        mapped_wiki_into_sidnormer(
            all_str_key, qnn_str_key, sid_normer, in_data["str2qnn"]
        )

    # Save order matters due to memory used for post-processing pre-aving
    for name, data in sid_sets.items():
        fu.dumppkl(data, f"{cfg.postp_dir}{name}_set.pkl")
    del sid_sets
    passage_data.save()
    del passage_data
    # These takes extra memory to save, so do these last
    sid_normer.save(force=True)
    del sid_normer
    qnn_str_key.save(force=True)
    del qnn_str_key
    all_str_key.save(force=True)
    del all_str_key
    logging.info(
        ">> Finished reducing Round 1: StrKeys, PassageData, SidNormer and Sets"
    )


def reducing__graphs(cfg, str_key, keys_to_run=None, test=False):
    keys_to_run = cfg.graphs if keys_to_run is None else keys_to_run
    files = get_all_mapped_files(input_dir)
    files = files[:3] if test else files

    def add_as_sids(in_dict, out_dict, conv_k=True, conv_v=True):
        for k, vs in in_dict.items():
            k_sid = str_key.get_str2sid(k) if conv_k else k
            for v in vs:
                if isinstance(v, tuple):
                    v_sid_tuple = (
                        str_key.get_str2sid(v[0]) if conv_v else v[0],
                        str_key.get_str2sid(v[1]) if conv_v else v[1],
                    )
                    out_dict[k_sid].add(v_sid_tuple)
                else:
                    v_sid = str_key.get_str2sid(v) if conv_v else v
                    out_dict[k_sid].add(v_sid)

    # Initialize the out dicts
    out_dicts = {}
    for graph_type in cfg.graph_types:
        out_dicts[graph_type] = {ktr: defaultdict(set) for ktr in keys_to_run}

    # Iterate through mapped files build the graphs
    for i, f in enumerate(files):
        ru.processed_log(i, len(files))
        in_data = fu.load_file(f)
        for graph_type in cfg.graph_types:
            for k in out_dicts[graph_type].keys():
                convert_k = True
                convert_v = True
                if k == "cid2children":
                    convert_k = False
                if k in ["ori2cids", "ent2cids"]:
                    convert_v = False
                add_as_sids(
                    in_data["graphs"][graph_type][k],
                    out_dicts[graph_type][k],
                    conv_k=convert_k,
                    conv_v=convert_v,
                )

    # Then dup results
    for graph_type in cfg.graph_types:
        for name, data in out_dicts[graph_type].items():
            fu.dumppkl(data, get_graph_data_path(cfg, graph_type, name)
    logging.info(">> Finished reducing round 2: All graph dicts")


def group_ori_by_ent(
    ori2cidentcount,# input
    str_key,        # input
    sid_normer,     # input
    ori2ent2tot,    # output
    ori2ent2cids,   # output
    ori2ent2pages,  # output
    qstr=False,
    qent=False,
):
    for ori, cidentcount in ori2cidentcount.items():
        # Get the version of ori we're using
        ori_sid_key = str_key.get_str2sid(ori)
        assert ori_sid_key is not None
        if qstr:
            ori_sid_key = sid_normer.get_sid2nsid(ori_sid_key)
            assert ori_sid_key is not None
        if ori_sid_key not in ori2ent2tot:
            ori2ent2tot[ori_sid_key] = {}
            ori2ent2cids[ori_sid_key] = {}
            ori2ent2pages[ori_sid_key] = {}
        for cid, entcount in cidentcount.items():
            page_id = cid.split("__")[0]
            for ent, count in entcount.items():
                ent_sid_key = str_key.get_str2sid(ent)
                assert ent_sid_key is not None
                if qent:
                    ent_sid_key = sid_normer.get_sid2nsid(ent_sid_key)
                    assert ent_sid_key is not None
                if ent_sid_key not in ori2ent2tot[ori_sid_key]:
                    ori2ent2tot[ori_sid_key][ent_sid_key] = 0
                    ori2ent2cids[ori_sid_key][ent_sid_key] = set()
                    ori2ent2pages[ori_sid_key][ent_sid_key] = set()
                ori2ent2tot[ori_sid_key][ent_sid_key] += count
                ori2ent2cids[ori_sid_key][ent_sid_key].add(cid)
                ori2ent2pages[ori_sid_key][ent_sid_key].add(page_id)

def reducing__ori2entdetailed(
    cfg,
    str_key,
    sid_normer,
    graph_types,
    data_types,
    test=False,
):
    files = get_all_mapped_files(cfg.wiki_mapped_dir)
    files = files[:3] if test else files
    
    # Get normalization information for data_types
    data_types_dict = {}
    for data_type in data_types:
        data_types_dict[data_type] = get_norm_bool_from_ent_str_type(data_type)
        
    # Make intermediate dicts for aggregating stats
    int_dicts = {}
    for graph_type in graph_types:
        int_dicts[graph_type] = {}
        for dict_type in data_types_dict.keys():
            int_dicts[graph_type][dict_type] = {
                "ori2ent2tot": {},
                "ori2ent2cids": {},
                "ori2ent2pages": {},
            }

    # First aggregate all the data
    for i, f in enumerate(files):
        ru.processed_log(i, len(files))
        in_file = fu.load_file(f)
        for graph_type in graph_types:
            # ori: cid: ent: num_ori_for_cid_ent
            in_data = in_file["graphs"][graph_type]["ori2entdetailed"]
            for dtname, dt_bools in data_types_dict.items():
                qstr, qent = dt_bools
                group_ori_by_ent(
                    ori2cidentcount=in_data,
                    str_key=str_key,
                    sid_normer=sid_normer,
                    ori2ent2tot=int_dicts[graph_type][dtname]["ori2ent2tot"],
                    ori2ent2cids=int_dicts[graph_type][dtname]["ori2ent2cids"],
                    ori2ent2pages=int_dicts[graph_type][dtname]["ori2ent2pages"],
                    qstr=qstr,
                    qent=qent,
                )

    # Then post-process it into the correct format
    logging.info(">> Begin postprocess to put in the correct form")
    out_dicts = {}
    for graph_type in graph_types:
        # for each graph_type and str_ent combo:
        # ori_sid: [(ent_sid, num_tot, num_unique_cids, num_unique_ent_sids)]
        out_dicts[graph_type] = defaultdict(dict)
    for gt, dt2intdata in int_dicts.items():
        for dt, intdata in dt2intdata.items():
            for ori_sid in intdata["ori2ent2tot"].keys():
                for ent_sid in intdata["ori2ent2tot"][ori_sid].keys():
                    num_tot = intdata["ori2ent2tot"][ori_sid][ent_sid]
                    num_unique_cids = len(intdata["ori2ent2cids"][ori_sid][ent_sid])
                    num_unique_pages = len(intdata["ori2ent2pages"][ori_sid][ent_sid])
                    if ori_sid not in out_dicts[gt][dt]:
                        out_dicts[gt][dt][ori_sid] = []
                    out_dicts[gt][dt][ori_sid].append(
                        Ori2Ent(
                            ent_sid=ent_sid,
                            num_total=intdata["ori2ent2tot"][ori_sid][ent_sid],
                            num_unique_cids=len(
                                intdata["ori2ent2cids"][ori_sid][ent_sid]
                            ),
                            num_unique_pages=len(
                                intdata["ori2ent2pages"][ori_sid][ent_sid]
                            ),
                        )
                    )

    # Finally, save the data
    for graph_type in graph_types:
        for name, data in out_dicts[graph_type].items():
            fu.dumppkl(data, get_ori2entdetailed_path(cfg, graph_type, name)


## --------- v2: Mapping Functions ----------##


def get_inds_in_chunk(chunk, strs, cid=None):
    str_to_inds = {}
    for st in strs:
        try:
            for m in re.finditer(re.escape(st), chunk):
                if st not in str_to_inds:
                    str_to_inds[st] = []
                str_to_inds[st].append((m.start(), m.end()))
        except:
            logging.info(f">> ERROR in re: {cid} {st} {chunk}")
            continue
    return str_to_inds


# Need: str: (str_range, tok_range, ent)
# Assume chunk does not contain title and that they'll be
# concatenated with a [SEP] token
def get_token_inds(tokenizer, title, chunk, strs, prep_title=None, cid=None):
    tok_data = {}

    # Handle title first
    title_len = len(title)
    tok_data["title_str_span"] = (0, title_len)
    title_toks = tokenizer.encode(title, add_special_tokens=False)
    tok_data["title_tok_span"] = (0, len(title_toks))
    tok_data["prep_title_str_span"] = None
    tok_data["prep_title_tok_span"] = None
    if prep_title is not None:
        tok_data["prep_title_str_span"] = (0, len(prep_title))
        prep_title_toks = tokenizer.encode(prep_title, add_special_tokens=False)
        tok_data["prep_title_tok_span"] = (0, len(prep_title_toks))

    # Then handle the chunk content
    strs_found = []
    tok_inds = []
    str_inds = []

    # Explode out all the inds, and sort by first indexg
    st_inds_dict = get_inds_in_chunk(chunk, strs, cid)
    st_inds_list = [(st, i) for st, il in st_inds_dict.items() for i in il]
    st_inds_list = sorted(st_inds_list, key=lambda x: (x[1][0], x[1][1]))
    for st, inds in st_inds_list:
        pre_str = chunk[: inds[0]]
        act_str = chunk[inds[0] : inds[1]]
        through_str = chunk[: inds[1]]

        before_ind = tokenizer.encode(pre_str, add_special_tokens=False)
        start_token_ind = len(before_ind)
        after_ind = tokenizer.encode(through_str, add_special_tokens=False)
        end_token_ind = len(after_ind)

        if start_token_ind == end_token_ind:
            continue

        if len(tok_inds) > 0:
            last_token_inds = tok_inds[-1]
            if (
                start_token_ind == last_token_inds[0]
                and end_token_ind == last_token_inds[1]
            ):
                continue

        strs_found.append(st)
        tok_inds.append((start_token_ind, end_token_ind))
        str_inds.append(inds)

    chunk_str_start_ind = title_len + len(tokenizer.sep_token)
    chunk_tok_start_ind = len(title_toks) + 1
    tok_data["passage_spans"] = {}
    for i, st in enumerate(strs_found):
        if st not in tok_data["passage_spans"]:
            tok_data["passage_spans"][st] = {
                "tok_spans": [],
                "str_spans": [],
            }
        tok_data["passage_spans"][st]["tok_spans"].append(
            tuple([ind + chunk_tok_start_ind for ind in tok_inds[i]])
        )
        tok_data["passage_spans"][st]["str_spans"].append(
            tuple([ind + chunk_str_start_ind for ind in str_inds[i]])
        )
    return tok_data


def split_cid(cid):
    page_id, chunk_num = cid.split("__")
    return page_id, int(chunk_num)


def chunk_data_to_passage_data(chunk_data, chunk_num=None):
    if chunk_num is None:
        _, chunk_num = split_cid(chunk_data["chunk_id"])
    out_dict = {
        k: v.strip()
        for k, v in chunk_data.items()
        if k
        in [
            "title",
            "prep_title",
            "qnn_title",
            "qnn_prep_title",
            "on_title",
            "qn_title",
        ]
    }
    if out_dict["qnn_title"] is None:
        out_dict["qnn_title"] = ""
    if out_dict["qnn_prep_title"] is None:
        out_dict["qnn_prep_title"] = ""
    out_dict["passage_list"] = [chunk_data["content"]]
    out_dict["chunk_num_list"] = [chunk_num]
    return out_dict


def chunk_data_to_token_data(chunk_data, tokenizer):
    prep_title = chunk_data["prep_title"].strip()
    tag_strs = list(set([s[0] for s in chunk_data["tags"]]))
    tags_and_links_strs = list(set(tag_strs + [s[0] for s in chunk_data["links"]]))
    just_tags_tok_data = get_token_inds(
        tokenizer,
        chunk_data["title"],
        chunk_data["content"],
        tag_strs,
        prep_title=prep_title,
        cid=chunk_data["chunk_id"],
    )
    tags_and_links_tok_data = get_token_inds(
        tokenizer,
        chunk_data["title"],
        chunk_data["content"],
        tags_and_links_strs,
        prep_title=prep_title,
        cid=chunk_data["chunk_id"],
    )
    return {
        "just_tags": just_tags_tok_data,
        "tags_and_links": tags_and_links_tok_data,
    }


def init_graphs(get_graph_key):
    graphs = {}
    for use_links in [True, False]:
        key = get_graph_key(use_links)
        graphs[key] = {
            "ori2ents": defaultdict(set),  # str -> {ent, }
            "ent2oris": defaultdict(set),  # ent -> {str, }
            "cid2children": defaultdict(set),  # cid -> {(str, ent), ...}
            "ori2cids": defaultdict(set),  # str -> {cid, }
            "ent2cids": defaultdict(set),  # ent -> {cid, }
            "ori2entdetailed": defaultdict(
                dict
            ),  # str -> {(cid, ent, num_strent_in_cid), }
        }
    return graphs


def init_sets(get_graph_key):
    sets = {}
    for use_links in [True, False]:
        key = get_graph_key(use_links)
        sets[key] = {
            "title": set(),
            "prep_title": set(),
            "linked_entity": set(),
            "ori_text": set(),
        }
    return sets


def update_with_chunk_data(
    pdl, tkns, graphs, sets, str2qnn, get_graph_key, tokenizer, chunk_data
):
    cid = chunk_data["chunk_id"]
    page_id, chunk_num = split_cid(cid)
    title = chunk_data["title"]
    prep_title = chunk_data["prep_title"].strip()

    # if '&quot;' in title or '&amp;' in title:
    #    print(f"Title: {title} vs", title.replace('&quot;', '"').replace('&amp;', '&'))

    # Passage Data List
    if page_id in pdl:
        pdl[page_id]["passage_list"].append(chunk_data["content"])
        pdl[page_id]["chunk_num_list"].append(chunk_num)
    else:
        pdl[page_id] = chunk_data_to_passage_data(chunk_data, chunk_num=chunk_num)

    # Tokens
    tkns[cid] = chunk_data_to_token_data(chunk_data, tokenizer)

    # Rest of graph creation (for tags and tags_links separately)
    for use_links in [True, False]:
        key = get_graph_key(use_links)
        if cid == "65526679__0":
            print(cid, use_links, key)

        # Title and prep title
        if chunk_data["qnn_title"] is None:
            str2qnn[title] = ""
        else:
            str2qnn[title] = chunk_data["qnn_title"]
        sets[key]["title"].add(title)
        graphs[key]["ori2ents"][title].add(title)
        graphs[key]["ent2oris"][title].add(title)
        graphs[key]["ori2cids"][title].add(cid)
        graphs[key]["ent2cids"][title].add(cid)

        if prep_title != title:
            if chunk_data["qnn_prep_title"] is None:
                qnn_prep_title = ""
            else:
                qnn_prep_title = chunk_data["qnn_prep_title"].strip()
            str2qnn[prep_title] = qnn_prep_title
            sets[key]["prep_title"].add(prep_title)
            graphs[key]["ori2ents"][prep_title].add(prep_title)
            graphs[key]["ent2oris"][prep_title].add(prep_title)
            graphs[key]["ori2cids"][prep_title].add(cid)
            graphs[key]["ent2cids"][prep_title].add(cid)

        # Links and Tags
        parsed_str_list = [[l for l in ll] for ll in chunk_data["tags"]]
        if use_links:
            for llist in chunk_data["links"]:
                normed_link_ent = norm_links(llist[1])
                if normed_link_ent == "":
                    continue
                new_llist = [
                    llist[i] if i != 1 else normed_link_ent for i in range(len(llist))
                ]
                parsed_str_list.append(new_llist)

        for ori_to_ent in parsed_str_list:
            ori, ent, qnn_ori, qnn_ent = ori_to_ent
            if qnn_ori is None:
                qnn_ori = ""
            if qnn_ent is None:
                qnn_ent = ""
            if ori is None:
                ori = ""
            if ent is None:
                ent = ""
            if "&quot;" in ori or "&amp;" in ori:
                ori = ori.replace("&quot;", '"').replace("&amp;", "&")
            if "&quot;" in ent or "&amp;" in ent:
                ent = ent.replace("&quot;", '"').replace("&amp;", "&")

            str2qnn[ori] = qnn_ori
            str2qnn[ent] = qnn_ent
            sets[key]["ori_text"].add(ori)
            sets[key]["linked_entity"].add(ent)
            graphs[key]["ori2ents"][ori].add(ent)
            graphs[key]["ent2oris"][ent].add(ori)
            graphs[key]["cid2children"][cid].add((ori, ent))
            graphs[key]["ori2cids"][ori].add(cid)
            graphs[key]["ent2cids"][ent].add(cid)
            if cid not in graphs[key]["ori2entdetailed"][ori]:
                graphs[key]["ori2entdetailed"][ori][cid] = {ent: 1}
            elif ent not in graphs[key]["ori2entdetailed"][ori][cid]:
                graphs[key]["ori2entdetailed"][ori][cid][ent] = 1
            else:
                graphs[key]["ori2entdetailed"][ori][cid][ent] += 1


def process_chunked_fixed_file(
    input_file, output_dir=None, tokenizer=None, verbose=False
):
    if tokenizer is None:
        tokenizer = tu.initialize_tokenizer(PATH_ARGS.tokenizer_config)

    if output_dir is None:
        output_dir = PATH_ARGS.wiki_chunked_fixed_parsed_dir

    input_filename = input_file.split("/")[-1][: -len(".jsonl")]
    get_graph_key = lambda use_links: "graph_data_" + (
        "tags_and_links" if use_links else "just_tags"
    )

    out_file = get_mapped_file(output_dir, input_filename)
    if os.path.exists(out_file):
        logging.info(f">> Out file already exists, skip: {out_file}")
        return

    # Initialize the data structures
    str2qnn = {}
    passage_data = {}  # page_id -> {}
    tokens = {}  # cid -> {}
    graphs = init_graphs(get_graph_key)
    sets = init_sets(get_graph_key)

    # Read in the file and process
    with jsonlines.open(input_file) as reader:
        for i, chunk_data in enumerate(reader):
            ru.processed_log(i, every=100)
            update_with_chunk_data(
                passage_data,
                tokens,
                graphs,
                sets,
                str2qnn,
                get_graph_key,
                tokenizer,
                chunk_data,
            )

    # Dump data
    for k in graphs.keys():
        for n in graphs[k].keys():
            graphs[k][n] = dict(graphs[k][n])
    output_data = {
        "passage_data": passage_data,
        "tokens": tokens,
        "graphs": graphs,
        "sets": sets,
        "str2qnn": str2qnn,
    }
    fu.dumppkl(output_data, out_file)


def process_all_chunked_fixed(input_dir, processes=100, test=False, mod=None, eq=None):
    all_files = sorted(glob.glob(f"{input_dir}wikipedia_chunks_*.jsonl"))
    if test:
        # all_files = all_files[:processes]
        all_files = [
            "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia_chunked_fixed/wikipedia_chunks_14384.jsonl",
            "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia_chunked_fixed/wikipedia_chunks_14641.jsonl",
            "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia_chunked_fixed/wikipedia_chunks_1496.jsonl",
            "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia_chunked_fixed/wikipedia_chunks_15009.jsonl",
        ]
    if mod is not None and eq is not None:
        all_files = [f for i, f in enumerate(all_files) if i % mod == eq]
    logging.info(f">> Processing {len(all_files)} with {processes} processes")

    with multiprocessing.Pool(processes=processes) as pool:
        result_list = []
        for _, res in enumerate(
            pool.imap_unordered(process_chunked_fixed_file, all_files)
        ):
            result_list.append(res)


## --------- Postprocess Wikipedia ----------##


def glob_alpha_segs(top_wiki_dir):
    return sorted(glob.glob(f"{top_wiki_dir}[A-Z][A-Z]"))


def glob_alpha_subsegs(alpha_path):
    return sorted(glob.glob(f"{alpha_path}/wiki_[0-9][0-9]"))


def glob_all_wiki_files(top_wiki_dir):
    return sorted(glob.glob(f"{top_wiki_dir}[A-Z][A-Z]/wiki_[0-9][0-9]"))


def process_tag(raw_tag):
    if raw_tag is not None:
        qn_tag = qmp_norm(raw_tag)
        nqn_tag = qmp_norm(normalize(raw_tag))
        if nqn_tag is not None and len(nqn_tag) > 0:
            return nqn_tag, qn_tag
    return None, None


def process_link(raw_link):
    if raw_link is not None:
        fixed_link = norm_links(raw_link)
        if fixed_link is not None and len(fixed_link) > 0:
            qn_link = qmp_norm(fixed_link)
            nqn_link = qmp_norm(normalize(fixed_link))
            if nqn_link is not None and len(nqn_link) > 0:
                return nqn_link, qn_link
    return None, None


# ---------------- Making and Dealing with Chunks ------------------#


def expand_ldata(orig_ldata):
    ldata = {**orig_ldata}

    ## Expand the title
    title = ldata["title"]
    prep_title = title.split("(")[0]  # pre_process_title
    ldata["prep_title"] = prep_title
    ldata["qnn_title"] = qmp_norm(normalize(title))
    ldata["qnn_prep_title"] = qmp_norm(normalize(prep_title))
    # for backwards compatability
    ldata["qn_title"] = qmp_norm(title)
    ldata["on_title"] = old_norm(title)

    ## Expand the links and tags
    extended_links = []
    for l in ldata["links"]:
        ori_text, ent_str = l
        qnn_ori_text = qmp_norm(normalize(ori_text))

        fixed_ent_str = norm_links(ent_str)
        if fixed_ent_str is None:
            qnn_ent_str = None
        else:
            qnn_ent_str = qmp_norm(normalize(fixed_ent_str))
        extended_links.append(
            [
                ori_text,
                ent_str,
                qnn_ori_text,
                qnn_ent_str,
            ]
        )
    extended_tags = []
    for t in ldata["tags"]:
        ori_text, ent_str = t
        qnn_ori_text = qmp_norm(normalize(ori_text))
        if ent_str is None:
            qnn_ent_str = None
        else:
            qnn_ent_str = qmp_norm(normalize(ent_str))
        extended_tags.append(
            [
                ori_text,
                ent_str,
                qnn_ori_text,
                qnn_ent_str,
            ]
        )
    ldata["links"] = extended_links
    ldata["tags"] = extended_tags
    return ldata


def long_chunk_to_small_chunks(
    in_ldata, max_num_words=200, goal_num_words=100, period_threshold=40
):
    chunk = in_ldata["content"]
    cid = in_ldata["chunk_id"]
    tags = in_ldata["tags"]
    links = in_ldata["links"]

    cid_base, cid_part = cid.split("__")
    chunk_left = chunk
    new_chunks = []
    while len(chunk_left) > 0:
        chunk_left_words = chunk_left.split()
        first_words = chunk_left_words[:goal_num_words]
        first_words_len = len(" ".join(first_words))
        end_ind = first_words_len
        switch_to_spaces = goal_num_words + period_threshold
        while end_ind < len(chunk_left) and (
            (
                len(chunk_left[:end_ind].split()) < switch_to_spaces
                and chunk_left[end_ind] != "."
            )
            or (
                len(chunk_left[:end_ind].split()) >= switch_to_spaces
                and chunk_left[end_ind] != " "
            )
        ):
            end_ind += 1
        end_ind += 1  # include the space or period in the prev chunk
        new_chunk = chunk_left[:end_ind]
        new_chunks.append(new_chunk)
        if end_ind == len(chunk_left):
            chunk_left = ""
        else:
            chunk_left = chunk_left[end_ind:]
    new_chunk_cids = [f"{cid_base}__{int(cid_part)+i}" for i in range(len(new_chunks))]

    all_tags = {
        "tags": tags,
        "links": links,
    }
    all_new_tags = {}
    for ttype, tgs in all_tags.items():
        new_tags = [[]]
        chunk_ind = 0
        num_tries = 0
        unfound = []
        for ori_text, ent_str in tags:
            if num_tries > 5:
                unfound.append([ori_text, ent_str])
                break
            if ori_text in new_chunks[chunk_ind]:
                new_tags[chunk_ind].append((ori_text, ent_str))
            else:
                if chunk_ind + 1 >= len(new_chunks):
                    chunk_ind = 0
                    num_tries += 1
                else:
                    chunk_ind += 1
                    new_tags.append([])
        all_new_tags[ttype] = [list(set(str_tuple_list)) for str_tuple_list in new_tags]
        if len(unfound) > 0:
            for ori_text, ent_str in unfound:
                logging.info(
                    f">> ERROR: unfound {ttype} for {cid}: {ori_text} {ent_str}"
                )
    new_tags_list = all_new_tags["tags"]
    new_links_list = all_new_tags["links"]

    new_ldata = []
    for i, new_chunk in enumerate(new_chunks):
        nldata = {**in_ldata}
        nldata["content"] = new_chunk
        nldata["chunk_id"] = new_chunk_cids[i]
        if i < len(new_tags_list):
            nldata["tags"] = new_tags_list[i]
        else:
            nldata["tags"] = []
        if i < len(new_links_list):
            nldata["links"] = new_links_list[i]
        else:
            nldata["links"] = []
        new_ldata.append(nldata)
    return new_ldata


def write_fixed_too_long_chunks(
    file_list,
    max_num_words=200,
    goal_num_words=100,
    period_threshold=40,
):
    fixed_out_dir = PATH_ARGS.wiki_fixed_chunk_dir
    for i, in_path in enumerate(file_list):
        ru.processed_log(i, len(file_list))
        out_path = fixed_out_dir + in_path.split("/")[-1]
        if os.path.exists(out_path):
            logging.info(f">> Found {out_path} so skip")
            continue
        with jsonlines.Writer(open(out_path, "w+"), flush=True) as writer:
            with jsonlines.open(in_path) as reader:
                for l in reader:
                    ldata = l["meta"]
                    chunk = ldata["content"]
                    if len(chunk) > max_num_words * 3:
                        if len(chunk.split()) <= max_num_words:
                            # actually, its fine, write and continue
                            writer.write(expand_ldata(ldata))
                            continue

                        # Process and write each new chunk then continue
                        new_ldata_list = long_chunk_to_small_chunks(
                            ldata,
                            max_num_words,
                            goal_num_words,
                            period_threshold,
                        )
                        for new_ldata in new_ldata_list:
                            writer.write(expand_ldata(new_ldata))
                        continue

                    # If its not too long, just write it directly
                    writer.write(expand_ldata(ldata))
    logging.info(">> Done!")
