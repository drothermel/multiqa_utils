import os

# Used for processing new wiki dump
import re
import html

from utils.parallel_utils import FileProcessor


class WikiChunker(FileProcessor):
    def __init__(cfg):
        self.cfg = cfg
        self.chunked_dir = cfg.wiki_processing.wiki_chunked_dir
        super().__init__(
            cfg.wiki_processing.num_threads,
            cfg.wiki_processing.num_procs,
            cfg.wiki_processing.proc_batch_size,
            skip_exists=cfg.wiki_processing.skip_exists,
        )

    def get_output_from_inpath(self, file_path):
        


    def get_chunked_filepath(init_filepath):
        # basepath/AA/wiki_00
        init_name = fu.get_file_from_path(init_filepath)
        init_dir = fu.get_dir_from_path(init_filepath)
        sub_dirname = init_dir.split(os.path.sep)
        return os.path.join(self.chunked_dir, f'{sub_dirname}_{init_name}.pkl')
    

    def _chunk_wiki_file(self, filepath):
        
        

# Pool fxn for an input file list
# Chunk one original wiki file per process
def chunk_wiki_files(input_files, output_dir, num_procs):
    os.makedirs(output_dir, exist_ok=True)
    fxn_args_list = list(zip(input_files, repeat(output_dir)))
    ru.apply_pool(get_chunks, fxn_args_list, len(input_files), processes=num_procs)

# From qampari github
# DataCreation/DataAlginment/utils/alignment_utils.py
def fix_parsed_wiki_text(sample):
    text_splits = sample["clean_text"].split("&gt;")
    test_splits_new = [
        text_splits[i].replace("&lt;/a", "")
        if i % 2 == 1
        else text_splits[i].split("&lt;a")[0]
        for i in range(len(text_splits))
    ]
    text_new = " ".join(test_splits_new)
    sample["text"] = text_new
    return sample

# From qampari github
# Pulled from DataCreation/DataAlginment/utils/alignment_utils.py
def read_parsed_wikipedia(input_file):
    total_data = list()
    with open(input_file, "r") as f:
        for line in f.readlines():
            curr_data = json.loads(line.strip())
            total_data.append(fix_text(curr_data))
    return total_data

# TODO: get rid of this
# From qampari github
def _read_parsed_wiki(file_path):
    dump = read_parsed_wikipedia(file_path)
    for el in dump:
        el["file_path"] = file_path
        yield el


# # ---- Old Mapper & Reducer Fxns ---- # #

def get_mapped_file(cfg, input_name):
    mapped_dir = cfg.wiki_processing.wiki_mapped_dir
    return f"{mapped_dir}{input_name}_parsed.pkl"


def get_all_mapped_files(cfg):
    return sorted(glob.glob(get_mapped_file(cfg, "*")))


def reducing__pagedata_strkeys_sidsets_sidnormer(cfg, test=False):
    files = get_all_mapped_files(cfg)
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

    # Iterate through mapped files adding to data classes
    for i, f in enumerate(files):
        ru.processed_log(i, len(files))
        in_data = fu.load_file(f)
        # Note order matters, make keys then use them for str -> sid only
        mapped_wiki_into_passage_data(passage_data, in_data["passage_data"])
        mapped_wiki_into_qnnstrkey(qnn_str_key, in_data["str2qnn"])
        mapped_wiki_into_strkey(all_str_key, in_data["sets"])
        mapped_wiki_into_sidnormer(
            all_str_key, qnn_str_key, sid_normer, in_data["str2qnn"]
        )

    # Save order matters due to memory used for post-processing pre-aving
    passage_data.save()
    del passage_data
    # These takes extra memory to save, so do these last
    sid_normer.save(force=True)
    del sid_normer
    qnn_str_key.save(force=True)
    del qnn_str_key
    all_str_key.save(force=True)
    del all_str_key
    logging.info(">> Finished Reducing Part 1: StrKeys, PassageData, and SidNormer")


def reducing__sets(
    cfg,
    all_str_key,
    return_struct=False,
    test=False,
):
    sid_sets = {}

    # Run Reduce
    mapped_files = get_all_mapped_files(cfg)
    if test:
        mapped_files = mapped_files[:3]
    for ind, mf in enumerate(mapped_files):
        ru.processed_log(ind, len(mapped_files))
        in_data = fu.load_file(mf, verbose=False)
        mapped_wiki_into_sidsets(
            cfg,
            all_str_key,
            sid_sets,
            in_data['sets'],
        )

    # Verify and dump
    logging.info(">> Verify Set Reduce Results:")
    for graph_t, sets_data in sid_sets.items():
        for set_t, data in sets_data.items():
            logging.info(f"   [{graph_t:30}][{set_t:10}] set size: {len(data):,}")
            out_path = get_set_data_path(cfg, graph_t, set_t)
            if test:
                logging.info(">> Running in test mode, don't dump")
                continue
            fu.dumppkl(data, out_path, verbose=False)

    logging.info("")
    logging.info(">> Finished reducing sets")
    if return_struct:
        return sid_sets


def reducing__graphs(cfg, str_key, input_dir, keys_to_run=None, test=False):
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

    # Then dump results
    for graph_type in cfg.graph_types:
        for name, data in out_dicts[graph_type].items():
            fu.dumppkl(data, get_graph_data_path(cfg, graph_type, name))
    logging.info(">> Finished Reducing Part 3: All graph dicts")


# Goal: mapping from norm_str to all matching sids (of any type)
# Need: sid_sets, all_str_key, qnn_str_key, sid_normer
# Return: graph_t: norm_t: norm_str: sids
def reducing__norm2sids(
    cfg,
    sid_sets,
    all_str_key,
    qnn_str_key,
    sid_normer,
    test=False,
):
    # Run Reduce & Verify
    logging.info(">> Verifying the norm2sid reduce:")
    all_norm_fxns = su.get_all_norm_fxns(
        all_str_key=all_str_key,
        qnn_str_key=qnn_str_key,
        sid_normer=sid_normer,
    )
    for graph_t, sets in sid_sets.items():
        all_sids = set()
        for sid_set in sets.values():
            all_sids.update(sid_set)

        for norm_t, norm_fxn in all_norm_fxns.items():
            norm2sids = defaultdict(set)
            for ind, sid in enumerate(all_sids):
                st = all_str_key.get_sid2str(sid)
                norm_str = norm_fxn(st)
                norm2sids[norm_str].add(sid)
                if test and ind > 30000:
                    break

            outpath = get_norm2sids_path(cfg, graph_t, norm_t)
            if not test:
                fu.dumppkl(dict(norm2sids), outpath, verbose=False)
            else:
                logging.info(">> Running in test mode, don't dump")

            # Calculate Verify stats
            metrics = Metrics()
            for norm_str, sids in norm2sids.items():
                metrics.add_to_hists_per('num_sids', None, len(sids))
            metrics.update_agg_stats()
            metrics_str = f" - [{graph_t:30}][{norm_t:20}]"
            for val_name, val in metrics.vals.items():
                metrics_str += f' {val_name}: {val:0.2f}'
            logging.info(metrics_str)

    logging.info("")
    logging.info(">> Finished reducing norm2sids")


def group_ori_by_ent(
    ori2cidentcount,  # input
    str_key,  # input
    sid_normer,  # input
    ori2ent2tot,  # output
    ori2ent2cids,  # output
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
            page_id, _ = split_cid(cid)
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
                            num_total=num_tot,
                            num_unique_cids=num_unique_cids,
                            num_unique_pages=num_unique_pages,
                        )
                    )

    # Finally, save the data
    for graph_type in graph_types:
        for name, data in out_dicts[graph_type].items():
            fu.dumppkl(data, get_ori2entdetailed_path(cfg, graph_type, name))

    logging.info(">> Finished Reducing Part 5: Ori2entdetailed")


# # --------- v2: Mapping Functions ----------# #


def get_inds_in_chunk(chunk, strs, cid=None):
    str_to_inds = {}
    for st in strs:
        try:
            for m in re.finditer(re.escape(st), chunk):
                if st not in str_to_inds:
                    str_to_inds[st] = []
                str_to_inds[st].append((m.start(), m.end()))
        except:  # noqa: E722
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
        # act_str = chunk[inds[0] : inds[1]]
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
        parsed_str_list = [[li for li in ll] for ll in chunk_data["tags"]]
        if use_links:
            for llist in chunk_data["links"]:
                normed_link_ent = su.norm_links(llist[1])
                if normed_link_ent == "":
                    continue
                new_llist = [
                    llist[ii] if ii != 1 else normed_link_ent
                    for ii in range(len(llist))
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
    input_file,
    tokenizer_config_path,
    wiki_chunked_fixed_parsed_dir,
    output_dir=None,
    tokenizer=None,
    verbose=False,
):
    if tokenizer is None:
        tokenizer = tu.initialize_tokenizer(tokenizer_config_path)

    if output_dir is None:
        output_dir = wiki_chunked_fixed_parsed_dir

    input_filename = input_file.split("/")[-1][: -len(".jsonl")]

    def get_graph_key(use_links):
        return "graph_data_" + ("tags_and_links" if use_links else "just_tags")

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
        base = "/scratch/ddr8143/wikipedia/tagme_dumps_qampari_wikipedia_chunked_fixed/"
        all_files = [
            f"{base}wikipedia_chunks_14384.jsonl",
            f"{base}wikipedia_chunks_14641.jsonl",
            f"{base}wikipedia_chunks_1496.jsonl",
            f"{base}wikipedia_chunks_15009.jsonl",
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


# # --------- Postprocess Wikipedia ----------# #


def glob_alpha_segs(top_wiki_dir):
    return sorted(glob.glob(f"{top_wiki_dir}[A-Z][A-Z]"))


def glob_alpha_subsegs(alpha_path):
    return sorted(glob.glob(f"{alpha_path}/wiki_[0-9][0-9]"))


def glob_all_wiki_files(top_wiki_dir):
    return sorted(glob.glob(f"{top_wiki_dir}[A-Z][A-Z]/wiki_[0-9][0-9]"))


def process_tag(raw_tag):
    if raw_tag is not None:
        qn_tag = su.qmp_norm(raw_tag)
        nqn_tag = su.qnn_norm(su.get_detokenizer(), raw_tag)
        if nqn_tag is not None and len(nqn_tag) > 0:
            return nqn_tag, qn_tag
    return None, None


def process_link(raw_link):
    if raw_link is not None:
        fixed_link = su.norm_links(raw_link)
        if fixed_link is not None and len(fixed_link) > 0:
            qn_link = su.qmp_norm(fixed_link)
            nqn_link = su.qnn_norm(su.get_detokenizer(), fixed_link)
            if nqn_link is not None and len(nqn_link) > 0:
                return nqn_link, qn_link
    return None, None


# ---------------- Chunks: Produce Chunked Fixed ------------------#


def expand_ldata(orig_ldata):
    ldata = {**orig_ldata}

    # # Expand the title
    dtk = su.get_detokenizer()
    title = ldata["title"]
    prep_title = title.split("(")[0]  # pre_process_title
    ldata["prep_title"] = prep_title
    ldata["qnn_title"] = su.qnn_norm(dtk, title)
    ldata["qnn_prep_title"] = su.qnn_norm(dtk, prep_title)
    # for backwards compatability
    ldata["qn_title"] = su.qmp_norm(title)
    ldata["on_title"] = su.old_norm(title)

    # # Expand the links and tags
    extended_links = []
    for link in ldata["links"]:
        ori_text, ent_str = link
        qnn_ori_text = su.qnn_norm(dtk, ori_text)

        fixed_ent_str = su.norm_links(ent_str)
        if fixed_ent_str is None:
            qnn_ent_str = None
        else:
            qnn_ent_str = su.qnn_norm(dtk, fixed_ent_str)
        extended_links.append(
            [
                ori_text,
                ent_str,
                qnn_ori_text,
                qnn_ent_str,
            ]
        )
    extended_tags = []
    for tag in ldata["tags"]:
        ori_text, ent_str = tag
        qnn_ori_text = su.qnn_norm(dtk, ori_text)
        if ent_str is None:
            qnn_ent_str = None
        else:
            qnn_ent_str = su.qnn_norm(dtk, ent_str)
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

# ---------------- Chunks: Re-Split to Fixed Sized Chunks ------------------#

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
    wiki_fixed_chunk_dir,
    max_num_words=200,
    goal_num_words=100,
    period_threshold=40,
):
    fixed_out_dir = wiki_fixed_chunk_dir
    for i, in_path in enumerate(file_list):
        ru.processed_log(i, len(file_list))
        out_path = fixed_out_dir + in_path.split("/")[-1]
        if os.path.exists(out_path):
            logging.info(f">> Found {out_path} so skip")
            continue
        with jsonlines.Writer(open(out_path, "w+"), flush=True) as writer:
            with jsonlines.open(in_path) as reader:
                for line in reader:
                    ldata = line["meta"]
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


# ##################################
# #    Process New Wikidump       ##
# ##################################

# After wikiextractor has already processed the wikidump then we can
# use this to create base files for a page index.
#
# HTML Escaping from:
# https://medium.com/@jorlugaqui/
# how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
# This will:
#    1) Remove any html tags remaining from the text
#    2) Append the keywords "Title:" and "Article:" along with the title to the text
#    3) Format the final output file into a .jsonl in the format expected by
#       pyserini index builder
def postprocess_wikipedia_segment_to_page_index(infile, outfile, verbose=True):
    clean = re.compile("<.*?>")
    orig_file = fu.load_file(infile, ending=".jsonl")

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
