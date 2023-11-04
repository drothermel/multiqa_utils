import glob
import os
import psutil
import time
import multiprocessing
import random
import logging

import utils.file_utils as fu

import maqa_utils.wiki_utils as wu
import maqa_utils.tokenizer_utils as tu

from maqa_utils.helper_classes import PassageData, StringKey, SidNormer


# ----- General Purpose Helpers ------ #
def print_dict(dct):
    for k, v in dct.items():
        if isinstance(v, dict):
            print(f" - {k}:")
            for kk, vv in v.items():
                print(f"     = {kk}: {vv}")
        elif isinstance(v, list):
            print(f" - {k}:")
            for vv in v:
                print(f"   = {vv}")
        else:
            print(f" - {k}: {v}")


def topline_keys(in_dict):
    for k, v in in_dict.items():
        print(f" - {k}")
        if isinstance(v, list):
            if isinstance(v[0], dict):
                for kk, vv in v[0].items():
                    print(f"     - {kk}")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"     - {kk}")


def get_current_process():
    curr_pid = os.getpid()
    for p in psutil.process_iter():
        if p.pid == curr_pid:
            return p
    return None


def get_process_mem_gb(process):
    md = process.memory_info()._asdict()
    # could return a dict of 'rss', 'vms', 'shared', 'data'
    return md["rss"] / 1e9


def get_curr_process_mem_gb():
    return get_process_mem_gb(get_current_process())


def load_time_space(load_lambda, arg_dict):
    start_time = time.time()
    start_mem_info = get_curr_process_mem_gb()
    obj = load_lambda(arg_dict)
    total_time = time.time() - start_time
    total_space_gb = get_curr_process_mem_gb() - start_mem_info
    return obj, total_time, total_space_gb


# ----- Test the Mapping Utils (wu and tu) ----- #


def test__chunk_data_to_passage_data(filename, chunk_ind=0, verbose=False):
    raw_chunks = fu.load_file(filename, verbose=verbose)
    test_chunk = raw_chunks[chunk_ind]

    if verbose:
        print("Raw Chunk:")
        print_dict(test_chunk)

    out_dict = wu.chunk_data_to_passage_data(test_chunk)
    if verbose:
        print()
        print()
        print("Output:")
        print_dict(out_dict)
    print(">> [Passed] Function ran, use verbose=True to see output")


def test__chunk_data_to_token_data(filename, chunk_ind=0, verbose=False):
    raw_chunks = fu.load_file(filename, verbose=verbose)
    test_chunk = raw_chunks[chunk_ind]

    if verbose:
        print("Raw Chunk:")
        print_dict(test_chunk)

    tokenizer = tu.initialize_tokenizer(wu.PATH_ARGS.tokenizer_config)
    out_dict = wu.chunk_data_to_token_data(test_chunk, tokenizer)

    if verbose:
        print()
        print()
        print("Output:")
        print_dict(out_dict)
        print()

    print("------ Tests ------")
    tags_vs_tags_and_links = len(out_dict["just_tags"]["passage_spans"]) <= len(
        out_dict["tags_and_links"]["passage_spans"]
    )
    print(f'{"--> Len tags and links >= len tags?:":50} [{tags_vs_tags_and_links}]')
    full_test_chunk = test_chunk["title"] + tokenizer.sep_token + test_chunk["content"]
    full_test_tokens = tokenizer.encode(full_test_chunk, add_special_tokens=False)
    all_pass_str = True
    all_pass_tok = True
    for key in ["just_tags", "tags_and_links"]:
        for st, spdict in out_dict[key]["passage_spans"].items():
            for i in range(len(spdict["str_spans"])):
                str_sp = spdict["str_spans"][i]
                tok_sp = spdict["tok_spans"][i]
                if full_test_chunk[str_sp[0] : str_sp[1]] != st:
                    all_pass_str = False
                tok_str_gt = tokenizer.decode(
                    tokenizer.encode(st, add_special_tokens=False)
                )
                tok_str = tokenizer.decode(full_test_tokens[tok_sp[0] : tok_sp[1]])
                if tok_str_gt != tok_str:
                    all_pass_tok = False
    print(f'{"--> All string spans match:":50} [{all_pass_str}]')
    print(f'{"--> All tok spans match:":50} [{all_pass_tok}]')


def test__update_with_chunk_data(filename, chunk_ind=0, verbose=False):
    raw_chunks = fu.load_file(filename, verbose=verbose)
    test_chunk = raw_chunks[chunk_ind]
    tokenizer = tu.initialize_tokenizer(wu.PATH_ARGS.tokenizer_config)

    def get_graph_key(use_links):
        return "tags_and_links" if use_links else "just_tags"

    str2qnn = {}
    pdl = {}  # page_id -> {}
    tkns = {}  # cid -> {}
    graphs = wu.init_graphs(get_graph_key)
    sets = wu.init_sets(get_graph_key)

    def print_data():
        if not verbose:
            return
        print("Graphs:")
        print_dict(graphs)

        print("Sets:")
        print_dict(sets)

        print("Str2QNN:")
        print_dict(str2qnn)

        print("Page data:")
        print_dict(pdl)

        print("Tokens:")
        print_dict(tkns)

    if verbose:
        print(" === Init === ")
    print_data()

    wu.update_with_chunk_data(
        pdl, tkns, graphs, sets, str2qnn, get_graph_key, tokenizer, test_chunk
    )

    if verbose:
        print()
        print(" === GT 1 === ")
        print_dict(test_chunk)

    if verbose:
        print()
        print(" === After first Chunk === ")
    print_data()

    test_chunk = raw_chunks[chunk_ind + 1]
    wu.update_with_chunk_data(
        pdl, tkns, graphs, sets, str2qnn, get_graph_key, tokenizer, test_chunk
    )

    if verbose:
        print()
        print(" === GT 2 === ")
        print_dict(test_chunk)

    if verbose:
        print()
        print(" === After second Chunk === ")
    print_data()
    print(">> [Passed] Function ran, use verbose=True to see output")


def test__process_chunked_fixed_file(file, verbose=False):
    tokenizer = tu.initialize_tokenizer(wu.PATH_ARGS.tokenizer_config)
    outdir = "/scratch/ddr8143/repos/multi-answer-qa/full_pipeline/notebooks/"
    wu.process_chunked_fixed_file(file, outdir, tokenizer)
    input_filename = file.split("/")[-1][: -len(".jsonl")]
    output_file = f"{outdir}{input_filename}_parsed.pkl"
    output = fu.load_file(output_file, verbose=verbose)
    if verbose:
        for k in output.keys():
            if len(output[k].keys()) > 10:
                print(k, len(output[k]))
            else:
                print(k, output[k].keys())
                for kk in output[k].keys():
                    if len(output[k][kk].keys()) > 10:
                        print(kk, len(output[k][kk]))
                    else:
                        print(kk, output[k][kk].keys())
    print(">> [Passed] Function ran, use verbose=True to see output")


def test_datastructs(verbose=False):
    # Load datasets (note if you use the full dataset this will take a LONG time)
    all_str_key = StringKey(wu.PATH_ARGS.stringkeys_dir, "ori_v0")
    qnn_str_key = StringKey(wu.PATH_ARGS.stringkeys_dir, "qnn_v0")
    passage_data = PassageData(
        wu.PATH_ARGS.passage_data_dir,
        "v0",
        prep2title=True,
        extra_norms=True,
    )
    sid_normer = SidNormer(wu.PATH_ARGS.sidnormer_dir, "v0")

    # Verify the consistency between str2sid and sid2str
    for i in [1, 24, 6666, 3994]:
        assert i == all_str_key.get_str2sid(all_str_key.get_sid2str(i))
    for i in [1, 24, 6666, 3994]:
        assert i == qnn_str_key.get_str2sid(qnn_str_key.get_sid2str(i))

    # Verify passage data getters and coherency
    page_id, page_data = next(iter(passage_data.page_data.items()))
    page_title = page_data.title
    assert passage_data.title2pid[page_title] == page_id
    assert page_title == passage_data.get_title_from_pid(page_id)
    page_cids = passage_data.get_cids_from_pid(page_id)
    assert len(page_cids) > 0
    for cid in page_cids:
        assert page_title == passage_data.get_title_from_cid(cid)
    assert page_title in passage_data.get_titles_from_prep_title(page_data.prep_title)
    assert page_data.prep_title == passage_data.get_prep_title_from_title(page_title)
    on_title, titles = next(iter(passage_data.on2titles.items()))
    for title in titles:
        assert title in passage_data.get_titles_from_on_title(on_title)
    qn_title, titles = next(iter(passage_data.qn2titles.items()))
    for title in titles:
        assert title in passage_data.get_titles_from_qn_title(qn_title)

    # Verify that sid_normer was built + coherencey
    assert sid_normer.get_sid2nsid(len(sid_normer)) is None
    for i in [0, 3555, 4332]:
        assert i in sid_normer.get_nsid2sids(sid_normer.get_sid2nsid(i))
    if verbose:
        print(
            f"Num nsids: {sid_normer.nsid2sids.shape[0]}, "
            + f"Max num shared sids: {sid_normer.nsid2sids.shape[1]}"
        )

    print(">> [Passed] Function ran, use verbose=True to see output")


# ----- Verify Mapping Ran Succesfully ----- #


def check_mapped_file(in_file):
    output_dir = wu.PATH_ARGS.wiki_chunked_fixed_parsed_dir
    input_filename = in_file.split("/")[-1][: -len(".jsonl")]
    out_file = f"{output_dir}{input_filename}_parsed.pkl"
    try:
        in_data = fu.load_file(in_file, verbose=False)
        out_data = fu.load_file(out_file, verbose=False)
        for line in in_data:
            page_id, chunk_num = line["chunk_id"].split("__")
            chunk_num = int(chunk_num)
            assert page_id in out_data["passage_data"]
            assert chunk_num in out_data["passage_data"][page_id]["chunk_num_list"]
    except:  # noqa: E722
        logging.info(f">> Missing: {in_file}")
        return in_file
    return None


def verify_all_processed(input_dir, processes=24, verbose=False):
    all_in_files = sorted(glob.glob(f"{input_dir}wikipedia_chunks_*.jsonl"))
    with multiprocessing.Pool(processes=processes) as pool:
        result_list = []
        for i, res in enumerate(pool.imap_unordered(check_mapped_file, all_in_files)):
            if i % 200 == 0 and verbose:
                logging.info(f">> Processing: {i * 100.0 / len(all_in_files):0.2f}%")
            if res is not None:
                result_list.append(res)
    return result_list


# ----- Verify Reducing Ran Succesfully ----- #


def verify_datastructs():
    print("Load Data Structs:")

    def load_str_fxn(args):
        return StringKey(**args)

    all_str_key, ask_time, ask_space = load_time_space(
        load_str_fxn,
        {"data_dir": wu.PATH_ARGS.stringkeys_dir, "name": "ori_v0"},
    )
    print(f">> {'all_str_key':15} [{ask_space:0.2f}GB] {ask_time/60.0:0.1f} min")
    qnn_str_key, qsk_time, qsk_space = load_time_space(
        load_str_fxn,
        {"data_dir": wu.PATH_ARGS.stringkeys_dir, "name": "qnn_v0"},
    )
    print(f">> {'qnn_str_key':15} [{qsk_space:0.2f}GB] {qsk_time/60.0:0.1f} min")

    def load_pd_fxn(args):
        return PassageData(**args)

    passage_data, pd_time, pd_space = load_time_space(
        load_pd_fxn,
        {
            "data_dir": wu.PATH_ARGS.passage_data_dir,
            "name": "v0",
            "prep2titles": True,
            "extra_norms": True,
        },
    )
    print(f">> {'passage_data':15} [{pd_space:0.2f}GB] {pd_time/60.0:0.1f} min")

    def load_sn_fxn(args):
        return SidNormer(**args)

    sid_normer, sn_time, sn_space = load_time_space(
        load_sn_fxn, {"data_dir": wu.PATH_ARGS.sidnormer_dir, "name": "v0"}
    )
    print(f">> {'sid_normer':15} [{sn_space:0.2f}GB] {sn_time/60.0:0.1f} min")

    print()
    print("Verify Data Structs:")

    # Verify string keys and normer
    num_nsids, max_num_sids_per = sid_normer.nsid2sids.shape
    print(f">> All Str Keys: {len(all_str_key):,} unique strings")
    print(f">> QNN Str Keys: {len(qnn_str_key):,} unique strings")
    print(
        f">> Sid Normer:   {len(sid_normer):,} sids and "
        + f"{num_nsids:,} nsids by {max_num_sids_per:,} max_num_sids_per"
    )
    for i in range(5):
        sid = random.randint(0, len(all_str_key) - 1)
        st = all_str_key.get_sid2str(sid)
        new_sid = all_str_key.get_str2sid(st)

        nsid = sid_normer.get_sid2nsid(sid)
        nstr = qnn_str_key.get_sid2str(nsid)
        un_sids = sid_normer.get_nsid2sids(nsid)
        un_strs = [all_str_key.get_sid2str(usid) for usid in un_sids]
        print(f"   - sid: {sid} str: {st} and back: {new_sid}")
        print(f"       nsid: {nsid} nstr: {nstr} and back: {un_sids}, {un_strs}")
        print()

    # Verify Passage Data
    print(f">> Passage Data: {len(passage_data):,} pages")
    num_passages_list = [len(pd.passage_dict) for pd in passage_data.page_data.values()]
    avg_passages = sum(num_passages_list) / len(passage_data)
    print(
        f"  - num_chunks: min {min(num_passages_list):,} "
        + f"avg {avg_passages:0.2f} max {max(num_passages_list):,}"
    )
    print()
    one_page_ind = random.randint(0, len(passage_data) - 1)
    for i, page_id in enumerate(passage_data.page_data.keys()):
        if i == one_page_ind:
            one_page_id = page_id
    title = passage_data.get_title_from_pid(one_page_id)
    prep_title = passage_data.get_prep_title_from_title(title)
    cids = passage_data.get_cids_from_pid(one_page_id)
    print(f"  Title: {title} | Prep Title: {prep_title}")
    for cid in cids:
        print(f"     - {cid} {passage_data.get_passage(cid)}")
    print()
    print(">> Finished verifying datastructs --------------")


def verify_graphs():
    print("Verify Graphs:")
    for graph_type in wu.GRAPH_TYPES:
        for data_type in wu.GRAPH_KEYS:
            out_dict = fu.load_file(
                wu.get_graph_data_path(graph_type, data_type),
                verbose=False,
            )
            len_lists = [len(v) for v in out_dict.values()]
            print(
                f">> [{graph_type:^30}][{data_type:^15}] num elems: "
                + f"{len(out_dict):30,} avg len lists: "
                + f"{sum(len_lists)/len(len_lists):0.2f}"
            )


def aggregate_ori2entdetailed(
    all_str_key,
    qnn_str_key,
    graph_type,
    str_ent_type,
    verbose=False,
):
    print(f"[{graph_type:^30}][{str_ent_type:^15}]")
    try:
        test_file = fu.load_file(
            wu.get_ori2entdetailed_path(graph_type, str_ent_type),
            verbose=False,
        )
    except:  # noqa: E722
        print(" ----- REDUCER ERROR: FAILED TO WRITE OUTPUT ------- ")
        return

    num_ents = 0
    tot_mma = [0, 0, 0]
    ucids_mma = [0, 0, 0]
    upgs_mma = [0, 0, 0]
    num_to_print = 3

    i = 0
    for ori, ori_data in test_file.items():
        num_ents += len(ori_data)
        if i < num_to_print and verbose:
            if "qstr" not in str_ent_type:
                print("   Original:", ori, all_str_key.get_sid2str(ori))
            else:
                print("   Original:", ori, qnn_str_key.get_sid2str(ori))
        ntot = []
        ncids = []
        npgs = []
        for j, ori_ent in enumerate(ori_data):
            ntot.append(ori_ent.num_total)
            ncids.append(ori_ent.num_unique_cids)
            npgs.append(ori_ent.num_unique_pages)
            if i < num_to_print and j < num_to_print and verbose:
                if "qent" not in str_ent_type:
                    ent_str = all_str_key.get_sid2str(ori_ent.ent_sid)
                else:
                    ent_str = (qnn_str_key.get_sid2str(ori_ent.ent_sid),)
                print(f"     - Ent: {ent_str:40} | {ori_ent}")
        tot_mma = [
            tot_mma[0] + min(ntot),
            tot_mma[1] + max(ntot),
            tot_mma[2] + (sum(ntot) / len(ntot)),
        ]
        ucids_mma = [
            ucids_mma[0] + min(ncids),
            ucids_mma[1] + max(ncids),
            ucids_mma[2] + (sum(ncids) / len(ncids)),
        ]
        upgs_mma = [
            upgs_mma[0] + min(npgs),
            upgs_mma[1] + max(npgs),
            upgs_mma[2] + (sum(npgs) / len(npgs)),
        ]
        i += 1

    print()
    print(f">> Total num ori strings: {len(test_file):,}")
    print(f">> Avg Num ents: {num_ents/i:0.4f}")
    print(
        ">> Avg Total Count min/avg/max: "
        + f"{tot_mma[0]/i:0.4f} {tot_mma[2]/i:0.4f} {tot_mma[1]/i:0.4f}"
    )
    print(
        ">> Avg UCids Count min/avg/max: "
        + f"{ucids_mma[0]/i:0.4f} {ucids_mma[2]/i:0.4f} {ucids_mma[1]/i:0.4f}"
    )
    print(
        ">> Avg UPgs  Count min/avg/max: "
        + f"{upgs_mma[0]/i:0.4f} {upgs_mma[2]/i:0.4f} {upgs_mma[1]/i:0.4f}"
    )
    print()


def verify_ori2entdetailed():
    print("Verify ori2entdetailed:")
    print(" - load all_str_key")
    all_str_key = StringKey(wu.PATH_ARGS.stringkeys_dir, "ori_v0")
    print(" - load qnn_str_key")
    qnn_str_key = StringKey(wu.PATH_ARGS.stringkeys_dir, "qnn_v0")
    print()

    print(">> Inspect graphs:")
    for graph_type in wu.GRAPH_TYPES:
        i = 0
        for str_ent_type in wu.STR_ENT_TYPES:
            verbose = False if i > 0 else True
            if verbose:
                print(
                    "-----> Example original strings with "
                    + "a few linked ents & Ori2Ent data"
                )
            aggregate_ori2entdetailed(
                all_str_key,
                qnn_str_key,
                graph_type,
                str_ent_type,
                verbose,
            )
            i += 1
        print()
