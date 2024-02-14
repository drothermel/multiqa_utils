from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.data import load
import os

# Used for processing new wiki dump
import re
import html

from utils.parallel_utils import FileProcessor
from utils.file_utils import fu
import multiqa_utils.string_utils as su


def flatten_list_of_lists(list_of_lists):
    return [l for sublist in list_of_list for l in sublist]


def merge_into_dict(dict1, dict2):
    for key, val_col2 in dict2.items():
        if key in dict1:
            dict1[key].update(val_col2)
        else:
            dict1[key] = val_col2


def sum_into_dict(dict1, dict2):
    for k, v2 in dict2.items():
        if k not in dict1:
            dict1[k] = v2
        else:
            dict1[k] += v2


class DataManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage_list = [
            'chunking',
            'linking',
            'mapping',
        ]
        self.processing_state = None

        self._load_or_create_processing_state()

        # then it needs more info about what to do
        # - [Currnently TODO] create sbatch for the next step in the data creation
        #   pipeline based on the state file
        #     -> see _write_stage_sbatch(self, stage) for the notes of whats next
        #     -> at same time modifying maqa/scripts/conf/wiki_processing/v0.yaml
        # - instantiate and shepherd the relevant stage's class
        #   - [next] create a RedisLogger
        #   - [next] create the stage class with the logger
        #   - [next] handle the coordination
        # - [future] load or use one of the data objects

    def save_processing_state(self):
        fu.dumpfile(self.processing_state, self.cfg.wiki_processing.state_path)

    def get_next_processsing_stage(self):
        for stage in self.stage_list:
            if not self._validate_stage_complete(stage):
                return stage
        return None

    # Get the next incomplete stage & write the expected sbatch file
    def write_next_sbatch(self):
        next_stage = self.get_next_processing_stage()
        if next_stage is None:
            logging.info(">> All stages completed already")
            return

        logging.info(f">> Next stage: {next_stage}")
        sbatch_filename = self._write_stage_sbatch(next_stage)
        logging.info(f">> Wrote sbatch: {sbatch_filename}")

    def run_chunking(self):
        pass

    def run_linking(self):
        pass

    def _load_or_create_processing_state(self):
        if os.path.exists(self.cfg.wiki_processing.state_path):
            self.processing_state = fu.load_file(self.cfg.wiki_processing.state_path)
        else:
            self.processing_state = {}

        for stage_num, stage in enumerate(self.stage_list):
            if stage in self.processing_state:
                # It was loaded from the state file
                continue

            # Initialize the missing state info
            self.processing_state[stage] = {
                'stage': stage,
                'stage_num': stage_num,
                'complete': False,
                'expected_runs': None,
                'written_runs': set(),
                'verified_runs': set(),
            }
        self.save_processing_state()

    def _validate_stage_complete(self, stage):
        # If the stage is marked complete then its complete
        if self.processing_state[stage]['complete']:
            return True

        # If expected runs is none then this stage hasn't be initialized
        if self.processing_state[stage]['expected_runs'] is None:
            return False

        # If all expected runs have been verified complete, stage is complete
        num_verified = self.processing_state[stage]['verified_runs']
        num_expected = self.processing_state[stage]['expected_runs']
        if len(num_verified - num_expected) == 0:
            self.processing_state[stage]['complete'] = True
            self.save_processing_state()
            return True

        # But otherwise the "complete = False" marker is correct
        return False

    def _write_stage_sbatch(self, stage):
        # TODO: something with redis checks too

        # For now, just using the state info
        # Check if expected runs is none -> kick off all shards
        # If not, compare written to expected -> kick off missing
        # Write the sbatch config params in the section specific area of the
        #    cfg file
        # Probably just break reducing into 3 stages and run them in sequence
        #    with 1 shard each

        # Then the sbatch file runs cfg.wiki_processing.data_manager_script_path
        #      (which should use the standard cfg params for distribution)
        # with wikiprocessing.stage_to_run=stage
        # and shard_num=XXX, shard_siz=XXX
        # Then everything should be written to an sbatch file that has a random
        #    component to its name so it doesn't overwrite previous ones
        sbatch_filename = ''

        # and the script should create a datamanager and call dm.run_<stage>()
        # which will handle the creation of the RedisLogger to make sure
        #   that this isn't already currently running and then mark ourselves as
        #   running
        #   and then create the relevant FileProcessor
        # each of the FileProcessors should have a way to mark written runs as
        #   verified (and we need to make sure the threads don't start until after
        #   this happens)
        # then only the cases where written runs aren't verified should actually
        #   continue.
        return sbatch_filename

    def _create_sbatch(self, stage):
        # TODO: pass these in somehow?
        array_start = 0
        array_end = 0
        run_time = '23:59:00'
        use_gpu = False
        mem = '220G'
        num_cpus = 8
        script_path = self.cfg.wiki_processing.data_manager_script_path
        # TODO: add the rest of the args
        script_args = {
            'wiki_processing.stage_to_run': stage,
            'shard_num': '${SLURM_ARRAY_TASK_ID}',
        }
        script_args_str = ' '.join([f'{name}={val}' for name, val in script_args.items()])

        # Setup the SBATCH args
        sbatch_params = {
            'job-name': stage,
            'array': f'{int(array_start)}-{int(array_end)}',
            'open-mode': 'append',
            'output': '/scratch/ddr8143/multiqa/slurm_logs/%x_%A_j%a.out',
            'error': '/scratch/ddr8143/multiqa/slurm_logs/%x_%A_j%a.err',
            'export': 'ALL',
            'time': run_time,
            'mem': mem,
            'nodes': '1',
            'tasks-per-node': '1',
            'cpus-per-task': str(num_cpus),
        }
        if use_gpu:
            sbatch_params['gres'] = 'gpu:rtx8000:1'
            sbatch_params['account'] = 'cds'

        # Build the rest of the file
        file_lines = ['#!/bin/bash\n']
        file_lines.extend([
            f'#SBATCH --{name}={val}\n' for name, val in sbatch_params.items()
        ])
        file_lines.extend([
            '\n',
            'singularity exec --nv --overlay $SCRATCH/overlay-50G-10M_v2.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "\n'
            '\n'
            'source /ext3/env.sh\n',
            'conda activate multiqa\n',
            '\n',
            f'python {script_path} {script_args_str} \n',
            '"\n',
        ])
        return file_lines


class WikiChunker(FileProcessor):
    def __init__(self, cfg):
        self.cfg = cfg
        self.chunked_dir = cfg.wiki_processing.wiki_chunked_dir
        self.target_words = cfg.wiki_processing.chunk_target_word_len
        self.max_words = cfg.wiki_processing.chunk_max_word_len

        # Unused, but might be needed for word_tokenize and sent_tokenize to work
        self.tokenizer = load(f"tokenizers/punkt/english.pickle")
        self.detokenizer = su.get_detokenizer()

        super().__init__(
            cfg.wiki_processing.num_threads,
            cfg.wiki_processing.num_procs,
            cfg.wiki_processing.proc_batch_size,
            skip_exists=cfg.wiki_processing.skip_exists,
        )

    def get_output_from_inpath(self, file_path):
        # basepath/AA/wiki_00
        init_name = fu.get_file_from_path(init_filepath)
        init_dir = fu.get_dir_from_path(init_filepath)
        sub_dirname = init_dir.split(os.path.sep)[-1]
        return os.path.join(self.chunked_dir, f'{sub_dirname}_{init_name}.pkl')

    # Necessary bc orig files are just "wiki_00" not "wiki_00.jsonl"
    def load_file(self, file_path):
        return fu.load_file(file_path, ending='jsonl')

    # Output list of pages within a fine, with lists of chunks for each page
    def merge_results(self, file_output_list):
        return flatten_list_of_lists(file_output_list)

    # CPU intensive task: chunking and tag/link matching
    def process_batch_elem(self, item):
        # Only process pages with text
        text = item['clean_text']
        if len(text) == 0 or len(text.strip()) == 0:
            return []
        title = item['title']

        # Normalize the text and title
        fixed_text = su.base_fix_string(self.detokenizer, text)
        fixed_title = su.base_fix_string(self.detokenizer, title)

        # Chunk the text
        sentences = sent_tokenize(fixed_text)
        raw_chunks = self._combine_sentences_into_chunks(sentences)

        # Convert to final output format
        chunks = [
            {
                'page_id': int(item['id']),
                'raw_title': title,
                'title': fixed_title,
                'chunk_num': c_i,
                'chunk_text': c_text,
                'chunk_start_ind': c_start,
                'chunk_end_ind': c_end,
            }
            for c_i, (c_start, c_end, c_text) in enumerate(raw_chunks)
        ]
        return chunks

    # Returns: [(chunk_start_ind, chunk_end_ind, chunk_text), ...]
    def _combine_sentences_into_chunks(self, sentences):
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_start = 0

        def finalize_current_chunk():
            nonlocal current_chunk, chunk_start, current_word_count
            chunk_end = chunk_start + len(' '.join(current_chunk))
            chunks.append((chunk_start, chunk_end, ' '.join(current_chunk)))
            chunk_start = chunk_end + 1
            current_chunk = []
            current_word_count = 0

        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            if len(sentence_words) > self.max_words:
                # Split a long sentence into smaller parts
                for word in sentence_words:
                    if current_word_count >= self.max_words:
                        finalize_current_chunk()
                    current_chunk.append(word)
                    current_word_count += 1
            else:
                # Handle normal sentences
                if current_word_count + len(sentence_words) > self.max_words:
                    finalize_current_chunk()

                current_chunk.append(sentence)
                current_word_count += len(sentence_words)

                if current_word_count >= self.target_words:
                    finalize_current_chunk()

        if current_chunk:
            finalize_current_chunk()

        return chunks


class WikiMapper(FileProcessor):
    def __init__(self, cfg):
        self.cfg = cfg
        self.graph_type = cfg.wiki_processing.graph_type
        self.chunked_dir = cfg.wiki_processing.wiki_chunked_dir
        self.mapped_dir = cfg.wiki_processing.wiki_mapped_dir[self.graph_type]
        self.tokenizer = tu.initialize_tokenizer(
            cfg.tokenizer_config_path,  # TODO: This doesn't work yet
        )

        super().__init__(
            cfg.wiki_processing.num_threads,
            cfg.wiki_processing.num_procs,
            cfg.wiki_processing.proc_batch_size,
            skip_exists=cfg.wiki_processing.skip_exists,
        )

    def get_output_from_inpath(self, file_path):
        # chunked_dir/AA_wiki_00.pkl
        init_name = fu.get_file_from_path(init_filepath)
        return os.path.join(self.mapped_dir, init_name)

    # Item is a single chunk info dict
    def process_batch_elem(self, item):
        chunk = item
        mapped_chunk = {}
        # Get the link types to use
        link_types = [self.graph_types.split('_and_')]

        # Get the token info
        token_info = self._get_token_info(chunk, link_types)

        cid = (chunk['page_id'], chunk['chunk_num'])
        # Build sets and graphs
        sets = {'spans': set(), 'ents': set()}
        graphs = {
            'span2ents': defaultdict(set),
            'ref_ent2cids': defaultdict(set),
            'ref_span2cids': defaultdict(set),
            'cid2children': defaultdict(set),
            'span2entdetailed': defaultdict(dict),
        }
        for link_type in link_types:
            for span, ent in chunk[link_type]:
                sets['spans'].add(span)
                sets['ents'].add(ent)
                graphs['span2ents'][span].add(ent)
                graphs['ref_ent2cids'][ent].add(cid)
                graphs['ref_span2cids'][span].add(cid)
                graphs['cid2children'][cid].add((span, ent))
                if cid not in graphs['span2entdetailed'][span]:
                    graphs['span2entdetailed'][span][cid] = {ent: 1}
                elif ent not in graphs['span2entdetailed'][span][cid]:
                    graphs['span2entdetailed'][span][cid][ent] = 1
                else:
                    graphs['span2entdetailed'][span][cid][ent] += 1

        # Create str2nstr
        base_str_set = set([title]) | sets['spans'] | sets['ents']
        str2nstr = {}
        for norm_type, norm_fxn in su.get_all_norm_fxns().items():
            str2nstr[norm_type] = {st: norm_fxn(st) for st in base_str_set}

        mapped_chunk = {
            **chunk,
            'cid': cid,
            'sets': sets,
            'graphs': graphs,
            'token_info': token_info,
            'str2nstr': str2nstr,
        }
        return mapped_chunk

    # This is where we combine any shared structures
    def merge_results(self, file_output_list):
        processed_chunks = file_output_list

        # Merge token data
        cid2tokeninfo = {
            (pc['pid'], pc['chunk_num']): pc['token_info'] for pc in processed_chunks
        }

        # Merge norm data
        str2nstr = {k: {} for k in processed_chunks[0]['str2nstr'].keys()}
        for pc in processed_chunks:
            for norm_type, s2ns in pc['str2nstr'].items():
                str2nstr[norm_type].update(s2ns)

        # Merge page data
        pid2chunk2passage = defaultdict(dict)
        pid2title = {}
        for pc in processed_chunks:
            pid2chunk2passage[pc['pid']][pc['chunk_num']] = pc['content']
            pid2title[pc['pid']] = pc['title']

        # Merge sets and graphs, build some new ones
        sets = {k: set() for k in pc['sets'].keys()}
        sets['titles'] = set()
        graphs = {k: {} for k in pc['graphs'].keys()}
        graphs['title_ent2cids'] = defaultdict(set)
        for pc in processed_chunks:
            sets['titles'].add(pc['title'])
            sets['spans'].update(pc['sets']['spans'])
            sets['ents'].update(pc['sets']['ents'])

            for gk in ['span2ents', 'ref_ent2cids', 'ref_span2cids', 'cid2children']:
                merge_into_dict(graphs[gk], pc['graphs'][gk])
            merge_span2entdetailed(
                graphs['span2entdetailed'], pc['graphs']['span2entdetailed']
            )

            graphs['title_ent2cids'][chunk['title']].append(pc['cid'])
        mapped_data = {
            'pid2chunk2passage': dict(pid2chunk2passage),
            'pid2title': pid2title,
            'sets': sets,
            'graphs': {k: dict(v) for k, v in graphs.items()},
            'cid2tokeninfo': cid2tokeninfo,
            'str2nstr': str2nstr,
        }
        return mapped_data

    def _get_token_info(self, chunk, link_types):
        title = chunk['title']
        contents = chunk['contents']

        title_toks = self.tokenizer.encode(title, add_special_tokens=False)
        encoded_w_offsets = self.tokenizer.encode(
            contents, add_special_tokens=False, return_offsets_mapping=True
        )
        token_info = {
            'title_toks': title_toks,
            'title_span_inds': (0, len(title)),
            'title_tok_inds': (0, len(title_toks)),
            'content_toks': encoded_w_offsets.ids,
        }
        spans = flatten_list_of_lists([chunk[lt] for lt in link_types])
        token_info['spans_w_tok_inds'] = self._find_token_ranges(
            passage,
            encoded_w_offsets,
            spans,
        )
        return token_info

    def _find_token_ranges(self, passage, encoded_w_offsets, spans):
        # Tokenize the passage and get a mapping of token indices to character ranges
        char_to_token = [None] * len(passage)
        for token_ind, (start_char, end_char) in enumerate(
            encoded_w_offsets.offset_mapping
        ):
            for char_ind in range(start_char, end_char):
                char_to_token[char_ind] = token_ind

        # Update each span with token indices
        updated_spans = []
        for span in spans:
            start_char_ind = span['start_ind']
            end_char_ind = span['end_ind'] - 1  # Adjust to be inclusive

            start_token_ind = char_to_token[start_char_ind]
            end_token_ind = char_to_token[end_char_ind]

            updated_spans.append(
                {
                    'span': span['span'],
                    'ent': span['ent'],
                    'span_inds': (
                        start_char_ind,
                        end_char_ind + 1,
                    ),  # Make end index exclusive again
                    'tok_inds': (
                        start_token_ind,
                        end_token_ind + 1,
                    ),  # Make end index exclusive
                }
            )

        return updated_spans

    def _merge_span2entdetailed(self, dict1, dict2):
        for key, subdict in dict2.items():  # span: {cid: ent: int}
            if key not in dict1:  # span
                dict1[key] = subdict
                continue

            for kk, subsubdict in subdict.items():  # cid: {ent: int}
                if kk not in dict1[key]:  # cid
                    dict1[key][kk] = subsubdict
                    continue

                # merge {ent: int} with {ent: int}
                sum_into_dict(dict1[key][kk], subsubdict)


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


def glob_alpha_segs(top_wiki_dir):
    return sorted(glob.glob(f"{top_wiki_dir}[A-Z][A-Z]"))


def glob_alpha_subsegs(alpha_path):
    return sorted(glob.glob(f"{alpha_path}/wiki_[0-9][0-9]"))


def glob_all_wiki_files(top_wiki_dir):
    return sorted(glob.glob(f"{top_wiki_dir}[A-Z][A-Z]/wiki_[0-9][0-9]"))


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
