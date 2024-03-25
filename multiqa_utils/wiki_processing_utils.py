from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.data import load
import os
import string
import random
import logging

# Used for processing new wiki dump
import re
import html

from utils.redis_utils import RedisLogger
from utils.parallel_utils import FileProcessor
from utils.util_classes import Metrics
import utils.file_utils as fu
import utils.run_utils as ru
import multiqa_utils.genre_utils as gu
import multiqa_utils.string_utils as su


def flatten_list_of_lists(list_of_lists):
    return [li for sublist in list_of_lists for li in sublist]


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
            'entity_set',
            'linking',
            'mapping',
        ]
        self.processing_state = None
        self.logger = None
        self.stage_class = None

        self._load_or_create_processing_state()

    def save_processing_state(self):
        fu.dumpfile(self.processing_state, self.cfg.wiki_processing.state_path)

    def set_stage_class_from_stage_name(self, stage_name):
        assert stage_name in self.stage_list
        if stage_name == 'chunking':
            self.stage_class = WikiChunker(self.cfg)
        elif stage_name in ['entity_set', 'linking']:
            self.stage_class = WikiLinker(self.cfg)
        else:
            assert False

    def get_next_processing_stage_name(self):
        for stage_name in self.stage_list:
            if not self._validate_stage_complete(stage_name):
                return stage_name
        return None

    # Get the next incomplete stage & write the expected sbatch file
    def write_next_sbatch(self):
        # Get next incomplete stage
        next_stage_name = self.get_next_processing_stage_name()
        if next_stage_name is None:
            logging.info(">> All stages completed already")
            return

        # Write sbatch for that stage
        logging.info(f">> Next stage: {next_stage_name}")
        self._write_stage_sbatch(next_stage_name)

    def run_stage(self, stage_name):
        if stage_name == 'sbatch':
            self.write_next_sbatch()
            return

        self.set_stage_class_from_stage_name(stage_name)
        job_starting = self._verify_job_start(stage_name)
        if not job_starting:
            logging.info(f">> {stage_name} already ran/is running, skipping")
            self.stage_class = None
            return
        logging.info(f">> Running {stage_name} job")
        self.stage_class.set_logger(self.logger)
        self.stage_class.select_job_files()
        self.stage_class.execute()
        run_status = self.stage_class.verify_run()
        shard_ind = self.cfg.shard_ind
        logging.info(
            f">> Completed {stage_name} shard {shard_ind} with status: {run_status}"
        )

    def _load_or_create_processing_state(self):
        if os.path.exists(self.cfg.wiki_processing.state_path):
            self.processing_state = fu.load_file(self.cfg.wiki_processing.state_path)
        else:
            self.processing_state = {}

        changes = False
        for stage_num, stage_name in enumerate(self.stage_list):
            if stage_name in self.processing_state:
                # It was loaded from the state file
                continue

            changes = True
            # Initialize the missing state info
            self.processing_state[stage_name] = {
                'stage_name': stage_name,
                'stage_num': stage_num,
                'complete': False,
                'expected_runs': None,
                'verified_runs': set(),
                'run_error': False,
            }
        if changes:
            self.save_processing_state()

    # Only call in the sbatch creation flow
    def _validate_stage_complete(self, stage_name):
        # If the stage is marked complete then its complete
        if self.processing_state[stage_name]['complete']:
            return True

        # If expected runs is none then this stage hasn't be initialized
        if self.processing_state[stage_name]['expected_runs'] is None:
            return False

        # If all expected runs have been verified complete, stage is complete
        num_verified = self.processing_state[stage_name]['verified_runs']
        num_expected = self.processing_state[stage_name]['expected_runs']
        if len(num_verified - num_expected) == 0:
            self.processing_state[stage_name]['complete'] = True
            self.save_processing_state()
            return True

        self.set_stage_class_from_stage_name(stage_name)
        job_status = self.stage_class.check_verified()
        if job_status == 'error':
            self.processing_state[stage_name]['run_error'] = True
            self.save_processing_state()
            logging.info(">> WARNING: this stage isn't complete, there was an error")
            self.stage_class = None
            return True

        if job_status == 'verified':
            self.processing_state[stage_name]['complete'] = True
            self.processing_state[stage_name]['verified_runs'] = num_expected
            self.save_processing_state()
            self.stage_class = None
            return True

        if job_status == 'incomplete':
            flag_dir = self.stage_class.flag_dir
            job_name = self.stage_class.get_job_name()
            job_data = fu.load_file(f'{flag_dir}{job_name}.incomplete.json')
            num_missing = len(job_data['missing_files'])
            num_verified = num_expected - num_missing
            self.processing_state[stage_name]['verified_runs'] = num_verified
            self.save_processing_state()
            self.stage_class = None
            return False

        self.stage_class = None
        return False

    # The sbatch writing only chekcs the state file, not RedisLogger, and
    # only considers the jobs that have already been marked verified as
    # complete.  Additional checks happen at startup of the jobs kicked off
    # by the sbatch script.
    def _write_stage_sbatch(self, stage_name):
        state = self.processing_state[stage_name]

        # Determine which inds to run (conservatively)
        inds_to_run = []
        if state['expected_runs'] is None:
            num_shards = self.cfg.wiki_processing[stage_name].sbatch.num_shards
            state['expected_runs'] = set([i for i in range(num_shards)])

        inds_to_run = state['expected_runs'] - state['verified_runs']
        if len(inds_to_run) == 0:
            logging.info(f">> All inds for this stage ({stage_name})have been run.")
            return

        # Then everything should be written to an sbatch file that has a random
        #    component to its name so it doesn't overwrite previous ones
        sbatch_file_lines = self._create_sbatch(stage_name, inds_to_run)
        rand_v = ''.join(
            random.choices(
                string.ascii_letters + string.digits,
                k=4,
            )
        )
        sbatch_filename = (
            f'{self.cfg.wiki_processing.sbatch_dir}{stage_name}.{rand_v}.sbatch'
        )
        fu.dump_file(sbatch_file_lines, sbatch_filename, ending='txt', verbose=True)

    def _create_sbatch(self, stage_name, shards_to_run):
        script_path = self.cfg.wiki_processing.data_manager_script_path
        if stage_name in ['entity_set', 'linking']:
            script_path = self.cfg.wiki_processing.data_manager_hydra_old_script_path
        script_args = {
            'wiki_processing.stage_to_run': stage_name,
            'shard_num': self.cfg.wiki_processing[stage_name].sbatch.num_shards,
            'shard_ind': '${SLURM_ARRAY_TASK_ID}',
        }

        conda_env = self.cfg.wiki_processing[stage_name].sbatch.conda_env
        sbatch_params = {
            'job-name': stage_name,
            'array': ru.get_job_array_str(shards_to_run),
            'time': self.cfg.wiki_processing[stage_name].sbatch.run_time,
            'mem': self.cfg.wiki_processing[stage_name].sbatch.mem,
            'cpus-per-task': str(self.cfg.wiki_processing[stage_name].sbatch.num_cpus),
        }
        if self.cfg.wiki_processing[stage_name].sbatch.use_gpu:
            sbatch_params['gres'] = 'gpu:rtx8000:1'

        return ru.make_sbatch_file(
            script_path=script_path,
            script_args=script_args,
            sbatch_new_params=sbatch_params,
            conda_env=conda_env,
        )

    def _verify_job_start(self, stage_name):
        self.logger = RedisLogger(
            config_file=self.cfg.redis_config_file,
            server_dir=self.cfg.redis_server_dir,
        )

        job_name = self.stage_class.get_job_name()
        if self.logger.check_is_job_running(job_name):
            return False

        # If job isn't running, call verify on stage class (may take a long time)
        job_status = self.stage_class.check_run_verified()
        if job_status in ['verified', 'error']:
            return False

        # Otherwise, try to mark that we're running and return if we should start
        tostart = self.logger.if_not_running_mark_return_tostart_flag(job_name)
        return tostart


class WikiStage(FileProcessor):
    def __init__(self, cfg, stage_name):
        super().__init__(
            cfg.wiki_processing.chunking.num_threads,
            cfg.wiki_processing.chunking.num_procs,
            cfg.wiki_processing.chunking.proc_batch_size,
            skip_exists=cfg.wiki_processing.chunking.skip_exists,
        )
        self.cfg = cfg
        self.stage_name = stage_name
        self.flag_dir = None
        self.logger = None
        self.file_metric_key = 'all_files_in_metrics'

    def set_logger(self, logger):
        self.logger = logger

    def get_job_name(self):
        return f'{self.stage_name}_{self.cfg.shard_ind}'

    def check_verified(self):
        job_name = self.get_job_name()
        flag_path_base = f'{self.flag_dir}{job_name}'
        if os.path.exists(f'{flag_path_base}.verified.json'):
            return 'verified'
        elif os.path.exists(f'{flag_path_base}.error.json'):
            return 'error'
        elif os.path.exists(f'{flag_path_base}.incomplete.json'):
            return 'incomplete'
        return None

    def check_run_verified(self):
        run_status = self.check_verified()
        if run_status is None:
            run_status = self.verify_run()
        return run_status

    def _get_test_results_dump_flag_file(self, test_res, extra_data):
        if self.file_metric_key not in test_res:
            test_res[self.file_metric_key] = False
            extra_data[
                self.file_metric_key
            ] = f'Expected key missing: {self.file_metric_key}'

        # Convert test results into flag
        dump_data = ['']
        test_results = 'verified'
        if not test_res[self.file_metric_key]:
            test_results = 'incomplete'
            dump_data = {'missing_files': extra_data[self.file_metric_key]}
        if not all(list(test_res.keys())):
            test_results = 'error'
            dump_data = extra_data

        # Dump results
        job_name = self.get_job_name()
        flag_path = f'{self.flag_dir}{job_name}.{test_results}.json'
        fu.dump_file(dump_data, flag_path, verbose=True)
        return test_results

    def _check_expected_vs_file_path_redis(self):
        expected_files = set(self.files)
        logged_files = set(
            self.logger.read_from_redis(
                data_type='list',
                name=f'{self.stage_name}.file_path',
            )
        )
        failed_files = list(expected_files - logged_files)
        return logged_files, failed_files

    def _standard_update_test_res_extra_data(
        self,
        tname,
        test_res,
        extra_data,
        failed_files,
    ):
        test_res[tname] = len(failed_files) == 0
        if not test_res[tname]:
            extra_data[tname] = failed_files

    # ---- Override these ---- #
    # execute() and other FileProcessor fxns too

    def select_job_files(self):
        self.files = None

    def verify_run(self):
        assert self.logger is not None
        test_results = None
        assert test_results in ['verified', 'incomplete', 'error']
        return test_results

    # ------------------------ #


class WikiChunker(WikiStage):
    def __init__(self, cfg, stage_name):
        super().__init__(cfg, stage_name)
        self.input_dir = cfg.wiki_processing.wiki_input_dir
        self.chunked_dir = cfg.wiki_processing.chunked_dir
        self.flag_dir = f'{self.chunked_dir}run_flags/'
        if not os.path.exists(self.flag_dir):
            os.make_dirs(self.flag_dir)

        self.target_words = cfg.wiki_processing.chunk_target_word_len
        self.max_words = cfg.wiki_processing.chunk_max_word_len
        self.regexp, self.pattterns = su.get_regexp_and_patterns()

        # Unused, but might be needed for word_tokenize and sent_tokenize to work
        self.tokenizer = load("tokenizers/punkt/english.pickle")
        self.detokenizer = su.get_detokenizer()

    def select_job_files(self):
        all_files = fu.get_recursive_files(self.input_dir)
        all_files = [f for f in all_files if '/wiki_' in f]
        shard_files = ru.get_curr_shard(
            all_files, self.cfg.shard_num, self.cfg.shard_ind
        )
        self.files = shard_files

    def get_output_from_inpath(self, file_path):
        # basepath/AA/wiki_00
        init_name = fu.get_file_from_path(file_path)
        init_dir = fu.get_dir_from_path(file_path)
        sub_dirname = init_dir.split(os.path.sep)[-1]
        return os.path.join(self.chunked_dir, f'{sub_dirname}_{init_name}.pkl')

    # Necessary bc orig files are just "wiki_00" not "wiki_00.jsonl"
    def load_file(self, file_path):
        return fu.load_file(file_path, ending='jsonl')

    # Output list of pages within a file, with lists of chunks for each page
    def merge_results(self, file_path, file_output_list):
        # Merge results
        all_chunks = flatten_list_of_lists(file_output_list)

        # Calculate metrics
        md = Metrics()
        md.increment_val('num_total_chunks', amount=len(all_chunks))
        for page_chunks in file_output_list:
            num_chunks = len(page_chunks)
            if num_chunks == 0:
                md.increment_val('num_pages_without_text')
            else:
                md.increment_val('num_pages_with_text')
                for chunk in page_chunks:
                    num_chars = len(chunk['chunk_text'])
                    num_toks = chunk['chunk_end_ind'] - chunk['chunk_start_ind']
                    md.add_to_metric(
                        'chars',
                        'chunk',
                        num_chars,
                        metric_type='hist',
                    )
                    md.add_to_metric(
                        'toks',
                        'chunk',
                        num_toks,
                        metric_type='hist',
                    )
            md.vals['max_num_chunks_per_page'] = max(
                md.vals['max_num_chunks_per_page'], num_chunks
            )
        md.vals['avg_num_chunks_per_page'] = (
            1.0 * md.vals['num_total_chunks'] / md.vals['num_pages_with_text']
        )
        md.update_agg_stats(no_len=True)

        metrics_to_log = [('list_elem', 'file_path', file_path)]
        metrics_to_log.extend([('list_elem', v_n, v_d) for v_n, v_d in md.vals.items()])
        log_success = self.logger.log_all_to_redis(
            metrics_to_log,
            prefix=self.stage_name,
        )
        if not log_success:
            logging.info(f">> Redis logging failed for metrics: {metrics_to_log}")
        return all_chunks

    # CPU intensive task: chunking
    def process_batch_elem(self, item):
        # Only process pages with text
        text = item['clean_text']
        if len(text) == 0 or len(text.strip()) == 0:
            return []
        title = item['title']

        # Normalize the text and title
        fixed_text = su.base_fix_string(
            self.detokenizer, text, self.regexp, self.patterns
        )
        fixed_title = su.base_fix_string(
            self.detokenizer, title, self.regexp, self.patterns
        )

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

    def verify_run(self):
        assert self.logger is not None
        logging.info(">> Running verify_run")

        # Get relevant stats
        files_set = set(self.files)
        cmets = self.logger.get_kv_pairs_with_prefix(self.stage_name)
        file_paths = cmets['file_path']
        all_stats = {}
        for i, fp in enumerate(file_paths):
            if fp not in files_set:
                continue
            all_stats[fp] = {
                'num_total_chunks': cmets['num_total_chunks'][i],
                'chars_per_chunk_min': cmets['chars_per_chunk_min'][i],
                'toks_per_chunk_min': cmets['toks_per_chunk_min'][i],
            }

        # -- Calculate test results -- #
        test_res = {}
        extra_data = {}

        # Verify all files are in chunking metrics
        failed_files = list(files_set - all_stats.keys())
        self._standard_update_test_res_extra_data(
            self.file_metric_key,
            test_res,
            extra_data,
            failed_files,
        )

        # Verify no chunks have fewer than one char or tok
        failed_files = []
        for fp, fd in all_stats.items():
            if fd['chars_per_chunk_min'] == 0:
                failed_files.append((fp, fd['chars_per_chunk_min']))
        self._standard_update_test_res_extra_data(
            'has_chars',
            test_res,
            extra_data,
            failed_files,
        )

        failed_files = []
        for fp, fd in all_stats.items():
            if fd['toks_per_chunk_min'] == 0:
                failed_files.append((fp, fd['toks_per_chunk_min']))
        self._standard_update_test_res_extra_data(
            'has_toks',
            test_res,
            extra_data,
            failed_files,
        )

        # Verify that the output file exists
        failed_files = []
        for fp, fd in all_stats.items():
            out_fp = self.get_output_from_inpath(fp)
            if not os.path.exists(out_fp):
                failed_files.append((fp, out_fp))
        self._standard_update_test_res_extra_data(
            'out_file_exists',
            test_res,
            extra_data,
            failed_files,
        )

        # Verify file content
        num_unique_chunks = {}
        num_chunks = {}
        for fp, fd in all_stats.items():
            out_fp = self.get_output_from_inpath(fp)
            if not os.path.exists(out_fp):
                continue
            out_data = fu.load_file(out_fp)
            cids = []
            for cdata in out_data:
                cids.append((cdata['page_id'], cdata['chunk_num']))
            num_chunks[fp] = len(cids)
            num_unique_chunks[fp] = len(set(cids))

        failed_files = []
        for fp, nc in num_chunks.items():
            nuc = num_unique_chunks[fp]
            if nc != nuc:
                failed_files.append((fp, nc, nuc))
        self._standard_update_test_res_extra_data(
            'all_cids_unique',
            test_res,
            extra_data,
            failed_files,
        )

        failed_files = []
        for fp, nuc in num_unique_chunks.items():
            expected_c = all_stats[fp]['num_total_chunks']
            if nuc != expected_c:
                failed_files.append((fp, nuc, expected_c))
        self._standard_update_test_res_extra_data(
            'expected_num_chunks',
            test_res,
            extra_data,
            failed_files,
        )
        return self._get_test_results_dump_flag_file(test_res, extra_data)

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


class WikiLinker(WikiStage):
    def __init__(self, cfg, stage_name):
        super().__init__(cfg, stage_name)
        self.chunked_dir = cfg.wiki_processing.chunked_dir
        self.link_type = cfg.wiki_processing.linking.link_type
        self.linked_dir = cfg.wiki_processing.linked_dirs[self.link_type]
        self.flag_dir = f'{self.linked_dir}run_flags/'
        if not os.path.exists(self.flag_dir):
            os.make_dirs(self.flag_dir)

        # Only used for linking
        # - for genre linking
        self.genre_model = None
        self.genre_cand_trie = None

    def select_job_files(self):
        # For either job type, get all the chunked files
        all_files = fu.get_recursive_files(self.chunked_dir)
        all_files = [f for f in all_files if '/wiki_' in f]
        if self.stage_name == 'entity_set':
            # One job for all files
            self.files = all_files

        if self.stage_name == 'linking':
            # Sharded jobs
            shard_files = ru.get_curr_shard(
                all_files, self.cfg.shard_num, self.cfg.shard_ind
            )
            self.files = shard_files

    def process_file(self, file_path):
        if self.stage_name == 'entity_set':
            return self._process_file_entity_set(file_path)

        if self.stage_name == 'linking':
            return self._process_file_linking(file_path)

    def execute(self):
        assert self.logger is not None
        # Linking needs to load some additional data
        if self.stage_name == 'linking':
            if self.link_type == 'genre':
                logging.info(">>  Loading genre model")
                self.genre_model = gu.load_genre_model(
                    self.cfg.wiki_processing.genre_model_path,
                    self.cfg.wiki_processing.genre_batch_size,
                )
                logging.info(">>  Loading genre candidate trie")
                self.genre_cand_trie = fu.load_file(
                    self.cfg.wiki_processing.genre_cand_trie_path
                )

        # Both versions need to run process file in threads
        all_results = super().execute()

        # Linking dumps the results per-file and returns nothing
        # Entity set needs to aggregate the results
        if self.stage_name == 'entity_set':
            self._entity_set_results_to_trie_and_dump(all_results)
        return None

    def verify_run(self):
        if self.stage_name == 'entity_set':
            return self._verify_entity_set_run()

        if self.stage_name == 'linking':
            return self._verify_linking_run()

    def _process_file_entity_set(self, file_path):
        data = self.load_file(file_path)
        all_titles = [c['title'] for c in data]
        # Log metric: file_path
        self.logger.log_one_to_redis(
            data_type='list_elem',
            name='file_path',
            data=file_path,
            prefix=self.stage_name,
        )
        return all_titles

    def _entity_set_results_to_trie_and_dump(self, results):
        # Aggregate all ents across files
        all_ents = set()
        for file_ents in results:
            all_ents.update(file_ents)

        # Log metric: num_ents
        self.logger.log_one_to_redis(
            data_type='key_val',
            name='num_ents',
            data=len(all_ents),
            prefix=self.stage_name,
        )

        # Make and dump trie, model needed for tokenization
        genre_model = gu.load_genre_model(
            self.cfg.wiki_processing.genre_model_path,
            self.cfg.wiki_processing.genre_batch_size,
        )
        cand_trie_outpath = self.cfg.wiki_processing.genre_cand_trie_path
        _ = gu.build_dump_return_cand_trie(all_ents, genre_model, cand_trie_outpath)

    def _verify_entity_set_run(self):
        assert self.logger is not None
        test_res = {}
        extra_data = {}

        # -- Run tests -- #
        _, failed_files = self._check_expected_vs_file_path_redis()
        self._standard_update_test_res_extra_data(
            self.file_metric_key,
            test_res,
            extra_data,
            failed_files,
        )

        tname = 'cand_trie_exists'
        failed_files = []
        cand_trie_path = self.cfg.wiki_processing.genre_cand_trie_path
        if not os.path.exists(cand_trie_path):
            failed_files.append(cand_trie_path)
        test_res[tname] = len(failed_files) == 0
        if not test_res[tname]:
            extra_data[tname] = {'missing_trie_file': cand_trie_path}

        tname = 'trie_size_match_logged_num_ents'
        failed_files = []
        if not os.path.exists(cand_trie_path):
            failed_files.append(cand_trie_path)
            extra_data[tname] = {'error': "size can't match if trie doesn't exist"}
        else:
            logged_num_ents = self.logger.read_from_redis(
                data_type='key_val',
                name=f'{self.stage_name}.num_ents',
            )
            cand_trie = fu.load_file(cand_trie_path)
            if len(cand_trie) != logged_num_ents:
                failed_files.append(cand_trie_path)
                extra_data[tname] = {
                    'logged_num_ents': logged_num_ents,
                    'cand_trie_len': len(cand_trie),
                }
        test_res[tname] = len(failed_files) == 0

        return self._get_test_results_dump_flag_file(test_res, extra_data)

    def _get_outpath_from_inpath_linking(self, file_path):
        init_name = fu.get_file_from_path(file_path)
        return os.path.join(self.linked_dir, f'{init_name}.pkl')

    def _process_file_linking(self, file_path):
        # Check if already done, load data
        outpath = self._get_outpath_from_inpath_linking(file_path)
        if self.skip_exists and os.path.exists(outpath):
            return

        results = []
        if self.link_type == 'genre':
            assert self.genre_model is not None
            assert self.genre_cand_trie is not None
            input_data = gu.load_and_prepare_wiki_data(file_path)
            results = gu.batched_predict(
                input_data,
                self.genre_model,
                self.genre_cand_trie,
            )
        else:
            assert False, f">> ERROR: Unknown link type: {self.link_type}"
        self.logger.log_one_to_redis(
            data_type='list_elem',
            name='file_path',
            data=file_path,
            prefix=self.stage_name,
        )
        fu.dump_file(results, outpath)
        return None

    def _verify_linking_run(self):
        assert self.logger is not None
        test_res = {}
        extra_data = {}

        # -- Run tests -- #
        logged_files, failed_files = self._check_expected_vs_file_path_redis()
        self._standard_update_test_res_extra_data(
            self.file_metric_key,
            test_res,
            extra_data,
            failed_files,
        )

        failed_files = []
        for fn in logged_files:
            logged_data = fu.load_file(fn)
            if any([len(pr) == 0 for pr in logged_data]):
                failed_files.append(fn)
        self._standard_update_test_res_extra_data(
            'all_chunks_have_preds',
            test_res,
            extra_data,
            failed_files,
        )
        return self._get_test_results_dump_flag_file(test_res, extra_data)


"""
class WikiMapper(FileProcessor):
    def __init__(self, cfg):
        self.cfg = cfg
        self.graph_type = cfg.wiki_processing.graph_type
        self.chunked_dir = cfg.wiki_processing.chunked_dir
        self.mapped_dir = cfg.wiki_processing.mapped_dir[self.graph_type]
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
        init_name = fu.get_file_from_path(file_path)
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
    def merge_results(self, file_path, file_output_list):
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
    mapped_dir = cfg.wiki_processing.mapped_dir
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
"""


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
