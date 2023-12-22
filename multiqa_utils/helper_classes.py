from collections import defaultdict, namedtuple

import os
import math
import numpy as np
import numbers
import logging
import pygtrie

import utils.file_utils as fu

def dict_to_1d_npy_array(in_dict, dtype):
    max_val = max(in_dict.keys())
	out_array = np.zeros([max_val], dtype=dtype)
	for k, v in in_dict.items():
		out_array[k] = v
	return out_array

class DataStruct:
    def __init__(
        self,
        data_dir,
        name,
        save_endings={},
        building=False,
    ):
        self.name = name
        self.data_dir = data_dir
        self.save_endings = save_endings
        self.building = building
        self.load()

    def __len__(self):
        return 0

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name}, len={len(self)})"


    def get_dir(self):
        return f"{self.data_dir}{self.name}/"

    def get_path(self, data_name, ending=None):
        save_dir = self.get_dir()
        if ending is None:
            ending = self.save_endings[data_name]
        return f"{save_dir}{data_name}.{ending}"


    def load(self):
        if self.building:
            return
        assert os.path.exists(self.get_dir())
        for name, data in vars(self):
            if name not in self.save_endings:
                continue
            loaded_data = fu.load_file(self.get_path(name))
            setattr(self, name, loaded_data)

    def save(self):
        assert self.building
        os.makedirs(self.get_dir(), exist_ok=True)
        for name, data in vars(self):
            if name not in self.save_endings:
                continue
            fu.dump_file(data, self.get_path(name))


class PassageDataV2(DataStruct):
    def __init__(
        self,
        data_dir,
        name,
        load_names=[],
        building=False,
    ):
        self.load_names = load_names
        # Completely overwrite save bc its complex
        # And then overwrite load to use load_names for loading
        super().__init__(
            data_dir,
            name,
            save_endings={},
            building,
        )

        # Make individual string type string_keys, per graph_type
        # title_ent_strings -> tsids, then ptsids (store max tsid value)
        # ref_ent_strings -> rsids
        # ori_strings -> osids (includes title strings, prep title strings, ori_strings
        #                       but not ref_ent_strings bc not predictabl)
        # --> make each one contiguous, keep a sid2Xsid and Xsid2sid dict
        #     dumped somewhere that we can optimize if we need to
        # --> also have a map sid2Xsidtypes and we can get sid_type2sids using
        #     the sid2Xsid keys for the given type
        # Note: nsids are a single group, for each sid type, map to the same
        #       nsid set

        # TODO: after adding all the pages and before creating data structs
        #       get the mapping from ori_pid to pid and use pids for everything

        # Step 1: load all the data in original form
        # original page_id -> not contiguous
        # oripid: (title, prep_title, ori_chunk_num_list, content_list)

        # Step 2: Map to new contiguous ids
        # ori_page_id -> page_id
        # ori_chunk_num -> chunk_id
        # ori_(page_id, chunk_num) -> passage_id

        # Non optimized, store just in case
        # ori_page_id -> ori_chunk_num_list
        # page_id -> chunk_id_list
        # page_id -> passage_id_list
        # ** page_id -> first_passage_id (as array)
        # ** passage_id -> page_id (as array)
        # passage_id -> chunk_id

        # Step 3: Use new ids to store data separately for easy loading
        #         dump everything just in case, especially for string case
        # page_id -> title
        # page_id -> prep_title
        # ** page_id -> [passage_ids]
        # ** passage_id -> content [h5py with variable length arrays eventually, pkl now]
        # ** passage_id -> token_data [pkl, map the ent and ori strings to sids]
        # title -> page_id
        # title -> prep_title
        # prep_title -> [page_id]
        # prep_title -> [titles]


        # Step 4: Convert to sids and dump as arrays
        #         Don't store map with general sids
        #         because these can be reconstructed from the strings if we
        #         need them which we shouldn't.
        # ** page_id -> t sid, pt sid     (page_id indexed 2 col array)
        # ** t sid -> page_id            (tsid indexed 1 col array)
        # pt sid -> [page_ids]         (dictionary) 

    #def add_page(self, page_id, title, prep_title):

class StringKey:
    def __init__(
        self,
        data_dir,
        name,
        extra_metadata={},
        building=False,
    ):
        self.metadata = {
            'next_sid': 0,
        }
        self.str2sid_trie = None
        self.sid2str_trie = None
        super().__init__(
            data_dir,
            name,
            save_endings={
                'metadata': 'json',
            },
            building=building,
        )

    def load_str2sid(self):
        self.str2sid_trie = fu.load_file(self.get_path('str2sid_trie', 'pkl'))

    def load_sid2str(self):
        self.sid2str_trie = fu.load_file(self.get_path('sid2str_trie', 'pkl'))

    def save(self):
        self._build_sid2str_from_str2sid()
        super().save()

    def add_str(self, st, allow_repeat=False):
        if st in self.str2sid_trie:
            assert allow_repeat
            return
        self.str2sid_trie[st] = self.metadata['next_sid']
        self.metadata['next_sid'] += 1

    def _build_sid2str_from_str2sid(self):
        del self.sid2str_trie
        self.sid2str_trie = pygtrie.Trie()
        for st, sid in self.str2sid_trie.items():
            self.sid2str_trie[self._int_to_digits(sid)] = st

    def _int_to_digits(self, sid):
        if sid == 1:
            return [1]
        elif sid % 10 == 0:
            return [int(char) for char in str(sid)]
        return [
            (sid // (10**i)) % 10 for i in range(
                math.ceil(math.log(sid, 10) ) - 1, -1, -1
            )
        ]


class SidNormer:
    def __init__(self, data_dir, name, building=False):
        self.sid2nsid_arr = None
        self.nsid2sids_dict = defaultdict(list)

        # Used only for building
        self.sid2nsid_dict = {} if building else None
        super().__init__(
            data_dir,
            name,
            save_endings={
                'sid2nsid_arr': 'npy',
                'nsid2sids_dict': 'pkl', 
            },
            building=building,
        )


    def __len__(self):
        if self.sid2nsid_arr is None:
            return 0
        return self.sid2nsid_arr.size

	# Get the nsids associated with the sids because sid2nsid is
	# a 1D array with inds corresponding to sids and vals as nsids
	def get_nsids_from_sid_array(self, sid_array):
		return np.take_along_axis(
			self.sid2nsid_arr,
			sid_array,
			axis=0,
		)

	def get_nsid_from_sid(self, sid):
		return self.sid2nsid_arr[sid]
        
	def add_sid_nsid(self, sid, nsid):
		self.sid2nsid_dict[sid] = nsid
		self.nsid2sid_dict[nsid].append(sid)

	def load(self):
		super().load()
		self.nsid2sids_dict = defaultdict(list, self.nsid2sids_dict)

	def save(self):
		self.sid2nsid_arr = dict_to_1d_npy_array(
			self.sid2nsid_dict, np.uint32
		)
        self.nsid2sids_dict = dict(self.nsid2sids_dict)
		super().save()


class Int2ContigIdKey(DataStruct):
    def __init__(self, data_dir, name, dtype='uint32', building=False):
        self.val2id_dict = {}
        self.id2val_arr = None
        self.metadata = {
            'dtype': dtype,
            'next_id': 0,
        }
        self.id2data = {} # extra data to load only when needed

        # Used for building only
        self.id2val_list = [] if building else None
        super().__init__(
            data_dir,
            name,
            save_endings={
                'metadata': 'json',
                'val2id_dict': 'pkl',
                'id2val_arr': 'npy',
            },
            building=building
        )


    def last_elem_id(self):
        return self.metadata['next_id'] - 1
        

    def add_val(self, val, allow_repeat=False):
        if val in self.val2id_dict:
            assert allow_repeat
            return
        self.val2id_dict[val] = self.metadata['next_id']
        self.id2val_list.append(val)
        self.metadata['next_id'] +=1

    def make_id2val_arr(self):
        assert self.id2val_list is not None
        assert len(self.id2val_list) == self.metadata['next_id']
        self.id2val_arr = np.array(
            self.id2val_list, dtype=self.metadata['dtype'],
        )

    def save(self):
        self.make_id2val_arr()
        super().save()


    def save_extra_data(self, name, ending):
        file_path = self.get_path(f'extra__{name}')
        assert name in self.id2data
        fu.dump_file(self.id2data[name], file_path)

    def load_extra_data(self, name, ending):
        file_path = self.get_path(f'extra__{name}')
        assert os.path.exits(file_path)
        self.id2data[name] = fu.load_file(file_path)


# # ----------------- Old Classes ----------------- # #

Cid = namedtuple("Cid", "page_id chunk_num")
Page = namedtuple("Page", "title prep_title passage_dict")
Ori2Ent = namedtuple("Ori2Ent", "ent_sid num_total num_unique_cids num_unique_pages")


def invert_dict(in_dict):
    inv_dict = defaultdict(set)
    for k, v in in_dict.items():
        if isinstance(v, list) or isinstance(v, set):
            for vv in v:
                inv_dict[vv].add(k)
        else:
            inv_dict[v].add(k)
    return dict(inv_dict)


def dict_to_1d_npy_array(in_dict, dtype):
    max_val = max(in_dict.keys())
    out_array = -np.ones([max_val + 1], dtype=dtype)
    for k, v in in_dict.items():
        out_array[k] = v
    return out_array


def dict_to_2d_npy_array(in_dict, dtype):
    max_val = max(in_dict.keys())
    max_len = max([len(v) for k, v in in_dict.items()])
    out_array = -np.ones([max_val + 1, max_len], dtype=dtype)
    for k, vlist in in_dict.items():
        for i, v in enumerate(vlist):
            out_array[k][i] = v
    return out_array


"""
class StringKey:
    def __init__(
        self,
        data_dir,
        name,
        string_norm=None,
        building=False,
    ):
        self.name = name
        self.data_dir = data_dir
        self.str2sid = None
        self.sid2str = None
        self.string_norm = None
        self.next_sid = 0
        self.last_save_next_sid = 0
        self.data_files = [
            "str2sid.pkl",
            "sid2str.npy",
            "metadata.json",
        ]

        self.initialize(sid2str=sid2str, str2sid=str2sid, reset=reset)

    def __len__(self):
        return self.next_sid

    def __str__(self):
        return f"StringKey(name={self.name}, len={len(self)})"

    def get_dir(self):
        return f"{self.data_dir}{self.name}/"

    def convert_trie_to_npy(self):
        str_list = list(range(self.next_sid))
        tot_strs = 0
        for st, sid in self.str2sid.items():
            str_list[sid] = st
            tot_strs += 1
        npy_out = np.array(str_list, dtype="U")
        assert tot_strs == self.next_sid
        return npy_out

    def int_to_digits(self, sid):
        if sid == 1:
            return [1]
        elif sid % 10 == 0:
            return [int(char) for char in str(sid)]
        return [
            (sid // (10**i)) % 10 for i in range(
                math.ceil(math.log(sid, 10) ) - 1, -1, -1
            )
        ]

    def convert_str2sid_trie_to_sid2str_trie(self):
        del self.sid2str
        self.sid2str = pygtrie.Trie()
        for st, sid in self.str2sid.items():
            self.sid2str[self.int_to_digits(sid)] = st

    def load(self, str2sid=True, sid2str=True, reset=False):
        save_dir = self.get_dir()
        assert os.path.exists(save_dir)
        if reset:
            return False
        for file_name in self.data_files:
            file = f"{save_dir}{file_name}"
            if not os.path.exists(file):
                logging.info(f">> Missing {file} so reinitialize strkey")
                return False
        metadata = fu.load_file(f"{save_dir}metadata.json", verbose=False)
        self.next_sid = metadata["next_sid"]
        self.last_save_next_sid = metadata["next_sid"]
        if str2sid:
            self.str2sid = fu.load_file(f"{save_dir}str2sid.pkl", verbose=False)
        if sid2str:
            self.sid2str = fu.load_file(f"{save_dir}sid2str.pkl", verbose=False)
        return True

    def save(self, convert=False):
        assert self.last_save_next_sid <= self.next_sid
        if self.last_save_next_sid == self.next_sid and not force:
            logging.info(">> Latest version already saved")
            return

        save_dir = self.get_dir()
        metadata = {"next_sid": self.next_sid}
        fu.dumpjson(metadata, f"{save_dir}metadata.json")
        fu.dumppkl(self.str2sid, f"{save_dir}str2sid.pkl")
        if convert:
            self.convert_str2sid_trie_to_sid2str_trie()
        fu.dumppkl(self.sid2str, f"{sav_dir}sid2str.pkl")

        #self.sid2str = self.convert_trie_to_npy()
        #logging.info(">> Rebuilt sid2str")
        #fu.dumpnpy(self.sid2str, f"{save_dir}sid2str.npy")
        self.last_save_next_sid = self.next_sid
        logging.info(f">> Finished saving string key: {self.name}")

    def initialize(self, sid2str=True, str2sid=True, reset=False):
        save_dir = self.get_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f">> Created save dir: {save_dir}")

        if not self.load(sid2str=sid2str, str2sid=str2sid, reset=reset):
            self.str2sid = pygtrie.CharTrie()

    def add_str(self, st):
        if st in self.str2sid:
            return self.str2sid[st]
        new_str_sid = self.next_sid
        self.str2sid[st] = new_str_sid
        self.next_sid += 1
        return new_str_sid

    def get_sid2str(self, sid):
        assert self.sid2str is not None
        if sid is None or sid not in self.sid2str:
            return None
        return self.sid2str[sid]

    def get_str2sid(self, st):
        assert self.str2sid is not None
        if st is None or st not in self.str2sid:
            return None
        return self.str2sid[st]
"""

def build_pid2tsid(tsid2pid):
    return {int(pid): tsid for tsid, pid in tsid2pid.items()}

def build_pid2ptsid(ptsid2pids, pid2tsid={}):
	pid2ptsid = {}
	for ptsid, pids in ptsid2pids.items():
		for pid in pids:
			pid = int(pid)
			# Only add ptsids that are different than pid
			if pid2tsid.get(pid, -1) != ptsid:
				pid2ptsid[pid] = ptsid
	return pid2ptsid

def build_pid2tsidptsid(max_pid, tsid2pid, ptsid2pids):
    pid2tsid = build_pid2tsid(tsid2pid)
    pid2ptsid = build_pid2ptsid(ptsid2pids, pid2tsid=pid2tsid)

    pid_tsid_ptsid = np.zeros([max_pid, 2], np.uint32)
    for pid in range(max_pid):
        if pid not in pid2tsid:
            continue
        tsid = pid2tsid[pid]
        ptsid = pid2ptsid.get(pid, tsid)
        pid_tsid_ptsid[pid] = (tsid, ptsid)
    return pid_tsid_ptsid


def save_pid2tsidptsid(cfg, pid_tsid_ptsid):
    data_dir = cfg.wiki_processing.passage_data.data_dir
    name = cfg.wiki_processing.passage_data.name
    fu.dumpnpy(pid_tsid_ptsid, f'{data_dir}{name}/pid_tsid_ptsid.npy')


def load_pid2tsidptsid(cfg):
    data_dir = cfg.wiki_processing.passage_data.data_dir
    name = cfg.wiki_processing.passage_data.name
    return fu.load_file(f'{data_dir}{name}/pid_tsid_ptsid.npy', verbose=False)


def load_tsid2pid(cfg):
    data_dir = cfg.wiki_processing.passage_data.data_dir
    name = cfg.wiki_processing.passage_data.name
    return fu.load_file(f"{data_dir}{name}/tsid2pid.pkl", verbose=False)


def load_ptsid2pids(cfg):
    data_dir = cfg.wiki_processing.passage_data.data_dir
    name = cfg.wiki_processing.passage_data.name
    return fu.load_file(f"{data_dir}{name}/ptsid2pids.pkl", verbose=False)




class PassageData:
    def __init__(
        self,
        data_dir,
        name,
        load_page_data=False,
        prep2titles=False,
        extra_norms=False,
        reset=False,
    ):
        self.data_dir = data_dir
        self.name = name
        self.page_data = {}
        self.title2pid = {}
        self.pid2title = {}
        self.data_files = [
            "page_data.pkl",
            "title2pid.pkl",
        ]
        self.prep2titles = None
        self.title2prep = None
        self.on2titles = None
        self.qn2titles = None

        self.initialize(
            prep2titles=prep2titles,
            extra_norms=extra_norms,
            load_page_data=load_page_data,
            reset=reset,
        )

    def __len__(self):
        return len(self.page_data)

    def __str__(self):
        return f"PassageData(name={self.name}, len={len(self)})"

    def get_dir(self):
        return f"{self.data_dir}{self.name}/"

    def load(
        self,
        prep2titles=False,
        extra_norms=False,
        load_page_data=False,
        reset=False
    ):
        save_dir = self.get_dir()
        assert os.path.exists(save_dir)
        if reset:
            return False
        for file_name in self.data_files:
            file = f"{save_dir}{file_name}"
            if not os.path.exists(file):
                logging.info(f">> File doesn't exist so reset PassageData: {file}")
                return False

        self.title2pid = fu.load_file(f"{save_dir}title2pid.pkl", verbose=False)
        if load_page_data:
            self.page_data = fu.load_file(f"{save_dir}page_data.pkl", verbose=False)
        else:
            self.pid2title = {v: k for k, v in self.title2pid.items()}

        if prep2titles:
            self.prep2titles = fu.check_exists_load(
                f"{save_dir}prep2title.pkl",
                default=defaultdict(set),
                verbose=False,
            )
            if not load_page_data:
                self.title2prep = {}
                for prep, titles in self.prep2titles.items():
                    for t in titles:
                        self.title2prep[t] = prep

        if extra_norms:
            self.on2titles = fu.check_exists_load(
                f"{save_dir}on2titles.pkl",
                default=defaultdict(set),
                verbose=False,
            )
            self.qn2titles = fu.check_exists_load(
                f"{save_dir}qn2titles.pkl",
                default=defaultdict(set),
                verbose=False,
            )
        return True

    def save(self):
        save_dir = self.get_dir()
        for data, name in [
            (self.page_data, "page_data"),
            (self.title2pid, "title2pid"),
            (self.prep2titles, "prep2title"),
            (self.on2titles, "on2titles"),
            (self.qn2titles, "qn2titles"),
        ]:
            if data is not None and len(data) > 0:
                fu.dumppkl(dict(data), f"{save_dir}{name}.pkl")
        logging.info(f">> Finished saving passage data: {self.name}")

    def initialize(
        self, prep2titles, extra_norms, load_page_data, reset=False
    ):
        save_dir = self.get_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f">> Created save dir: {save_dir}")

        if not self.load(
            prep2titles=prep2titles,
            extra_norms=extra_norms,
            load_page_data=load_page_data,
            reset=reset,
        ):
            if prep2titles:
                self.prep2titles = defaultdict(set)
            if extra_norms:
                self.on2titles = defaultdict(set)
                self.qn2titles = defaultdict(set)

    def add_page(self, page_id, title, prep_title, on_title, qn_title):
        if page_id in self.page_data:
            return False
        self.page_data[page_id] = Page(
            title=title,
            prep_title=prep_title,
            passage_dict={},
        )
        self.title2pid[title] = page_id
        if self.on2titles is not None:
            self.on2titles[on_title].add(title)
        if self.qn2titles is not None:
            self.qn2titles[qn_title].add(title)
        if self.prep2titles is not None:
            self.prep2titles[prep_title].add(title)
        return True

    def add_passage(self, page_id, chunk_num, content):
        assert page_id in self.page_data
        page = self.page_data[page_id]
        if chunk_num in page.passage_dict:
            return False
        page.passage_dict[chunk_num] = content
        return True

    def get_cids_from_pid(self, page_id):
        all_cids = []
        for chunk_num in self.page_data[page_id].passage_dict.keys():
            all_cids.append(Cid(page_id=page_id, chunk_num=chunk_num))
        return sorted(all_cids, key=lambda c: c.chunk_num)

    def get_title_from_pid(self, page_id):
        if len(self.page_data) == 0:
            return self.pid2title.get(page_id, None)
        return self.page_data[page_id].title

    def get_title_from_cid(self, cid):
        assert isinstance(cid, Cid)
        return self.get_title_from_pid(cid.page_id)

    def get_passage(self, cid):
        assert isinstance(cid, Cid)
        return self.page_data[cid.page_id].passage_dict[cid.chunk_num]

    def get_titles_from_prep_title(self, prep_title):
        assert self.prep2titles is not None
        if prep_title not in self.prep2titles:
            return None
        return self.prep2titles[prep_title]

    def get_prep_title_from_title(self, title):
        if self.title2prep is not None:
            return self.title2prep.get(title, None)

        if title not in self.title2pid:
            return None
        return self.page_data[self.title2pid[title]].prep_title

    def get_titles_from_on_title(self, on_title):
        assert self.on2titles is not None
        if on_title not in self.on2titles:
            return None
        return self.on2titles[on_title]

    def get_titles_from_qn_title(self, qn_title):
        assert self.qn2titles is not None
        if qn_title not in self.qn2titles:
            return None
        return self.qn2titles[qn_title]


class SidNormer:
    def __init__(self, data_dir, name, reset=False):
        self.name = name
        self.data_dir = data_dir
        self.sid2nsid = None
        self.nsid2sids = None
        self.data_files = [
            "sid2nsid.npy",
            "nsid2sids.npy",
        ]
        self.initialize(reset=reset)

    def __len__(self):
        if self.sid2nsid is None:
            return 0
        return self.sid2nsid.size

    def __str__(self):
        return f"SidNormer(name={self.name}, len={len(self)})"

    def get_dir(self):
        return f"{self.data_dir}{self.name}/"

    def load(self, reset=False):
        save_dir = self.get_dir()
        assert os.path.exists(save_dir)
        if reset:
            return False
        for file_name in self.data_files:
            file = f"{save_dir}{file_name}"
            if not os.path.exists(file):
                logging.info(f">> Missing {file} so reinitialize sid normer")
                return False
        self.sid2nsid = fu.load_file(f"{save_dir}sid2nsid.npy", mmm="r", verbose=False)
        self.nsid2sids = fu.load_file(
            f"{save_dir}nsid2sids.npy", mmm="r", verbose=False
        )
        return True

    def load_to_memory(self):
        sid2nsid_mem = self.sid2nsid.tolist()
        self.sid2nsid = {i: s for i, s in enumerate(sid2nsid_mem)}
        nsid2sids_mem = {}
        for i, sidlist in enumerate(self.nsid2sids.tolist()):
            # Get rid of padding
            nsid2sids_mem[i] = set([s for s in sidlist if s >= 0])
        self.nsid2sids = nsid2sids_mem
        print(">> sid2nsid and nsid2sids loaded to memory")

    def initialize(self, reset=False):
        save_dir = self.get_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f">> Created save dir: {save_dir}")

        if not self.load(reset=reset):
            self.sid2nsid = {}

    def add_sid_to_nsid(self, sid, nsid):
        self.sid2nsid[sid] = nsid

    def save(self, force=False):
        if not force:
            logging.info(">> If you really want to do this, call with force=True")
        save_dir = self.get_dir()
        sid2nsid_array = dict_to_1d_npy_array(self.sid2nsid, dtype=np.int64)
        nsid2sids_array = dict_to_2d_npy_array(
            invert_dict(self.sid2nsid), dtype=np.int64
        )
        fu.dumpnpy(sid2nsid_array, f"{save_dir}sid2nsid.npy")
        fu.dumpnpy(nsid2sids_array, f"{save_dir}nsid2sids.npy")
        logging.info(f">> Finished saving sidstring normer: {self.name}")

    # Easy getter bc 1:1 mapping from str -> qnn
    def get_sid2nsid(self, sid):
        if isinstance(self.sid2nsid, dict):
            return self.sid2nsid.get(sid, None)
        if sid >= len(self.sid2nsid):
            return None
        return self.sid2nsid[sid]

    # Harder getter bc 1:many mapping from qnn -> str
    def get_nsid2sids(self, nsid):
        if self.nsid2sids is None:
            return None
        if isinstance(self.nsid2sids, dict):
            return self.nsid2sids.get(nsid, None)
        if nsid >= self.nsid2sids.shape[0]:
            return None
        # Drop padding
        all_sids = set([s for s in self.nsid2sids[nsid] if s >= 0])
        return all_sids

    def get_sids2nsids(self, sids):
        all_nsids = set()
        missing_sids = set()
        for sid in sids:
            nsid = self.get_sid2nsid(sid)
            if nsid is None:
                missing_sids.add(sid)
            else:
                all_nsids.add(nsid)
        return all_nsids, missing_sids


class ConfigManager:
    def __init__(self, data_dir, name, config_types, reset=False):
        self.name = name
        self.data_dir = data_dir
        self.config_types = config_types

        # Loaded data
        self.all_ind2cfgs = {}
        self.data_files = [
            self.get_filename(conf_type) for conf_type in self.config_types
        ]

        # Created from loaded data
        self.all_cfg2inds = {}
        self.next_free_ind = {}
        self.initialize(reset=reset)

    def __len__(self):
        return len(self.config_types)

    def get_dir(self):
        return f"{self.data_dir}{self.name}/"

    def get_filename(self, conf_type):
        return f'{conf_type}_ind2cfg.pkl'

    # Assumes max depth of 1
    def get_cfgkey_from_cfg(self, metadata_cfg):
        key_list = []
        metadata_elems = sorted(list(metadata_cfg.items()))
        for mkey, melem in metadata_elems:
            if melem is None:
                continue
            if 'ind' in mkey and mkey.strip('_ind') in self.config_types:
                continue
            if isinstance(melem, str):
                key_list.extend([mkey, melem])
            elif isinstance(melem, numbers.Number):
                key_list.extend([mkey, str(melem)])
            elif isinstance(melem, dict):
                assert False, "cfgs of level 2+ not supported"
            else:
                # Assume its a collection of unique elems
                sorted_elem = sorted(list(melem))
                frozen_set_elem = frozenset(sorted_elem)
                key_list.extend([mkey, frozen_set_elem])
        return tuple(key_list)

    def build_cfg2ind_maps(self):
        for config_t, ind2cfg in self.all_ind2cfgs.items():
            self.all_cfg2inds[config_t] = {}
            for ind, cfg in ind2cfg.items():
                cfg_key = self.get_cfgkey_from_cfg(cfg)
                self.all_cfg2inds[config_t][cfg_key] = ind

    def build_next_free_ind(self):
        for config_t, ind2cfg in self.all_ind2cfgs.items():
            if len(ind2cfg) == 0:
                self.next_free_ind[config_t] = 0
            else:
                self.next_free_ind[config_t] = max(ind2cfg.keys()) + 1

    def fill_created_structs(self):
        self.build_cfg2ind_maps()
        self.build_next_free_ind()

    def load(self, reset=False):
        save_dir = self.get_dir()
        assert os.path.exists(save_dir)
        if reset:
            return False
        for file_name in self.data_files:
            filepath = f"{save_dir}{file_name}"
            if not os.path.exists(filepath):
                logging.info(
                    f">> Missing {save_dir}{filepath} so reinitialize config manager"
                )
                return False

        # Load the expected data
        self.all_ind2cfgs = {}
        for conf_t in self.config_types:
            filename = self.get_filename(conf_t)
            filepath = f'{save_dir}{filename}'
            self.all_ind2cfgs[conf_t] = fu.load_file(filepath, verbose=False)
        return True

    def initialize(self, reset=False):
        save_dir = self.get_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f">> Created save dir: {save_dir}")

        if not self.load(reset=reset):
            self.all_ind2cfgs = {config_t: {} for config_t in self.config_types}
        self.fill_created_structs()

    def save(self):
        save_dir = self.get_dir()
        for conf_t, ind2cfg in self.all_ind2cfgs.items():
            filename = self.get_filename(conf_t)
            fu.dumppkl(ind2cfg, f'{save_dir}{filename}', verbose=True)
        logging.info(f">> Finished saving config manager: {self.name}")

    def get_cfg_by_ind(self, ind, config_type):
        return self.all_ind2cfgs[config_type][ind]

    # Asserts that cfg ins in ind
    def get_cfg_ind(self, cfg, config_type):
        cfgkey = self.get_cfgkey_from_cfg(cfg)
        return self.all_cfg2inds[config_type][cfgkey]

    def add_cfg_get_ind(self, cfg, config_type):
        cfgkey = self.get_cfgkey_from_cfg(cfg)

        # If this config already has an index, add that to the sweep_cfg
        if cfgkey in self.all_cfg2inds[config_type]:
            return self.all_cfg2inds[config_type][cfgkey]

        # Otherwise, get the next index and add this cfg to indices
        cfg_ind = self.next_free_ind[config_type]
        self.next_free_ind[config_type] += 1
        self.all_cfg2inds[config_type][cfgkey] = cfg_ind
        self.all_ind2cfgs[config_type][cfg_ind] = cfg
        return cfg_ind
