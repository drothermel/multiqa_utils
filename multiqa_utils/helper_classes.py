from collections import defaultdict, namedtuple

import os
import numpy as np
import logging
import pygtrie

import utils.file_utils as fu

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


class StringKey:
    def __init__(self, data_dir, name, sid2str=True, str2sid=True, reset=False):
        self.name = name
        self.data_dir = data_dir
        self.str2sid = None
        self.sid2str = None
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
            self.sid2str = fu.load_file(
                f"{save_dir}sid2str.npy", mmm="r", verbose=False
            )
        return True

    def save(self, force=False):
        assert self.last_save_next_sid <= self.next_sid
        if self.last_save_next_sid == self.next_sid and not force:
            logging.info(f">> Latest version already saved")
            return

        save_dir = self.get_dir()
        metadata = {"next_sid": self.next_sid}
        fu.dumpjson(metadata, f"{save_dir}metadata.json")
        fu.dumppkl(self.str2sid, f"{save_dir}str2sid.pkl")

        self.sid2str = self.convert_trie_to_npy()
        logging.info(f">> Rebuilt sid2str")
        fu.dumpnpy(self.sid2str, f"{save_dir}sid2str.npy")
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
        return self.sid2str[sid]

    def get_str2sid(self, st):
        assert self.str2sid is not None
        if st not in self.str2sid:
            return None
        return self.str2sid[st]


class PassageData:
    def __init__(
        self, data_dir, name, prep2titles=False, extra_norms=False, 
        load_page_data=True, reset=False,
    ):
        self.data_dir = data_dir
        self.name = name
        self.page_data = {}
        self.title2pid = {}
        self.data_files = [
            "page_data.pkl",
            "title2pid.pkl",
        ]
        self.prep2titles = None
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

    def load(self, prep2titles=False, extra_norms=False, load_page_data=True, reset=False):
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

        if prep2titles:
            self.prep2titles = fu.check_exists_load(
                f"{save_dir}prep2title.pkl",
                default=defaultdict(set),
                verbose=False,
            )

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

    def initialize(self, prep2titles, extra_norms, load_page_data, reset=False):
        save_dir = self.get_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info(f">> Created save dir: {save_dir}")

        if not self.load(
            prep2titles=prep2titles, extra_norms=extra_norms,
            load_page_data=load_page_data, reset=reset
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
                logging.info(f">> Missing {file} so reinitialize strkey")
                return False
        self.sid2nsid = fu.load_file(f"{save_dir}sid2nsid.npy", mmm="r", verbose=False)
        self.nsid2sids = fu.load_file(
            f"{save_dir}nsid2sids.npy", mmm="r", verbose=False
        )
        return True

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
            return self.nsid2sids.get(qnn_sid, None)
        if nsid >= self.nsid2sids.shape[0]:
            return None
        # Drop padding
        all_sids = set([s for s in self.nsid2sids[nsid] if s >= 0])
        return all_sids
