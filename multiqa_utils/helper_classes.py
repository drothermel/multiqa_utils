from utils.data_structs import DataStruct, ArrayDict, Int2IntsDict, Str2IntsDict


class PassageData(DataStruct):
    def __init__(
        self,
        data_dir,
        name,
        building=False,
    ):
        # used for building
        self.pid2info = None        # dict
        self.pid2oripid = None      # ArrayDict
        self.paid2cid = None        # ArrayDict
        self.paid2pid = None        # ArrayDict
        self.pid2firstpaid = None   # ArrayDict
        self.paid2text = None       # dict
        self.tstr2pid = None        # Str2IntsDict
        self.paid2tokeninfo = None  # dict
        self.pid2tsidptsid = None   # ArrayDict

        super().__init__(
            data_dir,
            name,
            always_save_attrs={},
            building,
        )

class NormManager(DataStruct):
    def __init__(
        self,
        data_dir,
        name,
        building=False,
    ):
        self.norm_names = []
        super().__init__(
            data_dir,
            name,
            always_save_attrs=[
                'norm_names',
            ],
            building=building,
        )


    def get_norm_data_name(self, norm_name, data_type):
        return f'{norm_name}__{data_type}'

    def _create_attrs_ds_for_norm_type_data(self, norm_name, nsid_dtype):
        assert self.building
        self.norm_names.append(norm_name)
        # sid2nsid
        var_and_ds_name = get_norm_data_name(norm_name, 'sid2nsid')
        sid2nsid = ArrayDict(
            data_dir=self.get_dir(),
            name=var_and_ds_name,
            building=True,
            value_shape=1,
            dtype=nsid_dtype,
            drop_empty=False,
        )
        setattr(
            self,
            var_and_ds_name,
            sid2nsid,
        )

        # str2nsid
        var_and_ds_name = get_norm_data_name(norm_name, 'str2nsid')
        str2nsid = Str2IntsDict(
            data_dir=self.get_dir(),
            name=var_and_ds_name,
            building=True,
        )
        setattr(
            self,
            var_and_ds_name,
            str2nsid,
        )
        # nstr2nsid
        var_and_ds_name = get_norm_data_name(norm_name, 'nstr2nsid')
        nstr2nsid = Str2IntsDict(
            data_dir=self.get_dir(),
            name=var_and_ds_name,
            building=True,
        )
        setattr(
            self,
            var_and_ds_name,
            nstr2nsid,
        )
        return sid2nsid, str2nsid, nstr2nsid

    def add_norm_type_data(self, norm_name, str2nstr_dict, str2sid_arraydict):
        assert self.building
        
        # First build the norm data structs
        max_num_nsids = len(str2nstr_dict)
        nsid_dtype = smallest_np_int_type_for_range(0, max_num_nsids)
        sid2nsid, str2nsid, nstr2nsid = self._create_attrs_for_norm_type_data(
            norm_name, nsid_dtype,
        )
        next_nsid = 0
        for st, nst in str2nstr.items():
            sid = str2sid_arraydict.get_str2sid(st)
            # Get the associated nsid
            if nst not in nstr2nsid:
                nstr2nsid.add_str_vals(nstr, vals=[next_nsid])
                next_nsid += 1
            nsid = nstr2nsid.get_str2int(nst)
            # Build the other data structs
            sid2nsid.add_key_val(sid, nsid, allow_repeats=False)
            str2nsid.add_str_vals(st, vals=[nsid])

    def get_sid2compids(self, norm_names_list):
        assert len(norm_names_list) > 1

        # First load sid2nsid and create all the nsid2sids
        nsid2sids_list = []
        for norm_name in norm_names_list:
            var_name = get_norm_data_name(norm_name, 'sid2nsid')
            self.load_extra_data(var_name)
            var_data = getattr(self, var_name)
            var_data.build_data2id_from_id2data()
            nsid2sids_list.append(var_data.data2id_int2ints)

        # Then create the comps
        init_comps = list(nsid2sids_list[0].values())
        agg_comps = {i: set(sids) for i, sids in enumerate(init_comps)}
        sid2comp = {}
        for i, sids in agg_comps.items():
            for s in sids:
                sid2comp[s] = i

        # Start merging in new norm types
        for nsid2sids in nsid2sids_list[1:]:
            for sids in nsid2sids.values():
                comps_to_merge = set()
                for s in sids:
                    comps_to_merge.add(sid2comp[s])
                
                if len(comps_to_merge) > 1:
                    new_comp = min(comps_to_merge)
                    for s in sids:
                        sid2comp[s] = new_comp
                    for comp in comps_to_merge:
                        if comp != new_comp:
                            agg_comps[new_comp].update(agg_comps[comp])
                        del agg_comps[comp]

        # Then finally compress the range of comp ids to be contig
        num_comps = len(agg_comps)
        for new_id, (old_id, sids) in enumerate(agg_comps.items()):
            for s in sids:
                sid2comp[s] = new_id
        return sid2comp, num_comps


    

