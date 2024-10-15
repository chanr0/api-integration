import csv
import json
import logging
import os

import join_api
from join_api.common import *
import join_api.model
import util

l = logging.getLogger('ikg')

_APPCONNECT_IKG_CACHE_PKL = "data/ikg/join_api_ikg.pkl"

def JoinApiIKG(source,
                  product_database=None,
                  encoder_name='USE',
                  application_filter=None,
                  reload=False,
                  cache_file=_APPCONNECT_IKG_CACHE_PKL,
                  simulate=False):
    if not os.path.isfile(cache_file) or reload:
        ## cache file not present or forced reload requested, load
        ## configuration from source and recreate IKG from App Connect
        ## metadata

        # Generate base Knowledge Graph
        l.info('parsing data')
        t = util.Timer()
        ikg = join_api.model.JoinApiKG_Base(product_database)
        with open(source) as fd:
            source_config = json.load(fd)
        join_api.parser.parse(source_config,
                                ikg,
                                application_filter=application_filter,
                                max_workers=1)
        ikg.postprocess()
        ikg.postcheck(flags=join_api.F_PATCH)
        l.info(f"parsed data in {t()}s")

        # Create VSS index for GOFA
        l.info('creating VSS data...')
        t = util.Timer()
        ikg_vss_facet = join_api.model.JoinApiKG_Redis(ikg, encoder_name)
        ikg_vss_facet.build_embedding_index(simulate=simulate)
        ikg_vss_facet.save_state(cache_file)
        l.info(f"created VSS data in {t()}s")
        return [ikg, ikg_vss_facet]

    # load IKG from cached state file
    if not os.path.isfile(source):
        l.critical(f"no such file or directory: {source}")
        return None
    l.info(f"loading data from {cache_file}")
    t = util.Timer()
    ikg_vss_facet = join_api.model.JoinApiKG_Redis()
    ikg_vss_facet.load_state(cache_file)
    if encoder_name.lower() != ikg_vss_facet.encoder.name.lower():
        l.error(f"cached model uses a different encoder ({ikg_vss_facet.encoder.name}, not {encoder_name})")
    l.info(f"loaded data in {t()}s")
    return [ikg_vss_facet.ikg, ikg_vss_facet]

def result_records_to_str(results):
    res = '['
    sep = ' '
    for result in results:
        r_app = result['application']
        r_obj = result['object']
        r_int = result['interaction']
        res += sep + \
            '{\n' \
            f'  "application": {r_app},\n' \
            f'  "object": {r_obj},\n' \
            f'  "interaction": {r_int},\n' \
            f'  "interaction-type": {result["interaction-type"]},\n' \
            f'  "score": {result["score"]}\n' \
            '}'
        sep = ', '
    return res + ' ]'

def query_record_to_str(rec):
    app_name = rec.get("application")
    obj_name = rec.get("object")
    int_name = rec.get("interaction")
    int_type = rec.get("interaction-type")
    return f"{util.COL_YELLOW}a={app_name}, o={obj_name}, i={int_name}, it={int_type}{util.COL_END}";
