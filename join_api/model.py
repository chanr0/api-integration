#!/usr/bin/env python3

# sandbox:
# time ./src/join_api/model.py --source data/join_api/join_api_fs.json --rebuild --filter domino,email,gmail,googledrive,servicenow,slack,box,dropbox,msonedrive

import redis

import itertools
import json
import logging
import pickle
import sys
import time

from join_api.common import *
import join_api.parser
import join_api.model
import join_api.model_ipaas
import join_api.model_cp4i
import util
import util.encoders

l = logging.getLogger('ikg')

REDIS_HOST = "localhost"
REDIS_PORT = 6379

DESC_PREFIX='d'

WEIGHT_APP = 0.8
WEIGHT_OBJ = 0.6
WEIGHT_INT = 0.4

_R63 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_'
def _int2r63(i):
    if i == 0: return 'A'
    res = ''
    while i:
        res = _R63[i%63] + res
        i = i // 63
    return res

def _r632int(s):
    res = 0
    for c in reversed(s):
        d = _R63.find(c)
        res = res*len(_R63)+d
    return res

def _load_json(filename):
    l.log(25, f'loading {filename}')
    with open(filename) as fd:
        return json.load(fd)

def _save_pickle(filename, obj):
    l.debug(f'pickle.dump({filename})')
    with open(filename, 'wb') as fd:
        pickle.dump(obj, fd)

def _load_pickle(filename):
    l.debug(f'pickle.load({filename})')
    with open(filename, 'rb') as fd:
        obj = pickle.load(fd)
    return obj

def Application(id_, data, ac_type=None, products=None):
    if ac_type is None:
        if data['version'] == 'cp4i':
            ## HACK: This is a semi-converted CP4I object, semi-convert it back
            data['displayName'] = data['display_name']
            return join_api.model_cp4i.Application(id_, data, products=products)
        if 'display_name' in data:
            return join_api.model_ipaas.Application(id_, data, products=products)
        elif 'displayName' in data:
            return join_api.model_cp4i.Application(id_, data, products=products)
        else:
            l.error(f"cannot determine JoinApi metadata type for {data['name']}")
            sys.exit(-1)
    elif ac_type == 'ipaas':
            return join_api.model_ipaas.Application(id_, data, products=products)
    elif ac_type == 'cp4i':
        return join_api.model_cp4i.Application(id_, data, products=products)
    else:
        l.error(f"unknown JoinApi metadata type: {ac_type}")
        sys.exit(-1)

class JoinApiKG_Base:
    def __init__(self, product_database):
        l.debug(f'ACKG()')
        if type(product_database) is str:
            product_database = join_api.ProductDB(product_database)
        self.product_database = product_database

        ## all the applications known by the knowledge graph,
        ## applications store objects and objects store interactions
        self.applications = {}

        ## all the knowledge graph's nodes (applications, objects, or
        ## interactions) indexed by their ids
        self.elem_vec = []

        ## all the object ids indexed by the object's full name
        self.id_map = {}

    def postprocess(self):
        for app in self.applications.values():
            for obj in app.objects.values():
                optypenames = set()
                for op in obj.operations.values():
                    optypename = op.optypename
                    if optypename: optypenames.add(optypename)
                obj.optypenames = optypenames
        # return

    def postcheck(self, flags=0):
        l.log(25, 'checking data')
        for app_name, app in self.applications.items():
            app.check(name=app_name, flags=flags)
        # return

    def info(self):
        l.log(25, f"App Connect KG with {len(ikg_elements)} elements")
        for i in range(10):
            l.log(25, f"{i}. {self.elements[i]}")

    @property
    def elements(self):
        return self.elem_vec

    def element_by_name(self, name):
        name = name.split('/')
        app = self.applications.get(name[0])
        if len(name)==1 or app is None: return app

        obj = app.objects.get(name[1])
        if len(name)==2 or obj is None: return obj

        op = obj.interactions.get(name[2])
        if len(name)==3 or op is None: return op

        return None

    ################################################################
    # This should potentially be pushed to a separate class

    def add_app(self, app_name, app_data):
        l.log(25, f'KG.add_app({app_name})')
        if app_name in self.applications:
            l.warning(f"{app_name} already in KnowledgeGraph, ignored")
            return False
        if self.product_database is None:
            app = Application(len(self.elem_vec), app_data)
        else:
            products = self.product_database.products(app_name)
            app = Application(len(self.elem_vec), app_data, products=products)
        self.applications[app_name] = app
        self.elem_vec.append(app)
        # l.log(25, app.dumps())
        return True

    def add_app_bo(self, ao_name, obj_data):
        #l.log(25, f'KG.add_app_bo({ao_name}, ao_data)')
        app_name, obj_name = ao_name.split(join_api.parser.SEP)
        app = self.applications.get(app_name)
        if app is None:
            l.error(f"cannot find application {app_name} for object {ao_name}")
            return False
        obj = app.Object(len(self.elem_vec), app, obj_data)
        app_objs = app.objects
        app_objs[obj_name] = obj
        self.elem_vec.append(obj)
        # if len(app_objs) < 3:
        #     l.log(25, obj.dumps())
        return True

    def add_bo_op(self, aoo_name, op_data):
        # l.log(25, f'KG.add_bo_op({aoo_name}, op_data)')
        app_name, obj_name, op_name = aoo_name.split(join_api.parser.SEP)
        app = self.applications.get(app_name)
        if app is None:
            l.error(f"cannot find application {app_name} for operation {aoo_name}")
            return False

        obj = app.objects.get(obj_name)
        if obj is None:
            l.error(f"cannot find object {obj_name} for operation {aoo_name}")
            return False

        op = app.Operation(len(self.elem_vec), obj, op_name, op_data)

        ops = obj.operations
        ops[op_name] = op
        self.elem_vec.append(op)
        # if len(ops) < 2:
        #     l.log(25, json.dumps(op_data, indent=2))
        #     l.log(25, op.dumps(detailed=True))
        return True

    def add_bo_prop(self, bo_name, prop_name, prop_data):
        #l.log(25, f'KGD.add_bo_prop({bo_name}, {prop_name}, prop_data)')
        pass

    def add_op_prop(self, op_name, prop_name, prop_data):
        #l.log(25, f'KGD.add_op_prop({op_name}, {prop_name}, prop_data)')
        pass

_BATCH_SIZE = 100

def _chunk(it, size):
    it = iter(it)
    while True:
        p = list(itertools.islice(it, size))
        if not p:
            break
        yield p


class JoinApiKG_Redis:

    def __init__(self, ikg=None, encoder_name=None):
        import redis
        import redis.commands.search.field
        import redis.commands.search.query
        self.redis = redis
        self.TagField = self.redis.commands.search.field.TagField
        self.TextField = self.redis.commands.search.field.TextField
        self.VectorField = self.redis.commands.search.field.VectorField
        self.Query = self.redis.commands.search.query.Query

        self.ikg = ikg
        self.vector_field_name = 'v'
        if encoder_name is None:
            self.encoder = None
            self.vector_size = None
        else:
            self.encoder = util.encoders.get_encoder(encoder_name)
            self.vector_size = self.encoder.embedding_size
        self.r = self.redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

    def delete_database(self, simulate=False):
        print('deleting ', end='')
        if not simulate: p = self.r.pipeline(transaction=False)
        for idx, key in enumerate(self.r.scan_iter(DESC_PREFIX+":*")):
            print(key.decode("utf-8"), end=' ')
            #self.r.delete(key)
            if not simulate: p.delete(key)
            if idx%10==9:
                print('\n  ', end='')
                if idx%100==99 and not simulate: p.execute()
        if not simulate: p.execute()
        print()

    def _build_embeddings(self, simulate=False):
        ikg_elements = self.ikg.elements

        ## 1. Compute all the descriptions used such that we only
        ## compute the embeddings for each description once.  If
        ## multiple objects use the same description, let's iterate
        ## lateron over all these objects (through their ids).
        t = util.Timer()
        descs_ids = {}
        for elem in ikg_elements:
            for elem_desc in elem.get_descriptions():
                if elem_desc is None or len(elem_desc) == 0: continue
                elem_desc = elem_desc.lower()
                arr = descs_ids.get(elem_desc)
                if arr is None:
                    descs_ids[elem_desc] = set([elem.id_])
                else:
                    arr.add(elem.id_)
        l.log(25, f"got {len(descs_ids)} object descripions in {t()}s")

        ## 2. Iterate over all descriptions, compue their embeddings,
        ## and add the corresponding KG nodes to Redis.
        t = util.Timer()
        apps_wo_product = set()
        ints_wo_optypename = set()
        desc_idx = 0
        batch_idx = 0
        batch_max = (len(descs_ids.keys())+_BATCH_SIZE-1) // _BATCH_SIZE
        for batch_descs in _chunk(descs_ids.keys(), _BATCH_SIZE):
            l.debug(f'processing batch {batch_idx+1}/{batch_max}')
            if simulate:
                ## no need to commpute the embeddings if this function
                ## runs in simulation-mode
                batch_embs = None
            else:
                batch_embs = self.encoder.encode_batch(batch_descs)
                p = self.r.pipeline(transaction=False)

            for idx, desc in enumerate(batch_descs):
                if desc is None:
                    l.critical('assertion failed: (desc is None)')
                    #sys.exit(0)
                    continue
                if simulate:
                    el_v = None
                else:
                    el_v = self.encoder.to_numpy(batch_embs[idx]).tobytes()
                l.log(25, f"{desc}: {descs_ids[desc]}")
                for el_id in descs_ids[desc]:
                    el_obj = ikg_elements[el_id]
                    hash_key = DESC_PREFIX + ':' + _int2r63(desc_idx)
                    hash_data = { 'd': desc,
                                  'i': el_id,
                                  't': el_obj.typename }
                    products = el_obj.products
                    if products is None:
                        if el_obj.typename == 'op': app_obj = el_obj.object.application
                        elif el_obj.typename == 'bo': app_obj = el_obj.application
                        else: app_obj = el_obj
                        apps_wo_product.add(app_obj.name)
                    else:
                        hash_data['ps'] = ','.join(products)
                    if el_obj.typename is None:
                        ## this should never happen, we bail out
                        l.critical(f"assertion_failed: {el_obj.full_name}: typename is None")
                        sys.exit(0)
                    if el_obj.typename == TAG_OBJ:
                        hash_data['aid'] = el_obj.application.id_
                        itypenames = el_obj.optypenames
                        if itypenames is None or len(itypenames) == 0:
                            itypenames = set(['actions', 'triggers'])
                        hash_data['it'] = ','.join(itypenames)
                    elif el_obj.typename == TAG_INT:
                        hash_data['oid'] = el_obj.object.id_
                        if el_obj.optypename is None:
                            ints_wo_optypename.add(el_obj.full_name)
                        else:
                            hash_data['it'] = el_obj.optypename
                    l.debug(f"{hash_key} {' '.join([k+':'+str(v) for k,v in hash_data.items()])}")
                    hash_data[self.vector_field_name] = el_v
                    if not simulate: p.hset(hash_key, mapping=hash_data)
                    desc_idx += 1
            if not simulate: p.execute()
            batch_idx += 1
        l.log(25, f"created Redis index in {t()}s")
        if apps_wo_product:
            l.error("error: applications without product identification: " +
                    ', '.join(apps_wo_product))
        if ints_wo_optypename:
            l.error("error: interactions without operation type: " +
                    ', '.join(ints_wo_optypename))

    def _build_index_hnsw(self,
                         number_of_vectors,
                         distance_metric='L2',
                         M=40,
                         EF=200):
        try:
            self.r.ft().dropindex()
        except self.redis.exceptions.ResponseError as err:
            l.info(f"could not drop index (ignored): {err}")
        self.r.ft().create_index([
            self.VectorField(self.vector_field_name, "HNSW",
                             { 'TYPE': 'FLOAT32',
                               'DIM': self.vector_size,
                               'DISTANCE_METRIC': distance_metric,
                               'INITIAL_CAP': number_of_vectors,
                               'M': M,
                               'EF_CONSTRUCTION': EF }),
            self.TextField('d'),
            self.TagField('i'),
            self.TagField('t'),
            self.TagField('it'),
            self.TagField('aid'),
            self.TagField('oid'),
            self.TagField('ps')
            #TextField("item_name"),
            #TagField("country")     
        ])
        ## python api misses option to configure case sensitivity
        ## ft.dropindex idx
        ## ft.create idx SCHEMA v vector hnsw 14 type float32 dim 512|768 distance_metric cosine initial_cap 38424 m 40 ef_construction 200 ef_runtime 20 it TAG CASESENSITIVE t TAG a TAG CASESENSITIVE o TAG CASESENSITIVE
        ## ft.search idx "@oids:{D3}"

    def build_embedding_index(self, simulate=False):
        self._build_embeddings(simulate)
        if not simulate:
            self._build_index_hnsw(len(self.ikg.elements), 'COSINE')
        return

    def save_state(self, filename):
        _save_pickle(filename, [ self.ikg,
                                 self.encoder.name ])

    def load_state(self, filename):
        objs = _load_pickle(filename)
        self.ikg, encoder_name = objs
        if self.encoder is None:
            self.encoder = util.encoders.get_encoder(encoder_name)
            self.vector_size = self.encoder.embedding_size
        else:
            if self.encoder.name != encoder_name:
                raise Exception(f"encoder name mismatch {self.encoder.name} != {encoder_name}")

    ################################################################
    # GOFA code

    @property
    def applications(self):
        return self.ikg.applications

    @property
    def elements(self):
        return self.ikg.elements

    def get_element_by_name(self, names):
        if type(names) is str: names=names.split('/')
        if len(names) == 0: return None

        app = self.ikg.applications.get(names[0])
        if app is None: return None
        if len(names) == 1: return app

        obj = app.objects.get(names[1])
        if obj is None: return None
        if len(names) == 2: return obj

        op = app.operations.get(names[2])
        return op

    def lookup(self, desc, typenames=None, app_ids=None, obj_ids=None, it=None, product=None, k=5, v=1):
        q_vector = self.encoder.encode_batch([desc])[0]
        q_dict = { 'v': self.encoder.to_numpy(q_vector).tobytes() }
        q_attribs = []
        if typenames:
            q_attribs.append(f"@t:{{{typenames}}}")
        if app_ids:
            #q_apps = '|'.join([_int2r63(i) for i in app_ids])
            q_apps = '|'.join([str(i) for i in app_ids])
            q_attribs.append(f"@aid:{{{q_apps}}}")
        if obj_ids:
            #q_objs = '|'.join([_int2r63(i) for i in obj_ids])
            q_objs = '|'.join([str(i) for i in obj_ids])
            q_attribs.append(f"@oid:{{{q_objs}}}")
        if it:
            #q_attribs.append(f"@it:{{{'|'.join(it)}}}")
            q_attribs.append(f"@it:{{{it}}}")
        if product:
            q_attribs.append(f"@ps:{{{product}}}")
        # if desc.lower() == 'attendee':
        #     q_attribs.append('@i:{549}')
        #     print(f"{q_attribs}, v~{desc}")
        #     print(q_dict['v'])
        if len(q_attribs):
            q_str = '(' + ' '.join(q_attribs) + ')'
        else:
            q_str = '*'
        q_str += f" => [KNN {k} @{self.vector_field_name} $v AS v_score]"
        if v > 0: l.debug(f"{q_str}, v~{desc}")
        q = self.Query(q_str).sort_by('v_score').paging(0,k).return_fields('v_score',
                                                                           'i',
                                                                           'd',
                                                                           'aid',
                                                                           'oid',
                                                                           'it').dialect(2)
        q_res = None
        while q_res is None:
            try:
                q_res = self.r.ft().search(q, query_params=q_dict)
            except self.redis.exceptions.BusyLoadingError as err:
                print(f"Redis busy loading error: {err}, sleeping 8s")
                time.sleep(8)
        return q_res

    def lookup_app(self, desc, product=None, k=5, v=1):
        q_res = self.lookup(desc, TAG_APP,
                            product=product, k=k, v=v)
        # _print_results(q_res)
        q_proc = set() # ids we have processed already
        res = []
        for doc in q_res.docs:
            id_ = int(doc.i)
            if id_ in q_proc: continue
            q_proc.add(id_)
            a_s = max(float(doc.v_score), 0)
            score = WEIGHT_APP * a_s
            res.append([id_, score, doc.d])
            # a_obj = self.ikg.elements[id_]
            # print(f"  {id_:>3}. {a_obj.full_name}, s={WEIGHT_APP}*{a_s:4.2f}={score:4.2f}")
        return res

    def lookup_obj(self, desc, a_ids, it=None, product=None, k=5, v=1):
        q_res = self.lookup(desc, TAG_OBJ,
                            app_ids=a_ids.keys(), it=it, product=product, k=k, v=v)
        # _print_results(q_res)
        q_proc = set() # ids we have processed already
        res = []
        for doc in q_res.docs:
            id_ = int(doc.i)
            if id_ in q_proc: continue
            q_proc.add(id_)
            o_obj = self.ikg.elements[id_]
            o_s = max(float(doc.v_score), 0)
            if a_ids:
                a_s = a_ids[o_obj.application.id_]
                score = a_s + WEIGHT_OBJ * o_s
                # print(f"  {id_:>3}. {o_obj.full_name:24}, s={a_s:4.2f}+{WEIGHT_OBJ}*{o_s:4.2f}={score:4.2f}")
            else:
                score = 0 + WEIGHT_OBJ * o_s
                # print(f"  {id_:>3}. {o_obj.full_name:24}, s=0+{WEIGHT_OBJ}*{o_s:4.2f}={score:4.2f}")
            res.append([id_, score, doc.d])
        res.sort(key=lambda el:el[1])
        return res

    def lookup_int(self, desc, o_ids, it=None, product=None, k=5, v=1):
        q_res = self.lookup(desc, TAG_INT,
                            obj_ids=o_ids.keys(), it=it, product=product, k=k, v=v)
        # _print_results(q_res)
        q_proc = set() # ids we have processed already
        res = []
        for doc in q_res.docs:
            id_ = int(doc.i)
            if id_ in q_proc: continue
            q_proc.add(id_)
            i_obj = self.ikg.elements[id_]
            i_s = max(float(doc.v_score), 0)
            o_s = o_ids[i_obj.object.id_]
            score = o_s + WEIGHT_INT * i_s
            res.append([id_, score, doc.d])
            # print(f"  {id_:>3}. {i_obj.full_name:24}, s={o_s:4.2f}+{WEIGHT_INT}*{i_s:4.2f}={score:4.2f}")
        res.sort(key=lambda el:el[1])
        return res

    def match_record(self, record, product=None, topn=5, v=0):
        """
        Match a single record.  A record is of the form:
        { "entity": "servicenow",
          "model": "ticket",
          "interaction": "open",
          "interaction-type": "actions|triggers" }
        where the interaction-type field is optional and if present either
        actions or triggers.

        The result is of the form (sorted by score in descending order):
        [ ScoredItem(score=0.99, record={
            "entity": "servicenow",
            "model": "ticket",
            "interaction": "create"}),
          ... ]
        """

        app_name = record.get("application")
        obj_name = record.get("object")
        int_name = record.get("interaction")
        int_type = record.get("interaction-type")
        #if int_type: int_type = int_type[0]

        ## some hardcoded heuristics for translating requests

        ## trigger operations that start with new, typically map to
        ## created, ideally, our model should know that
        if int_type == 'triggers':
            if int_name == 'new' or int_name.startswith('new '): int_name = 'CREATED'

        ## slack slash commands almost always refer to messages
        ## TODO: this should be moved to the step after the app is
        ## identified and use the identified app for comparison
        if app_name == 'slack' and obj_name == 'slash command':
            obj_name = 'message'

        ## TODO? compute levenshtein distance to certain/all objects?
        ## Redis provides for levenshtein distance search

        ikg_elements = self.ikg.elements

        ################################################################
        #print('#### Applications')
        app_res = self.lookup_app(app_name, product=product, k=topn, v=v-1)
        if v>0:
            l.debug('r=' +
                    ', '.join([ f"{el[0]}:{ikg_elements[el[0]].full_name}={el[1]:4.2f}"
                                for el in app_res ]))

        #print('#### Objects')
        obj_res = self.lookup_obj(obj_name,
                                  a_ids={el[0]: el[1] for el in app_res},
                                  it=int_type, product=product, k=topn*topn, v=v-1)
        if v>0:
            l.debug('r=' +
                    ', '.join([ f"{el[0]}:{ikg_elements[el[0]].full_name}({el[2]})={el[1]:4.2f}"
                                for el in obj_res ]))
        #print('#### Interactions')
        int_res = self.lookup_int(int_name,
                                  o_ids={el[0]: el[1] for el in obj_res},
                                  it=int_type, product=product, k=4*topn*topn, v=v-1)
        if v>0:
            l.debug('r=' +
                    ', '.join([ f"{el[0]}:{ikg_elements[el[0]].full_name}({el[2]})={el[1]:4.2f}"
                                for el in int_res ]))

        results = []
        for i_id, i_s, i_exp in int_res[0:topn]:
            i_obj = ikg_elements[i_id]
            i_fn = i_obj.full_name

            o_obj = i_obj.object
            o_fn = o_obj.full_name
            o_id = o_obj.id_

            a_obj = o_obj.application
            a_fn = a_obj.full_name
            a_id = a_obj.id_
            
            results.append({
                'application': [a_id, a_fn],
                'object': [o_id, o_fn],
                'interaction': [i_id, i_fn],
                'interaction-type': i_obj.optypename,
                'score': i_s
                })
        return results


def Neo4JLoader(source, product_database=None, app_filter=None):
    if source[-5:] == '.json':
        # load from IKG json file, rcreate IKG from App Connect metadata
        l.info('parsing data')
        t = util.Timer()
        ikg = join_api.model.JoinApiKG_Base(product_database)
        source_config = _load_json(source)
        join_api.parser.parse(source_config,
                                ikg,
                                application_filter=appfilter,
                                max_workers=1)
        ikg.postprocess()
        ikg.postcheck(flags=join_api.F_PATCH)
        l.info(f"parsed data in {t()}s")

        ## @psc starting from here, you can use the ikg object, see
        ## JoinApiKG class from above
        neo4j_api = None
        
        ## populate Neo4J, see JoinApiIKG - EmbeddingSearch_Redis
        ## for Redis equivalent, save any state required on the python
        ## side for use with Neo4J (if any)

        return neo4j_api

    if source[-4:] == '.pkl':
        ## @psc load state from pkl file, assumes that Neo4J has already
        ## been loaded through the JSON file above, return Neo4J API
        ## that can be used to issue queries against
    
        neo4j_api = None
        return neo4j_api

    ## illegal request
    return None

def RedisGraphLoader(source, product_database=None, app_filter=None):
    from redisgraph import Node, Edge, Graph

    if source[-5:] == '.json':
        # load from IKG json file, rcreate IKG from App Connect metadata
        l.info('parsing data')
        t = util.Timer()
        ikg = join_api.model.JoinApiKG_Base(product_database)
        source_config = _load_json(source)
        join_api.parser.parse(source_config,
                                ikg,
                                application_filter=appfilter,
                                max_workers=1)
        ikg.postprocess()
        ikg.postcheck(flags=join_api.F_PATCH)
        l.info(f"parsed data in {t()}s")

        ## @psc starting from here, you can use the ikg object, see
        ## JoinApiKG class from above
        graphdb_driver = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        join_api_graph = Graph('join_api', graphdb_driver)
        #join_api_graph.delete()
        t = util.Timer()
        vertices_loaded = 0
        edges_loaded = 0
        for app in ikg.applications.keys():
            prop = {'name': ikg.applications[app].name,
                    'display_name': ikg.applications[app].display_name,
                    'description': ikg.applications[app].description}
            app_node = Node( label = ikg.applications[app].typename,
                             properties =  {k: v if v is not None else '' for k, v in prop.items()})
            join_api_graph.add_node(app_node)
            vertices_loaded += 1
            for bo in ikg.applications[app].objects:
                bo_object = ikg.applications[app].objects[bo]
                prop =  {'name': bo_object.name,
                         'display_name': bo_object.display_name,
                         'description': bo_object.description}
                bo_node = Node( label = bo_object.typename,
                                properties = {k: v if v is not None else '' for k, v in prop.items()})
                join_api_graph.add_node(bo_node)
                vertices_loaded += 1
                join_api_graph.add_edge(Edge(app_node, "business_object", bo_node))
                edges_loaded += 1
                for op in bo_object.interactions:
                    op_object = bo_object.interactions[op]
                    prop = {'name': op_object.name,
                            'display_name': op_object.display_name,
                            'description': op_object.description}
                    op_node = Node( label = op_object.typename,
                                    properties = {k: v if v is not None else '' for k, v in prop.items()} )
                    join_api_graph.add_node(op_node)
                    vertices_loaded += 1
                    join_api_graph.add_edge(Edge(bo_node, "operation", op_node))
                    edges_loaded += 1
        join_api_graph.commit()
        t.print_lap(f"loaded {vertices_loaded} vertices and {edges_loaded} edges to RedisGraph in {{}}s")
        graphdb_driver.close()


        redis_graph_api = None

        ## populate RedisGraph, see JoinApiIKG,
        ## EmbeddingSearch_Redis for Redis equivalent, save any state
        ## required on the python side for use with RedisGraph (if
        ## any)

        return redis_graph_api

    if source[-4:] == '.pkl':
        ## @psc load state from pkl file, assumes that Neo4J has already
        ## been loaded through the JSON file above, return RedisGraph API
        ## that can be used to issue queries against

        redis_graph_api = None
        return redis_graph_api

    ## illegal request
    return None

def _print_results(results):
    print(f"{len(results.docs)} results:")
    for doc in results.docs:
        print(util.COL_BOLD +   '- id  : ' + util.COL_END + doc.id)
        print(util.COL_YELLOW + '  i   : ' + util.COL_END + doc.i)
        print(util.COL_YELLOW + '  desc: ' + util.COL_END + doc.d)
        print(util.COL_YELLOW + '  aid : ' + util.COL_END + doc.aid)
        print(util.COL_YELLOW + '  oid : ' + util.COL_END + doc.oid)
        print(util.COL_YELLOW + '  it  : ' + util.COL_END + doc.it)
        print(util.COL_YELLOW + '  scre: ' + util.COL_END + doc.v_score)
