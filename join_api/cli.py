#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import join_api
import join_api.common
from join_api.model import _print_results
import join_api.testdata

import util

request_data = [e[0] for e in join_api.testdata.request_top_responses]

def _show_details(ikg):
    # Show the applications supported by different App Connect products
    applications = ikg.applications
    apps_by_product = dict()
    for key, application in applications.items():
        if key != application.name:
            print(f"key '{key}' does not match application name '{application.name}'")
            sys.exit(1)
        products = application.products
        for p in products:
            papps = apps_by_product.setdefault(p, [])
            papps.append(application.name)
    print('App Connect applications supported by IKG by product:')
    for pkey, papps in apps_by_product.items():
        print(f"{pkey}: {sorted(papps)}")

    # Show the applications that are missing in the IKG
    pd = ikg.ikg.product_database
    missing_apps = []
    for apps in pd.applications:
        if apps not in applications: missing_apps.append(apps)
    print('App Connect applications not supported by IKG:')
    print(f"  {sorted(missing_apps)}")

    # Compute the counts of the different element types
    elements = ikg.elements
    typenames = {
        join_api.common.TAG_APP: 0,
        join_api.common.TAG_OBJ: 0,
        join_api.common.TAG_INT: 0
    }
    for e in elements:
        typenames[e.typename] += 1

    print('Type counts:')
    for tn, tn_count in typenames.items():
        print(f"{tn:<3}: {tn_count:>5}")

if __name__ == '__main__':
    logging.basicConfig(# encoding='utf-8',
                        format='%(created)s %(name)s %(filename)s:%(lineno)s %(levelno)s %(message)s',
                        # filename='example.log',
                        level=logging.INFO)

    ################################
    # argument handling
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', dest='delete', action='store_true',
                        help='delete data from the vector store and exit')
    parser.set_defaults(delete=False)
    parser.add_argument('--simulate', dest='simulate', action='store_true',
                        help='only simulate execution, do change vector store')
    parser.set_defaults(simulate=False)
    parser.add_argument('--source', default='data/join_api/join_api_fs.json',
                        help='source data for the IKG (App Connect API or App Connect offline data)')
    parser.add_argument('--details', dest='details', action='store_true',
                        help='show detailed IKG information')
    parser.set_defaults(details=False)
    parser.add_argument('--neo4j', dest='neo4j', action='store_true',
                        help='use neo4j endpoint')
    parser.set_defaults(neo4j=False)
    parser.add_argument('--product', default='cp4i',
                        help='product to use: cp4i, ipaas, demo')
    parser.add_argument('--redisgraph', dest='redisgraph', action='store_true',
                        help='use redisgraph endpoint')
    parser.set_defaults(redisgraph=False)
    parser.add_argument('--rebuild', dest='rebuild', action='store_true',
                        help='rebuild vector store')
    parser.set_defaults(rebuild=False)
    parser.add_argument('--redistest', dest='redistest', action='store_true',
                        help='run simple Redis test search')
    parser.set_defaults(redistest=False)
    parser.add_argument('--amrtest', dest='amrtest', action='store_true',
                        help='run AMR tests')
    parser.set_defaults(amrtest=False)
    parser.add_argument('--encoder', default='USE',
                        help='word/sentence encoder to use (bert or USE)')
    parser.add_argument('--filter', default=None,
                        help='comma-separated list of applications to load (not supported for pickle source)')
    parser.add_argument('-i', '--interactive', dest='interactive', action='store_true',
                        help='start interactive python interpreter after creating the model')
    parser.set_defaults(interactive=False)

    args = parser.parse_args()

    global OPT_SIMULATE
    OPT_SIMULATE = args.simulate

    if args.filter is None:
        appfilter = None
    else:
        appfilter = {s.strip() for s in args.filter.split(',')}

    ################################
    # arguments handled, "main" code

    if args.delete:
        ikg = join_api.model.JoinApiKG_Redis()
        ikg.delete_database()
        sys.exit(0)

    if args.neo4j:
        neo4j_api = join_api.Neo4JLoader(args.source)
        ## @psc issue test queries against neo4j api
        sys.exit(0)

    if args.redisgraph:
        redis_graph_api = join_api.RedisGraphLoader(args.source)
        ## @psc issue test queries against RedisGraph api
        sys.exit(0)

    _, ikg = join_api.JoinApiIKG(args.source,
                                      'data/deployment_targets.csv',
                                      encoder_name=args.encoder,
                                      application_filter=appfilter,
                                      reload=args.rebuild,
                                      simulate=OPT_SIMULATE)
    # Initialize encoder
    ikg.encoder.encode_batch(["Hello World"])

    if args.redistest:
        t = util.Timer()
        _print_results(ikg.lookup('Create email', typenames='app|obj|int'))
        _print_results(ikg.lookup('message'))
        _print_results(ikg.lookup('slack', typenames='app'))
        _print_results(ikg.lookup('costerlingoms',typenames='app'))
        _print_results(ikg.lookup('email', typenames='app|obj'))
        _print_results(ikg.lookup('email', typenames='obj'))
        # _print_results(ikg.lookup('create message',
        #                           obj_ids=[1,2],
        #                           typenames=TAG_INT))
        print(f"queries took {t()}s")

    if args.details:
        _show_details(ikg)

    if args.amrtest:
        t = util.Timer()
        for record in request_data[:1]:
            print()
            print(join_api.query_record_to_str(record))
            res = ikg.match_record(record, args.product)
            print(join_api.result_records_to_str(res))
        t.print_lap()

    if args.interactive:
        print("ikg.match_record({'application': 'gmail', 'object': 'mail', 'interaction': 'send', 'interaction-type': 'actions'})[:5]")
        import code
        code.interact(local=locals())
