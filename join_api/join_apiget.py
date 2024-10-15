#!/usr/bin/env python3
# Author: Thomas Gschwind <thg@zurich.co.com>

import json
import logging
import os
import sys
import time

from concurrent.futures import ThreadPoolExecutor

from join_api.api import create_client
from join_api.parser import JoinApiProcessor

SOURCE_DEFAULT = "data/join_api/credentials/join_api_instance_credentials.json"
TARGET_DEFAULT = "data/join_api"

T_HOUR = 3600
T_DAY  = 24*T_HOUR

THROTTLE = 0.1

l = logging.getLogger('ikg')

def configure_basic_logging(log_level):
    logging.basicConfig(#encoding='utf-8',
                        format='%(created)s %(name)s %(filename)s:%(lineno)s %(levelno)s %(message)s',
                        level=log_level)
    logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

def configure_json_logging(log_level):
    log_fields = {
        'application': lambda r: r.get('name'),
        'message': lambda r: f'{r.get("levelno")}: {r.get("message")}',
        'filename': lambda r: r.get('filename'),
        'line': lambda r: str(r.get('lineno')),
        'time': lambda r: str(r.get('created'))
    }
    formatter = JSONFormatter(log_fields)
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(formatter)
    # l.addHandler(logHandler)
    # l.setLevel(logging.INFO)
    logging.getLogger('').addHandler(logHandler)
    logging.getLogger('ikg').setLevel(log_level)

def has_expired(filename, current_time, expiration_timeout):
    if not os.path.exists(filename): return True
    if expiration_timeout is None: return False
    mtime = os.path.getmtime(filename)
    l.debug(f"{filename}: {current_time - mtime}s old")
    return expiration_timeout < current_time - mtime

def load_json(filename):
    l.debug(f'loading {filename}')
    with open(filename) as fd:
        return json.load(fd)

class JoinApiDownloader(JoinApiProcessor):
    def __init__(self,
                 acc,
                 target_directory,
                 app_timeout=None,
                 obj_timeout=None,
                 application_filter=None,
                 max_workers=4):
        super().__init__(acc, max_workers)
        self.target_directory = target_directory
        if obj_timeout is not None and \
           (app_timeout is None or app_timeout > obj_timeout):
            l.error(f'app_timeout > obj_timeout, using obj_timeout for both')
            app_timeout = obj_timeout
        self.application_expiration = app_timeout
        self.object_expiration = obj_timeout
        self.application_filter = application_filter
        self.current_time = time.time()

    def download(self):
        super().retrieve_applications()

    def process_applications_data(self, applications):
        fn = os.path.join(self.target_directory,
                          'applications.json')
        with open(fn, 'w') as fd:
            json.dump(applications, fd, indent=2)

    def retrieve_application(self, app_name, app_summary):
        if self.application_filter and \
           app_name not in self.application_filter:
            # l.debug(f'{app_name}: filtered out')
            return
        filename = os.path.join(self.target_directory,
                                'application_objects',
                                f"{app_name}.json")
        if not has_expired(filename, self.current_time, self.application_expiration):
            l.info(f'{app_name}: not expired')
            return
        l.info(f'{app_name}: retrieving')
        super().retrieve_application(app_name, app_summary)

    def process_application_data(self, app_name, app_summary, app_objects):
        dirname = os.path.join(self.target_directory,
                               'application_objects')
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        filename = os.path.join(dirname, f"{app_name}.json")
        with open(filename, 'w') as fd:
            json.dump(app_objects, fd, indent=2)

    def retrieve_object(self, account, ao_summary):
        app_name = account.app_name
        ao_name = ao_summary.get('name', None)
        filename = os.path.join(self.target_directory,
                                'object_schemas',
                                app_name,
                                f"{ao_name}.json")
        if not has_expired(filename, self.current_time, self.object_expiration):
            l.info(f'{app_name}/{ao_name}: not expired')
            return
        l.info(f'{app_name}/{ao_name}: retrieving')
        super().retrieve_object(account, ao_summary)

    def process_object_data(self, app_name, ao_name, ao_summary, ao_schema):
        dirname = os.path.join(self.target_directory,
                               'object_schemas',
                               app_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        filename = os.path.join(dirname, f"{ao_name}.json")
        with open(filename, 'w') as fd:
            json.dump(ao_schema, fd, indent=2)

def get_app_data(acc, apps, accounts, throttle):
    # app_name = next(iter(accounts.keys()))
    # for idx, app_name in enumerate(accounts.keys()):
    for idx, app_name in enumerate(apps.keys()):
        if app_name not in accounts: continue

        # print()

        # print(f"Retrieving {idx}.{app_name} object...", end='')
        # time.sleep(throttle)
        # app_eps, com_eps = acc.get_application_endpoints(acc.get_accounts(app_name)[0])
        # print(com_eps)
        # print(json.dumps(app_eps, indent=2))

        print(f"Retrieving {idx}.{app_name} objects...", end='')
        time.sleep(throttle)
        app_objs, com_objs = acc.get_application_objects(acc.get_accounts(app_name)[0])
        print(com_objs)
        # print(json.dumps(app_objs, indent=2))

        # print('================================================================')

    
################################################################
if __name__ == '__main__':
    import argparse

    # Argument Parser and Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=SOURCE_DEFAULT,
                        help='JoinApi instance')
    parser.add_argument('--target', default=TARGET_DEFAULT,
                        help='target directory')
    parser.add_argument('--deployment-target', default=None, help='deployment target (IPAAS,CP4I)')
    parser.add_argument('--filter', default=None,
                        help='comma-separated list of applications to load')
    parser.add_argument('--app-timeout', type=int, default=None,
                        help='timeout to reload applications in seconds')
    parser.add_argument('--obj-timeout', type=int, default=None,
                        help='timeout to reload object schemas in seconds')
    parser.add_argument('--retries', type=int, default=1,
                        help='concurrent connections to use for downloading')
    parser.add_argument('--workers', type=int, default=4,
                        help='concurrent connections to use for downloading')
    parser.add_argument('--log-level', default='info',
                        help='log messages above log level (debug, info, warning, error, critical)')
    args = parser.parse_args()

    # Set up Logging
    log_levels={'debug': logging.DEBUG,
                'info': logging.INFO,
                'warning': logging.WARNING,
                'error': logging.ERROR,
                'critical': logging.CRITICAL}
    log_level = log_levels[args.log_level]
    configure_basic_logging(log_level)
    # configure_json_logging(log_level)

    # Argument Processing
    join_api_source = load_json(args.source)
    if join_api_source['connector'] == 'join_api_cp4i_api':
        debug_cp4i = True

    if args.filter is None:
        appfilter = None
    else:
        appfilter = {s.strip() for s in args.filter.split(',')}

    if args.deployment_target:
        appfilter = read_deployment_target('../../data/deployment_targets.csv',
                                           args.deployment_target,
                                           appfilter)

    if args.obj_timeout is not None and \
       (args.app_timeout is None or args.app_timeout > args.obj_timeout):
        print('ERROR: object-timeout must be greater than application-timeout')
        parser.print_help()
        sys.exit(1)

    acc = create_client(join_api_source,
                        os.path.dirname(args.source),
                        retries=args.retries)

    if debug_cp4i:
        apps = acc.get_applications()
        print(f"{len(apps)} applications of type {type(apps)}")
        #print(json.dumps(apps, indent=2))
        print('================================================================')

        print()
        all_accounts = acc.get_all_accounts(reload=False)
        print(f"{len(all_accounts)} accounts of type {type(all_accounts)}")
        #print(json.dumps(all_accounts, indent=2))
        print('================================================================')

        get_app_data(acc, apps, all_accounts, THROTTLE)

        # check for missing accounts
        accounts_missing = []
        for app_name in apps.keys():
            if app_name not in all_accounts: accounts_missing.append(app_name)
        # check accounts for which we don't have apps
        accounts_not_in_apps = []
        for app_name in all_accounts.keys():
            if app_name not in apps: accounts_not_in_apps.append(app_name)

        print(accounts_missing)
        print(f"missing accounts ({len(accounts_missing)    :>2}): {', '.join(accounts_missing)    }")
        print(f"extra accounts   ({len(accounts_not_in_apps):>2}): {', '.join(accounts_not_in_apps)}")

        sys.exit(0)

    downloader = JoinApiDownloader(acc, args.target,
                                      app_timeout=args.app_timeout,
                                      obj_timeout=args.obj_timeout,
                                      application_filter=appfilter,
                                      max_workers=args.workers)
    import time
    start = time.time()
    downloader.download()
    end = time.time()
    print(f'data retrieval took {end - start} seconds')
