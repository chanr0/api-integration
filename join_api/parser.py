#!/usr/bin/env python3
# Author: Thomas Gschwind <thg@zurich.co.com>

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from join_api.common import *
from join_api.api import create_client

l = logging.getLogger('ikg')

# disable/enable certain log messages
_LOG_MISSING_OBJECT_SCHEMA = False
_LOG_MISSING_PROPERTIES = False
_LOG_NEITHER_EXCLUDED_NOR_INCLUDED = False
_LOG_NO_REQUEST_PROPERTIES = False
_LOG_NO_RESPONSE_PROPERTIES = False
_LOG_NO_TYPE_FOR_FIELD = False

SEP = '/'

def get_property(props, prop_name):
    global log_errors

    prop_items = prop_name.split('.')
    nested_props = props.get(prop_items[0], None)
    if nested_props == None:
        return None
    for prop_item in prop_items[1:]:
        nested_props2 = nested_props.get("type", None)
        if nested_props2 == None:
            # only log error if enabled
            if _LOG_NO_TYPE_FOR_FIELD:
                l.error(f'no type field for {prop_item} in {prop_name}')
            return None

        if type(nested_props2) == list:
            ## TODO: the array type most likely indicates this is an
            ## array of a property of the nested type => no warning
            if len(nested_props2) != 1:
                l.error(f'got list for prop {prop_item} of len={len(nested_props2)} (expected len=1)')
            nested_props2 = nested_props2[0]

        if type(nested_props2) != dict:
            l.error(f'need dict to look up nested prop {prop_item} in {prop_name}')
            return None

        nested_props2 = nested_props2.get(prop_item, None)
        if nested_props2 == None:
            return None

        nested_props = nested_props2

    return nested_props


def parse_properties(app_name, bo_name, interaction_name,
                     prop_data, all_prop_data, KG):
    full_bo_name = f'{app_name}{SEP}{bo_name}'
    full_op_name = f'{app_name}{SEP}{bo_name}{SEP}{interaction_name}'

    # TODO: this is dirty; just to see the types of keys; will be
    # removed in the future
    for key in prop_data:
        if key == "excluded": continue
        if key == "included": continue
        if key == "mandatory": continue
        l.info(f'{full_op_name}: ignoring unknown property spec "{key}"')

    # TODO: currently only handling mandatory properties
    excluded = prop_data.get("excluded", None)
    included = prop_data.get("included", None)
    if excluded != None and included != None:
        l.info(f'{full_op_name}: found excluded and included property definition, using included')

    if included != None:
        # copy over all included properties
        for prop_name in included:
            # if not prop_name in all_prop_data:
            prop_data = get_property(all_prop_data, prop_name)
            if prop_data == None:
                # only log error if enabled
                if _LOG_MISSING_PROPERTIES:
                    l.info(f'{full_op_name}: missing "{prop_name}" property in properties')
                continue
            KG.add_op_prop(full_op_name,
                           f'{full_bo_name}{SEP}{prop_name}',
                           prop_data)
    elif excluded != None:
        # 1. sanity check, check whether excluded properties actually exist
        for prop_name in excluded:
            if not prop_name in all_prop_data:
                # only log error if enabled
                if _LOG_MISSING_PROPERTIES:
                    l.warning(f'{full_op_name}: missing "{prop_name}" property in properties')
                continue

        # 2. copy over all non-excluded properties
        for prop_name in all_prop_data:
            if prop_name in excluded: continue
            KG.add_op_prop(full_op_name,
                           f'{full_bo_name}{SEP}{prop_name}',
                           all_prop_data[prop_name])
    else:
        # only log error if enabled
        if _LOG_NEITHER_EXCLUDED_NOR_INCLUDED:
            l.info(f'{full_op_name}: neither excluded nor included property definition found')


def parse_bo_data(app_name, bo_name, bo_summary, bo_schema, KG):
    full_bo_name = f'{app_name}{SEP}{bo_name}'
    if bo_schema is None:
        ## if we don't have the detailed bo_schema, use the
        ## interactions data from the summary object; it's the same
        ## minus the request and response properties
        interactions = bo_summary.get('interactions')
    else:
        all_prop_data = bo_schema.get('properties', {})
        if len(all_prop_data) == 0:
            l.info(f'{full_bo_name}: no properties')
        else:
            for prop_name, prop_data in all_prop_data.items():
                KG.add_bo_prop(full_bo_name,
                               f'{full_bo_name}{SEP}{prop_name}',
                               prop_data)
        interactions = bo_schema.get('interactions')

    # interactions
    if not interactions or len(interactions) == 0:
        l.info(f'{full_bo_name}: no interactions')
        return

    for interaction_name, interaction_data in interactions.items():
        # gmail/mail/CREATED/triggers patch
        interaction_type = interaction_data.get("type")
        if app_name == 'gmail' and \
           bo_name == 'mail' and \
           interaction_name == 'CREATED':
            l.warning(f'changing gmail/mail/CREATED interaction type from {interaction_type} to triggers')
            interaction_data["type"] = interaction_type = 'triggers'

        if not interaction_type:
            l.warning(f'no interaction type for {bo_name}{SEP}{interaction_name}')
        if interaction_type not in OP_TYPES:
            l.debug(f'ignored {bo_name}{SEP}{interaction_name} of type {interaction_type}')
            continue
        if interaction_name not in OP_TYPES[interaction_type]:
            l.debug(f'ignored {bo_name}{SEP}{interaction_name} of type {interaction_type}')
            continue
        op_summary = None
        if bo_summary: op_summary = bo_summary.get('interactions').get(interaction_name)
        op_details = None
        if bo_schema: op_details = bo_schema.get('interactions').get(interaction_name)
        # KG.add_bo_op(f'{full_bo_name}{SEP}{interaction_name}',
        #              interaction_data)
        KG.add_bo_op(f'{full_bo_name}{SEP}{interaction_name}',
                     [op_summary, op_details])

        req_prop_data = interaction_data.get("requestProperties")
        if req_prop_data == None:
            # only log error if enabled
            if _LOG_NO_REQUEST_PROPERTIES:
                l.info(f'{full_bo_name}{SEP}{interaction_name}: no request properties')
        else:
            parse_properties(app_name, bo_name, interaction_name,
                             req_prop_data, all_prop_data, KG)

        resp_prop_data = interaction_data.get("responseProperties")
        if resp_prop_data == None:
            # only log error if enabled
            if _LOG_NO_RESPONSE_PROPERTIES:
                l.info(f'{full_bo_name}{SEP}{interaction_name}: no response properties')
        else:
            parse_properties(app_name, bo_name, interaction_name,
                             resp_prop_data, all_prop_data, KG)

################################################################

def no_ao_comparison(app_name, obj1, obj2):
    """Do nothing with (i.e., do not compare) the application objects
    passed to the function.
    """
    pass

def compare_ao_interactions(app_name, ao_summary, ao_schema):
    ao_name = ao_summary.get('name', None)

    # 1. check for each ao_summary interaction whether it is found in
    # the ao_schema
    for interaction in ao_summary['interactions']:
        if interaction not in ao_schema['interactions']:
            l.warning(f'{app_name}/{ao_name}/{interaction} in application_objects but not in object_schema')

    # 2. check for each ao_schema interaction whether it is found in
    # the ao_summary
    for interaction in ao_schema['interactions']:
        if interaction not in ao_summary['interactions']:
            l.warning(f'{app_name}/{ao_name}/{interaction} in object_schema but not in application_objects')


################################################################

class JoinApiProcessor:
    def __init__(self, acc, max_workers=4):
        self.acc = acc
        self.max_workers = max_workers

    def retrieve_applications(self):
        applications = self.acc.get_applications()
        self.process_applications_data(applications)
        app_names = applications.keys()

        if self.max_workers == 0: return

        if self.max_workers <= 1:
            # not using a ThreadPoolExecutor for a single worker
            self.executor = None
            for app_name in app_names:
                self.retrieve_application(app_name, applications[app_name])
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            def retrieve_application(app_name):
                self.retrieve_application(app_name, applications[app_name])
            # NOTE: the for loop is necessary to pick up the futures'
            # results; otherwise the thread pool executor may be shut
            # down before subsequent schema requests can be submitted
            # by the pending futures, additionally, exceptions from
            # the workers are collected here
            for r in self.executor.map(retrieve_application, app_names):
                r.result()
            self.executor.shutdown()
        
    def process_applications_data(self, applications):
        print('got applications object')

    def retrieve_application(self, app_name, app_summary):
        accounts = self.acc.get_accounts_for_application(app_name)
        if len(accounts) == 0:
            l.warning(f'{app_name}: no accounts, skipping')
            return
        if len(accounts) > 1:
            l.warning(f'{app_name}: multiple accounts using first')
        account = accounts[0]

        app_objs = self.acc.get_application_objects(account)
        if app_objs == None:
            l.warning(f'get_application_objects({app_name}) returned None')
            return
        if type(app_objs) is not list:
            if 'objects' not in app_objs:
                l.warning(f'get_application_objects({app_name}) returned object without objects key')
                return
            app_objs = app_objs['objects']
        self.process_application_data(app_name, app_summary, app_objs)

        for ao in app_objs:
            if self.executor == None:
                self.retrieve_object(account, ao)
            else:
                def retrieve_object(account, ao):
                    return self.retrieve_object(account, ao)
                self.executor.submit(retrieve_object, account, ao)

    def process_application_data(self, app_name, app_summary, app_objects):
        print(f"got application data for {app_name}")

    def retrieve_object(self, account, ao_summary):
        app_name = account.app_name
        ao_name = ao_summary.get('name', None)
        # l.info(f'requesting {app_name}/{ao_name}...')

        ao_schema = self.acc.get_object_schema(account, ao_name)
        if ao_schema == None:
            # no need to log the error, was already logged by the
            # get_object_schema function
            if _LOG_MISSING_OBJECT_SCHEMA:
                l.info(f"no object schemas for {app_name}/{ao_name}")
            # TBD: return?
        self.process_object_data(app_name, ao_name, ao_summary, ao_schema)

    def process_object_data(self, app_name, ao_name, ao_summary, ao_schema):
        print(f"got object schema {app_name}/{ao_name}")


################################################################

class JoinApiParser(JoinApiProcessor):
    def __init__(self,
                 acc,
                 kg,
                 application_filter=None,
                 compare_app_objects=no_ao_comparison,
                 max_workers=4):
        super().__init__(acc, max_workers)
        self.kg = kg
        self.application_filter = application_filter
        self.compare_app_objects = compare_app_objects

    def parse(self):
        super().retrieve_applications()

    def process_applications_data(self, applications):
        # nothing to do here, application data is added to the IKG
        # when the application's data will be processed
        pass

    def retrieve_application(self, app_name, app_summary):
        if self.application_filter and \
           app_name not in self.application_filter:
            return
        super().retrieve_application(app_name, app_summary)

    def process_application_data(self, app_name, app_summary, app_objects):
        self.kg.add_app(app_name, app_summary)

    def process_object_data(self, app_name, ao_name, ao_summary, ao_schema):
        ao_data = [ao_summary, ao_schema]
        self.compare_app_objects(app_name, ao_summary, ao_schema)
        ## The application objects data is an array of the app object
        ## summary (as contained in the application data) and the
        ## application object's schema
        self.kg.add_app_bo(f'{app_name}{SEP}{ao_name}', ao_data)
        parse_bo_data(app_name, ao_name, ao_summary, ao_schema, self.kg)


def parse(source, kg,
          application_filter=None,
          compare_app_objects=no_ao_comparison,
          max_workers=4):
    acc = create_client(source)
    if acc is None:
        l.error(f"could not create join_api client from {source}")
        return
    parser = JoinApiParser(acc, kg,
                              application_filter=application_filter,
                              compare_app_objects=compare_app_objects,
                              max_workers=max_workers)
    parser.parse()

## backwards compatibility
parse_join_api_data = parse

################################################################

class KGDebugger:
    def __init__(self):
        l.log(25, f'KGD()')

    def add_app(self, app_name, app_data):
        """Add an application object to the knowledge graph."""
        l.log(25, f'KGD.add_apps({app_name}, app_data)')

    def add_app_bo(self, obj_name, obj_data):
        l.log(25, f'KG.add_app_bo({obj_name}, obj_data)')

    def add_bo_op(self, op_name, op_data):
        l.log(25, f'KGD.add_bo_op({op_name}, op_data)')

    def add_bo_prop(self, bo_name, prop_name, prop_data):
        l.log(25, f'KGD.add_bo_prop({bo_name}, {prop_name}, prop_data)')

    def add_op_prop(self, op_name, prop_name, prop_data):
        l.log(25, f'KGD.add_op_prop({op_name}, {prop_name}, prop_data)')


if __name__ == '__main__':
    filename = "../../connector-schemas"
    ikg = KGDebugger()
    parse_join_api_data_from_directory(filename, ikg)
