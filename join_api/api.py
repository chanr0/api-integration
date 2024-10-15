#!/usr/bin/env python3
# Author: Thomas Gschwind <thg@zurich.co.com> and 
#         Riza Özçelik <riz@zurich.co.com>

import http.client
import http.cookies
import json
import logging
import os
import requests
import ssl
import sys
import time
import urllib.parse

from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

l = logging.getLogger('ikg')

Account = namedtuple('Account', ['app_name', 'name', 'id'])

def _load_json(filename):
    # l.debug(f'loading {filename}')
    with open(filename) as fd:
        return json.load(fd)

def _create_error_message(response):
    code = response.status_code
    try:
        error_messages = response.json()['errors']
        message = '. '.join([m['message'] for m in error_messages])
    except:
        message = response.text

    return f"Status Code: {code}. Message: {message}"


################################################################

_IPAASCLIENT_PATHS = {
    "base": {
        "US": "https://firefly-api-prod.join_api.cocloud.com",
        "UK": "https://firefly-api-produk.eu-gb.join_api.cocloud.com"
    },
    "create_access_token": "tokens",
    "get_applications": "api/v1/applications",
    "get_application_accounts": "api/v1/applications/{app_name}/accounts",
    "get_application_objects": "api/v1/applications/{app_name}/accounts/{account_name}/objects",
    "get_object_schema": "api/v1/internal-connectors/discovery/api/v0/applications/{app_name}/accounts/{account_id}/models/{object_name}"
}

class IPAASClient:
    def __init__(self,
                 configuration,
                 configuration_dir,
                 retries=3):
        self.instance_credentials = join_api_configuration
        self.access_token_filename = \
            os.path.join(configuration_dir, configuration.get('access_token_filename', 'access_token.json'))
        self.retries = retries

        instance_id = join_api_configuration.get('instance_id')
        location = join_api_configuration.get('instance_location').upper()
        locations = _IPAASCLIENT_PATHS['base']

        if location not in locations:
            raise ValueError(f'unsupported instance location: {location}')
        self.base = f'{locations[location]}/{instance_id}'

        if access_token_filename is None or \
           not os.path.exists(access_token_filename):
            self._create_access_token()
        else:
            self.access_token = _load_json(access_token_filename)

        self.app_names = self.get_applications()

    def _send_get_request(self, endpoint):
        # l.debug(f'Requesting {endpoint}')
        response = requests.get(endpoint, headers=self.access_token)
        # Check token expiration
        if response.status_code == 401:
            l.info('existing API token expired, requesting new one')
            self._create_access_token()
            response = requests.get(endpoint, headers=self.access_token)
        # Retry if request is unsuccessful.
        n_retries = 0
        while response.status_code != 200 and \
              response.status_code != 400 and \
              n_retries < self.retries:
            # increase the sleep time with the number of retries
            time.sleep(0.1*(1<<n_retries))
            l.debug(f'resending request to {endpoint} (retry {n_retries})')
            response = requests.get(endpoint, headers=self.access_token)
            n_retries += 1

        return response

    def _create_access_token(self):
        endpoint = f'{self.base}/{_IPAASCLIENT_PATHS["create_access_token"]}'
        response = requests.post(endpoint,
                                 data={'grant_type': 'password',
                                       'username': 'apiKey',
                                       'password': self.instance_credentials['apikey']})
        if response.status_code != 200:
            raise ConnectionError(f'cannot request new access token: {_create_error_message(response)}')

        access_token = response.json()['access_token']
        self.access_token = { 'Authorization': f'bearer {access_token}' }
        if self.access_token_filename and \
           os.path.exists(self.access_token_filename):
            with open(self.access_token_filename, 'w') as fd:
                json.dump(self.access_token, fd, indent=4)
            l.info('created and saved access token')
        else:
            l.info('created access token')

    def get_applications(self):
        endpoint = f'{self.base}/{_IPAASCLIENT_PATHS["get_applications"]}'
        response = self._send_get_request(endpoint)

        if response.status_code != 200:
            raise ConnectionError(f'cannot retrieve application names: {_create_error_message(response)}')

        app_names = {application['name']: application
                     for application in response.json()['applications']}

        # remove this dummy app, if it is returned
        app_names.pop('streaming-connector-scheduler', None)
        return app_names

    def get_accounts_for_application(self, app_name):
        if app_name not in self.app_names:
            raise ValueError(f'{app_name} is not an JoinApi application')
        s = _IPAASCLIENT_PATHS['get_application_accounts']
        objects_endpoint = s.format(app_name=app_name)
        endpoint = f'{self.base}/{objects_endpoint}'
        response = self._send_get_request(endpoint)

        if response.status_code != 200:
            raise ConnectionError(f'cannot retrieve accounts for {app_name}. {_create_error_message(response)}')

        # l.debug(f'retrieved {len(response.json().get("accounts",[]))} accounts')
        accounts = response.json()['accounts']
        return [Account(app_name=app_name, name=account['name'], id=account['id']) for account in accounts]

    def get_application_objects(self, account):
        app_name = account.app_name
        account_name = account.name
        # retrieve application objects
        s = _IPAASCLIENT_PATHS['get_application_objects']
        objects_endpoint = s.format(app_name=app_name, account_name=account_name)
        endpoint = f'{self.base}/{objects_endpoint}'
        response = self._send_get_request(endpoint)

        if response.status_code != 200:
            l.error(f'cannot retrieve application objects of {app_name} (account {account_name}): {_create_error_message(response)}')
            return None

        # l.debug(f'application objects for {app_name}: {response.json()}')
        return response.json()

    def get_object_schema(self, account, object_name, savedir=None):
        account_id = account.id
        app_name = account.app_name
        s = _IPAASCLIENT_PATHS['get_object_schema']
        schema_endpoint = s.format(app_name=app_name, account_id=account_id, object_name=object_name)
        endpoint = f'{self.base}/{schema_endpoint}'
        # l.debug(f'connecting to {endpoint}')
        response = self._send_get_request(endpoint)
        if response.status_code != 200:
            l.error(f'cannot retrieve object schema {app_name}/{object_name} (account {account.name}): {_create_error_message(response)}')
            return None

        # l.debug(f'Object schemas for {app_name}: {response.json()}')
        return response.json()


################################################################

def _accounts2dict(accounts):
    res = {}
    for el in accounts:
        app = el['connectorType']
        if app in res: res[app].append(el)
        else: res[app] = [el]
    return res

def _encode_account(accountid):
    # join_api account encoding is not following the HTTP standard
    # return urllib.parse.quote_plus(account.id)
    return accountid.replace(' ', '%20')

class CP4IClient:
    def __init__(self,
                 configuration,
                 configuration_dir,
                 retries=3):
        self.configuration = configuration
        self.retries = retries
        self.ssl_verify = False

        self.apps_url = configuration['applications_url']
        self.cas_host = configuration['cas_host']
        self.lcp_host = configuration['lcp_host']
        self.cookie_header = {'Cookie': f"jwt={configuration['jwt_cookie']}"}
        self.cache_dir = configuration['cache_dir']

        ################################################################
        # Defining certificate related stuff and host of endpoint
        certificate_file = os.path.join(configuration_dir, 'tls.crt')
        key_file = os.path.join(configuration_dir, 'tls.key')
        # certificate_secret= 'your_certificate_secret'

        # # Defining parts of the HTTP request
        # request_url='/a/http/url'
        # request_headers = {
        #     'Content-Type': 'application/json'
        # }

        ################################################################
        # Define the client certificate settings for https connection
        context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        # context.verify_mode = ssl.CERT_REQUIRED
        # self.context.load_cert_chain(certfile=certificate_file, password=certificate_secret)
        context.load_cert_chain(certfile=certificate_file, keyfile=key_file)
        self.context = context

        self.access_token = None
        self.applications = None
        self.accounts = None

        # create cache directory
        dirs = [ self.cache_dir,
                 os.path.join(self.cache_dir, 'application_objects'),
                 os.path.join(self.cache_dir, 'application_endpoints') ]
        for d in dirs:
            if not os.path.isdir(d): os.mkdir(d)

    def _send_get_request(self, endpoint):
        # l.debug(f'Requesting {endpoint}')
        if self.access_token:
            response = requests.get(endpoint,
                                    headers=self.access_token,
                                    verify=self.ssl_verify)
            # Check token expiration
            if response.status_code == 401:
                l.info('existing API token expired, requesting new one')
                self._create_access_token()
                response = requests.get(endpoint,
                                        headers=self.access_token,
                                        verify=self.ssl_verify)
        else:
            response = requests.get(endpoint,
                                    verify=self.ssl_verify)

        # Retry if request is unsuccessful.
        n_retries = 0
        while response.status_code != 200 and \
              response.status_code != 400 and \
              n_retries < self.retries:
            print('NOT IMPLEMENTED')
            sys.exit(1)
            # increase the sleep time with the number of retries
            time.sleep(0.1*(1<<n_retries))
            l.debug(f'resending request to {endpoint} (retry {n_retries})')
            response = requests.get(endpoint, headers=self.access_token)
            n_retries += 1

        return response

    def get_applications(self, reload=False):
        cached_response_file = os.path.join(self.cache_dir, 'applications.json')
        if not reload:
            # use cached applications list
            if self.applications is not None:
                return self.applications
            if os.path.isfile(cached_response_file):
                response_json = _load_json(cached_response_file)
                self.applications = { application['name']: application
                                      for application in response_json }
                return self.applications

        endpoint = self.apps_url
        response = self._send_get_request(endpoint)
        if response.status_code != 200:
            raise ConnectionError(f'cannot retrieve application names: {_create_error_message(response)}')
        response_json = response.json()
        with open(cached_response_file, 'w') as fd:
            json.dump(response_json, fd, indent=2)

        self.applications = {application['name']: application
                             for application in response_json}
        return self.applications

    def get_accounts_for_application(self, app_name, reload=False):
        return self.get_accounts(app_name, reload)

    def get_accounts(self, app_name, reload=False):
        accounts = self.get_all_accounts(reload=reload)
        app_accounts = accounts.get(app_name, [])
        return [Account(app_name=app_name, name=account['name'], id=account['_id']) for account in app_accounts]

    def _http_get_json(self, host, url, headers={}):
        # Create a connection to submit HTTP requests
        connection = http.client.HTTPSConnection(host, port=443, context=self.context)
        connection.request("GET", url, headers=headers)
        try:
            response = connection.getresponse()
        except http.client.RemoteDisconnected:
            return None, None
        content_type = response.getheader('content-type')
        content_type = content_type.split(';')
        if content_type[0].lower() != 'application/json':
            print('ERROR: unexpected content-type: {content_type[0]}')
            return response, None

        # the default charset for HTTP is ISO-8859-1
        charset='iso-8859-1'
        for param in content_type[1:]:
            param=param.strip()
            if param.startswith('charset='): charset=param[8:].lower()
        if charset not in {'iso8859-1', 'utf-8'}:
            print('ERROR: response with unknown charset: {charset}')
            return response, None

        data = response.read().decode(charset)
        return response, json.loads(data)

    def get_all_accounts(self, reload=False):
        cached_response_file = os.path.join(self.cache_dir, 'accounts.json')
        if not reload:
            # use cached account data
            if self.accounts is not None:
                return self.accounts
            if os.path.isfile(cached_response_file):
                accounts = _load_json(cached_response_file)
                if type(accounts) is list:
                    self.accounts = _accounts2dict(accounts)
                else:
                    self.accounts = accounts
                return self.accounts

        url = f"https://{self.cas_host}/v2/accounts"
        _, json_data = self._http_get_json(self.cas_host, url, headers=self.cookie_header)
        with open(cached_response_file, 'w') as fd:
            json.dump(json_data, fd, indent=2)
        self.accounts = _accounts2dict(json_data)
        return self.accounts

    def get_application_endpoints(self, account, reload=False):        
        app_name = account.app_name
        account_name = account.name

        cached_response_file = os.path.join(self.cache_dir, 'application_endpoints', app_name.lower()+'.json')
        if not reload:
            # use cached response
            if os.path.isfile(cached_response_file):
                response = _load_json(cached_response_file)
                return response, 'cached'

        # l.info('================')
        # l.info(f"{app_name}, {account_name}")
        # retrieve application endpoints
        url = f"https://{self.lcp_host}/api/v0/applications/{urllib.parse.quote_plus(app_name)}/accounts/{_encode_account(account.id)}/models/endpoint"
        print(url)
        _, json_data = self._http_get_json(self.lcp_host, url)
        if not json_data or 'status' in json_data:
            print(f"ERROR: {json_data['status']}")
            print(json.dumps(json_data, indent=2))
            return json_data, 'unexpected'
        with open(cached_response_file, 'w') as fd:
            json.dump(json_data, fd, indent=2)
        return json_data, 'new'
        
    def get_application_objects(self, account, reload=False):
        app_name = account.app_name
        account_name = account.name

        cached_response_file = os.path.join(self.cache_dir, 'application_objects', app_name.lower()+'.json')
        if not reload:
            # use cached response
            if os.path.isfile(cached_response_file):
                response = _load_json(cached_response_file)
                return response, 'cached'

        # l.info('================')
        # l.info(f"{app_name}, {account_name}")
        # retrieve application objects
        url = f"https://{self.lcp_host}/api/v0/applications/{urllib.parse.quote_plus(app_name)}/accounts/{_encode_account(account.id)}/objects"
        print(url)
        _, json_data = self._http_get_json(self.lcp_host, url)
        if type(json_data) is not list:
            if type(json_data) is dict: print(f"ERROR: {json_data['status']}")
            else: print('ERROR:')
            print(json.dumps(json_data, indent=2))
            return json_data, 'unexpected'
        with open(cached_response_file, 'w') as fd:
            json.dump(json_data, fd, indent=2)
        return json_data, 'new'

    def get_object_schema(self, account, object_name, savedir=None):
        # object schema not supported by this connector
        print("NOT IMPLEMENTED!!!")
        not_implemented()
        return None


################################################################

# disable/enable certain log messages
_LOG_UNAVAILABLE_OBJECT = False

_FSCLIENT_DEFAULT_CONFIGURATION = {
    'applications': 'data/join_api/applications.json',
    'application_objects': 'data/join_api/application_objects/{app_name}.json',
    'object_schemas': 'data/join_api/object_schemas/{app_name}/{object_name}.json'
}

def _remove_suffix(s, suffix):
    return s[:s.index(suffix)]

class FSClient:
    def __init__(self, configuration=None, configuration_dir=None):
        if configuration is None:
            self.paths = _FSCLIENT_DEFAULT_CONFIGURATION
        else:
            self.paths = configuration

        l.info('using offline JoinApiAPI')
        l.info('loading application names dump')
        self.app_names = _load_json(self.paths['applications'])

        application_dir = self.paths['application_objects'].replace('{app_name}.json', '')
        self.available_apps = {_remove_suffix(app_name, '.json') for app_name in os.listdir(application_dir)}

    def get_applications(self):
        return self.app_names

    def get_accounts_for_application(self, app_name):
        template = self.paths['application_objects']
        filename = template.format(app_name=app_name)
        if not os.path.isfile(filename):
            return []
        return [Account(app_name=app_name, name='DummyAccount', id='DummyId')]

    def get_application_objects(self, account):
        if isinstance(account, str):
            # Assume that this the app name
            account = Account(account, None, None)

        app_name = account.app_name
        if app_name not in self.app_names:
            l.error(f"no such application: {app_name}")
            return None

        if app_name not in self.available_apps:
            l.error(f"application is unavailable: {app_name}")
            return None

        template = self.paths['application_objects']
        filename = template.format(app_name=account.app_name)
        json = _load_json(filename)

        ## sanity checking
        ## CP4I application object data is simply a list of the form: [ /* list of objects */ ]
        ## IPAAS application object data is of the form: { "objects": [ /* list of objects */ ] }
        if type(json) is not list and \
           ('errors' in json or 'objects' not in json):
            l.error(f'got error document for {app_name}')
            l.error(json)
            return None

        return json

    def get_object_schema(self, account, object_name):
        if isinstance(account, str):
            # Assume that this the app name
            account = Account(account, None, None)

        app_name = account.app_name
        app_objects = self.get_application_objects(account)
        if app_objects == None: return None

        ## TODO: check whether object_schema is in app_objects to
        ## correctly report as non-existing rather than unavailable
        
        template = self.paths['object_schemas']
        filename = template.format(app_name=account.app_name,
                                   object_name=object_name)
        try:
            json = _load_json(filename)
        except FileNotFoundError as e:
            if _LOG_UNAVAILABLE_OBJECT:
                l.error(f"object schema is unavailable: {account.app_name}/{object_name}, {e}")
            return None

        # sanity checking
        if 'errors' in json and \
           not 'interactions' in json and \
           not 'properties' in json:
            l.error(f'got error document for {account.app_name}/{object_name}')
            l.error(json)
            return None

        return json

    def get_available_business_objects(self, account):
        if isinstance(account, str):
            account = Account(account, None, None)

        app_name = account.app_name
        app_objects = self.get_application_objects(account)
        if app_objects == None: return None

        template = self.paths['object_schemas']
        dirname = _remove_suffix(template.format(app_name=app_name,
                                                 object_name=''), '.json')
        filenames = os.listdir(dirname)
        ## TODO: build union of objects listed in directory and app_objects
        return [_remove_suffix(fn, '.json') for fn in filenames]

    # def get_bos_of_apps(self):
    #     def get_bos_of_an_app(app_name):
    #         # objects = join_api.get_application_objects(app_name).get('objects', {})
    #         bo_names = self.get_available_business_objects(app_name)
    #         return [BO(app_name, bo_name, self.get_object_schema(app_name, bo_name))
    #                 for bo_name in bo_names]

    #     app_names = list(self.available_apps)
    #     app_names.remove('msdynamicscrm-mock')
    #     app_names.remove('sugarcrm')
    #     return {app_name: get_bos_of_an_app(app_name) for app_name in app_names}


################################################################

def create_client(spec, spec_dir=None, retries=3):
    if spec['connector'] == 'join_api_ipaas':
        return IPAASClient(spec, spec_dir, retries=retries)
    if spec['connector'] == 'join_api_cp4i':
        return CP4IClient(spec, spec_dir, retries=retries)
    if spec['connector'] == 'join_api_fs':
        return FSClient(spec, spec_dir)
    l.error(f'unknown connector type: {spec["connector"]}')
    return None
