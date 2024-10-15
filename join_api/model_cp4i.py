import json
import logging                    

from join_api.common import *
import util

l = logging.getLogger('ikg')

class Application:
    ac_type = 'cp4i'

    def __init__(self, id_, data, objects=None, products=None):
        self._id = id_
        self._data = data
        if objects is None:
            self._objects = {}
        else:
            self._objects = objects
        self._products = products

    def check(self, name=None, flags=0):
        l.log(25, f"checking app {self.name}")
        res = []
        if name is not None and name != self.name:
            res.append(f"name ({name}) and name ({self.name}) differ")
        
        for key in ['name', 'displayName', 'description']:
            if key not in self._data:
                res.append(f"missing {key}")

        if res:
            l.warning(f"warnings in {self.full_name}: {', '.join(res)}")
        
        for obj_name, obj in self._objects.items():
            obj.check(name=obj_name, flags=flags)

        return res == []

    def dumps(self, detailed=False):
        res = \
            f"App({self.name}, {len(self.objects)} objs (CP4I), " \
            f"{self.display_name}\n" \
            f"    {self.description}"
        if detailed:
            res += '\n' + json.dumps(self._data, indent=2)
        return res + ')'

    @property
    def id_(self):
        return self._id

    @property
    def data(self):
        return self._data

    @property
    def products(self):
        return self._products

    @property
    def name(self):
        return self._data['name']

    @property
    def typename(self):
        return TAG_APP

    @property
    def full_name(self):
        return self._data['name']

    @property
    def display_name(self):
        return self._data['displayName']

    @property
    def description(self):
        return self._data.get('description')

    @property
    def objects(self):
        """The application objects pertaining to this application."""
        return self._objects

    @objects.setter
    def objects(self, new_objects):
        self._objects = new_objects

    @objects.deleter
    def objects(self):
        self._objects = set()
        ## del self._x

    def get_descriptions(self):
        return set([ util.get_path(self._data, ['name']),
                     util.get_path(self._data, ['displayName']),
                     util.get_path(self._data, ['description']) ])

    def __repr__(self):
        return f'App(CP, {self.name}, {self.objects})'

    def __str__(self):
        return f'App(CP, {self.name}, {self.objects})'


    class Object:
        def __init__(self, id_, application, data):
            self._id = id_
            self._application = application
            self._data = data
            self._optypenames = {}
            self._operations = {}
            self._properties = {}

        def check(self, name=None, flags=0):
            res = []
            d0 = self._data[0]
            d0_n = util.get_path(d0, ['name'])
            if name is not None and name != d0_n:
                res.append(f"name ({name}) and name ({d0_n}) differ")

            d0_dn = util.get_path(d0, ['displayName'])
            d0_d_n = util.get_path(d0, ['display', 'name'])
            if d0_dn != d0_d_n:
                res.append(f"displayName ({d0_dn}) and display/name ({d0_d_n}) differ")

            for key in ['name', 'displayName', 'description', 'display/description', 'display/name']:
                key_split = key.split('/')
                val = util.get_path(d0, key_split)
                if val is None:
                    res.append(f"missing {key}")
                    if flags&F_PATCH and \
                       key == 'displayName' and \
                       d0_d_n is not None:
                        d0['displayName'] = d0_d_n
                        res[-1] = res[-1] + ' (patched)'

            if res:
                l.warning(f"warnings in {self.full_name}: {', '.join(res)}")

            for op_name, op in self._operations.items():
                op.check(name=op_name, flags=flags)

            return res == []

        def dumps(self, detailed=False):
            res = \
                f"Obj({self.full_name} (CP4I), " \
                f"{self.display_name}\n" \
                f"    {self.description}"
            if detailed:
                res += '\n' + json.dumps(self._data, indent=2)
            return res + ')'

        @property
        def id_(self):
            return self._id

        @property
        def data(self):
            return self._data

        @property
        def products(self):
            return self.application.products

        @property
        def typename(self):
            return TAG_OBJ

        @property
        def optypenames(self):
            return self._optypenames

        @optypenames.setter
        def optypenames(self, value):
            self._optypenames = value

        @property
        def name(self):
            return self._data[0]['name']

        @property
        def full_name(self):
            return f"{self.application.name}/{self._data[0]['name']}"

        @property
        def display_name(self):
            return util.get_path(self._data[0], ['displayName'])

        @property
        def description(self):
            return self._data[0].get('description')

        @property
        def application(self):
            return self._application

        @property
        def operations(self):
            return self._operations

        @property
        def interactions(self):
            return self._operations

        def get_descriptions(self):
            return set([ util.get_path(self._data[0], ['name']),
                         util.get_path(self._data[0], ['displayName']),
                         util.get_path(self._data[0], ['description']),
                         util.get_path(self._data[0], ['display', 'name']),
                         util.get_path(self._data[0], ['display', 'description']) ])

        # all the parameters (properties) that are used by any operation
        # of this application object
        def get_properties(self):
            return self.properties

        def __repr__(self):
            return f'BO({self.application.name}/{self.name})'

    class Operation:
        def __init__(self, id_, app_object, op_name, op_data):
            self._id = id_
            self._object = app_object
            self._name = op_name
            self._data = op_data
            self._properties = {}

        def check(self, name=None, flags=0):
            #l.log(25, f"checking op {self.name}")
            res = []
            d0 = self._data[0]
            if name is not None and self._name != name:
                res.append(f"name ({name}) and name ({d0_n}) differ")

            for key in ['display/description', 'display/name']:
                key_split = key.split('/')
                val = util.get_path(d0, key_split)
                if val is None:
                    res.append(f"missing {key}")

            if self._data[1] is not None:
                res.append('unexpected detailed metadata')

            if res:
                l.warning(f"warnings in {self.full_name}: {', '.join(res)}")

            return res == []

        def dumps(self, detailed=False):
            res = \
                f"Op({self.full_name} (CP4I), " \
                f"{self.display_name}\n" \
                f"    {self.description}"
            if detailed:
                res += '\n' + json.dumps(self._data, indent=2)
            return res + ')'

        @property
        def id_(self):
            return self._id

        @property
        def data(self):
            return self._data

        @property
        def products(self):
            return self._object.application.products

        @property
        def name(self):
            return self._name

        @property
        def typename(self):
            return TAG_INT

        @property
        def optypename(self):
            res = util.get_path(self._data[0], ['type'])
            if res is not None: return res
            if self._name in OP_TYPES['actions']: return 'actions'
            if self._name in OP_TYPES['triggers']: return 'triggers'
            return res

        @property
        def full_name(self):
            return f"{self._object.full_name}/{self._name}"

        @property
        def display_name(self):
            return util.get_path(self._data[0], ['display', 'name'])

        @property
        def description(self):
            return util.get_path(self._data[0], ['display', 'description'])

        @property
        def object(self):
            return self._object

        def get_descriptions(self):
            return set([ self._name,
                         util.get_path(self._data[0], ['display', 'name']),
                         util.get_path(self._data[0], ['display', 'description']) ])

            
