import csv
import logging

l = logging.getLogger('ikg')

F_PATCH=0x01

TAG_APP='app'
TAG_OBJ='bo'
TAG_INT='op'

OP_TYPES = {
    'actions': { n for n in [ 'CREATE', 'RETRIEVEALL', 'UPDATE', 'UPDATEALL',
                              'DELETEALL', 'UPSERTWITHWHERE' ] },
    'triggers': { n for n in [ 'CREATED', 'CREATEDORUPDATED', 'UPDATED',
                               'DELETED', 'CREATED_POLLER', 'UPDATED_POLLER',
                               'CREATEDORUPDATED_POLLER' ] }
}

class ProductDB:

    def __init__(self, products_file='data/deployment_targets.csv', app_filter=None):
        targets = {}
        with open(products_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for idx, row in enumerate(csvreader):
                if row[0] == '?':
                    print(f'applications.csv:{idx}: ignoring "{row[0]},{row[1]}"')
                    continue
                if app_filter and row[0] not in app_filter:
                    continue
                row[2] = row[2].split(';')
                targets[row[0]] = row
        self.targets = targets

    @property
    def applications(self):
        return self.targets.keys()

    def products(self, application):
        res = self.targets.get(application.lower())
        if res is None:
            print(f"no product data for '{application}'")
            return None
        return res[2]

def read_deployment_target(deployments_file, deployment_target, app_filter=None):
    # TODO: add stack trace to this warning
    l.warning(f'read_deployment_target is deprecated')
    result = []
    deployment_target = deployment_target.lower()
    with open(deployments_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for idx, row in enumerate(spamreader):
            if row[0] == '?':
                print(f'applications.csv:{idx}: ignoring "{row[0]},{row[1]}"')
                continue
            if app_filter and row[0] not in app_filter:
                continue
            targets = row[2].split(';')
            if deployment_target in targets:
                result.append(row[0])
    return result
