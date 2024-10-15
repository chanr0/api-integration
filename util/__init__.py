COL_BOLD = '\033[1m'
COL_UNDERLINE = '\033[4m'

COL_PURPLE = '\033[95m'
COL_CYAN = '\033[96m'
COL_DARKCYAN = '\033[36m'
COL_BLUE = '\033[94m'
COL_GREEN = '\033[92m'
COL_YELLOW = '\033[93m'
COL_RED = '\033[91m'

COL_END = '\033[0m'


class Timer:
    import time
    def __init__(self):
        self.start = self.time.time()
    def reset(self):
        self.start = self.time.time()
    def print_lap(self, fstrg='{}'):
        end = self.time.time()
        print(fstrg.format(f"{end-self.start}"))
    def __call__(self):
        return self.time.time()-self.start


def get_path(adict, path, default=None):
    cdict = adict
    for idx, elem in enumerate(path):
        if type(cdict) is not dict: return default
        cdict = cdict.get(elem)
    return cdict
