#!/usr/bin/env python3

import json
import logging

class JSONFormatter(logging.Formatter):

    def __init__(self, fields):
        self.fields = fields
        logging.Formatter.__init__(self)

    def serialize_log_record(self, log_record):
        return json.dumps(log_record,
                          indent=None)

    # def formatException(self, exc_info):
    #     result = super(JSONFormatter, self).formatException(exc_info)
    #     return repr(result)

    def format(self, record):
        # does the record have an exception associated with it; still
        # need to figure out how to process this; currently, we just
        # dump it in whatever format
        if record.exc_info:
            print(f"r.exc_info={self.formatException(record.exc_info)}")
        if record.exc_text:
            print(f"r.exc_text={record.exc_text}")
        if record.stack_info:
            print(f"r.stack_info={self.formatException(record.stack_info)}")

        record.message = record.getMessage()
        record_dict = record.__dict__
        log_record = {}
        for key, get_key_fun in self.fields.items():
            log_record[key] = get_key_fun(record_dict)
        return self.serialize_log_record(log_record)

if __name__ == '__main__':
    # filename='example.log',
    # format='%(asctime)s %(name)s %(levelname)s:%(message)s',
    # logging.basicConfig(encoding='utf-8',
    #                     level=logging.DEBUG)

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
    l.addHandler(logHandler)
    l.setLevel(logging.DEBUG)
    l.info('hi there')
    l.error('nothing to log')
