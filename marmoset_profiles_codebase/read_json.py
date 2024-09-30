import json


def read_json(fname):
    f = open(fname)
    data = json.loads(f.read())
    f.close()
    return data
