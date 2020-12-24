import hashlib

def hash_args(*args) -> str:
    contents = ""
    for c in args:
        contents = f"{contents}_{str(c)}"
    contents = hashlib.md5(contents.encode('utf-8')).hexdigest()
    return contents
