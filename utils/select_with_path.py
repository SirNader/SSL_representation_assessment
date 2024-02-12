def select_with_path(obj, path):
    if path is not None:
        for p in path.split("."):
            obj = getattr(obj, p)
    return obj
