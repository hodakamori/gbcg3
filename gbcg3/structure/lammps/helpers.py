def append_to_dict(dic, key, val):
    if key in dic:
        dic[key].append(val)
    else:
        dic[key] = [val]
