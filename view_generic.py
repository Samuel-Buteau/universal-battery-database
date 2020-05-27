def conditional_register(ar, name, content):
    if name not in ar.keys():
        ar[name] = content
