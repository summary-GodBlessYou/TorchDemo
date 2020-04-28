

def to_string(title, **kwargs):
    for k in kwargs:
        print(k + ":", kwargs[k])
    print("-" * 20, title, "---end" "\n")


