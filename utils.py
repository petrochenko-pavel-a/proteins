import pickle
def save(p, d):
    with open(p, "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)
