import pickle


class Pickle:

    @staticmethod
    def load(filename: str) -> dict:
        f = open(filename, "rb")
        data = pickle.load(f)
        f.close()
        return data

    @staticmethod
    def dump(filename: str, data: dict) -> None:
        f = open(filename, "wb+")
        pickle.dump(data, f)
        f.close()
