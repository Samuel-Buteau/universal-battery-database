# TODO(harvey): Need a better protocol for loading
import pickle


class Pickle:

    @staticmethod
    def load(filename: str, object_count: int) -> tuple:
        f = open(filename, "rb")
        objects = []
        for _ in range(object_count):
            objects.append(pickle.load(f))
        f.close()
        return tuple(objects)

    @staticmethod
    def dump(filename: str, objects: tuple) -> None:
        f = open(filename, "wb+")
        for o in objects:
            pickle.dump(o, f)
        f.close()
