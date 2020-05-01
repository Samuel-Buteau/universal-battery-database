import pickle


class Pickle:

    # TODO (harvey):
    #   Make a protocol to takes some info (like a path_to_plot_data)
    #   that includes two methods.
    #   One of the methods takes in a big mapping of barcodes to computed
    #   results or just data and it dumps it to the right files.
    #   The other method looks into the path and loads all the data in the same
    #   format that it was put in.
    #   For instance, if each barcode produces a file and different iteration
    #   ids have different files, then it loads the data into a dictionary with
    #   a key being the iteration id and the other key being the barcode.

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
