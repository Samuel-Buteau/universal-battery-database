import matplotlib.pyplot as plt

from plot import savefig
from Key import *


class LossRecord:
    def __init__(self):
        self.data = []
        self.labels = [
            Key.Loss.Q_CC, Key.Loss.Q_CV, Key.Loss.V_CC, Key.Loss.V_CV,
            Key.Loss.Q, Key.Loss.SCALE, Key.Loss.R, Key.Loss.SHIFT,
            Key.Loss.CELL, Key.Loss.RECIP, Key.Loss.PROJ, Key.Loss.OOB
        ]

    def record(self, count, losses):
        self.data.append((count, losses))

    def print_recent(self, fit_args):
        if len(self.data) > 0:
            count, losses = self.data[-1]
            print("Count {}:".format(count))
            for i in range(len(losses)):
                print("\t{}:{}. coeff:{}".format(
                    self.labels[i],
                    losses[i],
                    fit_args["coeff_" + self.labels[i].split("_loss")[0]]
                ))

    def plot(self, count, fit_args):
        fig = plt.figure(figsize = [11, 10])
        ax = fig.add_subplot(111)
        ax.set_yscale("log")
        for i in range(len(self.labels)):
            ax.plot(
                [s[0] for s in self.data],
                [s[1][i] for s in self.data],
                label = self.labels[i]
            )

        ax.legend()

        savefig("losses_Count_{}.png".format(count), fit_args)
        plt.close(fig)
