import argparse
import pickle
import matplotlib.pyplot as plt

import utils

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=str)

    args =argparser.parse_args()

    anoms = pickle.load(open(args.file,"rb"))
    plt.plot(anoms)
    plt.plot(utils.running_mean(anoms, 100))
    # for ax in fig.axes:
    #     for line in ax.lines:
    #         if line.get_label() == "diff":
    #             line.set_visible(False)
    #             fig.canvas.draw()
    plt.show()