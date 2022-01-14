import argparse
import pickle
import matplotlib.pyplot as plt

import utils

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=str)

    args =argparser.parse_args()

    data = pickle.load(open(args.file,"rb"))
    anoms = data["anom_scores"]

    #plt.plot(anoms)
    plt.plot(utils.running_mean(anoms, 100), label="Anomaly Score")

    if "anom_markers" in data:
        anom_markers = data["anom_markers"]
        for anom_marker in anom_markers:
            print("aaaa")
            plt.axvline(anom_marker, c="red")
    plt.legend()
    # for ax in fig.axes:
    #     for line in ax.lines:
    #         if line.get_label() == "diff":
    #             line.set_visible(False)
    #             fig.canvas.draw()
    plt.show()