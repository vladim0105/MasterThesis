import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

import utils

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=str)

    args =argparser.parse_args()

    data = pickle.load(open(args.file,"rb"))
    anoms = data["anom_scores"]

    fig, ax = plt.subplots()

    anom_score_plot, = ax.plot(utils.trailing_average(anoms, 100), label="Anomaly Score", alpha=0.75)
    ax.set_ylabel("Anomaly Score")
    ax.set_xlabel("Frames")

    anom_marker_plots = None
    if "anom_markers" in data:
        anom_marker_plots = []
        anom_markers = data["anom_markers"]
        for anom_marker in anom_markers:
            plot = plt.axvline(anom_marker, c="red", alpha=0.3)
            anom_marker_plots.append(plot)
    plt.legend([anom_score_plot, tuple(anom_marker_plots) if "anom_markers" in data else None], ["HTM Anomaly Score", "Anomalies"])
    #plt.legend()
    plt.savefig("figure.eps")
    plt.show()