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

    #plt.plot(anoms)
    #plt.ylim(0, 1.1)
    anom_score_plot, = plt.plot(utils.trailing_average(anoms, 50)[100:], label="Anomaly Score")

    anom_marker_plots = []
    if "anom_markers" in data:
        anom_markers = data["anom_markers"]
        for anom_marker in anom_markers:
            plot = plt.axvline(anom_marker, c="red", alpha=0.5)
            anom_marker_plots.append(plot)
    #n_predictions = plt.bar(np.arange(len(data["predicted_things"])), data["predicted_things"], alpha=0.1)
    plt.legend([anom_score_plot, tuple(anom_marker_plots) if "anom_markers" in data else None], ["HTM Anomaly Score", "Anomalies"])
    # for ax in fig.axes:
    #     for line in ax.lines:
    #         if line.get_label() == "diff":
    #             line.set_visible(False)
    #             fig.canvas.draw()
    plt.xlabel("Frames")
    plt.show()