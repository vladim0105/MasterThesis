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
    #l1_scores = data["l1_scores"][100:]

    # #plt.plot(anoms)
    # #plt.ylim(0, 1.1)
    # anom_score_plot, = plt.plot(utils.trailing_average(anoms, 50), label="Anomaly Score")
    # _, = plt.plot(utils.trailing_average(l1_scores, 50), label="L1 Scores")
    #
    # anom_marker_plots = []
    # if "anom_markers" in data:
    #     anom_markers = data["anom_markers"]
    #     for anom_marker in anom_markers:
    #         plot = plt.axvline(anom_marker, c="red", alpha=0.5)
    #         anom_marker_plots.append(plot)
    # #n_predictions = plt.bar(np.arange(len(data["predicted_things"])), data["predicted_things"], alpha=0.1)
    # #plt.legend([anom_score_plot, tuple(anom_marker_plots) if "anom_markers" in data else None], ["HTM Anomaly Score", "Anomalies"])
    # # for ax in fig.axes:
    # #     for line in ax.lines:
    # #         if line.get_label() == "diff":
    # #             line.set_visible(False)
    # #             fig.canvas.draw()
    # plt.xlabel("Frames")

    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    anom_score_plot, = ax.plot(utils.trailing_average(anoms, 50), color='blue', label="Anomaly Score", alpha=0.75)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylabel("Anomaly Score")
    ax.set_xlabel("Frames")
    # Generate a new Axes instance, on the twin-X axes (same position)
    #ax2 = ax.twinx()

    # Plot exponential sequence, set scale to logarithmic and change tick color
    #ax2.plot(utils.trailing_average(l1_scores, 100), color='green', label="L1 Error", alpha=0.75)
    #ax2.tick_params(axis='y', labelcolor='green')
    #ax2.set_ylabel("L1 Error")
    # anom_marker_plots = None
    # if "anom_markers" in data:
    #     anom_marker_plots = []
    #     anom_markers = data["anom_markers"]
    #     for anom_marker in anom_markers:
    #         plot = plt.axvline(anom_marker, c="red", alpha=0.3)
    #         anom_marker_plots.append(plot)
    #plt.legend([anom_score_plot, tuple(anom_marker_plots) if "anom_markers" in data else None], ["HTM Anomaly Score", "Anomalies"])
    plt.show()