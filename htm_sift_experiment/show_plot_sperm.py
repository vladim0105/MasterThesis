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
    offset = 200
    moving_avg = 100
    anoms = data["anom_scores"][offset:]
    l1_scores = data["l1_scores"][offset:]



    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    anom_score_plot, = ax.plot(utils.trailing_average(anoms, moving_avg), color='blue', label="Anomaly Score", alpha=0.75)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylabel("Anomaly Score")
    ax.set_xlabel("Frames")
    # Generate a new Axes instance, on the twin-X axes (same position)
    ax2 = ax.twinx()

    # Plot exponential sequence, set scale to logarithmic and change tick color
    l1_plot, = ax2.plot(utils.trailing_average(l1_scores, moving_avg), color='green', label="L1 Error", alpha=0.75)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylabel("L1 Error")

    anom_marker_plots = None
    if "anom_markers" in data:
        anom_marker_plots = []
        anom_markers = data["anom_markers"]
        for anom_marker in anom_markers:
            plot = plt.axvline(anom_marker-offset, c="red", alpha=0.3)
            anom_marker_plots.append(plot)

    plt.legend([anom_score_plot, l1_plot, tuple(anom_marker_plots) if "anom_markers" in data else None], ["Grid HTM", "L1 Error", "Segments"])
    plt.show()