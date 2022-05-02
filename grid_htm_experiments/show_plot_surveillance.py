import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import model
import utils
plt.rcParams["figure.figsize"] = (10, 4.8)

def test(anoms, axis=None):
    return np.sum(anoms, axis=axis)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=str)

    args =argparser.parse_args()

    data = pickle.load(open(args.file,"rb"))
    offset = 200
    moving_avg = 100

    anoms = np.array(data["raw_anoms"])
    anoms = np.mean(anoms, axis=(1,2))[offset:]
    #anoms[anoms < 0.025] = 0
    #anoms = model.grid_mean_aggr_func(anoms, (1,2))[offset:]
    #anoms = test(anoms, (1,2))[offset:]*model.grid_mean_aggr_func(anoms, (1,2))[offset:]
    x_vals = np.array(data["x_vals"])[offset-moving_avg+1:]
    frame_freeze = data["frame_freeze"]-offset
    #anoms = data["anom_scores"][offset:]

    fig, ax = plt.subplots()

    # Plot linear sequence, and set tick labels to the same color
    anom_score_plot, = ax.plot(x_vals, utils.trailing_average(anoms, moving_avg), color='blue', label="Anomaly Score", alpha=0.75)
    ax.set_ylabel("Anomaly Score")
    ax.set_xlabel("Frames")
    print(f"Frame Freeze IDX: "+str(int(frame_freeze)))
    frame_freeze_plot = plt.axvline(frame_freeze, c="green", alpha=0.3)


    anom_marker_plots = []
    anom_markers = data["anom_markers"]
    for anom_marker in anom_markers:
        plot = plt.axvline(anom_marker-offset, c="red", alpha=0.3)
        anom_marker_plots.append(plot)

    plt.legend([anom_score_plot, tuple(anom_marker_plots), frame_freeze_plot], ["Grid HTM", "Segments", "Frame Freeze"])
    plt.savefig("surveillance_results/figure.eps")
    plt.show()