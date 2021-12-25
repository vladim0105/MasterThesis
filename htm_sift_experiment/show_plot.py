import argparse
import pickle
import matplotlib.pyplot as plt



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("file", type=str)

    args =argparser.parse_args()

    fig = pickle.load(open(args.file,"rb"))
    fig.show()
    # for ax in fig.axes:
    #     for line in ax.lines:
    #         if line.get_label() == "diff":
    #             line.set_visible(False)
    #             fig.canvas.draw()
    plt.show()