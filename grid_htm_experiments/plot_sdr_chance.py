import math

import numpy as np
import matplotlib.pyplot as plt

def sigma(n, w, b):
    return math.comb(w, b)*math.comb(n-w, w-b)
def false_positive_chance(n, w, threshold):
    sum = 0
    for b in range(threshold, w):
        sum+=sigma(n, w, b)
    return sum/math.comb(n, w)
print(false_positive_chance(100, 32, 16))
w_vals = [16, 32, 64]
for w in w_vals:
    x_vals = []
    y_vals = []
    threshold = math.floor(w*0.3)
    for n in range(max(w_vals)+1, 2000):
        y = false_positive_chance(n, w, threshold)
        x_vals.append(n)
        y_vals.append(y)
    plt.plot(x_vals, y_vals, label=f"w={w}")
plt.yscale("log")
plt.xlim(max(w_vals)+1)
a = [64]
b = np.arange(500, 2001, 500)
plt.xticks(np.concatenate([a,b]), fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("n", fontsize=14)
plt.ylabel("False Positive Chance", fontsize=14)
plt.title("$\\theta$=0.3", fontsize=20)
plt.tight_layout()
plt.legend(fontsize=14)
plt.savefig("sdr_chance.eps")
plt.show()