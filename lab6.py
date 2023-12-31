import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def huber(x: np.array, k: float):
    phi = lambda elem: elem if abs(elem) < k else k * np.sign(elem)

    return np.mean([phi(elem) for elem in x])

def sigm3_rull(x: np.array):

    xmean = np.mean(x)
    xvar = np.std(x)
    is_in_3sigm = lambda xi: np.abs(xi - xmean) < 3*xvar

    clear_res = [val for val in x if is_in_3sigm(val)]

    return clear_res

def boxplot_rull(x: np.array):
    outlyings = plt.boxplot(x)
    clear_res = [i for i in x if not i in outlyings["fliers"]]
    # x = list(x)
    # for xx in outlyings["fliers"]:
    #     x
    return clear_res

def double_stage_mean(x: np.array):
    # x = sigm3_rull(x)
    x = boxplot_rull(x)
    # x = huber(x, 1.44)
    return np.mean(x)

distributions = {
    "Нормальное": stats.norm.rvs,
    "Коши": stats.cauchy.rvs,
    "Смесь": lambda size: 0.9 * stats.norm.rvs(size=size) + 0.1 * stats.cauchy.rvs(size=size) 
}

measures = {
    "Среднее": np.mean,
    "Медиана": np.median,
    "Хубер": lambda x: huber(x, 1.44),
    "Двухэтапная": double_stage_mean
}

def monte_karlo(N: int, sample_size: int, dist_grvs, measure):
    means = [measure(dist_grvs(size=sample_size)) for _ in range(N)]

    return np.mean(means), np.var(means)

def main(N : int = 10000, sample_size: int = 100):
    for dname, grvs in distributions.items():
        print(dname)
        for mname, measure in measures.items():
            mu, var = monte_karlo(N, sample_size, grvs, measure)
            print(f"\t{mname}")
            print(f"\t\tСреднее:\t{mu:.6f}")
            print(f"\t\tДисперсия:\t{var:.6f}")
        print("")

if __name__ == "__main__":
    main()