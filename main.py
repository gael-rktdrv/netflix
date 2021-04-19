import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt


def main():
    X = np.loadtxt("toy_data.txt")
    Ks, seeds, res = np.array([1,2,3,4]), list(range(5)), {}

    # for seed in seeds:
    #     # import pdb; pdb.set_trace()
    #     mixtures = []
    #     for k in K:
    #         mixture, post = common.init(X, k, seed)
    #         mixtures.append(mixture)
    #     res[seed] = mixtures

    # for key, value in res.items():
    #     print(f"Seed {key}:\n{value}")
    # K =  int(input("Type K: "))
    costs = {}

    for K in Ks:
        temp = []
        for seed in seeds:
            mixture, post = common.init(X, K=K, seed=seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            common.plot(X=X, mixture=mixture, post=post, title=f'K=1 | seed={seed} | cost={cost:.3f}')
            plt.show()
            temp.append(round(cost, 3))
        costs[K] = min(temp)
    print(costs)


if __name__=="__main__":
    main()