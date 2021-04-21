import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt


def kmeans(X, Ks, seeds):
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


def naive_em(X, Ks, seeds):
    costs = {}
    
    for K in Ks:
        temp = []
        for seed in seeds:
            mixture, post = common.init(X, K=K, seed=seed)
            mixture, post, _, cost = naive_em.run(X, mixture, post)
            common.plot(X=X, mixture=mixture, post=post, title=f'K=1 | seed={seed} | cost={cost:.3f}')
            plt.show()
            temp.append(round(cost, 3))
        costs[K] = min(temp)
    print(costs)


def main():
    X = np.loadtxt("toy_data.txt")
    Ks, seeds = np.array([1,2,3,4]), list(range(5))
    naive_em(X, Ks, seeds):


if __name__=="__main__":
    main()