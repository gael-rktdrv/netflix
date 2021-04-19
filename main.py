import numpy as np
import kmeans
import common
import naive_em
import em


def main():
    X = np.loadtxt("toy_data.txt")
    K, seeds, res = np.array([1,2,3,4]), list(range(5)), {}

    for seed in seeds:
        # import pdb; pdb.set_trace()
        mixtures = []
        for k in K:
            mixture, post = common.init(X, k, seed)
            mixtures.append(mixture)
        res[seed] = mixtures

    for key, value in res.items():
        print(f"Seed {key}:\n{value}")


if __name__=="__main__":
    main()