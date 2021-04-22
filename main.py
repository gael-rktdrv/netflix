import numpy as np
import kmeans
import common
import naive_em
import em
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots


def km(X, Ks, seeds, decim):
    costs = {}
    
    for K in Ks:
        temp = []
        for seed in seeds:
            mixture, post = common.init(X, K=K, seed=seed)
            mixture, post, cost = kmeans.run(X, mixture, post)
            title = f'KM: K={K} | seed={seed} | cost={cost:.3f}'
            name = f'K{K}-seed{seed}'
            common.plot(X=X, mixture=mixture, post=post, title=title)
            plt.savefig('/home/Gael/Documents/Projects/MITx/netflix/km/' + name + '.png')
            # time.sleep(2)
            # plt.close()
            # plt.show()
            temp.append(round(cost, decim))
        costs[K] = min(temp)
    print(costs)


def n_em(X, Ks, seeds, decim):
    LL = {}
    
    for K in Ks:
        temp = []
        for seed in seeds:
            mixture, post = common.init(X, K=K, seed=seed)
            mixture, post, _, new_ll = naive_em.run(X, mixture)
            title = f'EM: K={K} | seed={seed} | likelihood={new_ll:.3f}'
            common.plot(X=X, mixture=mixture, post=post, title=title)
            name = f'K{K}-seed{seed}'
            plt.savefig('/home/Gael/Documents/Projects/MITx/netflix/em/' + name + '.png')
            # time.sleep(2)
            # plt.close()
            temp.append(round(new_ll, decim))
        LL[K] = max(temp)
    print(f"Likelihoods: \n{LL}")


def main():
    X = np.loadtxt("toy_data.txt")
    Ks, seeds = np.array([1,2,3,4]), list(range(5))
    decim = 6  # Decimals points after the comma
    args = (X, Ks, seeds, decim)
    # print('Choose method: \n1-Kmeans\n2-EM algorithm')
    choice = int(input("Choose method: \n1-Kmeans\n2-EM algorithm\nBoth\nChoice: "))
    if choice == 1:
        km(*args)  # Running Kmeans
    elif choice == 2:
        n_em(*args) # Running naive_em


if __name__=="__main__":
    main()


# def n_em(X, Ks, seeds):
#     LL = {}
    
#     for K in Ks:
#         # fig = make_subplots(rows=2, cols=2)
#         # a, b = 1, 1
#         temp = []
#         for seed in seeds:
#             mixture, post = common.init(X, K=K, seed=seed)
#             mixture, post, ll, _ = naive_em.run(X, mixture)
#             title = f'K={K} | seed={seed} | likelihood={ll:.3f}'
#             # Plotting all 4 seeds
#             # fig.add_trace(
#             #     common.plot(X=X, mixture=mixture, post=post, title=title), 
#             #     row=a, col=b
#             #     )
#             common.plot(X=X, mixture=mixture, post=post, title=title)
#             name = f'K{K}-seed{seed}'
#             plt.savefig('/home/Gael/Documents/Projects/MITx/netflix/em/' + name + '.png')
#             time.sleep(2)
#             plt.close()
#             # plt.show()
#             temp.append(round(ll, 3))
#             # Managing rows and columns
#             # if a < 2:
#             #     a += 1
#             # else:
#             #     b += 1
#         # fig.update_layout(height=600, width=800, title_text=f"K={K}")
#         # fig.show()
#         LL[K] = max(temp)
#     print(f"Likelihoods: \n{LL}")
