def main():
    import numpy as np
    import pandas as pd
    from scipy.special import expit as sigmoid
    import igraph as ig
    import random
    import logging

    # 设置日志配置
    logging.basicConfig(
        filename="simu-catnum.log",
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    def is_dag(W):
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def simulate_dag(d, s0, graph_type):
        """Simulate random DAG with some expected number of edges.

        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF, BP

        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """

        def _random_permutation(M):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(
                np.eye(M.shape[0])
            )  # np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(
                _random_permutation(B_und), k=-1
            )  # `np.tril()`函数中的参数`k`用于指定要提取的下三角部分相对于对角线的偏移量。当`k=0`时，将包括对角线上的元素；当`k>0`时，将提取对角线以上的部分；当`k<0`时，将提取对角线以下的部分。

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)  # 获取邻接矩阵的数据

        if graph_type == "ER":
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == "SF":
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
            B = _graph_to_adjmat(G)
        elif graph_type == "BP":
            # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
            top = int(0.2 * d)
            G = ig.Graph.Random_Bipartite(
                top, d - top, m=s0, directed=True, neimode=ig.OUT
            )
            B = _graph_to_adjmat(G)
        else:
            raise ValueError("unknown graph type")
        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm

    def simulate_parameter(B, alpha=1):  # 改了
        """Simulate SEM parameters for a DAG.

        Args:
            B (np.ndarray): [d, d] binary adj matrix of DAG
            w_ranges (tuple): disjoint weight ranges

        Returns:
            W (np.ndarray): [d, d] weighted adj matrix of DAG
        """
        w_ranges = ((-2.0 * alpha, -0.5 * alpha), (0.5 * alpha, 2.0 * alpha))
        W = np.zeros(B.shape)
        S = np.random.randint(
            len(w_ranges), size=B.shape
        )  # which range # 生成一个形状为B.shape的随机整数数组，范围在[0, len(w_ranges))之间
        for i, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            W += B * (S == i) * U
        return W

    def simulate_linear_sem(W, n, Sem_Type, noise_scale=None):
        """Simulate samples from linear SEM with specified type of noise.

        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

        Args:
            W (np.ndarray): [d, d] weighted adj matrix of DAG
            n (int): num of samples, n=inf mimics population risk
            sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
            noise_scale (np.ndarray): scale parameter of additive noise, default all ones

        Returns:
            X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
        """

        def _simulate_single_equation(X, w, sem_type, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            if sem_type == "gauss":
                z = np.random.normal(scale=scale, size=n)  # scale标准差
                x = X @ w + z
            elif sem_type == "exp":
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "gumbel":
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "uniform":
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = X @ w + z
            elif sem_type == "logistic":
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            elif sem_type == "poisson":
                x = np.random.poisson(np.exp(X @ w)) * 1.0
            else:
                raise ValueError("unknown sem type")
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError("noise scale must be a scalar or has length d")
            scale_vec = noise_scale
        if not is_dag(W):
            raise ValueError("W must be a DAG")
        if np.isinf(n):  # population risk for linear gauss SEM
            if sem_type == "gauss":
                # make 1/d X'X = true cov
                X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
                return X
            else:
                raise ValueError("population risk not available")

        if Sem_Type is None:
            type_vec = ["gauss"] * d
        elif len(Sem_Type) == 1:
            type_vec = Sem_Type * d
        else:
            if len(Sem_Type) != d:
                raise ValueError("Sem_Type must be a scalar or has length d")
            type_vec = Sem_Type

        G = ig.Graph.Weighted_Adjacency(W.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            X[:, j] = _simulate_single_equation(
                X[:, parents], W[parents, j], type_vec[j], scale_vec[j]
            )
        return X

    default_save_path = "D:\\桌面\\研二\\DAG既往汇总\\DAG simulation new\\simu-catnum"
    import os

    # graph_type = "ER"
    d = 20
    s0 = d
    # k=1 固定二分类变量个数为1, d=20
    for n in [1000]:
        for distribution in ["gauss", "exp", "gumbel"]:
            # for graph_type in ["SF"]:
            # for graph_type in ["ER", "SF"]:
            filenameB = os.path.join(
                default_save_path, f"B_true_n{n}_{distribution}.csv"
            )
            filenameW = os.path.join(
                default_save_path, f"W_true_n{n}_{distribution}.csv"
            )
            # B_true = simulate_dag(d, s0, graph_type)
            # np.savetxt(filenameB, B_true, delimiter=",")
            B_true = pd.read_csv(filenameB, header=None)  # 用原先的
            B_true = B_true.values
            # W_true = simulate_parameter(B_true)
            # np.savetxt(filenameW, W_true, delimiter=",")
            W_true = pd.read_csv(filenameW, header=None)  # 用原先的
            W_true = W_true.values
            random.seed(42)
            print(f"distribution={distribution}")
            logging.info(f"distribution={distribution}")
            sem_type = ["logistic"] + [distribution] * (d - 1)
            random.shuffle(sem_type)
            print(f"sem_type:{sem_type}")
            logging.info(f"sem_type:{sem_type}")

            for i in range(10):  # 10 replications
                filenameX = os.path.join(
                    default_save_path, f"X_n{n}_{distribution}_i{i}.csv"
                )
                X = simulate_linear_sem(W_true, n, sem_type)
                np.savetxt(filenameX, X, delimiter=",")

    # # k=0,2,4,10; d=20 （对应分类变量的比0,10%，20%，50%）
    # d = 20
    # s0 = d
    # for n in [1000]:
    #     for k in [0, 2, 4, 10]:
    #         filenameB = os.path.join(default_save_path, f"B_true_n{n}_k{k}.csv")
    #         filenameW = os.path.join(default_save_path, f"W_true_n{n}_k{k}.csv")
    #         # B_true = simulate_dag(d, s0, graph_type)
    #         # np.savetxt(filenameB, B_true, delimiter=",")
    #         B_true = pd.read_csv(filenameB, header=None)  # 用原先的
    #         B_true = B_true.values
    #         # W_true = simulate_parameter(B_true)
    #         # np.savetxt(filenameW, W_true, delimiter=",")
    #         W_true = pd.read_csv(filenameW, header=None)  # 用原先的
    #         W_true = W_true.values
    #         random.seed(42)
    #         sem_type = ["logistic"] * k + ["gauss"] * (d - k)
    #         random.shuffle(sem_type)
    #         print(f"k={k}")
    #         logging.info(f"k={k}")
    #         print(f"sem_type:{sem_type}")
    #         logging.info(f"sem_type:{sem_type}")
    #         for i in range(10):  # 10 replications
    #             filenameX = os.path.join(default_save_path, f"X_n{n}_k{k}_i{i}.csv")
    #             X = simulate_linear_sem(W_true, n, sem_type)
    #             np.savetxt(filenameX, X, delimiter=",")


main()
