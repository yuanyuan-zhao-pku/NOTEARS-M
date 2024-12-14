"""  
This algorithm is an extension of the NOTEARS algorithm.  
The original NOTEARS algorithm was developed by Xun Zheng, et al.  
This implementation is authored by Yuanyuan Zhao.  
"""

"""Simulation for different proportion of catgeorical nodes (catratio)"""
from notearsm.NOTEARS_M import notears, is_dag, count_accuracy, count_accuracy_und

import os
import csv
import time
import logging
import numpy as np
import pandas as pd

default_save_path = "."

file_acc_k = os.path.join(default_save_path, "acc_k_NOTEARS-M.csv")


def get_loss_type(k):
    """Define node types: "gauss" for continuous nodes, "logistic" for binary nodes, and "multi-logistic" for multinomial nodes."""
    if k == 0:
        return ["gauss"] * 20
    elif k == 2:
        return ["gauss"] * 12 + ["logistic"] + ["gauss"] * 5 + ["logistic"] + ["gauss"]
    elif k == 4:
        return (
            ["gauss"] * 12
            + ["logistic"]
            + ["gauss"]
            + ["logistic"]
            + ["gauss"] * 3
            + ["logistic"] * 2
        )
    elif k == 10:
        return [
            "gauss",
            "logistic",
            "gauss",
            "logistic",
            "logistic",
            "gauss",
            "gauss",
            "gauss",
            "logistic",
            "gauss",
            "gauss",
            "gauss",
            "logistic",
            "gauss",
            "logistic",
            "gauss",
            "logistic",
            "logistic",
            "logistic",
            "logistic",
        ]
    else:
        raise ValueError("Invalid value for k: {}".format(k))


d = 20
with open(file_acc_k, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "n",
            "k",
            "fdr",
            "tpr",
            "fpr",
            "shd",
            "pred_size",
            "F1_score",
            "fdr_und",
            "tpr_und",
            "fpr_und",
            "shd_und",
            "pred_size",
            "F1_score_und",
        ]
    )
    for n in [1000]:
        for k in [0, 2, 4, 10]:
            start_time = time.time()
            ACC = np.zeros([10, 6])
            ACC_und = np.zeros([10, 6])
            filenameB = os.path.join(default_save_path, f"B_true_n{n}_k{k}.csv")
            B_true = pd.read_csv(filenameB, header=None)
            B_true = B_true.values
            for i in range(10):
                filenameWE = os.path.join(
                    default_save_path, f"W_est_n{n}_k{k}_i{i}.csv"
                )
                filenameX = os.path.join(default_save_path, f"X_n{n}_k{k}_i{i}.csv")
                loss_type = get_loss_type(k)
                X = pd.read_csv(filenameX, header=None)
                X = X.values
                W_est = notears(
                    X,
                    lambda1=0.1,
                    loss_type=loss_type,
                    m_vec=[1] * d,
                )
                assert is_dag(W_est)
                np.savetxt(filenameWE, W_est, delimiter=",")
                ACC[i, :] = count_accuracy(B_true, W_est != 0)
                ACC_und[i, :] = count_accuracy_und(B_true, W_est != 0)
                print(ACC[i, :])
                print(ACC_und[i, :])
                logging.info(ACC[i, :])
                logging.info(ACC_und[i, :])
            acc = np.mean(ACC, axis=0)
            acc_und = np.mean(ACC_und, axis=0)
            print(f"n={n},k={k},acc={acc}")
            print(f"n={n},k={k},acc_und={acc_und}")
            logging.info(f"n={n},k={k},acc={acc}")
            logging.info(f"n={n},k={k},acc_und={acc_und}")
            end_time = time.time()
            exe_time = end_time - start_time
            print(f"d={d} execution time: {exe_time}")
            logging.info(f"d={d} execution time: {exe_time}")
            writer.writerow(
                [
                    n,
                    k,
                    acc[0],
                    acc[1],
                    acc[2],
                    acc[3],
                    acc[4],
                    acc[5],
                    acc_und[0],
                    acc_und[1],
                    acc_und[2],
                    acc_und[3],
                    acc_und[4],
                    acc_und[5],
                ]
            )
