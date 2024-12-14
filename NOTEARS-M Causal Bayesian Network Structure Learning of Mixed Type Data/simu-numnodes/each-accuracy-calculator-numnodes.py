"""  
This algorithm is an extension of the NOTEARS algorithm.  
The original NOTEARS algorithm was developed by Xun Zheng, et al.  
This implementation is authored by Yuanyuan Zhao.  
"""

"""Calculate accuracy metrics for each replication."""
import logging
import os
import time
import numpy as np
import pandas as pd
import csv
from notearsm.NOTEARS_M import count_accuracy, count_accuracy_und

default_save_path = "."

file_acc_d_each = os.path.join(default_save_path, "each_acc_d_NOTEARS-M.csv")

with open(file_acc_d_each, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "n",
            "d",
            "i",
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
        for d in [5, 10, 20, 40]:
            start_time = time.time()
            ACC = np.zeros([10, 6])
            ACC_und = np.zeros([10, 6])
            filenameB = os.path.join(default_save_path, f"B_true_n{n}_d{d}.csv")
            B_true = pd.read_csv(filenameB, header=None)
            B_true = B_true.values
            for i in range(10):
                filenameWE = os.path.join(
                    default_save_path, f"W_est_n{n}_d{d}_i{i}.csv"
                )
                filenameX = os.path.join(default_save_path, f"X_n{n}_d{d}_i{i}.csv")
                X = pd.read_csv(filenameX, header=None)
                X = X.values
                W_est = pd.read_csv(filenameWE, header=None)
                W_est = W_est.values
                ACC[i, :] = count_accuracy(B_true, W_est != 0)
                ACC_und[i, :] = count_accuracy_und(B_true, W_est != 0)
                print(ACC[i, :])
                print(ACC_und[i, :])
                logging.info(ACC[i, :])
                logging.info(ACC_und[i, :])
                writer.writerow(
                    [
                        n,
                        d,
                        i,
                        ACC[i, 0],
                        ACC[i, 1],
                        ACC[i, 2],
                        ACC[i, 3],
                        ACC[i, 4],
                        ACC[i, 5],
                        ACC_und[i, 0],
                        ACC_und[i, 1],
                        ACC_und[i, 2],
                        ACC_und[i, 3],
                        ACC_und[i, 4],
                        ACC_und[i, 5],
                    ]
                )
            end_time = time.time()
            exe_time = end_time - start_time
            print(f"d={d} execution time: {exe_time}")
            logging.info(f"d={d} execution time: {exe_time}")
