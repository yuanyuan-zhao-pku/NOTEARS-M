library(bnlearn)
# 定义诊断指标函数----------------------
count_accuracy <- function(B_true, B_est) {  
  # Argument validation  
  if (!all(B_est %in% c(0, 1))) {  
    stop("B_est should take value in {0,1}")  
  }  
  
  d <- nrow(B_true)  
  # Linear index of nonzeros  
  pred <- which(B_est != 0)  
  cond <- which(B_true != 0)  
  cond_reversed <- which(t(B_true) != 0)  
  cond_skeleton <- unique(c(cond, cond_reversed))  
  
  # True positive, false positive, and reverse  
  true_pos <- intersect(pred, cond)  
  false_pos <- setdiff(pred, cond_skeleton)  
  extra <- setdiff(pred, cond)  
  reverse <- intersect(extra, cond_reversed)  
  
  # Compute accuracy metrics  
  pred_size <- length(pred)  
  cond_neg_size <- 0.5 * d * (d - 1) - length(cond)  
  fdr <- (length(reverse) + length(false_pos)) / max(pred_size, 1)  
  tpr <- length(true_pos) / max(length(cond), 1)  
  fpr <- (length(reverse) + length(false_pos)) / max(cond_neg_size, 1)  
  # 获取下三角矩阵  
  get_lower_triangle <- function(mat) {  
    if (!is.matrix(mat)) {  
      stop("Input must be a matrix.")  
    }  
    
    # 创建一个与输入矩阵相同的矩阵，初始化为零  
    lower_triangle <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))  
    
    # 将下三角部分的值复制到新矩阵中  
    lower_triangle[lower.tri(mat, diag = TRUE)] <- mat[lower.tri(mat, diag = TRUE)]  
    
    return(lower_triangle)  
  }
  pred_lower <- which(get_lower_triangle(B_est + t(B_est)) != 0)  
  cond_lower <- which(get_lower_triangle(B_true + t(B_true)) != 0)  
  extra_lower <- setdiff(pred_lower, cond_lower)  
  missing_lower <- setdiff(cond_lower, pred_lower)  
  shd <- length(extra_lower) + length(missing_lower) + length(reverse)  
  precision <- 1 - fdr  
  recall <- tpr  
  f1 <- 2 * precision * recall / max((precision + recall), 1)  
  
  # Output and logging  
  cat(sprintf("fdr: %.3f, tpr: %.3f, fpr: %.3f, shd: %d, nnz: %d, f1: %.3f\n",  
              fdr, tpr, fpr, shd, pred_size, f1))  
  return(c(fdr = fdr, tpr = tpr, fpr = fpr, shd = shd, nnz = pred_size, f1 = f1))  
}
count_accuracy_und <- function(B_true, B_est) {  
  # Argument validation  
  if (!all(B_est %in% c(0, 1))) {  
    stop("B_est should take value in {0,1}")  
  }  
  
  d <- nrow(B_true)  
  # Linear index of nonzeros  
  pred <- which(B_est != 0)  
  cond <- which(B_true != 0)  
  cond_reversed <- which(t(B_true) != 0)  
  cond_skeleton <- unique(c(cond, cond_reversed))  
  
  # True positive, false positive, and reverse  
  true_pos <- intersect(pred, cond)  
  false_pos <- setdiff(pred, cond_skeleton)  
  extra <- setdiff(pred, cond)  
  reverse <- intersect(extra, cond_reversed)  
  
  # Compute accuracy metrics  
  pred_size <- length(pred)  
  cond_neg_size <- 0.5 * d * (d - 1) - length(cond)  
  fdr_und <- (length(false_pos)) / max(pred_size, 1)  
  tpr_und <- (length(reverse) + length(true_pos)) / max(length(cond), 1)  
  fpr_und <- (length(false_pos)) / max(cond_neg_size, 1)  
  # 获取下三角矩阵  
  get_lower_triangle <- function(mat) {  
    if (!is.matrix(mat)) {  
      stop("Input must be a matrix.")  
    }  
    
    # 创建一个与输入矩阵相同的矩阵，初始化为零  
    lower_triangle <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))  
    
    # 将下三角部分的值复制到新矩阵中  
    lower_triangle[lower.tri(mat, diag = TRUE)] <- mat[lower.tri(mat, diag = TRUE)]  
    
    return(lower_triangle)  
  }
  pred_lower <- which(get_lower_triangle(B_est + t(B_est)) != 0)  
  cond_lower <- which(get_lower_triangle(B_true + t(B_true)) != 0)  
  extra_lower <- setdiff(pred_lower, cond_lower)  
  missing_lower <- setdiff(cond_lower, pred_lower)  
  shd_und <- length(extra_lower) + length(missing_lower)
  precision_und <- 1 - fdr_und  
  recall_und <- tpr_und  
  f1_und <- 2 * precision_und * recall_und / max((precision_und + recall_und), 1)  
  
  # Output and logging  
  cat(sprintf("fdr_und: %.3f, tpr_und: %.3f, fpr_und: %.3f, shd_und: %d, nnz: %d, f1_und: %.3f\n",  
              fdr_und, tpr_und, fpr_und, shd_und, pred_size, f1_und))  
  return(c(fdr_und = fdr_und, tpr_und = tpr_und, fpr_und = fpr_und, 
           shd_und = shd_und, nnz = pred_size, f1_und = f1_und))  
}
## MMHC-------------------------------------
# 定义路径和文件名-------------------------------------------------  
default_save_path <- "D:\\桌面\\研二\\DAG既往汇总\\DAG simulation new\\simu-samplesize\\MMHC"  
# 循环处理----------------------------------------------------  
# 创建一个空的数据框来存储结果  
results <- data.frame(  
  n = numeric(),  
  d = numeric(),
  i = numeric(),
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()
) 
for (n in c(100,500,1000)) {  
  for (d in c(20)) { 
    print(d)
    library(tidyverse)
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_d%d.csv", n, d))  
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(i)
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_d%d_i%d.csv", n, d, i))  
      # filenameX <- file.path(default_save_path, sprintf("X_n%d_d%d_i%d.csv", n, d, i))  
      
      # loss_type <- sem_type_dict[as.character(d)] 
      # X <- read.csv(filenameX, header = FALSE) # %>% as.matrix()  
      W_est <- read.csv(filenameWE,header = T,row.names = 1) %>% as.matrix()
      
      # write.csv(W_est, filenameWE)  
      ACC[i+1, ] <- count_accuracy(B_true, W_est)  # 
      ACC_und[i+1, ] <- count_accuracy_und(B_true, W_est)  # 
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         d = d,
                         i = i,
                         fdr = ACC[i+1, ][1], 
                         tpr = ACC[i+1, ][2], 
                         fpr = ACC[i+1, ][3], 
                         shd = ACC[i+1, ][4], 
                         pred_size = ACC[i+1, ][5], 
                         F1_score = ACC[i+1, ][6], 
                         fdr_und = ACC_und[i+1, ][1], 
                         tpr_und = ACC_und[i+1, ][2], 
                         fpr_und = ACC_und[i+1, ][3], 
                         shd_und = ACC_und[i+1, ][4], 
                         pred_size = ACC_und[i+1, ][5], 
                         F1_score_und = ACC_und[i+1, ][6]
                       ) 
      ) 
      
      # print(ACC[i+1, ])  
      cat(sprintf("n=%d,d=%d,acc=%s\n", n, d, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,d=%d,acc_und=%s\n", n, d, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    # acc <- colMeans(ACC)  
    # acc_und <- colMeans(ACC_und)  
    # cat(sprintf("n=%d,d=%d,acc=%s\n", n, d, paste(acc, collapse = ",")))  
    # cat(sprintf("n=%d,d=%d,acc_und=%s\n", n, d, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("d=%d execution time: %.2f seconds\n", d, exe_time))  
    
 
  }  
}
# 将结果写入一个 CSV 文件  
write.csv(results, file.path(default_save_path, "each_acc_n_MMHC.csv"), row.names = FALSE) 

## mDAG-------------------------------------------------------------
# 定义路径和文件名-------------------------------------------------  
default_save_path <- "D:\\桌面\\研二\\DAG既往汇总\\DAG simulation new\\simu-samplesize\\mDAG"  
# 循环处理----------------------------------------------------  
# 创建一个空的数据框来存储结果  
results <- data.frame(  
  n = numeric(),  
  d = numeric(),
  i = numeric(),
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()
) 
for (n in c(100,500,1000)) {  
  for (d in c(20)) { 
    print(paste0("n=",n))
    library(tidyverse)
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_d%d.csv", n, d))  
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(i)
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_d%d_i%d.csv", n, d, i))  
      # filenameX <- file.path(default_save_path, sprintf("X_n%d_d%d_i%d.csv", n, d, i))  
      
      # loss_type <- sem_type_dict[as.character(d)] 
      # X <- read.csv(filenameX, header = FALSE) # %>% as.matrix()  
      W_est <- read.csv(filenameWE,header = T,row.names = 1) %>% as.matrix()
      
      # write.csv(W_est, filenameWE)  
      ACC[i+1, ] <- count_accuracy(B_true, W_est)  # 
      ACC_und[i+1, ] <- count_accuracy_und(B_true, W_est)  # 
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         d = d,
                         i = i,
                         fdr = ACC[i+1, ][1], 
                         tpr = ACC[i+1, ][2], 
                         fpr = ACC[i+1, ][3], 
                         shd = ACC[i+1, ][4], 
                         pred_size = ACC[i+1, ][5], 
                         F1_score = ACC[i+1, ][6], 
                         fdr_und = ACC_und[i+1, ][1], 
                         tpr_und = ACC_und[i+1, ][2], 
                         fpr_und = ACC_und[i+1, ][3], 
                         shd_und = ACC_und[i+1, ][4], 
                         pred_size = ACC_und[i+1, ][5], 
                         F1_score_und = ACC_und[i+1, ][6]
                       ) 
      ) 
      
      # print(ACC[i+1, ])  
      cat(sprintf("n=%d,d=%d,acc=%s\n", n, d, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,d=%d,acc_und=%s\n", n, d, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    # acc <- colMeans(ACC)  
    # acc_und <- colMeans(ACC_und)  
    # cat(sprintf("n=%d,d=%d,acc=%s\n", n, d, paste(acc, collapse = ",")))  
    # cat(sprintf("n=%d,d=%d,acc_und=%s\n", n, d, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("d=%d execution time: %.2f seconds\n", d, exe_time))  
    
    
  }  
}
# 将结果写入一个 CSV 文件  
write.csv(results, file.path(default_save_path, "each_acc_n_mDAG.csv"), row.names = FALSE) 

## DAGbagM---------------------
# 定义路径和文件名-------------------------------------------------  
default_save_path <- "D:\\桌面\\研二\\DAG既往汇总\\DAG simulation new\\simu-samplesize\\DAGBagM"  
# 循环处理----------------------------------------------------  
# 创建一个空的数据框来存储结果  
results <- data.frame(  
  n = numeric(),  
  d = numeric(),
  i = numeric(),
  fdr = numeric(),  
  tpr = numeric(),  
  fpr = numeric(),  
  shd = numeric(),  
  pred_size = numeric(),  
  F1_score = numeric(),
  fdr_und = numeric(),  
  tpr_und = numeric(),  
  fpr_und = numeric(),  
  shd_und = numeric(),  
  pred_size = numeric(),  
  F1_score_und = numeric()
) 
for (n in c(100,500,1000)) {  
  for (d in c(20)) { 
    print(paste0("n=",n))
    library(tidyverse)
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_d%d.csv", n, d))  
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(i)
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_d%d_i%d.csv", n, d, i))  
      # filenameX <- file.path(default_save_path, sprintf("X_n%d_d%d_i%d.csv", n, d, i))  
      
      # loss_type <- sem_type_dict[as.character(d)] 
      # X <- read.csv(filenameX, header = FALSE) # %>% as.matrix()  
      W_est <- read.csv(filenameWE,header = T,row.names = 1) %>% as.matrix()
      
      # write.csv(W_est, filenameWE)  
      ACC[i+1, ] <- count_accuracy(B_true, W_est)  # 
      ACC_und[i+1, ] <- count_accuracy_und(B_true, W_est)  # 
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         d = d,
                         i = i,
                         fdr = ACC[i+1, ][1], 
                         tpr = ACC[i+1, ][2], 
                         fpr = ACC[i+1, ][3], 
                         shd = ACC[i+1, ][4], 
                         pred_size = ACC[i+1, ][5], 
                         F1_score = ACC[i+1, ][6], 
                         fdr_und = ACC_und[i+1, ][1], 
                         tpr_und = ACC_und[i+1, ][2], 
                         fpr_und = ACC_und[i+1, ][3], 
                         shd_und = ACC_und[i+1, ][4], 
                         pred_size = ACC_und[i+1, ][5], 
                         F1_score_und = ACC_und[i+1, ][6]
                       ) 
      ) 
      
      # print(ACC[i+1, ])  
      cat(sprintf("n=%d,d=%d,acc=%s\n", n, d, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,d=%d,acc_und=%s\n", n, d, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    # acc <- colMeans(ACC)  
    # acc_und <- colMeans(ACC_und)  
    # cat(sprintf("n=%d,d=%d,acc=%s\n", n, d, paste(acc, collapse = ",")))  
    # cat(sprintf("n=%d,d=%d,acc_und=%s\n", n, d, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("d=%d execution time: %.2f seconds\n", d, exe_time))  
    
    
  }  
}
# 将结果写入一个 CSV 文件  
write.csv(results, file.path(default_save_path, "each_acc_n_DAGBagM.csv"), row.names = FALSE) 
