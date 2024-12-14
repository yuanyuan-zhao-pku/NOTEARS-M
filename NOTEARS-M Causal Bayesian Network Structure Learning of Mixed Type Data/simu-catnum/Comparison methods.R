# Define accuracy metrics function ----------------------  
count_accuracy <- function(B_true, B_est) {  
  # Validate function arguments  
  if (!all(B_est %in% c(0, 1))) {  
    stop("B_est should take value in {0,1}")  
  }  
  
  d <- nrow(B_true)  
  # Linear index of non-zero elements  
  pred <- which(B_est != 0)  
  cond <- which(B_true != 0)  
  cond_reversed <- which(t(B_true) != 0)  
  cond_skeleton <- unique(c(cond, cond_reversed))  
  
  # True positives, false positives, and reversals  
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
  
  # Get lower triangle matrix  
  get_lower_triangle <- function(mat) {  
    if (!is.matrix(mat)) {  
      stop("Input must be a matrix.")  
    }  
    
    # Create a matrix with the same dimensions as the input matrix, initialized to zero  
    lower_triangle <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))  
    
    # Copy the lower triangle values to the new matrix  
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
  # Validate function arguments  
  if (!all(B_est %in% c(0, 1))) {  
    stop("B_est should take value in {0,1}")  
  }  
  
  d <- nrow(B_true)  
  # Linear index of non-zero elements  
  pred <- which(B_est != 0)  
  cond <- which(B_true != 0)  
  cond_reversed <- which(t(B_true) != 0)  
  cond_skeleton <- unique(c(cond, cond_reversed))  
  
  # True positives, false positives, and reversals  
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
  
  # Get lower triangle matrix  
  get_lower_triangle <- function(mat) {  
    if (!is.matrix(mat)) {  
      stop("Input must be a matrix.")  
    }  
    
    # Create a matrix with the same dimensions as the input matrix, initialized to zero  
    lower_triangle <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))  
    
    # Copy the lower triangle values to the new matrix  
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
# MMHC (treat all loss_type as continuous) ------------------------  
library(bnlearn)  

# Define MMHCtoW function to learn the Bayesian network structure and return adjacency matrix estimate W  
MMHCtoW <- function(x) {  
  mmhc_net = bnlearn::mmhc(x, whitelist = NULL, blacklist = NULL, restrict.args = list(),  
                           maximize.args = list(), debug = FALSE)   
  nodes <- paste0("V", 1:ncol(x))  
  
  W <- as.matrix(matrix(0, ncol = length(nodes),   
                        nrow = length(nodes)))  
  for (i in 1:nrow(mmhc_net$arcs)) {  
    from <- which(nodes == mmhc_net$arcs[i, 1])  
    to <- which(nodes == mmhc_net$arcs[i, 2])  
    W[from, to] <- 1  
  }  
  return(W)  
}  

# Define path and filename  
default_save_path <- "./MMHC"  

# Create empty data frames to store results  
results <- data.frame(  
  n = numeric(),  
  m = numeric(),  
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
results_avg <- data.frame(  
  n = numeric(),  
  m = numeric(),  
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

# Loop for different m 
for (n in c(1000)) {  
  for (m in c(2,3,4)) {   
    print(paste0("m=",m))  
    library(tidyverse)  
    start_time <- Sys.time()  
    ACC <- matrix(0, nrow = 10, ncol = 6)  
    ACC_und <- matrix(0, nrow = 10, ncol = 6)  
    
    filenameB <- file.path(default_save_path, sprintf("B_true_n%d_m%d.csv", n, m))
    B_true <- read.csv(filenameB, header = FALSE) %>% as.matrix()  
    
    for (i in 0:9) {  
      print(paste0("i=", i))  
      
      filenameWE <- file.path(default_save_path, sprintf("W_est_n%d_m%d_i%d.csv", n, m, i))  
      filenameX <- file.path(default_save_path, sprintf("X_n%d_m%d_i%d.csv", n, m, i))        
      X <- read.csv(filenameX, header = FALSE)   
      W_est <- MMHCtoW(X)  
      
      write.csv(W_est, filenameWE)  
      ACC[i + 1, ] <- count_accuracy(B_true, W_est)  
      ACC_und[i + 1, ] <- count_accuracy_und(B_true, W_est)  
      
      results <- rbind(results,  
                       data.frame(  
                         n = n,  
                         m = m,  
                         i = i,  
                         fdr = ACC[i + 1, ][1],   
                         tpr = ACC[i + 1, ][2],   
                         fpr = ACC[i + 1, ][3],   
                         shd = ACC[i + 1, ][4],   
                         pred_size = ACC[i + 1, ][5],   
                         F1_score = ACC[i + 1, ][6],   
                         fdr_und = ACC_und[i + 1, ][1],   
                         tpr_und = ACC_und[i + 1, ][2],   
                         fpr_und = ACC_und[i + 1, ][3],   
                         shd_und = ACC_und[i + 1, ][4],   
                         pred_size = ACC_und[i + 1, ][5],   
                         F1_score_und = ACC_und[i + 1, ][6]   
                       )   
      )   
      
      cat(sprintf("n=%d,m=%d,acc=%s\n", n, m, paste(ACC[i+1, ], collapse = ",")))  
      cat(sprintf("n=%d,m=%d,acc_und=%s\n", n, m, paste(ACC_und[i+1, ], collapse = ",")))  
    }  
    
    acc <- colMeans(ACC)  
    acc_und <- colMeans(ACC_und)  
    cat(sprintf("n=%d,m=%d,acc=%s\n", n, m, paste(acc, collapse = ",")))  
    cat(sprintf("n=%d,m=%d,acc_und=%s\n", n, m, paste(acc_und, collapse = ",")))  
    
    end_time <- Sys.time()  
    exe_time <- as.numeric(end_time - start_time, units = "secs")  
    cat(sprintf("m=%d execution time: %.2f seconds\n", m, exe_time))
    
    results_avg <- rbind(results_avg,  
                         data.frame(  
                           n = n,  
                           m = m,  
                           fdr = acc[1],   
                           tpr = acc[2],   
                           fpr = acc[3],   
                           shd = acc[4],   
                           pred_size = acc[5],   
                           F1_score = acc[6],   
                           fdr_und = acc_und[1],   
                           tpr_und = acc_und[2],   
                           fpr_und = acc_und[3],   
                           shd_und = acc_und[4],   
                           pred_size = acc_und[5],   
                           F1_score_und = acc_und[6]   
                         )   
    )  
  }  
}  

# Write results to CSV files  
write.csv(results_avg, file.path(default_save_path, "acc_m_MMHC.csv"), row.names = FALSE)   
write.csv(results, file.path(default_save_path, "each_acc_m_MMHC.csv"), row.names = FALSE)   
