# =============================================================================
# BB Fictional Loans — Logistic Regression for BadFlagCategory
# =============================================================================
# Dataset: 119,810 loans | 16 columns
# Target : BadFlagCategory — "0" (Good loan) / "1" (Bad loan)
# =============================================================================

# ---- 0. PACKAGES ------------------------------------------------------------

required_packages <- c("tidyverse", "caret", "pROC", "corrplot", "scales", "ggplot2")
installed <- installed.packages()[, "Package"]
for (pkg in required_packages) {
  if (!pkg %in% installed) install.packages(pkg, repos = "https://cloud.r-project.org")
}

library(tidyverse)
library(caret)
library(pROC)
library(corrplot)
library(scales)


# ---- 1. LOAD DATA -----------------------------------------------------------

loans_raw <- read.csv("BB_fictional_loans.csv", stringsAsFactors = FALSE)

cat("Dimensions:", nrow(loans_raw), "rows ×", ncol(loans_raw), "columns\n")
cat("Columns:\n"); print(names(loans_raw))
cat("\nFirst rows:\n"); print(head(loans_raw, 3))
cat("\nSummary:\n"); print(summary(loans_raw))


# ---- 2. EDA -----------------------------------------------------------------

# 2.1 Target class balance
cat("\n--- Target distribution ---\n")
tbl <- table(loans_raw$BadFlagCategory)
print(tbl)
print(prop.table(tbl))
# Note: 78.3% Good (0) vs 21.7% Bad (1) — imbalanced dataset

ggplot(as.data.frame(tbl), aes(x = Var1, y = Freq, fill = Var1)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(round(Freq / sum(Freq) * 100, 1), "%")),
            vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("0" = "#4b8bbe", "1" = "#e05252")) +
  labs(title = "Loan Quality Distribution",
       x = "BadFlagCategory (0 = Good, 1 = Bad)", y = "Count") +
  theme_minimal() + theme(legend.position = "none")


# 2.2 Missing values — none present, but Education = "-1" encodes unknown
cat("\nMissing values per column:\n")
print(colSums(is.na(loans_raw)))
cat("\nEducation '-1' (unknown) count:", sum(loans_raw$Education == "-1"), "\n")


# 2.3 Numeric distributions by target
numeric_vars <- c("Age", "Amount", "ExternalCreditScore",
                  "LoanDuration", "IncomeTotal", "LiabilitiesTotal")

for (v in numeric_vars) {
  p <- ggplot(loans_raw, aes_string(x = "BadFlagCategory", y = v,
                                     fill = "BadFlagCategory")) +
    geom_boxplot(outlier.alpha = 0.2, outlier.size = 0.5) +
    scale_fill_manual(values = c("0" = "#4b8bbe", "1" = "#e05252")) +
    labs(title = paste(v, "by Loan Quality"), x = "BadFlagCategory") +
    theme_minimal() + theme(legend.position = "none")
  print(p)
}


# 2.4 Categorical bad-loan rates
cat_vars <- c("NewCreditCustomer", "ExternalPaymentDefaultRemarks",
              "Country", "Gender", "Education")

for (v in cat_vars) {
  p <- loans_raw %>%
    mutate(is_bad = as.integer(BadFlagCategory == "1")) %>%
    group_by(across(all_of(v))) %>%
    summarise(bad_rate = mean(is_bad), n = n(), .groups = "drop") %>%
    ggplot(aes_string(x = v, y = "bad_rate")) +
    geom_col(fill = "#e05252") +
    geom_text(aes(label = paste0(round(bad_rate * 100, 1), "%\n(n=", n, ")")),
              vjust = -0.3, size = 3) +
    scale_y_continuous(labels = percent_format(), limits = c(0, 0.6)) +
    labs(title = paste("Bad loan rate by", v), y = "Bad Rate (%)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  print(p)
}


# 2.5 Correlation matrix (numeric variables + target)
numeric_data <- loans_raw %>%
  select(all_of(numeric_vars)) %>%
  mutate(BadFlag = as.integer(loans_raw$BadFlagCategory))

cor_mat <- cor(numeric_data, use = "pairwise.complete.obs")
cat("\nCorrelation matrix:\n")
print(round(cor_mat, 3))

corrplot(cor_mat, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, addCoef.col = "black",
         number.cex = 0.65, title = "Correlation — Numeric Variables",
         mar = c(0, 0, 2, 0))

# Key finding: IncomeTotal and LiabilitiesTotal correlation
cat("\nIncomeTotal vs LiabilitiesTotal correlation:",
    round(cor(loans_raw$IncomeTotal, loans_raw$LiabilitiesTotal), 3), "\n")


# ---- 3. DATA CLEANING & FEATURE ENGINEERING ---------------------------------

loans <- loans_raw

# 3.1 Parse date; extract loan month (seasonal effect)
loans$LoanDate  <- as.Date(loans$LoanDate)
loans$LoanMonth <- month(loans$LoanDate)
loans$LoanYear  <- year(loans$LoanDate)

# 3.2 Replace Education "-1" with NA, then treat as "Unknown" level
loans$Education[loans$Education == "-1"] <- "Unknown"

# 3.3 Outlier handling — cap numeric at 99th percentile
cap99 <- function(x) {
  upper <- quantile(x, 0.99, na.rm = TRUE)
  pmin(x, upper)
}
loans$Amount_c    <- cap99(loans$Amount)
loans$Income_c    <- cap99(loans$IncomeTotal)
loans$Liab_c      <- cap99(loans$LiabilitiesTotal)
loans$CreditScore_c <- cap99(loans$ExternalCreditScore)

# 3.4 Feature engineering
loans$DebtToIncome <- ifelse(loans$Income_c > 0,
                              loans$Liab_c / loans$Income_c,
                              NA)

# 3.5 Bin numeric variables into ordered categories
loans$Age_bin <- cut(loans$Age,
                     breaks = c(0, 24, 34, 44, 54, 64, Inf),
                     labels = c("18-24", "25-34", "35-44", "45-54", "55-64", "65+"),
                     right = TRUE)

loans$Amount_bin <- cut(loans$Amount_c,
                        breaks = quantile(loans$Amount_c,
                                          probs = c(0, 0.25, 0.5, 0.75, 1),
                                          na.rm = TRUE),
                        labels = c("Q1_Low", "Q2_MidLow", "Q3_MidHigh", "Q4_High"),
                        include.lowest = TRUE)

loans$CreditScore_bin <- cut(loans$CreditScore_c,
                              breaks = c(-Inf, 400, 600, 750, Inf),
                              labels = c("Poor", "Fair", "Good", "Excellent"),
                              right = TRUE)

loans$Duration_bin <- cut(loans$LoanDuration,
                           breaks = c(0, 12, 24, 36, Inf),
                           labels = c("Short_12m", "Mid_24m", "Long_36m", "VeryLong"),
                           right = TRUE)

loans$Income_bin <- cut(loans$Income_c,
                         breaks = quantile(loans$Income_c,
                                           probs = c(0, 0.25, 0.5, 0.75, 1),
                                           na.rm = TRUE),
                         labels = c("Low", "MidLow", "MidHigh", "High"),
                         include.lowest = TRUE)

loans$DTI_bin <- cut(loans$DebtToIncome,
                      breaks = c(-Inf, 0.2, 0.5, 1.0, Inf),
                      labels = c("DTI_Low", "DTI_Moderate", "DTI_High", "DTI_VeryHigh"),
                      right = TRUE)
# Replace NA DebtToIncome (zero income) with separate level
loans$DTI_bin <- addNA(loans$DTI_bin)
levels(loans$DTI_bin)[is.na(levels(loans$DTI_bin))] <- "DTI_Unknown"

# 3.6 Convert target and categoricals to factors
loans$BadFlagCategory <- factor(loans$BadFlagCategory, levels = c("0", "1"))
loans$NewCreditCustomer <- factor(loans$NewCreditCustomer)
loans$Gender  <- factor(loans$Gender)
loans$Country <- factor(loans$Country)
loans$Education <- factor(loans$Education)
loans$ExternalPaymentDefaultRemarks <- factor(loans$ExternalPaymentDefaultRemarks)


# ---- 4. MODELLING DATASET ---------------------------------------------------

model_vars <- c(
  "BadFlagCategory",
  "NewCreditCustomer",
  "Gender",
  "Country",
  "Education",
  "ExternalPaymentDefaultRemarks",
  "Age_bin",
  "Amount_bin",
  "CreditScore_bin",
  "Duration_bin",
  "Income_bin",
  "DTI_bin",
  "LoanMonth"
)

model_data <- loans %>%
  select(all_of(model_vars)) %>%
  mutate(LoanMonth = factor(LoanMonth)) %>%
  drop_na()

cat("\nFinal modelling dataset:", nrow(model_data), "rows\n")
cat("Class balance:\n"); print(prop.table(table(model_data$BadFlagCategory)))


# ---- 5. TRAIN / TEST SPLIT --------------------------------------------------
# Stratified 70/30 split to preserve bad-loan proportion

set.seed(42)
idx_train <- createDataPartition(model_data$BadFlagCategory,
                                  p = 0.70, list = FALSE)

train <- model_data[idx_train, ]
test  <- model_data[-idx_train, ]

cat("\nTrain:", nrow(train), "| Test:", nrow(test))
cat("\nTrain bad rate:", round(mean(train$BadFlagCategory == "1"), 4))
cat("\nTest  bad rate:", round(mean(test$BadFlagCategory  == "1"), 4), "\n")


# ---- 6. LOGISTIC REGRESSION — FULL MODEL ------------------------------------

cat("\nFitting full logistic regression model...\n")
logit_full <- glm(BadFlagCategory ~ .,
                  data = train,
                  family = binomial(link = "logit"))

cat("\n=== Full Model Summary ===\n")
print(summary(logit_full))


# ---- 7. REDUCE MODEL — SIGNIFICANT PREDICTORS ONLY -------------------------

full_coefs <- summary(logit_full)$coefficients
sig_terms  <- rownames(full_coefs)[full_coefs[, "Pr(>|z|)"] < 0.05]
sig_terms  <- sig_terms[sig_terms != "(Intercept)"]

# Map significant terms back to base variable names
all_preds <- setdiff(names(train), "BadFlagCategory")
sig_vars  <- Filter(
  function(v) any(startsWith(sig_terms, v)),
  all_preds
)
cat("\nSignificant base variables (p < 0.05):\n")
cat(paste(sig_vars, collapse = ", "), "\n")

formula_reduced <- as.formula(
  paste("BadFlagCategory ~", paste(sig_vars, collapse = " + "))
)
logit_reduced <- glm(formula_reduced,
                     data = train,
                     family = binomial(link = "logit"))

cat("\n=== Reduced Model Summary ===\n")
print(summary(logit_reduced))


# ---- 8. MODEL EVALUATION ----------------------------------------------------

evaluate_model <- function(model, test_df, label = "Model", threshold = 0.5) {
  probs  <- predict(model, newdata = test_df, type = "response")
  preds  <- factor(ifelse(probs >= threshold, "1", "0"), levels = c("0", "1"))
  actual <- test_df$BadFlagCategory

  cm      <- confusionMatrix(preds, actual, positive = "1")
  roc_obj <- roc(as.numeric(actual) - 1, probs, quiet = TRUE)
  auc_val <- auc(roc_obj)

  cat("\n===========================================\n")
  cat("  ", label, "\n")
  cat("===========================================\n")
  print(cm$table)
  cat(sprintf("  Accuracy  : %.4f\n", cm$overall["Accuracy"]))
  cat(sprintf("  Precision : %.4f\n", cm$byClass["Precision"]))
  cat(sprintf("  Recall    : %.4f\n", cm$byClass["Recall"]))
  cat(sprintf("  F1 Score  : %.4f\n", cm$byClass["F1"]))
  cat(sprintf("  AUC-ROC   : %.4f\n", auc_val))

  # ROC curve
  plot(roc_obj, col = "#4b8bbe", lwd = 2.5,
       main = paste("ROC Curve —", label))
  abline(a = 0, b = 1, lty = 2, col = "grey60")
  legend("bottomright",
         legend = paste("AUC =", round(auc_val, 3)),
         col = "#4b8bbe", lwd = 2)

  invisible(list(cm = cm, roc = roc_obj, auc = auc_val, probs = probs))
}

res_full    <- evaluate_model(logit_full,    test, "Full Model")
res_reduced <- evaluate_model(logit_reduced, test, "Reduced Model")


# ---- 9. COEFFICIENT VISUALISATION ------------------------------------------

coef_df <- broom::tidy(logit_reduced) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate))) %>%
  head(20)

ggplot(coef_df, aes(x = reorder(term, estimate), y = estimate,
                    fill = estimate > 0)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#e05252", "FALSE" = "#4b8bbe"),
                    labels = c("Protective factor", "Risk factor")) +
  labs(title = "Top 20 Coefficients — Reduced Logistic Regression",
       subtitle = "Positive = increases bad-loan probability",
       x = NULL, y = "Log-odds coefficient", fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")


# ---- 10. BONUS: DECISION TREE COMPARISON ------------------------------------
# Quick comparison with rpart (if available)

if (requireNamespace("rpart", quietly = TRUE)) {
  library(rpart)
  tree_model <- rpart(BadFlagCategory ~ ., data = train,
                      method = "class",
                      control = rpart.control(cp = 0.001, maxdepth = 6))
  tree_probs  <- predict(tree_model, newdata = test, type = "prob")[, "1"]
  tree_roc    <- roc(as.numeric(test$BadFlagCategory) - 1,
                     tree_probs, quiet = TRUE)
  cat("\nDecision Tree AUC-ROC:", round(auc(tree_roc), 4), "\n")
  cat("Logistic Regression AUC-ROC:", round(res_reduced$auc, 4), "\n")
} else {
  cat("\nInstall 'rpart' to compare with a decision tree: install.packages('rpart')\n")
}


# ---- 11. SUMMARY FINDINGS ---------------------------------------------------

cat("\n\n============================================================\n")
cat("  KEY FINDINGS\n")
cat("============================================================\n")
cat("Dataset:     119,810 loans | 21.7% bad loans (imbalanced)\n")
cat("No missing values. Education '-1' recoded as 'Unknown'.\n\n")
cat("Full model  AUC-ROC:", round(res_full$auc, 4), "\n")
cat("Reduced model AUC-ROC:", round(res_reduced$auc, 4), "\n\n")
cat("Most predictive variables (top 5 by |coefficient|):\n")
print(coef_df %>% select(term, estimate, p.value) %>% head(5))
cat("\nCorrelation IncomeTotal ~ LiabilitiesTotal:",
    round(cor(loans_raw$IncomeTotal, loans_raw$LiabilitiesTotal), 3), "\n")
cat("\nOutlier handling: values capped at the 99th percentile.\n")
cat("Class imbalance: 78.3% good vs 21.7% bad loans.\n")
cat("  → Consider SMOTE or class-weighting for production models.\n")
cat("============================================================\n")
