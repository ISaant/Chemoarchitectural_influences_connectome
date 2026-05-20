# Script to test time per spin with and without rotation
library(tidyverse) #For reading csv
library(lme4) #For linear mix effect model
library(reshape2)
library(ggplot2)
library(reticulate)
# THIS VERSION DO NOT USE PARALLEL DISTRIBUTION

# Start timer
start_time <- Sys.time()

# Set number of cores to use for parallel processing
numcores <- 8
options(mc.cores = numcores)

# Capture command-line arguments (Hungarian spins path and column index)
args <- commandArgs(trailingOnly = TRUE)

# Set path to your Python script to use with reticulate
script_path <- "/home/isaant/scratch/python_scripts/pyFun_R_reticulate.py"

# Load Python helper functions via reticulate
source_python(script_path)

# Set up paths
# parent_path <- normalizePath("../")
parent_path <- '/home/isaant/scratch/'
output_dir <- file.path(parent_path, "Cam-CAN", "meg_outputs", "lm_results", "beta_Interaction")

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

subjects_path <- file.path(parent_path, "Cam-CAN", "meg_outputs", "compacted_data", "schaefer200_17networks_neurochem-similarity_AEC-fc_all-subs")

# Load all subject filenames
subjects <- sort(list.files(subjects_path))

# Load Hungarian spin matrix
hungSpins_path <- args[1]
hungSpins <- as.matrix(read_csv(hungSpins_path, show_col_types = FALSE))
stopifnot(is.matrix(hungSpins))
col <- as.integer(args[2]) + 1

# Initialize final DataFrame to collect all results
final_df <- data.frame()

# Check which subject files are readable and contain valid IDs
skip_list <- c()
na_data_list <- c()
for (i in seq_along(subjects)) {
  sub_file <- file.path(subjects_path, subjects[i])
  df <- tryCatch(read_csv(sub_file, show_col_types = FALSE), error = function(e) NULL)
  if (is.null(df) || any(is.na(df$ID))) {
    skip_list <- c(skip_list, subjects[i])
    next
  }
  if (any(is.na(df$age)) || any(is.na(df$cattell)) || any(is.na(df$acer))) {
    na_data_list <- c(na_data_list, subjects[i])
    next
  }
}

# Define frequency bands of interest
AECs <- c("delta", "theta", "alpha", "beta", "g_low", "g_high")

# Prepare to store beta coefficients per spin

# Select the appropriate spin column and adjust for 1-based indexing in R
spin <- hungSpins[,col]
spin <- spin + 1  # Adjust index from Python (0-based) to R (1-based)

all_data <- list()

# Loop through each subject, skip unreadable ones
for (sub in subjects) {
  if (sub %in% c(skip_list, na_data_list)) next
  sub_file <- file.path(subjects_path, sub)
  df <- tryCatch(read_csv(sub_file, show_col_types = FALSE), error = function(e) NULL)

  # Preprocess the subject data: normalize values and drop unnecessary columns
  df <- df %>%
    mutate(ID = as.factor(ID)) %>%
    select(-any_of(c("age_full_time_edu_comp", "degree", "from_ROI", "to_ROI"))) %>%
    mutate(across(all_of(AECs), ~as.vector(scale(.)))) %>%
    mutate(euclidean_dist = as.vector(scale(euclidean_dist)))

  # Reconstruct and rotate connectivity matrices for each band
  for (band in AECs) {
    band
    full_matrix <- reconstruct_symmetric_matrix(df[[band]])
    rotated <- full_matrix[spin, spin]
    df[[band]] <- rotated[lower.tri(rotated)]
  }
  all_data[[length(all_data) + 1]] <- df
}

# Combine all subjects' data into a single DataFrame
df_all <- bind_rows(all_data)

# Quality Checks
cat("Quality Checks (df_all):\n")
print(head(df_all))
cat("Summary of age:\n")
print(summary(df_all$age))
cat("Summary of cattell:\n")
print(summary(df_all$cattell))
cat("Summary of acer:\n")
print(summary(df_all$acer))

has_age <- all(!is.na(df_all$age))
has_cattell <- all(!is.na(df_all$cattell))
has_acer <- all(!is.na(df_all$acer))

cat("Subjects skipped due to missing ID:\n")
print(skip_list)
cat("Subjects skipped due to missing age, cattell, or acer:\n")
print(na_data_list)

# Regress out age from Cattell and Acer scores, or set NA if not possible
df_all <- df_all %>% mutate(cattell_resid = resid(lm(cattell ~ age)))
df_all <- df_all %>% mutate(acer_resid = resid(lm(acer ~ age)))

## Fit linear mixed-effects model for interaction with age
beta_spins_age <- list()
ranef_age_list <- list()
if (has_age) {
  for (band in AECs) {
    formula <- as.formula(paste0("neurochem_corr ~ ", band, " * age + sex + euclidean_dist + (", band, "*age|ID)"))
    model <- lmer(formula, data = df_all, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    coefs <- fixef(model)
    beta_spins_age[[band]] <- if (!is.null(coefs[[band]])) coefs[[band]] else NA
    ranefs <- ranef(model)$ID %>%
      rownames_to_column("ID") %>%
      pivot_longer(-ID, names_to = "term", values_to = "random_slope") %>%
      mutate(band = band, interaction = "age", spin = col - 1)
    ranef_age_list[[length(ranef_age_list) + 1]] <- ranefs
  }
} else {
  beta_spins_age <- setNames(as.list(rep(NA, length(AECs))), AECs)
}
names(beta_spins_age) <- AECs

# Cattell model: only if both age and cattell are available
beta_spins_cattell <- list()
ranef_cattell_list <- list()
if (has_age && has_cattell) {
  for (band in AECs) {
    formula <- as.formula(paste0("neurochem_corr ~ ", band, " *cattell_resid + age + sex + euclidean_dist + (", band, "*cattell_resid|ID)"))
    model <- lmer(formula, data = df_all, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    coefs <- fixef(model)
    beta_spins_cattell[[band]] <- if (!is.null(coefs[[band]])) coefs[[band]] else NA
    ranefs <- ranef(model)$ID %>%
      rownames_to_column("ID") %>%
      pivot_longer(-ID, names_to = "term", values_to = "random_slope") %>%
      mutate(band = band, interaction = "cattell", spin = col - 1)
    ranef_cattell_list[[length(ranef_cattell_list) + 1]] <- ranefs
  }
} else {
  beta_spins_cattell <- setNames(as.list(rep(NA, length(AECs))), AECs)
}
names(beta_spins_cattell) <- AECs

# Acer model: only if both age and acer are available
beta_spins_acer <- list()
ranef_acer_list <- list()
if (has_age && has_acer) {
  for (band in AECs) {
    formula <- as.formula(paste0("neurochem_corr ~ ", band, " *acer_resid + age + sex + euclidean_dist + (", band, "*acer_resid|ID)"))
    model <- lmer(formula, data = df_all, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))
    coefs <- fixef(model)
    beta_spins_acer[[band]] <- if (!is.null(coefs[[band]])) coefs[[band]] else NA
    ranefs <- ranef(model)$ID %>%
      rownames_to_column("ID") %>%
      pivot_longer(-ID, names_to = "term", values_to = "random_slope") %>%
      mutate(band = band, interaction = "acer", spin = col - 1)
    ranef_acer_list[[length(ranef_acer_list) + 1]] <- ranefs
  }
} else {
  beta_spins_acer <- setNames(as.list(rep(NA, length(AECs))), AECs)
}
names(beta_spins_acer) <- AECs

# Store beta coefficients in wide format
age_df <- data.frame(
  band = names(beta_spins_age),
  age = unlist(beta_spins_age)
)

cattell_df <- data.frame(
  band = names(beta_spins_cattell),
  cattell = unlist(beta_spins_cattell)
)

acer_df <- data.frame(
  band = names(beta_spins_acer),
  acer = unlist(beta_spins_acer)
)

# Merge into one final dataframe by band
final_df <- reduce(list(age_df, cattell_df, acer_df), full_join, by = "band")
final_df$spin <- col - 1

# Save all beta coefficients for the current spin to CSV
spin_str <- sprintf("%03d", col - 1)
write_csv(final_df, file.path(output_dir, paste0("spin_", spin_str, "_all_interactions_beta_coefficients.csv")))
cat("Saved all beta coefficients to single file:", file.path(output_dir, paste0("spin_", spin_str, "_all_interactions_beta_coefficients.csv")), "\n")

all_ranef_df <- bind_rows(ranef_age_list, ranef_cattell_list, ranef_acer_list)
write_csv(all_ranef_df, file.path(output_dir, paste0("spin_", spin_str, "_all_interactions_random_effects.csv")))

# End timer and report total execution time
end_time <- Sys.time()
total_time <- end_time - start_time
cat("Total execution time:", total_time, "\n")