# Libraries
library(tidyverse)
library(lme4)
library(pbapply)

# User settings
project_dir <- "/Users/isaant/Documents/PosDoc/Projects/Shaping_aging_fc" #UPDATE THIS

# Input/output folders
data_dir   <- file.path(project_dir, "meg_outputs", "schaefer200_17networks_neurochem-similarity_AEC-fc_all-subs")
output_dir <- file.path(project_dir, "meg_outputs", "lm_results", "beta_Interaction")
models_dir <- file.path(project_dir, "meg_outputs", "models")

# Frequency bands to analyze
bands <- c("delta", "theta", "alpha", "beta", "g_low", "g_high")

# Columns to remove if they exist
columns_to_remove <- c("age_full_time_edu_comp", "degree", "from_ROI", "to_ROI") #UPDATE THIS... Do you need any column to remove?

# Number of cores
options(mc.cores = 8)

# Create output folders if needed
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

# Start timer
start_time <- Sys.time()

# Load subject file list
subject_files <- sort(list.files(data_dir, full.names = TRUE))

# Lists to track excluded subjects
missing_id_files <- c()
missing_required_data_files <- c()

# Store valid data here
all_data <- list()

# Read and clean each subject file
for (file_path in subject_files) {
  
  file_name <- basename(file_path)
  
  # Try reading file
  df <- tryCatch(
    read_csv(file_path, show_col_types = FALSE),
    error = function(e) NULL
  )
  
  # Skip unreadable files
  if (is.null(df)) {
    missing_id_files <- c(missing_id_files, file_name)
    next
  }
  
  # Check required columns exist
  required_cols <- c("ID", "age", "cattell", "acer", "sex", "euclidean_dist", "neurochem_corr") #UPDATE THIS
  missing_cols <- setdiff(required_cols, names(df))
  
  if (length(missing_cols) > 0) {
    missing_required_data_files <- c(missing_required_data_files, file_name)
    next
  }
  
  # Check missing values in key variables
  if (any(is.na(df$ID))) {
    missing_id_files <- c(missing_id_files, file_name)
    next
  }
  
  if (any(is.na(df$age)) || any(is.na(df$cattell)) || any(is.na(df$acer))) {
    missing_required_data_files <- c(missing_required_data_files, file_name)
    next
  }
  
  # Keep only existing band columns
  band_cols_present <- bands[bands %in% names(df)]
  
  # Clean and scale data
  df <- df %>%
    mutate(ID = as.factor(ID)) %>%
    select(-any_of(columns_to_remove)) %>%
    mutate(across(all_of(band_cols_present), ~ as.numeric(scale(.)))) %>%
    mutate(euclidean_dist = as.numeric(scale(euclidean_dist)))
  
  all_data[[length(all_data) + 1]] <- df
}

# Combine all valid subject data
df_all <- bind_rows(all_data)

df_all <- df_all %>%
  mutate(age_z = scale(age))

cat("\nFiles skipped because they were unreadable or had missing ID:\n")
print(missing_id_files)

cat("\nFiles skipped because they had missing required data:\n")
print(missing_required_data_files)

cat("\nNumber of valid files loaded:", length(all_data), "\n")
cat("Total rows in combined dataset:", nrow(df_all), "\n")

# Fit one model per frequency band
reference_betas <- list()

for (band in bands) {
  
  # Skip band if column is not present
  if (!(band %in% names(df_all))) {
    cat("Skipping band:", band, "- column not found in data.\n")
    next
  }
  
  cat("Fitting model for band:", band, "\n")
  
  model_formula <- as.formula(
    paste0("neurochem_corr ~ ", band, " + age_z + sex + euclidean_dist + (", band, " | ID)")
  )
  
  model <- lmer(
    model_formula,
    data = df_all,
    REML = FALSE,
    control = lmerControl(
      optimizer = "bobyqa",
      optCtrl = list(maxfun = 2e5)
    )
  )
  
  # Save fixed-effect beta for this band
  reference_betas[[band]] <- fixef(model)[band]
  
  # OPTIONAL: save full model object
  saveRDS(model, file.path(models_dir, paste0("model_", band, ".rds")))
}

# Save summary table
reference_df <- tibble(
  band = names(reference_betas),
  beta = unlist(reference_betas)
)

output_file <- file.path(output_dir, "reference_beta_coefficients.csv")
write_csv(reference_df, output_file)

cat("\nSaved reference beta coefficients to:\n", output_file, "\n")

# End timer
end_time <- Sys.time()
cat("\nTotal runtime:\n")
print(end_time - start_time)