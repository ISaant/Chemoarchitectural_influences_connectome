# Script to test time per spin with and without rotation
library(tidyverse) #For reading csv
library(lme4) #For linear mix effect model
library(reshape2) 
library(ggplot2)
library(reticulate)
library(tibble)
library(ggdist)
# THIS VERSION DO NOT USE PARALLEL DISTRIBUTION 
# Based on model correction script 

#FUNCTION TO BE DEFINED to run the models
run_model <- function(formula, data, beta_df, column_prefix){
  #NOTE: A model with an interaction term like (neurochem ~ band * control0 + control1 + control2...) expands to neurochem_corr ~ band + term + band:term + ...
  #.     Fix effects of the band (fixef(model)[band]) of this model vs a mode with no interaction ((neurochem ~ band + control0 + control1 + control2...)) will ...
  #      WILL NOT be equal

  model <- lmer(formula, data = data, REML = FALSE, control = lmerControl(optimizer = "bobyqa"))
  chr_formula <- as.character(formula)[3]
  if (grepl("\\*", chr_formula)) {
    coefs <- fixef(model)
    term <- str_match(chr_formula, "\\*([^+]+)\\+")[,2]
    term <- trimws(term) 
    if (term == "sex"){term <- "sexMALE"}
    if (term == "intra_inter7"){term <- "intra_inter7inter"}
    interaction_term <- paste0(bands, ":",term)
    beta_df[beta_df$band == bands, paste0(column_prefix,term,"-interaction")] <- coefs[[interaction_term]]
  } 

  #else if (grepl("\\*", chr_formula))

  else {beta_df[beta_df$band == bands, paste0(column_prefix,"-no_interaction")] <- fixef(model)[bands]}
  
  return (beta_df)

}


add_intra_inter_labels <- function(df, Networks) {
  
  
  # Flatten subnetwork-to-main-network mapping
  sub_to_main <- setNames(rep(names(Networks), lengths(Networks)), unlist(Networks))
  
  # Helper: extract the prefix of a region (e.g., "VisPeri" from "VisPeri_StriCal_1")
  get_subnetwork <- function(roi) sub("^([A-Za-z0-9]+)_.*$", "\\1", roi)
  
  # Apply to all rows
  df$sub <- sapply(df$from_ROI, get_subnetwork)
  df$sub_to   <- sapply(df$to_ROI,   get_subnetwork)
  
  # Subnetwork-level classification (17)
  df$intra_inter17 <- ifelse(df$sub == df$sub_to, "intra", "inter")
  
  # Main network-level classification (7)
  df$main <- sub_to_main[df$sub]
  df$main_to   <- sub_to_main[df$sub_to]
  
  df$intra_inter7 <- ifelse(df$main == df$main_to, "intra", "inter")
  
  # Drop helper columns if you want
  #df$sub <- NULL
  df$sub_to <- NULL
  #df$main <- NULL
  df$main_to <- NULL
  
  return(df)
}

filter_intra_inter <- function(df, remove_type = c("intra", "inter")) {
  stopifnot("intra_inter7" %in% names(df))
  remove_type <- match.arg(remove_type)
  df %>% filter(.data$intra_inter7 != remove_type)
}

effects_df_generations <- function (AECs, df){
  effects_df <- tibble()

  for (bands in AECs) {
    formula <- as.formula(paste0("neurochem_corr ~ ", bands, " + age_z + sex + euclidean_dist + (", bands, " | ID)"))
    model <- lmer(formula, data = df, REML = FALSE,
                  control = lmerControl(optimizer = "bobyqa"))
    
    # Efectos fijos
    fixed_b <- fixef(model)[bands]
    fixed_intercept <- fixef(model)["(Intercept)"]
    
    # Efectos aleatorios
    ranefs <- ranef(model)$ID
    random_effects_band <- ranefs[, bands, drop = FALSE]
    random_effects_intercept <- ranefs[, "(Intercept)", drop = FALSE]
    
    # Nombres dinámicos
    fixed_effect_col <- paste0("fixed_effect_", bands)
    random_effect_col <- paste0("random_effect_", bands)
    total_effect_col <- paste0("total_effect_", bands)
    fixed_intercept_col <- paste0("fixed_intercept_", bands)
    random_intercept_col <- paste0("random_intercept_", bands)
    total_intercept_col <- paste0("total_intercept_", bands)
    
    # Crear tibble con nombres dinámicos
    random_effects <- tibble(
      ID = rownames(ranefs),
      !!fixed_effect_col := fixed_b,
      !!random_effect_col := random_effects_band[[1]],
      !!total_effect_col := fixed_b + random_effects_band[[1]],
      !!fixed_intercept_col := fixed_intercept,
      !!random_intercept_col := random_effects_intercept[[1]],
      !!total_intercept_col := fixed_intercept + random_effects_intercept[[1]],
    )
    
    if (nrow(effects_df) == 0) {
      effects_df <- random_effects
    } else {
      effects_df <- left_join(effects_df, random_effects, by = "ID")
    }
  }
  return(effects_df)
}

prepare_random_effect_data <- function(intra_df, inter_df, band) {
  random_col <- paste0("random_effect_", band)
  fixed_col <- paste0("fixed_effect_", band)
  total_col <- paste0("total_effect_", band)
  
  stopifnot(all(c("ID", random_col, fixed_col, total_col) %in% names(intra_df)))
  stopifnot(all(c("ID", random_col, fixed_col, total_col) %in% names(inter_df)))
  
  common_ids <- intersect(intra_df$ID, inter_df$ID)
  if (length(common_ids) == 0) stop("No overlapping IDs between intra and inter data frames.")
  
  intra_fixed <- intra_df[[fixed_col]]
  inter_fixed <- inter_df[[fixed_col]]
  intra_fixed <- intra_fixed[!is.na(intra_fixed)][1]
  inter_fixed <- inter_fixed[!is.na(inter_fixed)][1]
  
  random_df <- bind_rows(
    intra_df %>%
      filter(ID %in% common_ids) %>%
      transmute(ID, network = "Intra", total_effect = .data[[total_col]]),
    inter_df %>%
      filter(ID %in% common_ids) %>%
      transmute(ID, network = "Inter", total_effect = .data[[total_col]])
  ) %>%
    mutate(
      network = factor(network, levels = c("Intra", "Inter")),
      network_index = as.numeric(network)
    ) %>%
    group_by(ID) %>%
    mutate(
      jitter = runif(1, 0.06, 0.2),
      x_jitter = network_index + if_else(network == "Inter", -jitter, jitter)
    ) %>%
    ungroup()
  
  fixed_df <- tibble(
    network = factor(c("Intra", "Inter"), levels = c("Intra", "Inter")),
    network_index = as.numeric(network),
    fixed_value = c(intra_fixed, inter_fixed)
  )
  
  list(random_df = random_df, fixed_df = fixed_df)
}

create_random_effect_plot <- function(random_df, fixed_df, band) {
  base_color <- custom_palette[[band]]
  color_map <- c(
    "Intra" = lighten_color(base_color, 0.18),
    "Inter" = darken_color(base_color, 0.18)
  )
  
  ggplot(random_df, aes(x = network_index, y = total_effect, fill = network)) +
    # Half-eye density for Intra (left)
    ggdist::stat_halfeye(
      data = dplyr::filter(random_df, network == "Intra"),
      adjust = 0.6,
      width = 0.6,
      .width = 0,
      justification = 1.1,
      point_colour = NA,
      slab_color = NA,
      alpha = 0.85,
      side = "left"
    ) +
    # Half-eye density for Inter (right)
    ggdist::stat_halfeye(
      data = dplyr::filter(random_df, network == "Inter"),
      adjust = 0.6,
      width = 0.6,
      .width = 0,
      justification = -0.1,
      point_colour = NA,
      slab_color = NA,
      alpha = 0.85,
      side = "right"
    ) +
    # Boxplot in center
    geom_boxplot(
      width = 0.05,
      outlier.shape = NA,
      alpha = 0.55,
      color = "#3d3d3d"
    ) +
    # Lines between points per subject
    geom_line(
      data = random_df,
      aes(x = x_jitter, y = total_effect, group = ID),
      inherit.aes = FALSE,
      color = "grey75",
      alpha = 0.35,
      linewidth = 0.45
    ) +
    # Points per subject
    geom_point(
      data = random_df,
      aes(x = x_jitter, y = total_effect, color = network),
      inherit.aes = FALSE,
      size = 1.7,
      alpha = 0.85
    ) +
    # Fixed effect line
    geom_line(
      data = fixed_df,
      aes(x = network_index, y = fixed_value, group = 1),
      color = "#d62728",
      linewidth = 1.1
    ) +
    geom_point(
      data = fixed_df,
      aes(x = network_index, y = fixed_value),
      color = "#d62728",
      size = 2.6
    ) +
    scale_fill_manual(values = color_map, guide = "none") +
    scale_color_manual(values = color_map, guide = "none") +
    scale_x_continuous(
      breaks = fixed_df$network_index,
      labels = levels(fixed_df$network)
    ) +
    labs(
      title = paste("Total effects ·", band, "band"),
      subtitle = "Intra vs Inter networks",
      x = NULL,
      y = "Fixed + random effect"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )
}

generate_random_effect_comparison_plots <- function(intra_df, inter_df) {
  setNames(purrr::map(AECs, function(band) {
    prepared <- prepare_random_effect_data(intra_df, inter_df, band)
    create_random_effect_plot(prepared$random_df, prepared$fixed_df, band)
  }), AECs)
}

save_random_effect_plots <- function(plot_list, output_path, prefix) {
  if (!dir.exists(output_path)) dir.create(output_path, recursive = TRUE)
  purrr::iwalk(plot_list, function(plot_obj, band) {
    file_name <- file.path(output_path, paste0(prefix, "_", band, "_total_effect_raincloud.pdf"))
    ggsave(
      filename = file_name,
      plot = plot_obj,
      width = 8,
      height = 6,
      dpi = 300
    )
  })
}

AECs <- c("delta", "theta", "alpha", "beta", "g_low", "g_high")

custom_palette <- c(
  delta  = "#E1A36F",
  theta  = "#DEC484",
  alpha  = "#E2D8A5",
  beta   = "#6F9F9C",
  g_low  = "#577E89",
  g_high = "#B8A078"
)

mix_colors <- function(color, mix_with, weight = 0.25) {
  weight <- max(min(weight, 1), 0)
  base_rgb <- grDevices::col2rgb(color) / 255
  mix_rgb <- grDevices::col2rgb(mix_with) / 255
  result <- base_rgb * (1 - weight) + mix_rgb * weight
  grDevices::rgb(result[1], result[2], result[3])
}

lighten_color <- function(color, amount = 0.25) mix_colors(color, "#FFFFFF", amount)
darken_color  <- function(color, amount = 0.2)  mix_colors(color, "#000000", amount)

# Dictionary of functional networks
Networks <- list(
  Cont = c("ContA", "ContB", "ContC"),
  Default = c("DefaultA", "DefaultB", "DefaultC"),
  DorsAttn = c("DorsAttnA", "DorsAttnB"),
  Limbic = c("LimbicA", "LimbicB"),
  SalVentAttn = c("SalVentAttnA", "SalVentAttnB"),
  SomMot = c("SomMotA", "SomMotB"),
  TempPar = c("TempPar"),
  Vis = c("VisCent", "VisPeri")
)


# Set up paths
# parent_path <- normalizePath("../")     
parent_path <- '/Users/isaant/Documents/PosDoc/Projects/Shaping_aging_fc'
output_dir <- file.path(parent_path, "meg_outputs", "lm_results", "beta_Interaction_IntraInter")
output_dir_figures  <- file.path(parent_path, "meg_outputs", "lm_results", "FinalFigures")
# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

#Path to subjects data
subjects_path <- file.path(parent_path, "meg_outputs", "schaefer200_17networks_neurochem-similarity_AEC-fc_all-subs_LOO")

# Load all subject filenames
files <- sort(list.files(subjects_path))
subjects <- files[2:length(files)]

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

all_data <- list()

# Loop through each subject, skip unreadable ones
for (sub in subjects) {
  if (sub %in% c(skip_list, na_data_list)) next
  sub_file <- file.path(subjects_path, sub)
  df <- tryCatch(read_csv(sub_file, show_col_types = FALSE), error = function(e) NULL)

  # Preprocess the subject data: normalize values and drop unnecessary columns
  df <- df %>%
    mutate(ID = as.factor(ID)) %>%
    select(-any_of(c("age_full_time_edu_comp", "degree"))) %>%
    mutate(across(all_of(AECs), ~as.vector(scale(.)))) %>%
    mutate(euclidean_dist = as.vector(scale(euclidean_dist)))

  df <- add_intra_inter_labels(df,Networks)
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
df_all <- df_all %>% mutate(age_z = scale(age))
df_all$sex <- factor(df_all$sex, levels = c("FEMALE", "MALE"))
df_all$intra_inter7 <- factor(df_all$intra_inter7, levels = c("intra", "inter"))  # para forzar el orden
df_intra <- filter_intra_inter(df_all, "inter")
df_inter <- filter_intra_inter(df_all, "intra")
# Initialize empty data frame

effects_intra_df <- effects_df_generations(AECs, df_intra)
effects_inter_df <- effects_df_generations(AECs, df_inter)

raincloud_output_dir <- file.path(output_dir_figures, "raincloud_plots")

comparison_plots <- generate_random_effect_comparison_plots(effects_intra_df, effects_inter_df)
save_random_effect_plots(comparison_plots, raincloud_output_dir, "intra_vs_inter")
write_csv(effects_intra_df, file.path(output_dir,"random-effects_LOO_Intre-ieOnlyIntra.csv"))
write_csv(effects_inter_df, file.path(output_dir,"random-effects_LOO_Intra-ieOnlyInter.csv")) 
#write_csv(beta_df, file.path(output_dir, paste0(col,"_spin_Interaction_IntraInter.csv")))
#write_csv(effects_df, file.path(output_dir, paste0(col,"_spin_7_Networks_randEffects.csv")))
