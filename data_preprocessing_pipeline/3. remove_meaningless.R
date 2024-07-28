library(magrittr)
library(tidyverse) # load this first for most/all things
library(peacesciencer) # the package of interest
library(stevemisc) # a dependency, but also used for standardizing variables for better interpretation
library(tictoc)
library(countrycode)
library(ggplot2)

# read cm_features with name
cm_features <- read_csv("data/cm_features_v2.2.csv")

# Define columns to exclude from the analysis
manually_excluded_columns <- c("month_id", "country_id", "date", 'year', 'ccode', 'gw_statename', 'gleditsch_ward')
# Dynamically find columns that contain the word "decay"
columns_with_decay <- names(cm_features)[grep("decay|splag", names(cm_features))]
dependent_variable_columns <- names(cm_features)[grep("ged|acled", names(cm_features))]
# Combine manually specified exclusions with those that contain "decay"
columns_to_exclude <- unique(c(manually_excluded_columns, columns_with_decay, dependent_variable_columns))
# Identify all other columns (i.e., those not to exclude)
columns_to_check <- setdiff(names(cm_features), columns_to_exclude)

initial_cm_features_with_missing_percentage <- cm_features %>%
  rowwise() %>%
  mutate(Zero_Percentage = sum(across(all_of(columns_to_check), ~.x == 0)) / length(columns_to_check)) %>%
  ungroup()

# group initial_cm_features_with_missing_percentage by country_id and calculate the mean of Zero_Percentage
mean_zero_percentage_by_country_id <- initial_cm_features_with_missing_percentage %>%
  group_by(country_id, gw_statename) %>%
  summarise(mean_zero_percentage = round(mean(Zero_Percentage * 100, na.rm = TRUE), 2),
            min_date = min(date, na.rm = TRUE),
            max_date = max(date, na.rm = TRUE),
            max_fatalities = max(ged_sb, na.rm = TRUE),
            .groups = 'drop')

# round float to two decimal places
mean_zero_percentage_by_country_id$mean_zero_percentage <- round(mean_zero_percentage_by_country_id$mean_zero_percentage, 2)
# to mean_zero_percentage_by_country_id add date two date columns to identify what min and max date this country spans.
# stop()


# print rows for which wdi_sp_pop_totl is 0
# weird <- cm_features %>%
#   filter(wdi_sp_pop_totl == 0)

# print for each column percentage of 0 values
# for (col in colnames(cm_features)) {
#   print(paste(col, sum(cm_features[[col]] == 0) / nrow(cm_features) * 100))
# }

# drop Kosovo (all country id)
cm_features <- cm_features %>%
  filter(country_id != 232)
# Drop Chechoslovakia (all country id)
cm_features <- cm_features %>%
  filter(country_id != 187)
# Drop some weird Soviet Union (all country id). Drop 254, 248,252,253, 250
cm_features <- cm_features %>%
  filter(country_id != 254) %>%
  filter(country_id != 248) %>%
  filter(country_id != 252) %>%
  filter(country_id != 253) %>%
  filter(country_id != 250)
# Drop some weird Soviet Union2 (all country id). This one has spacial lag variables but no other variables
cm_features <- cm_features %>%
  filter(country_id != 189)
# Drop Yugoslavia (all country id 247, 227,188). It has high ged_sb in this period but no other variables
cm_features <- cm_features %>%
  filter(country_id != 247) %>%
  filter(country_id != 227) %>%
  filter(country_id != 188)
# Drop Bahamas for which dependent variable is always 0 and it has 70% of 0 values
cm_features <- cm_features %>%
  filter(country_id != 26)
# Brop Brunei as it has 60% of 0 values
cm_features <- cm_features %>%
  filter(country_id != 140)
# Drop Belize as it has 58% of 0 values
cm_features <- cm_features %>%
  filter(country_id != 27)
# Drop Taiwan as it has 43% of 0 values. Maybe restore later but with more data
cm_features <- cm_features %>%
  filter(country_id != 198)
# Drop German Democratic Republic as it existed for 1990-10-01 and has 49% of 0 values
cm_features <- cm_features %>%
  filter(country_id != 186)
# Drop German Federal Republic. It has 26% of 0 values but it has no fatalities and does not make sense in context of dropping GDR
cm_features <- cm_features %>%
  filter(country_id != 185)
# Drop People's Republic Yemen as it has 49% of 0 values
cm_features <- cm_features %>%
  filter(country_id != 197)
# drop Yemen (Arab Republic of Yemen). 25% missing values but make no sense in context of dropping PR of Yemen
# but only until 1990-05-01. Later it's good cause it was joined.
cm_features <- cm_features %>%
  filter(!(country_id == 196 & date <= as.Date("1990-05-01")))


# NOTE on Yemen: The present Republic of Yemen came into being in May 1990, when the Yemen Arab Republic (North Yemen)
# merged with the People's Democratic Republic of Yemen (South Yemen).

# Note: Normal Germany starts at 1990-10-01 with country_id 184

# save to csv
cm_features %>% write_csv("data/cm_features_v2.3.csv")
print("Saved to CSV")


cm_features_with_missing_percentage <- cm_features %>%
  rowwise() %>%
  mutate(Zero_Percentage = sum(across(all_of(columns_to_check), ~.x == 0)) / length(columns_to_check)) %>%
  ungroup()


print("Done!")
print("Unique country ids:")
print(length(unique(cm_features$country_id)))