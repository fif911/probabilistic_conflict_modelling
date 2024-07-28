# Adds ccode ID to the dataset based on GW id; Removes countries that have
library(magrittr)
library(tidyverse) # load this first for most/all things
library(peacesciencer) # the package of interest
library(stevemisc) # a dependency, but also used for standardizing variables for better interpretation
library(tictoc)
library(countrycode)
library(ggplot2)

# read cm_features.csv
cm_features <- read_csv("data/cm_features_v2.1.csv")
# read translation table
translation_table <- read_csv("data/translation_table_cow_gw_name.csv")
# prolong the translation table to 2022. the last year is 2020. Just copy the 2020 data and add 2021 and 2022

translation_table_2020 <- translation_table %>%
  filter(year == 2020)
translation_table_2021 <- translation_table_2020 %>%
  mutate(year = 2021)
translation_table_2022 <- translation_table_2020 %>%
  mutate(year = 2022)
translation_table_2023 <- translation_table_2020 %>%
  mutate(year = 2023)
translation_table_2024 <- translation_table_2020 %>%
  mutate(year = 2024)
translation_table_2025 <- translation_table_2020 %>%
  mutate(year = 2025)
translation_table <- rbind(translation_table, translation_table_2021, translation_table_2022, translation_table_2023, translation_table_2024, translation_table_2025)
# remove from memory
rm(translation_table_2020, translation_table_2021, translation_table_2022, translation_table_2023, translation_table_2024, translation_table_2025)



# add year column to cm_features based on date
cm_features <- cm_features %>%
  mutate(year = year(date))
# left_join(translation_table, by = c("gleditsch_ward" = "gwcode", "year" = "year"))

# based on the translation table, add the COW country codes to the cm_features data
# cm_features gw code column is called gleditsch_ward and the translation table gw code column is called gwcode
# account for year
# add only ccode column
cm_features <- cm_features %>%
  left_join(translation_table %>% select(gwcode, year, ccode, gw_statename), by = c("gleditsch_ward" = "gwcode", "year" = "year"))

# THERE ARE 6 COUNTRIES WITH NO CCODE AND GED_OS NOT 0 or acled_os NOT 0
# all of them have high percentage of missing values, except for a 1 (country_id 80)
# TODO: check what country is this
filtered_cm_features <- cm_features %>%
  filter(is.na(ccode) & (acled_os != 0 | ged_os != 0))

# FILTER the rows with no ccode and ged_os = 0. Usually just small islands
# print country IDs of states that do not have a ccode
# 5   6  19  20  21  22  35  36  77  80 152 153 174 178 181 182  86 144
print(unique((cm_features %>%
  filter(is.na(ccode)))$country_id))

# Remove all countries that do not have CCODE
cm_features_filtered_with_ccode <- cm_features %>%
  filter(!(is.na(ccode)))


# save to csv
cm_features_filtered_with_ccode %>% write_csv("data/cm_features_v2.2.csv")
read_cm_features_filtered_with_ccode <- read_csv("data/cm_features_v2.2.csv")

# check if the shape is the same
print(dim(cm_features_filtered_with_ccode))
print(dim(read_cm_features_filtered_with_ccode))


print("Done!")