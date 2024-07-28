# from 0.4 to 1.0 cm features version
# adds region and region23 to cm_features, one-hot encodes them, and fixes Germ Fed Rep ccode
library(magrittr)
library(tidyverse) # load this first for most/all things
library(peacesciencer) # the package of interest
library(stevemisc) # a dependency, but also used for standardizing variables for better interpretation
library(tictoc)
library(countrycode)
library(ggplot2)
library(fastDummies) # added for one-hot encoding

# read cm_features.csv
cm_features <- read_csv("data/cm_features_v2.3.csv")

# For Germ Fed Rep that has ccode 260, set the ccode to 255
cm_features <- cm_features %>%
  mutate(ccode = ifelse(ccode == 260, 255, ccode))

# based on countrycode package, add region to cm_features data
cm_features <- cm_features %>%
  mutate(region = countrycode::countrycode(ccode, origin = "cown", destination = "region")) %>%
  mutate(region23 = countrycode::countrycode(ccode, origin = "cown", destination = "region23"))

# set Southern Europe for Serbia ccode 345 for region23
cm_features <- cm_features %>%
  mutate(region23 = ifelse(ccode == 345, "Southern Europe", region23))

# get unique regions
unique_regions7 <- unique(cm_features$region)
unique_regions23 <- unique(cm_features$region23)

# print rows with NA region
rows_with_na_region <- cm_features %>%
  filter(is.na(region))
rows_with_na_region23 <- cm_features %>%
  filter(is.na(region23))


# assert there are no rows with NA region
stopifnot(nrow(rows_with_na_region) == 0)
stopifnot(nrow(rows_with_na_region23) == 0)
# assert there are no rows with NA ged_sb
stopifnot(sum(is.na(cm_features$ged_sb)) == 0)

# One hot encoding for region and region23, drop the first dummy to avoid multicollinearity
cm_features <- dummy_cols(cm_features, select_columns = c("region", "region23"), remove_selected_columns = FALSE, remove_first_dummy = TRUE)


print("Success")
cm_features %>% write_csv("data/cm_features_v2.4.csv")
print("Saved to data/cm_features_v2.4.csv")