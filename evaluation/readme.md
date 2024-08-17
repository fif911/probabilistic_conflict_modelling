# Evaluation

## Evaluation of the model

After the submission folder is created with corresponding submission file, the model needs to be evaluated to calculate
CRPS, IGN and MIS metrics.

To perform the evaluation run `evaluate_submissions.py` script with the following arguments (from the root project
folder):

```bash
python evaluation/evaluate_submissions.py -s ./submission -a ./actuals -e 1000 -t cm
```

This script will go over each submission file in the `./submission` folder and calculate the metrics for each of them
for each prediction year available.
It will create `eval` folder in for each of the submissions and save three metric files for each of the prediction
windows.

The resulting structure will look like this (assume `submission_1` has only 2018 and 2019 prediction windows):

```
submission
├── submission_1
|   ├── cm
│   │   ├── window=Y2018
│   │   │   ├── submission_1_Y2018.parquet
│   │   ├── window=Y2019
│   │   │   ├── submission_1_Y2019.parquet
│   ├── eval
│   │   ├── 2018
│   │   │   ├── crps.csv
│   │   │   ├── ign.csv
│   │   │   └── mis.csv
│   │   ├── 2019
│   │   │   ├── crps.csv
│   │   │   ├── ign.csv
│   │   │   └── mis.csv
```

## Comparing evaluations

To compare the evaluations of the models, run `compare_submissions.ipynb` notebook. It will read the evaluation files
from the `./submission` folder and make tables with the metrics for each of the prediction windows.

This will allow you to estimate which model for specific years and overall.