import datetime
import itertools
import logging
import os
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq
import yaml

logging.getLogger(__name__)

TargetType = Literal["cm", "pgm"]


def get_submission_details(submission: str | os.PathLike) -> dict:
    """Reads the submission_details.yml file in a submission and returns a dictionary with its items.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template

    Returns
    -------
    dict
        A dictionary with the elements in the YAML file.
    """
    with open(submission / "submission_details.yml") as f:
        submission_details = yaml.safe_load(f)
    return submission_details


def date_to_views_month_id(date: datetime.date) -> int:
    """Takes a date and converts it to a "month_id", which is a count of months starting at 1 on January 1980. Works on vectors/columns of dates.

    Parameters
    ----------
    date : datetime.date

    Returns
    -------
    int
        The month_id of the given date.
    """
    views_start_date = datetime.date(1980, 1, 1)
    r = relativedelta(date, views_start_date)
    return (r.years * 12) + r.months + 1


def views_month_id_to_year(month_id: int) -> int:
    """Converts a month_id to the calendar year of the month_id. Works on vectors/columns of month_ids.

    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.

    Returns
    -------
    int
        The calendar year of the month_id.
    """
    return 1980 + (month_id - 1) // 12


def views_month_id_to_month(month_id: int) -> int:
    """Converts a month_id to the calendar month of the month_id

    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.

    Returns
    -------
    int
        The calendar month of the month_id.
    """
    return ((month_id - 1) % 12) + 1


def views_month_id_to_date(month_id: int) -> datetime.date:
    """Converts a month_id to a datetime.date format.

    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.

    Returns
    -------
    datetime.date
        The date of the month_id, using first day of the month format.
    """
    df = pd.DataFrame()
    df["year"] = views_month_id_to_year(month_id)
    df["month"] = views_month_id_to_month(month_id)
    df["day"] = 1
    return pd.to_datetime(df, format="%Y%M")


def migrate_submission_from_old(submission: str | os.PathLike) -> None:
    """Helper function to migrate old submission folder structure to new structure based on Apache Hive.

    Parameters
    ----------
    submission : str | os.PathLike
        Folder following the old submission_template setup
    """

    submission = Path(submission)
    eval_data = itertools.chain(
        submission.glob(f"**/cm/*/eval/*.parquet"),
        submission.glob(f"**/pgm/*/eval/*.parquet"),
    )

    def folder_rename(d):
        new_name = d.parent / f'window=Y{d.name.split("_")[-1]}'
        d.rename(new_name)

    # Rename window folders in Apache Hive format
    window_folders = itertools.chain(
        submission.glob(f"cm/test_window_*"), submission.glob(f"pgm/test_window_*")
    )
    [folder_rename(d) for d in window_folders]
    # Delete evaluation files (must be re-estimated)
    [f.unlink() for f in eval_data]
    [f.unlink() for f in submission.glob(f"eval*.parquet")]

    # Cleanup old folders (only deletes if empty)
    [d.rmdir() for d in submission.glob(f"**/cm/*/eval/")]
    [d.rmdir() for d in submission.glob(f"**/pgm/*/eval/")]


def list_submissions(submissions_folder: str | os.PathLike) -> list[os.PathLike]:
    """Creates a list of paths to folders inside the submissions_folder.

    Parameters
    ----------
    submissions_folder : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template

    Returns
    -------
    list[os.PathLike]
        List of paths to submissions.
    """
    submissions_folder = Path(submissions_folder)
    return [
        submission
        for submission in submissions_folder.iterdir()
        if submission.is_dir() and not submission.stem == "__MACOSX"
    ]


def read_parquet(data_path: str | os.PathLike, filters=None) -> pd.DataFrame:
    """This function does not need to be used directly, it is mostly to document how to read folders with parquet files or single parquet files with a filter.

    Notes
    -----

    See https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions for filter examples

    Examples
    --------
    >>> import pyarrow.compute as pac
    >>> import pyarrow.parquet as pq
    >>> filter = pac.field("year") >= 2017
    >>> df = pq.ParquetDataset(path_to_folder_with_apache_hive_structured_parquet_files, filters = filter).read().to_pandas()

    """

    table = pq.ParquetDataset(data_path, filters=filters)
    print(f"filters: {filters}")
    print(table.schema)
    return table.read().to_pandas()


def is_parquet_in_target(submission: str | os.PathLike, target: TargetType) -> bool:
    """Test if there are any .parquet-files in the {submission}/{target} sub-folders.

    Parameters
    ----------
    submission : str | os.PathLike
        Path to a folder only containing folders structured like a submission_template
    target : TargetType
        A string, either "pgm" for PRIO-GRID-months, or "cm" for country-months.

    Returns
    -------
    bool
        True if there are any .parquet files in target sub-folders.
    """
    return any((submission / target).glob("**/*.parquet"))


def get_target_data(
        submission: str | os.PathLike, target: TargetType, filters=None
) -> pd.DataFrame:
    """Reads folders with a "pgm" or "cm" sub-folder containing Apache Hive structured parquet-files.

    Notes
    -----
    See https://arrow.apache.org/docs/python/compute.html#filtering-by-expressions for filter examples.

    Examples
    --------
    >>> import pyarrow.compute as pac
    >>> import pyarrow.parquet as pq
    >>> from utilities import list_submissions, get_target_data

    >>> filter = pac.field("year") >= 2017
    >>> subs = list_submissions(./submissions/")
    >>> df = get_target_data(subs[0], target = "pgm", filters = filter)

    """
    submission = Path(submission)
    print("get_target_data")
    print(submission)
    print(submission / target)
    # if target.endswith(".csv"):
    #     return pd.read_csv(submission / target)
    # elif target.endswith(".parquet"):
    #     return read_parquet(submission / target, filters=filters)
    # else:
    return read_parquet(submission / target, filters=filters)


def get_window_filters(window):
    pass


import pandas as pd


def aggregate_country_month_data(dyad_df):
    """
    Aggregates a dyad DataFrame by country and month, including specified shifted values for ged_sb.

    Parameters:
    - dyad_df: The input DataFrame containing dyadic data.
    - a_ged_sb_shifted_name: The column name for shifted a_ged_sb values.
    - b_ged_sb_shifted_name: The column name for shifted b_ged_sb values.

    Returns:
    - A DataFrame aggregated by month and country, including mean ged_sb and shifted ged_sb values.
    """
    # Step 1: Prepare separate DataFrames for each country role
    df_a = dyad_df[['month_id', 'country_id_a', 'a_country_name', 'a_ged_sb_15_shifted', 'a_ged_sb_pred']].copy()
    df_b = dyad_df[['month_id', 'country_id_b', 'b_country_name', 'b_ged_sb_15_shifted', 'b_ged_sb_pred']].copy()

    # Step 2: Rename columns for consistency
    df_a.rename(columns={'country_id_a': 'country_id', 'a_country_name': 'country_name',
                         'a_ged_sb_15_shifted': 'ged_sb_actual', 'a_ged_sb_pred': 'ged_sb_pred'}, inplace=True)
    df_b.rename(columns={'country_id_b': 'country_id', 'b_country_name': 'country_name',
                         'b_ged_sb_15_shifted': 'ged_sb_actual', 'b_ged_sb_pred': 'ged_sb_pred'}, inplace=True)

    # Step 3: Concatenate and group
    combined_country_df = pd.concat([df_a, df_b], ignore_index=True)
    aggregated_df = combined_country_df.groupby(['month_id', 'country_id', 'country_name']).agg(
        {'ged_sb': 'mean', 'y_shifted': 'mean'}).reset_index()

    return aggregated_df
