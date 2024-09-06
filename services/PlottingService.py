import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import shap
from CRPS import CRPS as pscore
from dateutil.relativedelta import relativedelta
from ngboost.distns import Normal
from services.ModelEvaluationService import model_evaluation_service
from services.DataTransformationService import data_transformation_service


class PlottingService:
    vector_length: int = 1000

    def plot_prediction_and_actuals_histograms(
        self,
        data: pd.DataFrame,
        target_column: str,
        indices_to_plot: list[int],
        show_plots: bool,
        save_figures: bool,
        directory: str,
    ) -> None:
        plt.rcParams.update({"font.size": 10})

        for index_id in indices_to_plot:
            row: pd.DataFrame = data.iloc[[index_id], ::]

            actual_pred = row[target_column].values[0]

            hist_data_temp = row.iloc[::, -self.vector_length :].values[0]
            # keep only samples that are in 95% confidence interval

            # Plot histogram of the sampled predictions using plt
            plt.figure(figsize=(10, 6))
            plt.hist(hist_data_temp, bins=50, alpha=0.7, label="NGB Predictions")

            # Plot vertical lines for actual value, mean value, and other relevant predictions
            plt.axvline(
                x=actual_pred,
                color="black",
                linestyle="dashed",
                linewidth=2,
                label="Actual Value",
            )
            plt.axvline(
                x=row["ngb_predictions"].values[0],
                color="blue",
                linestyle="dashed",
                linewidth=2,
                label="Mean Value (NGB)",
            )

            formatted_date = row["date"].dt.strftime("%Y-%m").values[0]
            country_name: str = row["country_name"].values[0]
            plt.title(
                f'Histogram of NGB Predictions and Actuals for {country_name} on {formatted_date} (month_id: {int(row["month_id"].values[0])})'
            )
            plt.xlabel("Predicted Value")
            plt.ylabel("Frequency")
            plt.legend()

            if save_figures:
                plt.savefig(
                    f'{directory}/prediction_and_actuals_for_{country_name}_month_id_{int(row["month_id"].values[0])}.png',
                    dpi=300,
                )

            if show_plots:
                plt.show()

        if not show_plots:
            plt.close("all")

    def plot_line_plots(
        self,
        country_list: pd.DataFrame,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_column: str,
        prediction_window: int,
        cutoff_date,  # Timestamp
        log_transform: bool,
        plot_std: bool,
        show_plots: bool,
        save_figures: bool,
        directory: str,
    ):
        plt.rcParams.update({"font.size": 10})

        image_count: int = 1
        country_ids: list[int] = test["country_id"].unique().tolist()
        countries: list[str] = []

        # Settings
        num_plots_per_figure: int = 4
        plt.figure(figsize=(15, 10))  # New figure
        plots_added: int = 0

        # Continue looping until all countries have been considered
        max_date_train = pd.to_datetime(train["date"].max())
        min_date_test = pd.to_datetime(test["date"].min())

        # 1 year buffer because of validation set
        expected_min_date_test = max_date_train + relativedelta(
            years=0, months=prediction_window + 3
        )  # 15 is window size + 1 is from Sep to Oct

        # assert the different is exactly 15 months
        assert min_date_test == expected_min_date_test

        for index, country_id in enumerate(country_ids):
            this_country_test: pd.DataFrame = test.loc[
                test["country_id"] == country_id, ::
            ]
            this_country_train: pd.DataFrame = train.loc[
                train["country_id"] == country_id, ::
            ].tail(24)

            if log_transform:
                this_country_target = this_country_test["original_" + target_column][
                    this_country_test["date"] <= cutoff_date
                ]
                predictions_vector = this_country_test[
                    "ngb_predictions_inverselog"
                ].copy()
                predictions_vector[predictions_vector < 0] = 0
            else:
                this_country_target = this_country_test[target_column][
                    this_country_test["date"] <= cutoff_date
                ]
                predictions_vector = this_country_test["ngb_predictions"]

            std_vector = (
                this_country_test["ngb_predictions_std"]
                if not log_transform
                else this_country_test["ngb_predictions_inverselog_std"]
            )

            country_name: str = (
                country_list.loc[country_list["country_id"] == country_id, ::]
                .loc[::, "name"]
                .values[0]
            )

            # Check if country should be skipped due to no data
            if this_country_test[target_column].sum() == 0:
                if show_plots:
                    print(f"Skipping {country_name} as all actual are 0")
                continue

            # Prepare the subplot for non-skipped countries
            plt.subplot(2, 2, plots_added + 1)

            # Plotting data
            plt.plot(
                this_country_train["date"],
                this_country_train["ged_sb"],
                label="Train",
                color="gray",
                linestyle="-",
                marker="",
            )
            plt.plot(
                this_country_test["date"].apply(lambda x: x - relativedelta(months=14)),
                this_country_test["ged_sb"],
                label=f"Test Input",
                color="black",
                linestyle="--",
                marker="",
            )
            plt.plot(
                this_country_test["date"][this_country_test["date"] <= cutoff_date],
                this_country_target,
                label=f"Actual",
                color="black",
                linestyle="-",
                marker="",
            )
            plt.plot(
                this_country_test["date"],
                predictions_vector,
                label=f"Model Output",
                color="blue",
                linestyle="-",
                marker="",
            )

            # plot std
            if plot_std:
                plt.fill_between(
                    this_country_test["date"],
                    predictions_vector - std_vector,
                    predictions_vector + std_vector,
                    color="blue",
                    alpha=0.2,
                )

            # Adding title and labels
            plt.title(f"{country_name} Actual vs Predicted")
            plt.xlabel("Date")

            # turn dates 90 degrees
            plt.xticks(rotation=45)

            # make ticks more readable
            plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m"))

            # add vertical lines for the training and testing split
            plt.axvline(
                x=min_date_test, color="gray", linestyle="--", label="16 months gap"
            )
            plt.axvline(x=max_date_train, color="gray", linestyle="--")

            plt.ylabel("Number of fatalities")
            plt.legend()

            # add light grid
            plt.grid(alpha=0.3)

            # Increment counters
            plots_added += 1
            countries.append(country_name)

            if plots_added % num_plots_per_figure == 0 or index == len(country_ids) - 1:
                # Adjust layout and display the figure
                plt.tight_layout()

                if save_figures:
                    plt.savefig(
                        f"{directory}/line_plot"
                        + "".join([f"_{country}" for country in countries])
                        + ".png",
                        dpi=300,
                    )
                    image_count += 1

                if show_plots:
                    plt.show()

                plt.figure(figsize=(15, 10))  # New figure
                plots_added = 0
                countries = []

        if save_figures and len(countries) != 0:
            plt.savefig(
                f"{directory}/line_plot"
                + "".join([f"_{country}" for country in countries])
                + ".png",
                dpi=300,
            )

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    def plot_predictions_distribution(
        self, distributions: np.ndarray, index: int, show_plots: bool
    ) -> None:
        plt.rcParams.update({"font.size": 10})

        plt.figure(figsize=(10, 6))
        plt.hist(distributions[index], bins=50, alpha=0.7, label="NGB Predictions")

        plt.title("Histogram of NGB Predictions")
        plt.xlabel("Predicted Value")
        plt.ylabel("Frequency")
        plt.legend()

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    def shap_force_plot_countries(
        self,
        explainer,
        shap_values,
        initial_data: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        countries: list[str],
        show_plots: bool,
        save_figures: bool,
        directory: str,
    ) -> None:
        plt.rcParams.update({"font.size": 10})
        
        for country_name in countries:
            indexes_for_country: list[int] = initial_data.loc[
                initial_data["country_name"].str.contains(country_name), ::
            ].index.tolist()

            for m, index in enumerate(indexes_for_country, start=1):
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[index],
                    X.iloc[index],
                    matplotlib=True,
                    show=False,
                )
                # add vectical of true value
                # plt.axvline(x=y.iloc[index], color='red', linestyle='--')
                plt.legend([f"Actual: {y.iloc[index]}"], loc="upper left")

                if save_figures:
                    plt.savefig(
                        f"{directory}/NGBoost_shap_fp_{country_name}_{m}_{index}.png",
                        dpi=300,
                    )

                if show_plots:
                    plt.show()
                    plt.close()

                plt.close()

        if not show_plots:
            plt.close("all")

    def shap_summary_plot(
        self,
        shap_values,
        data: pd.DataFrame,
        max_display: int,
        show_plots: bool,
        save_figures: bool,
        path: str,
    ) -> None:
        plt.rcParams.update({"font.size": 10})

        shap.initjs()
        shap.summary_plot(
            shap_values,
            data,
            feature_names=data.columns,
            show=False,
            max_display=max_display,
        )
        if save_figures:
            plt.savefig(path, dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close("all")

    def predictions_plot(
        self,
        data: pd.DataFrame,
        y: pd.Series,
        sampled_dist: np.ndarray,
        target_column: str,
        unique_months: list[int],
        grid_size_x: int,
        grid_size_y: int,
        plot_std: bool,
        inverse_log_transform: bool,
        save_figures: bool,
        path: str,
    ) -> None:
        plt.rcParams.update({"font.size": 11})

        # Set overall figure size
        plt.figure(figsize=(grid_size_x * 6, grid_size_y * 3))

        for index, month_id in enumerate(unique_months, start=1):
            this_month: pd.DataFrame = data.loc[data["month_id"] == month_id, ::]

            current_date = this_month["date"].iloc[0]

            if not inverse_log_transform:
                this_month_target = this_month[target_column]
                this_month_predictions = this_month["ngb_predictions"]
            else:
                this_month_target = data_transformation_service.inverse_log_transform(
                    this_month[target_column]
                )
                this_month_predictions = (
                    data_transformation_service.inverse_log_transform(
                        this_month["ngb_predictions"]
                    )
                )

            # get this month max and min index
            month_start_index = this_month.index.min()
            month_end_index = this_month.index.max()

            mean_crps = model_evaluation_service.calculate_crps(
                sampled_dist,
                y,
                month_start_index,
                month_end_index,
                inverse_log_transform,
            )

            # Create subplot for current month
            plt.subplot(grid_size_x, grid_size_y, index)
            plt.scatter(
                this_month_target,
                this_month_predictions,
                color="blue",
                label="Actual vs Predicted",
                alpha=0.5,
            )

            if plot_std:
                if not inverse_log_transform:
                    this_month_std = this_month["ngb_predictions_std"]
                else:
                    this_month_std = this_month["ngb_predictions_inverselog_std"]

                plt.errorbar(
                    this_month_target,
                    this_month_predictions,
                    yerr=this_month_std,
                    fmt="o",
                    color="blue",
                    alpha=0.5,
                )

            # print current_date in YY/MM format
            print_date = current_date.strftime("%Y-%m")

            plt.title(f"Month: {print_date}; mean CRPS: {mean_crps:.2f}")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")

            line_length = max(this_month_target.max(), this_month_predictions.max())

            plt.plot(
                [0, line_length], [0, line_length], color="red", label="45 degree line"
            )
            plt.legend()
            plt.xticks(rotation=45)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save_figures:
            plt.savefig(path, dpi=300)

        plt.close("all")

    def plot_cut_comparison(
        self,
        instance_to_plot: int,
        data: pd.DataFrame,
        new_predictions: pd.DataFrame,
        target_column: str,
        prediction_window: int,
        show_plots: bool,
        save_figures: bool,
        directory: str,
    ) -> None:
        plt.rcParams.update({"font.size": 18})

        # hist_color = 'blue'
        mean_line_color: str = "blue"
        actual_value_color: str = "black"

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        raw_predictions = data.iloc[instance_to_plot, -self.vector_length :]
        processed_predictions = new_predictions.loc[
            (
                data.iloc[instance_to_plot]["month_id"] + prediction_window,
                data.iloc[instance_to_plot]["country_id"],
            )
        ][["outcome"]]
        metadata = data.iloc[instance_to_plot][
            ["month_id", "country_id", "country_name", "date", target_column]
        ]
        formatted_date = metadata["date"].strftime("%Y-%m")

        # Set title for the row of histograms
        # fig.suptitle(f'{metadata["country_name"]} on {formatted_date} (month_id: {int(metadata["month_id"])})', fontsize=16)

        # Raw predictions histogram
        raw_predictions = raw_predictions.apply(pd.to_numeric, errors="coerce")
        raw_prediction_crps = pscore(
            raw_predictions, metadata[target_column]
        ).compute()[0]

        axs[0].hist(raw_predictions, bins=30, alpha=0.7)
        axs[0].axvline(
            raw_predictions.mean(),
            color=mean_line_color,
            linestyle="dashed",
            linewidth=2,
        )
        axs[0].axvline(metadata[target_column], color=actual_value_color, linewidth=2)
        axs[0].set_title(
            f"Histogram of Raw Predictions\nCRPS: {raw_prediction_crps:.2f}"
        )
        axs[0].set_xlabel("Predicted Values")
        axs[0].set_ylabel("Frequency")
        axs[0].legend(
            [
                f"Mean Value {raw_predictions.mean():.2f}",
                f"Actual Value {metadata[target_column]}",
            ]
        )

        # Processed predictions histogram
        processed_predictions = processed_predictions.apply(
            pd.to_numeric, errors="coerce"
        )
        processed_prediction_crps = pscore(
            processed_predictions["outcome"], metadata[target_column]
        ).compute()[0]

        axs[1].hist(processed_predictions, bins=30, alpha=0.7)
        axs[1].axvline(
            processed_predictions.mean().iloc[0],
            color=mean_line_color,
            linestyle="dashed",
            linewidth=2,
        )
        axs[1].axvline(metadata[target_column], color=actual_value_color)
        axs[1].set_title(
            f"Histogram of Resampled Predictions\nCRPS: {processed_prediction_crps:.2f}"
        )
        axs[1].set_xlabel("Predicted Values")
        axs[1].legend(
            [
                f"Mean Value {processed_predictions.mean().iloc[0]:.2f}",
                f"Actual Value {metadata[target_column]}",
            ]
        )

        # Raw predictions histogram with non-negative values
        raw_predictions_non_neg = raw_predictions.clip(lower=0)
        raw_prediction_non_neg_crps = pscore(
            raw_predictions_non_neg, metadata[target_column]
        ).compute()[0]
        axs[2].hist(raw_predictions_non_neg, bins=30, alpha=0.7)
        axs[2].axvline(
            raw_predictions_non_neg.mean(),
            color=mean_line_color,
            linestyle="dashed",
            linewidth=2,
        )
        axs[2].axvline(metadata[target_column], color=actual_value_color)
        axs[2].set_title(
            f"Histogram of Clipped Predictions\nCRPS: {raw_prediction_non_neg_crps:.2f}"
        )
        axs[2].set_xlabel("Predicted Values")
        axs[2].legend(
            [
                f"Mean Value {raw_predictions_non_neg.mean():.2f}",
                f"Actual Value {metadata[target_column]}",
            ]
        )

        plt.tight_layout()

        if save_figures:
            plt.savefig(
                f'{directory}/histograms_{metadata["country_name"]}_{formatted_date}.png'
            )

        if show_plots:
            plt.show()
        else:
            plt.close("all")


plotting_service = PlottingService()
