# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy import stats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint



def plot_data(
    pairs,
    ax,
    metric_name,
    outcome_name,
    data_points_per_bucket,
    fontsize=15,
    show_confidence_intervals=True,
    metric_label_reformatter=lambda x: x
):

    pairs = sorted(pairs, key=lambda x: x[0])
    metric_values, results = zip(*pairs)

    num_datapoints = len(pairs)
    num_buckets = num_datapoints // data_points_per_bucket
    if num_buckets < 2:
        return (0.0, 1.0)
    bucket_size = num_datapoints // num_buckets

    # Show the distribution of values in the background
    hist, bin_edges = np.histogram(metric_values)
    hist_normalized = [x * 1.0 / max(hist) for x in hist]
    ax.bar(
        x=bin_edges[:-1],
        height=hist_normalized,
        width=np.array(bin_edges[1:] - bin_edges[:-1]),
        alpha=0.5,
        label="Distribition of metric in corpus"
    )

    # Bucketize data points by the value of the metric and plot avg outcome for each bucket
    # If we set data_points_per_bucket to 1, we'll get a scatter plot instead.
    mean_metrics = []
    accuracies = []
    ci_bars = []
    for i in range(num_buckets):
        mean_metrics.append(
            sum(metric_values[i * bucket_size : (i + 1) * bucket_size])
            * 1.0
            / bucket_size
        )

        num_correct = sum(float(x) for x in results[i * bucket_size : (i + 1) * bucket_size]) * 1.0
        accuracy = num_correct / bucket_size
        ci = proportion_confint(count=num_correct, nobs=bucket_size, alpha=0.05)
        accuracies.append(accuracy)
        ci_bars.append(ci[1] - accuracy)

    if data_points_per_bucket > 1:
        if show_confidence_intervals:
            ax.errorbar(
                x=mean_metrics, y=accuracies,
                yerr=ci_bars,
                capsize=2,
                label=f"Bucketized {outcome_name} (5% binomial CI)",
            )
        else:
            ax.plot(mean_metrics, accuracies, label=f"Bucketized {outcome_name}")
    else:
        ax.scatter(mean_metrics, accuracies, label=f"Outcome ({outcome_name}) vs Metric ({metric_name})")

    # Show Pearson correlation between metric values and outcomes.
    # Color the title green if positive, red if negative, black if neutral
    coeff, p = stats.pearsonr(mean_metrics, accuracies)
    color = (0, 0, 0, 1) if np.isnan(coeff) else (max(-coeff, 0), max(coeff, 0), 0, 1.0) 

    ax.set_title(
        f"[Pearson Correlation: {coeff:.2f}]",
        fontdict={"color": color},
        fontsize=fontsize,
    )
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(f"Metric: {metric_label_reformatter(metric_name)}", fontsize=fontsize)
    ax.set_ylabel(f"Outcome: {outcome_name}", fontsize=fontsize)

    return coeff, p


def plot_metric_distributions_and_correlations(
    metric_outcome_pairs,
    n_columns=3,
    data_points_per_bucket=100,
    figsize=(7, 7),
    fontsize=15,
    savefig_path=None,
    metric_label_reformatter=lambda x: x
):

    n_metrics = len(metric_outcome_pairs)
    n_columns = min(n_metrics, n_columns)  # Adjust number of columns if we can't even fill up one row
    n_rows = (n_metrics - 1) // n_columns + 1
    fig, axs = plt.subplots(n_rows , n_columns, figsize=(n_columns * figsize[0], n_rows * figsize[1]), squeeze=False)

    cnt = 0
    for (metric_name, outcome_name), pairs in metric_outcome_pairs.items():
        row = cnt // n_columns
        col = cnt % n_columns
        try:
            plot_data(
                pairs,
                axs[row][col],
                metric_name,
                outcome_name,
                data_points_per_bucket=data_points_per_bucket,
                fontsize=fontsize,
                metric_label_reformatter=metric_label_reformatter
            )
            cnt += 1
        except Exception as e:
            print(f"Metric {metric_name} failed to render: {e}")

    fig.show()

    if savefig_path is not None:
        plt.savefig(savefig_path, facecolor='white', transparent=False)


def show_pairwise_metric_correlations(metrics_df):
    plt.figure(figsize=(30, 30))
    sns.heatmap(
        metrics_df.corr().sort_index().sort_index(axis = 1),
        annot=True,
        cmap='PiYG',
        center=0,
    )

class PredictFromCharacteristicsAnalysis:
    def __init__(self, characteristics_df, outcomes_df, model_type="logistic_regression", model_args=None):

        #self.characteristics_df = characteristics_df.drop(columns=["text_key"])
        self.characteristics_df = characteristics_df.pivot(columns='text_key')
        
        self.outcomes_df = outcomes_df
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.characteristics_df,
            self.outcomes_df
        )
        
        # We will fit a model for each column in outcomes
        self.predictive_models = {
            col: self.init_model(model_type)
            for col in outcomes_df
        }
        
        self.model_type = model_type
        
        print(f"Analysis initialized! There are:\n {len(self.X_train.columns)} features available\n {len(self.X_train)} samples for fitting a predictive model\n {len(self.X_test)} to evaluate the fit.")
        
    
    def show_individual_metric_correlations(
        self, 
        metrics=None,
        outcome_cols=None,
        n_columns=5,
        data_points_per_bucket=100,
        figsize=(7, 7),
        fontsize=16,
        metric_label_reformatter=lambda x: x,
        savefig_path=None,
    ):
        if metrics is None:
            metrics = self.characteristics_df.columns
        if outcome_cols is None:
            outcome_cols = self.outcomes_df.columns
        
        metric_outcome_pairs = {}
        for metric in metrics:
            for outcome_col in outcome_cols:
                metric_outcome_pairs[(str(metric), str(outcome_col))] = zip(
                    self.characteristics_df[metric],
                    self.outcomes_df[outcome_col],
                )
                
        plot_metric_distributions_and_correlations(
            metric_outcome_pairs,
            n_columns=n_columns,
            data_points_per_bucket=data_points_per_bucket,
            figsize=figsize,
            fontsize=fontsize,
            metric_label_reformatter=metric_label_reformatter,
            savefig_path=savefig_path,
        )
        
        
    def init_model(self, model_type, model_args=None):
            
        if model_type == "logistic_regression":

            # These are some default parameters we found to work well
            if model_args is None:
                model_args = {
                    "C": 0.1,
                    "penalty": "l1",
                    "solver": "liblinear",
                }

            return make_pipeline(
                StandardScaler(),
                LogisticRegression(**model_args)
            )

        elif model_type == "random_forest":

            # TODO not sure how universal these are
            if model_args is None:
                model_args = {
                    "max_depth": 4,
                    "min_samples_leaf": 100,
                }
            return RandomForestClassifier(**model_args)


        elif model_type == "linear_regression":
            # These are some default parameters we found to work well
            if model_args is None:
                model_args = {
                    "fit_intercept" : True
                }

            return make_pipeline(
                StandardScaler(),
                LinearRegression(**model_args)
            )
        
        else:
            raise NotImplementedError
            

    def fit_predictive_models(self, test_indices=None):
        
        for outcome_col, model in self.predictive_models.items():
            model.fit(self.X_train, self.y_train[outcome_col])
        
    def get_coefficients(self):
        coeffs = []
        for outcome_col, model in self.predictive_models.items():
            if isinstance(model.steps[1][1], LinearRegression):
                coeffs.append(
                    pd.Series(model.steps[1][1].coef_, index=self.characteristics_df.columns, name=outcome_col)
                )
            else:
                coeffs.append(
                    pd.Series(model.steps[1][1].coef_[0], index=self.characteristics_df.columns, name=outcome_col)
                )
        return pd.concat(coeffs, axis=1)
    

    def show_coefficients(self, num_rows=None, num_trees=10, figsize=(7, 7), savefig_path=None):
        
        if self.model_type == "logistic_regression" or self.model_type == "linear_regression":
            coeffs_df = self.get_coefficients()
            
            if num_rows is not None:
                coeffs_df["max_abs_val"] = -coeffs_df.abs().max(axis=1)
                coeffs_df = coeffs_df.sort_values(by=["max_abs_val"])[:num_rows]
                coeffs_df.drop(columns=["max_abs_val"], inplace=True)

            if figsize is None:
                figsize = (len(coeffs_df.columns)*5, len(coeffs_df.index)*0.3)

            plt.figure(figsize=figsize)

            plot = sns.heatmap(
                coeffs_df,
                annot=True,
                cmap='PiYG',
                center=0,
            )

            if savefig_path is not None:
                plot.get_figure().savefig(savefig_path, bbox_inches = "tight")
            
        elif self.model_type == "decision_trees":
            for outcome_col, model in self.predictive_models.items():
                    
                for tree in random.sample(model.estimators_, num_trees):
                    fig, ax = plt.subplots(1, 1, figsize=(num_trees * 5, 10))
                    plot_tree(
                        tree,
                        ax=ax,
                        fontsize=12,
                        feature_names=self.X_test.columns,
                        filled=True,
                        max_depth=3
                    )

        else:
            raise NotImplementedError
        

    def plot_predictictions(
        self,
        data_points_per_bucket=100,
        figsize=(7, 7),
        fontsize=16,
        savefig_path=None,
    ):
        
        metric_outcome_pairs = {}
        for outcome_col, model in self.predictive_models.items():
            metric_outcome_pairs[("Regression Score", outcome_col)] = zip(
                model.predict_proba(self.X_test)[:, 1],
                self.y_test[outcome_col],
            )

        plot_metric_distributions_and_correlations(
            metric_outcome_pairs,
            data_points_per_bucket=data_points_per_bucket,
            figsize=figsize,
            fontsize=fontsize,
            savefig_path=savefig_path,
        )
