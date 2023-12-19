import warnings

import pandas as pd
import seaborn as sns
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)


class Metrics:
    """ Class containing the metrics of a classifier.

    Attributes:
        classifier_name (str): the name of the classifier.
        metrics (list): a list of Metrics.Metric objects.
    """

    class Metric:
        """ Class containing the name and the score of a metric.

        Attributes:
            name (str): the name of the metric.
            score (float): the score of the metric.
        """

        def __init__(self, name: str, score: float):
            self.name = name
            self.score = score

    def __init__(self, classifier_name):
        """ Constructor for the Metrics object.

        :param classifier (sklearn classifier): the classifier to evaluate.
        :param data (Data): the data to use for the evaluation.
        """

        self.classifier_name = classifier_name
        self.metrics = []

    def __str__(self):
        """ String representation of the Metrics object.

        :return: the string representation of the Metrics object.
        """

        value: str = 'Metric scores of classifier ' + self.classifier_name + ' : '
        for metric in self.metrics:
            value += f'\n\t{metric.name} : {metric.score}'

        return value

    def add_metric(self, name: str, score: float):
        """ Add a metric to the metrics list.

        :param name (str): the name of the metric.
        :param score (float): the score of the metric.
        """

        self.metrics.append(Metrics.Metric(name, score))

    def get_dataframe(self) -> pd.DataFrame:
        """ Get the metrics as a pandas dataframe.

        :return: the metrics as a pandas dataframe. With the following columns : 'Classifiers', 'Metrics', 'Scores'.
        """

        data = []
        for metric in self.metrics:
            data.append([self.classifier_name, metric.name, metric.score])

        return pd.DataFrame(data, columns=['Classifiers', 'Metrics', 'Scores'])

    @staticmethod
    def show_metrics_list(metrics_list):
        """ Show the metrics as a bar plot.

        :param [Metrics] metrics_list: a list of Metrics objects.
        """

        metrics_list: [Metrics] = metrics_list
        
        data = pd.DataFrame()
        for metrics in metrics_list:
            data = pd.concat([data, metrics.get_dataframe()])

        ax = sns.barplot(data=data, y='Classifiers', x='Scores', hue='Metrics')
        ax.set(xlim=(0, 1.09))
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

        if len(metrics_list) > 5:
            plt.rc('font', size=5)
            
        for bars in ax.containers:
            ax.bar_label(bars, fmt='%.3f')

        plt.rc('font', size=12)
