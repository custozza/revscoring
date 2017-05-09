import logging

from tabulate import tabulate

from ... import util
from .scaled_classification_matrix import ScaleClassificationMatrix

logger = logging.getLogger(__name__)


class ScaledPredictionStatistics(ScaledClassificationMatrix):
    FIELDS = ['match_rate', 'filter_rate',
              'precision', '!precision',
              'recall', '!recall',
              'accuracy', 'fpr',
              'f1', '!f1']
    """
    The set of available fields that can be requested via
    :func:`~revscoring.scoring.statistics.classification.LabelStatistics.get_stat`.
    """  # noqa

    def __init__(self, y_pred, y_trues, population_rate=None):
        """
        Construct a basic set of statistics about a classification matrix.

        :Parameters:
            y_pred : [ `bool` ]
                A sequence of predictions where `True` represents a matched
                observation for a specific label.
            y_trues : [ `bool` ]
                A sequence of labels where `True` represents a positive
                observation.
            population_rate : `float`
                The rate at which the observed class appears in the population.
                This value will be used to re-scale the number of y_trues
                across all metrics.
        """
        super().__init__(y_pred, y_trues, population_rate=population_rate)

    def format(self, *args, formatting="str", **kwargs):
        """
        Format the set of statistics in a useful way.
        """
        if formatting == "str":
            return self.format_str(*args, **kwargs)
        elif formatting == "json":
            return self.format_json(*args, **kwargs)
        else:
            raise ValueError("Unknown formatting {0!r}".format(formatting))

    def format_json(self, path_tree, ndigits=3, **kwargs):
        return {stat_name: util.round(self[stat_name], ndigits)
                for stat_name in path_tree.keys() or self.FIELDS}

    def format_str(self, path_tree, ndigits=3, **kwargs):
        table_data = [[util.round(self[stat_name], ndigits)
                       for stat_name in path_tree.keys() or self.FIELDS]]
        return tabulate(table_data, headers=path_tree.keys() or self.FIELDS)

    def __getitem__(self, stat_name):
        """
        Gets a statistic based on a name.  E.g. "!recall" will call the right
        method.
        """
        method_name = stat_name.replace("!", "_")
        if stat_name in self:
            return self[stat_name]
        elif hasattr(self, method_name):
            return getattr(self, method_name)()
        else:
            raise KeyError(stat_name)

    def match_rate(self):
        """
        The proportion of observations that are matched in prediction.

            match-rate = positives / n
        """
        return (self.positives / self.n) if self.n is not 0 else None

    def filter_rate(self):
        """
        The proportion of observations that are not matched.

            filter-rate = 1 - match-rate
        """
        return (1 - self.match_rate()) \
               if self.match_rate() is not None else None

    def accuracy(self):
        """
        The proportion of predictions that were right.

            accuracy = correct / n
        """
        return (self.correct / self.n) if self.n != 0 else None

    def recall(self):
        """
        The proportion of the target class that the classifier matches.
        AKA "true-positive rate" and "sensitivity".

            recall = true-positives / target-class
        """
        return (self.tp / self.trues) if self.trues != 0 else None

    def _recall(self):
        """
        The inverse recall.  The proportion of non-target class items that are
        not matched.

            !recall = true-negatives / !target-class
        """
        return (self.tn / self.falses) if self.falses != 0 else None

    def fpr(self):
        """
        False-positive rate.  The proportion of proportion of non-target class
        items that are not matched.

            fpr = false-positives / !target-class
        """
        return (self.fp / self.falses) if self.falses != 0 else None

    def precision(self):
        """
        The proportion of matched observations that are correctly matched.
        AKA "positive predictive value".

            precision = true-positives / true-predicions
        """
        return (self.tp / self.positives) if self.positives != 0 else None

    def _precision(self):
        """
        The proportion of non-matched observations that are correctly not
        matched.  AKA "negative predictive value"

            !precision = true-negatives / false-predictions
        """
        return (self.tn / self.negatives) if self.negatives != 0 else None

    def f1(self):
        """
        An information theoretic statistic that balances specificity with
        sensitivity.
        """
        return (2 * ((self.precision() * self.recall()) /
                     (self.precision() + self.recall()))
                if self.precision() is not None and
                   self.recall() is not None and
                   self.precision() + self.recall() > 0 else None)

    def _f1(self):
        """
        The inverse f1.  The same information theoretic statistic applied to
        non-matched observations.
        """
        return (2 * ((self._precision() * self._recall()) /
                     (self._precision() + self._recall()))
                if self._precision() is not None and
                   self._recall() is not None and
                   self._precision() + self._recall() > 0 else None)
