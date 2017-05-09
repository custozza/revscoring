"""
Classification statistics can be generated for "Classifiers" -- models
that produce factors (aka levels) as an ouput along with a decision function
where thresholds can be arbitrarily placed.  Decision functions are usually
a probability or maybe a distance from some classification threshold.

.. autoclass:: revscoring.scoring.statistics.ThresholdClassification
    :members:
    :member-order:

.. autoclass:: revscoring.scoring.statistics.ThresholdOptimization
    :members:
    :member-order:

.. autoclass:: revscoring.scoring.statistics.threshold_classification.ThresholdStatistics
    :members:
    :member-order:

.. autoclass:: revscoring.scoring.statistics.threshold_classification.ThresholdStatList
    :members:
    :member-order:
"""  # noqa
import logging
import re

import tabulate
from numpy import all, diff, interp, linspace
from sklearn.metrics import auc

from . import util
from .classification import Classification, LabelStatistics, MicroMacroStat

logger = logging.getLogger(__name__)


class ThresholdClassification(Classification):
    FIELDS = ['roc_auc', 'pr_auc', 'thresholds']

    def __init__(self, *args, decision_key="probability", max_thresholds=200,
                 threshold_optimizations=None, **kwargs):
        """
        Construct a set of statistics for classifiers with a decision function
        output in the score doc.

        :Parameters:
            prediction_key : `str`
                A key into a score doc under which a scalar decision value
                can be found for each potential class.
            decision_key : `str`
                A key into a score doc under which a scalar decision value
                can be found for each potential class.
            labels : [ `mixed` ]
                A sequence of labels that are in-order.  Order is used when
                formatting statistical outputs.
            population_rates : `dict`
                A mapping of label classes with float representing the rate
                that each class occurs in the target population.  Rates
                observed in the sample will be scaled to match the population
                rates.  This is useful when training a model with different
                sample rates than the target population rates.
            max_thresholds : `int`
                The maximum number of thresholds to report.  Thresholds will
                be distributed uniformly across actually output probabilities.
        """
        super().__init__(*args, **kwargs)
        self.decision_key = decision_key
        self.max_thresholds = max_thresholds
        self.threshold_optimizations = {
            str(to): to for to in (threshold_optimizations or [])}

    def fit(self, score_labels):
        """
        Fit to scores and labels.

        :Parameters:
            score_labels : [( `dict`, `mixed` )]
                A collection of scores-label pairs generated using
                :class:`revscoring.Model.score`.  Note that fitting is usually
                done using data withheld during model training
        """
        super().fit(score_labels)

        threshold_stats = {}
        for label in self.labels:
            threshold_stats[label] = ThresholdStatistics(
                [s[self.decision_key][label] for s, l in score_labels],
                [l == label for s, l in score_labels],
                population_rate=self.population_rates.get(label),
                threshold_optimizations=self.threshold_optimizations)

        for stat_name in ThresholdStatistics.FIELDS:
            self[stat_name] = MicroMacroStat(stat_name, threshold_stats)

        for op_name in self.threshold_optimizations:
            self[op_name] = MicroMacroStat(op_name, threshold_stats)

        self['thresholds'] = {
            label: ThresholdStatList(tstats, self.max_thresholds)
            for label, tstats in threshold_stats.items()}

    def format_str(self, fields=None, ndigits=3, **kwargs):
        fields = fields or (Classification.FIELDS +
                            list(self.threshold_optimizations.keys()) +
                            ThresholdClassification.FIELDS)
        formatted = super().format_str(
            fields=fields, ndigits=ndigits, **kwargs)
        for field in fields:
            if field == "thresholds":
                formatted += "thresholds:\n"
                for label in self.labels:
                    formatted += util.tab_it_in(repr(label))
                    table_str = self['thresholds'][label] \
                                .format_str(ndigits=ndigits, **kwargs)
                    formatted += util.tab_it_in(table_str, 2)
                formatted += "\n"
            elif field in ThresholdClassification.FIELDS or \
                 field in self.threshold_optimizations:
                formatted += self[field].format_str(
                    self.labels, ndigits=ndigits, **kwargs)
                formatted += "\n"
        return formatted

    def format_json(self, fields=None, ndigits=3, **kwargs):
        fields = fields or (Classification.FIELDS +
                            ThresholdClassification.FIELDS +
                            list(self.threshold_optimizations.keys()))
        stats_doc = super().format_json(
            fields=fields, ndigits=ndigits, **kwargs)

        for field in fields:
            if field == "thresholds":
                stats_doc['thresholds'] = {
                    label: ltstats.format_json(ndigits=ndigits, **kwargs)
                    for label, ltstats in self['thresholds'].items()}
            elif field in ThresholdClassification.FIELDS:
                stats_doc[field] = self[field].format_json(
                    ndigits=ndigits, **kwargs)

        return stats_doc





class ThresholdStatistics(list):
    FIELDS = ['roc_auc', 'pr_auc']

    def __init__(self, y_decisions, y_trues, population_rate=None,
                 threshold_optimizations=None):
        """
        Construct a sequence of ThresholdStatistics

        :Parameters:
            y_decisions : [ `float` ]
                A sequence of decision-weights that represent confidence in
                a target class prediction
            y_trues : [ `bool` ]
                A sequence of labels where `True` represents a positive
                observation.
            population_rate : `float`
                The rate at which the observed class appears in the population.
                This value will be used to re-scale the number of y_trues
                across all metrics.
            threshold_optimizations : [ :class:`~revscoring.scoring.statistics.ThresholdOptimization` ]
                The threshold optimizations that should be computed.
        """  # noqa
        super().__init__()
        if population_rate is None:
            self.trues = sum(y_trues)
        else:
            n_true = sum(y_trues)
            observed_rate = n_true / len(y_trues)
            self.trues = sum(y_trues) * (population_rate / observed_rate)
        self.threshold_optimizations = threshold_optimizations or {}
        unique_thresholds = sorted(set(y_decisions))

        for threshold in unique_thresholds:
            self.append((threshold, LabelStatistics(
                [decision >= threshold for decision in y_decisions],
                 y_trues, population_rate=population_rate))
            )

    def roc_auc(self):
        return zero_to_one_auc([stat.fpr() for t, stat in self],
                               [stat.recall() for t, stat in self])

    def pr_auc(self):
        return zero_to_one_auc([stat.recall() for t, stat in self],
                               [stat.precision() for t, stat in self])



def zero_to_one_auc(x_vals, y_vals):
    x_space = linspace(0, 1, 50)
    if all(diff(x_vals) > 0):
        y_interp = interp(x_space, x_vals, y_vals)
    else:
        y_interp = interp(
            x_space, list(reversed(x_vals)), list(reversed(y_vals)))
    return auc(x_space, y_interp)
