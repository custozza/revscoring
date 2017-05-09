import logging

from tabulate import tabulate

from ... import util
from ...model_info import ModelInfo

logger = logging.getLogger(__name__)


class MicroMacroStats(ModelInfo):

    def __init__(self, stats, stat_name):
        """
        Constructs a micro-average and macro-average for a specific statistic
        based on the name.  Works like a dictionary with fields

         * micro : the micro-average
         * macro : the macro-average
         * labels : a mapping of labels to their individual statistics

        :Parameters:
        """  # noqa
        self.stat_name = stat_name
        try:
            self['micro'] = (
                sum(lstats[stat_name] * lstats.trues
                    for lstats in stats.values()) /
                sum(lstats.trues for lstats in stats.values()))
        except Exception as e:
            logger.warn("Could not generate micro-average of {0}: {1}"
                        .format(stat_name, str(e)))
            self['micro'] = None

        try:
            self['macro'] = (
                sum(lstats[stat_name] for lstats in stats.values()) /
                len(stats))
        except Exception as e:
            logger.warn("Could not generate macro-average of {0}: {1}"
                        .format(stat_name, str(e)))
            self['macro'] = None

        self['labels'] = {label: lstats[stat_name]
                          for label, lstats in stats.items()}

    def format_str(self, labels, ndigits=3, **kwargs):
        formatted = "{0} (micro={1}, macro={2}):\n" \
                     .format(self.stat_name,
                             util.round(self['micro'], ndigits=ndigits),
                             util.round(self['macro'], ndigits=ndigits))
        table_str = tabulate.tabulate(
            [[util.round(self['labels'][l], ndigits) for l in labels]],
            headers=labels)
        formatted += util.tab_it_in(table_str)
        return formatted

    def format_json(self, ndigits=3):
        return {
            'micro': util.round(self['micro'], ndigits),
            'macro': util.round(self['micro'], ndigits),
            'labels': {l: util.round(self['labels'][l], ndigits)
                       for l in self['labels']}
        }


class MicroMacroOptimizationStats(ModelInfo):

    def __init__(self, threshold_stats, optimization):
        self.stat_name = str(optimization)

        self['micro'] = (
            sum(optimization.optimize_from(tstats) * tstats.trues
                for tstats in threshold_stats.values()) /
            sum(tstats.trues for tstats in threshold_stats.values()))

        self['macro'] = (
            sum(optimization.optimize_from(tstats)
                for tstats in threshold_stats.values()) /
            len(threshold_stats))

        self['labels'] = {label: optimization.optimize_from(tstats)
                          for label, tstats in threshold_stats.items()}

    def format_str(self, path_tree, ndigits=3, **kwargs):
        formatted = str(self.optimization) + "\n"
        for key in path_tree.keys() or self.label_thresholds.keys():
            sub_tree = path_tree[key]
            if len(sub_tree) > 0:
                raise ValueError("Path past base of tree at {0!r}"
                                 .format(tuple(sub_tree.keys())))
            formatted += "\t{0}: {1}\n"
        return formatted
