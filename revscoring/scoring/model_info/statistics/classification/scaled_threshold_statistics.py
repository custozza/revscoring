

class ScaledThresholdStatistics(list):

    def __init__(self, y_decisions, y_trues, population_rate=None):
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
        """  # noqa
        super().__init__()
        if population_rate is None:
            self.trues = sum(y_trues)
        else:
            n_true = sum(y_trues)
            observed_rate = n_true / len(y_trues)
            self.trues = sum(y_trues) * (population_rate / observed_rate)
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

    def get_stat(self, stat_name):
        if not hasattr(self, stat_name):
            if stat_name in self.threshold_optimizations:
                optimization = self.threshold_optimizations[stat_name]
                return optimization.optimize_from(self)
            else:
                raise KeyError(stat_name)
        else:
            return getattr(self, stat_name)()

    def best(self, stat_name, at_value):
        for threshold, lstats in self:
            target_stat = lstats.get_stat(stat_name)
            if target_stat >= at_value:
                return threshold, lstats

    def optimize(self, optimize, stat_name, at_value):
        best_threshold = self.best(stat_name, at_value)
        if best_threshold is None:
            return None
        else:
            threshold, lstats = best_threshold
            return lstats.get_stat(stat_name)


def zero_to_one_auc(x_vals, y_vals):
    x_space = linspace(0, 1, 50)
    if all(diff(x_vals) > 0):
        y_interp = interp(x_space, x_vals, y_vals)
    else:
        y_interp = interp(
            x_space, list(reversed(x_vals)), list(reversed(y_vals)))
    return auc(x_space, y_interp)
