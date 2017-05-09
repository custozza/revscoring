
logger = logging.getLogger(__name__)

class ScaledClassificationMatrix(dict):

    def __init__(self, y_preds, y_trues, population_rate=None):
        """
        Constructs a classification matrix and scales the values based on a
        population rate.

        :Parameters:
            y_preds : [ `bool` ]
                A sequence of boolean values representing whether or not the
                target label was predicted
            y_trues : [ `bool` ]
                A sequence of boolean values representing whether or not the
                traget label was the supervised label provided
            population_rate : `float`
                The rate at which this label occurs in the population.  If
                not provided, the sample rate will be assumed to reflect the
                population rate.
        """
        self.population_rate = population_rate

        # Generate counts of basic classification metrics
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        for y_pred, y in zip(y_preds, y_trues):
            self.tp += y_pred and y  # win
            self.fn += not y_pred and y  # fail
            self.tn += not y_pred and not y  # win
            self.fp += y_pred and not y  # fail

        if population_rate is not None:
            observed_rate = ((self.tp + self.fn) /
                             (self.tp + self.fn + self.tn + self.fp))
            sample_rate = observed_rate / population_rate
            non_sample_rate = (1 - observed_rate) / (1 - population_rate)

            # Apply scaling to obtain expected population rate
            self.tp = self.tp / sample_rate
            self.fn = self.fn / sample_rate
            self.tn = self.tn / non_sample_rate
            self.fp = self.fp / non_sample_rate
            '''
            orig_tp = self.tp
            orig_fn = self.fn
            orig_tn = self.tn
            orig_fp = self.fp
            logger.debug("Scaled true-positives ({0}) by sample_rate {1}: {2}"
                         .format(orig_tp, sample_rate, self.tp))
            logger.debug("Scaled false-negatives ({0}) by sample_rate {1}: {2}"
                         .format(orig_fn, sample_rate, self.fn))
            logger.debug("Scaled true-negatives ({0}) by sample_rate {1}: {2}"
                         .format(orig_tn, sample_rate, self.tn))
            logger.debug("Scaled false-positives ({0}) by sample_rate {1}: {2}"
                         .format(orig_fp, sample_rate, self.fp))
            '''

        # Useful variables
        self.n = self.tp + self.tn + \
                 self.fp + self.fn
        self.positives = self.tp + self.fp
        self.negatives = self.tn + self.fn
        self.trues = self.tp + self.fn
        self.falses = self.fp + self.tn
        self.correct = self.tp + self.tn
