"""
Classification statistics can be generated for "Classifiers" -- models
that produce factors (aka levels) as an ouput.  E.g. True and False or
"A", "B", or "C".

.. autoclass:: revscoring.scoring.statistics.Classification
    :members:
    :member-order:

.. autoclass:: revscoring.scoring.statistics.classification.MicroMacroStat
    :members:
    :member-order:

.. autoclass:: revscoring.scoring.statistics.classification.LabelStatistics
    :members:
    :member-order:

.. autoclass:: revscoring.scoring.statistics.classification.ScaledClassificationMatrix
    :members:
    :member-order:
"""  # noqa
import logging
from collections import defaultdict

import tabulate

from . import util
from .statistics import Statistics

logger = logging.getLogger(__name__)
