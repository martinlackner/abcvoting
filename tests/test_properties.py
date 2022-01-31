"""
Unit tests for: properties.py
"""

import pytest
import os
import re
import random
from abcvoting.abcrules_cvxpy import cvxpy_thiele_methods
from abcvoting.abcrules_gurobi import _gurobi_thiele_methods
from abcvoting.output import VERBOSITY_TO_NAME, WARNING, INFO, DETAILS, DEBUG, output
from abcvoting.preferences import Profile, Voter
from abcvoting import abcrules, misc, scores, fileio

# have the input committees here in a list, for different properties

# then create parametrized tests. these will have say 2 different inputs