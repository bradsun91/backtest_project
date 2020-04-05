#!/usr/bin/python
# -*- coding: utf-8 -*-

# portfolio.py


from __future__ import print_function

import datetime
from math import floor
try:
    import Queue as queue
except ImportError:
    import queue

import numpy as np
import pandas as pd

from event import FillEvent, OrderEvent
from performance import create_sharpe_ratio, create_drawdowns


print("Executing plots.py")


# Added by Brad on 20191007:
class Plots(object):
    """
    """
    def __init__():
        pass


    def draw_summary_plots(self):
        print("Testing draw_summary_plots")
        self.equity_curve['equity_curve'].plot(figsize = (18,6))