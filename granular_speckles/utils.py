#! /usr/bin/python
# -*- coding: utf-8 -*-
# This file belongs to DWGranularSpeckles project.
# The software is realeased with MIT license.

import time

def timeit(method):
    '''this decorator measures operation time'''

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r %2.2f sec' %
              (method.__name__, te-ts))
        return result
    return timed


def test_speed(self):
    second = self.parallelReducedTime()
    first = self.reducedTime()
    if (first == second).all():
        return first
