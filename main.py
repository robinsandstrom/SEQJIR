from SEQIJR import SEQIJCR
from FileReader import FileReader
from parameters import *
import numpy as np
import Plotter

h = 1 / 5
t_start = min(time_confirmed)
P = 0
t_end = max(time_confirmed) + P

model = SEQIJCR(**parameters[country])

actions = \
    {
        10: {'b': parameters[country]['b']/2,
             'w_E': 0.5}
    }

'''
,
        20: {'b': parameters[country]['b']/10,
             'w_E': 0.2}
'''

model.set_actions(actions)

y_0 = model.get_y_0(min(cases_confirmed))

x, S, E, Q, I, J, C, R, D = model.prediction(y_0, t_start, t_end, h)

m = int(9 / h)
infected = (Q + I + J + C + R + D)/2
infected[m:np.shape(infected)[0]] = (infected[m:np.shape(infected)[0]]-infected[m])/2.5 + infected[m]

Plotter.plot_cases_deaths(x, infected, D)
