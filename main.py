from SEQIJR import SEQIJR
from FileReader import FileReader
#from ParameterEstimator import ParameterEstimator
import numpy as np
import matplotlib.pyplot as plt
from parameters import*

covid19_filename = 'COVID-19-geographic-disbtribution-worldwide-2020-03-20.xlsx'
population_filename = 'PopulationByCountry.xlsx'

country = 'Sweden'


files = FileReader(covid19_filename, population_filename)

x_confirmed = files.create_t_vector(country)
y_confirmed = files.cases(country)
d_confirmed = files.deaths(country)
h_confirmed = files.healthys(country)

# Greater Toronto Area
N = 1 * files.population(country.replace('_', ' '))


model = SEQIJR(N, Pi, mu, b,
                e_E, e_Q, e_J,
                g_1, g_2,
                s_1, s_2,
                k_1, k_2,
                d_1, d_2)

print('R_0 = ' + str(model.R_0()))
print('R_c = ' + str(model.R_c()))

S_0 = N
E_0 = 0
Q_0 = 0
I_0 = 1
J_0 = 0
R_0 = 0
aJ_0 = 0
aIJ_0 = 1
aD_0 = 0

y_0 = np.array([S_0,
                E_0,
                Q_0,
                I_0,
                J_0,
                R_0,
                aJ_0,
                aIJ_0,
                aD_0], dtype=float).transpose()

h = 1 / 2
x_1, S_p_1, E_p_1, Q_p_1, I_p_1, J_p_1, R_p_1, aJ_p_1, aIJ_p_1, aD_p_1 = model.prediction(y_0, 0, 365, h)

m = np.argmax(aJ_p_1 > min(y_confirmed))

y_0 = np.array([S_p_1[m],
                E_p_1[m],
                Q_p_1[m],
                I_p_1[m],
                J_p_1[m],
                R_p_1[m],
                aJ_p_1[m],
                aIJ_p_1[m],
                aD_p_1[m]], dtype=float).transpose()

P = 100

t_start = min(files.create_t_vector(country))
t_end = max(files.create_t_vector(country)) + P

x, S_p, E_p, Q_p, I_p, J_p, R_p, aJ_p, aIJ_p, aD_p = model.prediction(y_0, t_start, t_end, h)

plot_L = True


print(model.R_0())

fig = plt.figure()

plt.subplot(2, 2, 1)
plt.plot(x, aIJ_p)
plt.plot(x_confirmed, y_confirmed, 'bo', fillstyle='none')

plt.subplot(2, 2, 2)
plt.plot(x, aD_p)
plt.plot(x_confirmed, d_confirmed, 'bo', fillstyle='none')

plt.subplot(2, 2, 3)
plt.plot(x, R_p)
plt.plot(x_confirmed, h_confirmed, 'bo', fillstyle='none')


plt.subplot(2, 2, 4)
plt.plot(x, J_p + I_p)
#plt.plot(x_confirmed, h_confirmed, 'bo', fillstyle='none')


plt.show()
