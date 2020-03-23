# Greater Toronto Area
Pi = 0
mu = 1 / (80 * 365)       # 1/(80*365) (average age in days)^(-1)
b = 0.61 # Smittgraden
e_E = 0.25 # Smittsamhet för exposed
e_Q = 0.1 # Smittsamhet för karantänade
e_J = 0.5 # Smittsamhet för isolerade

g_1 = 1 / 10 # 1/ Antalet dagar gå från exposed till quarantined
g_2 = 1 / 2 # 1/ Antalet dagar gå från infective till isolated (J)

s_1 = 1 / 2 # 1 / Antalet dagar tills tillfrisknad som infective
s_2 = 1 / 10 # 1 / Antalet dagar att tillfriskna som isolerad

k_1 = 1 / 10 # 1 / Antalet dagar få från exponerad till infective (inkubationstid)
k_2 = 1 / 8 # 1 / Antalet dagar från karantän till isolerad

d_1 = 0.0 #  Dödlighet i infective
d_2 = 0.05 #  Dödlighet i isolated
