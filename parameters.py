from FileReader import FileReader

covid19_filename = 'COVID-19-geographic-distribution-sweden-2020-03-31.xlsx'
population_filename = 'PopulationBySwedishRegion.xlsx'
country = 'Sverige'

files = FileReader(covid19_filename, population_filename)

time_confirmed = files.create_t_vector(country)
cases_confirmed = files.cases(country)
deaths_confirmed = files.deaths(country)
in_hospital_confirmed = files.in_hospital(country)
in_intensive_care_confirmed = files.in_intensive_care(country)

parameters = \
    {'Sverige': {'N': files.population('Sverige'),
                 'b': .8,
                 'e_E': .2, 'e_Q': .1, 'e_J': .2, 'e_C': .1,
                 'T_E': 5, 'T_Q': 5, 'T_I': 5, 'T_J': 5, 'T_C': 4,
                 'w_E': .2, 'w_Q': .1, 'w_I': .1, 'w_J': .2, 'w_C': .6,
                 'mu': 0,
                 'Pi': 0,
                 'pi': 0},
    'Sverige_old': {'N': files.population('Sverige'),
               'b': .7,
                'e_E': .3, 'e_Q': .2, 'e_J': .3, 'e_C': .2,
                'T_E': 5, 'T_Q': 4, 'T_I': 4, 'T_J': 2, 'T_C': 2,
                'w_E': .2, 'w_Q': .1, 'w_I': .1, 'w_J': .2, 'w_C': .6,
                'mu': 0,
                'Pi': 0,
                'pi': 0},
    }
