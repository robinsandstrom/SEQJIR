import xlrd
import numpy as np


class FileReader:
    def __init__(self, covid19_filename, population_filename):
        self.covid19 = self.read_covid19_file(covid19_filename)
        self.populations = self.read_population_file(population_filename)

    # Returns dictionary on the form {'Sweden' : [np.array(# smittade), np.array(# doda)]}
    @staticmethod
    def read_covid19_file(filename):
        workbook = xlrd.open_workbook(filename)
        worksheet = workbook.sheet_by_index(0)
        current_country = ''
        dictionary = {}
        cases_col = -1
        deaths_col = -1
        country_col = -1
        for col in range(worksheet.ncols):
            temp = worksheet.cell_value(0, col).casefold()
            #print(temp)
            if (temp == 'cases') or (temp == 'newconfcases'):
                cases_col = col
            elif (temp == 'deaths') or (temp == 'newdeaths'):
                deaths_col = col
            elif (temp == 'countries and territories') or (temp == 'countryexp'):
                country_col = col
        if (cases_col == -1) or (deaths_col == -1) or (country_col == -1):
            print('Counld not find all columns')
        else:
            for row in range(1, worksheet.nrows):
                temp_country = worksheet.cell_value(row, country_col)
                if temp_country == current_country:
                    cases = np.append(cases, np.array([worksheet.cell_value(row, cases_col)], dtype=int), 0)
                    deaths = np.append(deaths, np.array([worksheet.cell_value(row, deaths_col)], dtype=int), 0)
                else:
                    if current_country != '':
                        dictionary[current_country] = [np.cumsum(np.flip(cases)), np.cumsum(np.flip(deaths))]
                    if worksheet.cell_value(row, 2) != '':
                        cases = np.array([worksheet.cell_value(row, cases_col)], dtype=int)
                    else:
                        cases = np.array([0], dtype=int)
                    if worksheet.cell_value(row, 3) != '':
                        deaths = np.array([worksheet.cell_value(row, deaths_col)], dtype=int)
                    else:
                        deaths = np.array([0], dtype=int)
                    current_country = temp_country
        return dictionary

    # Returns dictionary on the form {'Sweden' : int(population size)}
    @staticmethod
    def read_population_file(filename):
        workbook = xlrd.open_workbook(filename)
        worksheet = workbook.sheet_by_index(0)
        dictionary = {}
        for row in range(1, worksheet.nrows):
            dictionary[worksheet.cell_value(row, 0)] = int(worksheet.cell_value(row, 1))
        return dictionary
        return dictionary

    def population(self, country):
        return self.populations[country]

    def cases(self, country, min_cases=100):
        return self.covid19[country][0][self.covid19[country][0] >= min_cases]

    def deaths(self, country, min_cases=100):
        return self.covid19[country][1][self.covid19[country][0] >= min_cases]

    def number_of_points(self, country, min_cases=100):
        return np.shape(self.cases(country, min_cases))[0]

    def create_t_vector(self, country, min_cases=100):
        return np.arange(self.number_of_points(country, min_cases))
