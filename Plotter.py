from FileReader import FileReader
import matplotlib.pyplot as plt
import matplotlib
from parameters import time_confirmed
from parameters import cases_confirmed
from parameters import deaths_confirmed


def millions(x, pos):
    x = int(x)
    return '{:,}'.format(x).replace(',', ' ')


def plot_cases_deaths(x, infected, D):
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(time_confirmed, cases_confirmed, 'o', label='Uppmätt')
    axs[0].plot(x, infected, label='Prognos')
    axs[1].plot(time_confirmed, deaths_confirmed, 'o', label='Uppmätt')
    axs[1].plot(x, D, label='Prognos')

    axs[0].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(millions))
    axs[0].legend(loc="upper left")
    axs[1].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(millions))
    axs[1].legend(loc="upper left")
    plt.xlabel('Dagar sedan 100 bekräftade fall', fontsize=14)
    axs[0].set_ylabel('Bekräftat smittade', fontsize=14)
    axs[1].set_ylabel('Bekräftat döda', fontsize=14)
    # plt.title('Prognos: intensivvårdssökande i Sverige', fontsize=18)
    fig.show()
