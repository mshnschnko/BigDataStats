import math
import numpy as np
import pandas as pd
import scipy as sp
from dateutil import parser, rrule
from datetime import datetime, time, date
import scipy.linalg
import csv
from scipy import fft
from matplotlib import pyplot as plt
from hurst import compute_Hc, random_walk
N = 200
h = 0.02


def prony(x: np.array, T: float):
    if len(x) % 2 == 1:
        x = x[:len(x)-1]

    p = len(x) // 2

    shift_x = [0] + list(x)
    a = scipy.linalg.solve([shift_x[p+i:i:-1] for i in range(p)], -x[p::])

    z = np.roots([*a[::-1], 1])

    h = scipy.linalg.solve([z**n for n in range(1, p + 1)], x[:p])

    f = 1 / (2 * np.pi * T) * np.arctan(np.imag(z) / np.real(z))
    alfa = 1 / T * np.log(np.abs(z))
    A = np.abs(h)
    fi = np.arctan(np.imag(h) / np.real(h))

    return f, alfa, A, fi


def generateSample():
    return np.array([sum([(k * np.exp(-h * i / k) * np.cos(4 * np.pi * k * h * i + np.pi / k)) for k in range(1, 4)]) for i in range(1, N + 1)])


def slideMedian(sample, m):
    res = []
    for i in range(sample.size):
        if i < m:
            res.append(np.median(sample[0 : 2 * i + 1]))
        elif i >= sample.size - m - 1 :
            res.append(np.median(sample[i - (sample.size - i) : sample.size]))
        else:
            res.append(np.median(sample[i - m : i + m + 1]))
    return res


def calculateRotationPoints(sample):
    res = []
    for i in range(1, len(sample) - 2):
        if (sample[i] > sample[i - 1] and sample[i] > sample[i + 1]) or (sample[i] < sample[i - 1] and sample[i] < sample[i + 1]):
            res.append(sample[i])
    return res


def checkRotationPoints(sample, trend):
    tail = sample - trend
    rotationPoints = calculateRotationPoints(tail)

    pMean = (2.0 / 3.0) * (len(sample) - 2)
    pDisp = (16 * len(sample) - 29) / 90.0
    pSize = len(rotationPoints)

    print("Количествво поворотных точек: ", pSize)
    if pSize < pMean + pDisp and pSize > pMean - pDisp:
        print("\nРяд случаен\n")
    elif pSize > pMean + pDisp:
        print("\nРяд является быстро колеблющимся\n")
    elif pSize < pMean - pDisp:
        print("\nЗначения ряда положительно коррелированы\n")


def checkKendall(data):
    p = 0
    n = len(data)
    for i in range(len(data) - 1):
        for j in range(i+1, len(data)):
            if data[j] > data[i]:
                p += 1
    E_t = 0
    D_t = 2*(2*(len(data))+5)/(9*(len(data))*len(data)-1)
    s_t = np.sqrt(D_t)
    t = 4*p/(n*(n-1)) - 1
    print(f"t = {t}")
    if t > E_t + s_t:
        print("Возрастяющий тренд")
    elif t < E_t - s_t:
        print("Убывающий тренд")
    else:
        print("Ряд случаен")


if __name__ == "__main__":
    print("Task 1: \n")
    sample = generateSample()
    print(sample)
    plt.figure()
    plt.title("Исходный ряд")
    plt.plot(sample)
    plt.grid()
    plt.xlabel("k")
    plt.ylabel("$x_k$")
    
    f, alfa, A, fi = prony(sample, 0.1)
    plt.figure()
    plt.stem(A)
    plt.plot()
    plt.grid()
    plt.xlabel("k")
    plt.ylabel("$A_k$")
    plt.title("Амплитуды")
    plt.show()

    data_raw = pd.read_csv('LONDON.csv')
    data_raw['mean_temp'] = data_raw['mean_temp'].astype(float)
    data_raw['date'] = [datetime.strptime(str(d), '%Y%m%d') for d in data_raw['date']]
    data = data_raw.loc[:, ['date', 'mean_temp']]
    print(data)

    plt.plot(data_raw['date'], data['mean_temp'], label = 'Температура')
    trend = slideMedian(np.array(data['mean_temp']), 55)
    plt.plot(data_raw['date'], trend, label = 'Тренд')
    plt.xlabel("Дата")
    plt.ylabel("$Температура$")
    plt.title("Среднесуточная температура температура")
    plt.grid()
    plt.legend() 

    checkRotationPoints(trend, np.array(data['mean_temp']))
    checkKendall(np.array(data['mean_temp']))

    H, c, data = compute_Hc(trend, kind='random_walk', simplified=False)

    print(f"Hurst = {H}")
    if math.isclose(H, 0.5, rel_tol=1e-5):
        print("Ряд случаен")
    elif H > 0.5:
        print("Персистентный ряд (сохраняет тренд)")
    elif H < 0.5:
        print("Антиперсистентный ряд")
    
    print(len(data[0]))
    plt.figure()
    plt.plot(data[0], c*data[0]**H, color="deepskyblue", label="$cn^H$")
    plt.scatter(data[0], data[1], color="purple", label="$R/s\ факт.$")
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("$R/s$")
    plt.grid()
    plt.show()