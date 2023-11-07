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

def turning_points(data):
    p = 0
    for i in range(len(data) - 2):
        if data[i] < data[i+1] and data[i+1] > data[i+2]:
            p += 1
        if data[i] > data[i+1] and data[i+1] < data[i+2]:
            p += 1
    return p

def checkRotationPoints(sample, trend):
    tail = sample - trend
    rotationPoints = calculateRotationPoints(tail)
    print("\nПоворотные точки")
    print(f"Всего точек: {len(sample)}")
    pMean = (2.0 / 3.0) * (len(sample) - 2)
    pDisp = (16 * len(sample) - 29) / 90.0
    pSize = len(rotationPoints)
    pSize = turning_points(tail)

    print("Количество поворотных точек: ", pSize)
    if pSize < pMean + pDisp and pSize > pMean - pDisp:
        print("\nРяд случаен\n")
    elif pSize > pMean + pDisp:
        print("\nРяд является быстро колеблющимся\n")
    elif pSize < pMean - pDisp:
        print("\nЗначения ряда положительно коррелированы\n")


def checkKendall(data):
    print("\nКендел")
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
        print("Возрастяющий тренд\n")
    elif t < E_t - s_t:
        print("Убывающий тренд\n")
    else:
        print("Ряд случаен\n")

def ema(data, a):
    y = []
    y.append((data[0] + data[1])/2)
    for i in range(1, len(data)):
        y.append(a*data[i] + (1-a)*y[i-1])
    return y

def sma(data, m):
    sma = [0] * len(data)
    sma[0] = data[0]
    for i in range(1, len(data)-1):
        w = m
        while (i - w < 0) or (i + w > len(data) - 1):
            w -= 1
        sma[i] = sum(data[i-w:i+w]) / (2*w+1)
    sma[-1] = data[-1]
    return sma

def calculate_turning_points(sample):
    res = []
    for i in range(1, len(sample) - 2):
        if (sample[i] > sample[i - 1] and sample[i] > sample[i + 1]) or (sample[i] < sample[i - 1] and sample[i] < sample[i + 1]):
            res.append(sample[i])
    return res

def check_kendall(sample, trend):
    tail = sample - trend
    turning_points = calculate_turning_points(tail)
    p_mean = (2.0 / 3.0) * (len(sample) - 2)
    p_disp = (16 * len(sample) - 29) / 90.0
    p_size = len(turning_points)
    p_type = ""
    if p_size < p_mean + p_disp and p_size > p_mean - p_disp:
        p_type ="Randomness"
    elif p_size > p_mean + p_disp:
        p_type ="Rapidly oscillating"
    elif p_size < p_mean - p_disp:
        p_type = "Positively correlated"
    return p_size, p_type

def FFT(x):
    fft = np.fft.fft(x)
    # a: np.complex128 = [1, 2]
    # print(a)
    abs_fft = []
    for i in range(len(fft)):
        # abs_fft.append(fft[i].real)
        abs_fft.append(abs(fft[i]))
    return np.array(abs_fft)

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

    data_raw = pd.read_csv('volgograd.csv')
    data_raw['mean_temp'] = data_raw['mean_temp'].astype(float)
    data_raw['date'] = [datetime.strptime(str(d), '%Y%m%d') for d in data_raw['date']]
    data = data_raw.loc[:, ['date', 'mean_temp']]
    print(data)

    plt.plot(data_raw['date'], data['mean_temp'], label = 'Температура')
    # trend = slideMedian(np.array(data['mean_temp']), 55)
    trend = sma(np.array(data['mean_temp']), 55)
    # trend = ema(np.array(data['mean_temp']), 0.07)
    plt.plot(data_raw['date'], trend, label = 'Тренд')
    plt.xlabel("Дата")
    plt.ylabel("Температура")
    plt.title("Среднесуточная температура температура")
    plt.grid()
    plt.legend()
    plt.show()

    FFT_orig = FFT(np.array(data['mean_temp']))
    print(len(FFT_orig))
    ordi = np.linspace(0, 0.5, len(FFT_orig)//2)
    
    print(f"Главная частота = {ordi[np.argmax(FFT_orig[1:len(FFT_orig)//2])+1]}")
    plt.figure()
    plt.plot(ordi[1:], FFT_orig[1:len(FFT_orig)//2]/len(FFT_orig), label='FFT(x)')
    plt.grid()
    plt.show()

    # plt.figure()
    # plt.plot(np.array(data['mean_temp'])-trend)
    # checkRotationPoints(trend, np.array(data['mean_temp']))
    checkKendall(np.array(data['mean_temp'])-trend)
    # print(*check_kendall(trend, np.array(data['mean_temp'])))

    H, c, data = compute_Hc(data['mean_temp'], kind='random_walk', simplified=False)

    print(f"Hurst = {H}")
    if math.isclose(H, 0.5, rel_tol=1e-5):
        print("Ряд случаен")
    elif H > 0.5:
        print("Персистентный ряд (сохраняет тренд)")
    elif H < 0.5:
        print("Антиперсистентный ряд")
    
    plt.figure()
    plt.plot(data[0], c*data[0]**H, color="deepskyblue", label="$cn^H$")
    plt.scatter(data[0], data[1], color="purple", label="$R/s\ факт.$")
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("$R/s$")
    plt.grid()
    plt.show()