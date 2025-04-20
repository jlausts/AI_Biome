import numpy as np
from matplotlib import pyplot as plt




dir = '/media/pi/40F8EEB9F8EEAC7A/Users/jlaus/Documents/Programming/Data/Normalized/input/'
#dir = '/home/pi/Data/Data/Normalized1/Normalized/input/'
#dir = '/media/pi/40F8EEB9F8EEAC7A/Users/jlaus/Documents/Programming/Data/TrainingData/input/'

def getTicker(ticker):
    info = np.fromfile(f"{dir}{ticker}.bin", np.float32)
    #return info
    return np.reshape(info, (int(info.shape[0] / 256), 256))


a = getTicker('ABBV')


print(a[:,2][:390*2])
