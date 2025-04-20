from os import system, listdir
from sys import argv
import numpy as np

a = 1.5

files = [i for i in listdir() if '.' not in i and 'Cuda' != i]
                                                                                       
command = \
   'python3 -m gcc main.c ' + ' '.join([f'{i}/{i}.c' for i in files if '__' not in i and 'main' != i]) + \
      ' -lpthread -lm all ' + ' '.join(argv[1:])

try:
   system(command)
except KeyboardInterrupt:
   quit()
# C:\Users\jlaus\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\LocalCache\local-packages\Python38\site-packages\gcc\