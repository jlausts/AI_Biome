
from time import perf_counter as pc, sleep
from rich import print
from sys import argv, stdin
from os import path
from matplotlib import pyplot as plt
from get_error import error
from re import findall


# run the excecutable and time its execution
start = pc()

try:

    # parse the numbers from the terminal output and print the non numbers to the console
    numbers = []
    break_numbers_line_index = [0,]
    result = []
    for i in stdin:
        result.append(i)

    # get the labels if they are there
    lables = None
    if ',' in result[0]:
        lables = result[0].split(',')
        result = result[1:]

    for n, line in enumerate(result):

        if 'new' in line:
            break_numbers_line_index.append(n)
            continue

        line_numbers = findall('\-?\d+\.?\d*', line)

        if len(line_numbers) == 0 and line != 'new':
            print(line)

        else:
            numbers.append([float(i) for i in line_numbers])
    break_numbers_line_index.append(n)

    chunks = []
    for a,b in zip(break_numbers_line_index, break_numbers_line_index[1:]):
        if len(numbers[a:b]):
            chunks.append(numbers[a:b])


    # format the plot area
    plt.rcParams["figure.figsize"] = [round(1920/180), round(1080/180)]
    plt.rcParams["figure.autolayout"] = True
    
    plt.rcParams['axes.titley'] = 1.0   
    plt.rcParams['axes.titlepad'] = -28 
    
    num_plots = 0
    for chunk in chunks:
        # determine if there are multiple columns in the rows of numbers
        lengths_of_rows = [len(i) for i in chunk[::3]]
        num_plots = round(sum(lengths_of_rows) / len(lengths_of_rows))

        print(f'Plotting: {num_plots} graph{"s" if num_plots > 1 else ""}')

    if num_plots == 0:
        print('Nothing to Plot.')
        return
    
    _, ax = plt.subplots(num_plots, gridspec_kw = {'wspace':0, 'hspace':0})

    # set the lables to an empty string if there were none passed in
    if lables is None:
        lables = ['' for i in range(num_plots)]

    if len(lables) != num_plots:
        print(f'number of lables passed in does not match the number of plots.\n# of lables: {len(lables)}\n# of plots:  {num_plots}\nMake sure that they are CSV.\n')
        return

    plot_count = 0
    for chunk in chunks:

        chunk = [i for i in chunk if len(i) == num_plots]

        # reshape the array
        reshaped = [[] for i in range(num_plots)]
        for line in chunk:
            for n, i in enumerate(line):
                reshaped[n].append(i)


        # plot the numbers
        if num_plots > 1:
            for nums in reshaped:
                ax[plot_count].plot(nums)
                ax[plot_count].text(0.05, 0.95, lables[plot_count], verticalalignment='top', horizontalalignment='left', transform=ax[plot_count].transAxes)
                plot_count += 1
        else:
            ax.plot(reshaped[0])
            ax.set_title(lables[0])
        
    plt.show()

except:
    print(error())

