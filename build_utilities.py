from os import system, listdir, path
from concurrent.futures import ThreadPoolExecutor
import subprocess
from get_error import error




def run_cmd(cmd):
    '''
    run a command in the terminal on another core of the machine
    '''
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {cmd}: {e}\n{error()}\n")




# just compile the .c into .o
# generate a list of terminal commands
commands = []
for folder in [i for i in listdir() if path.isdir(i)]:
    for file in [file for file in [i for i in  listdir(folder) if '.c' in i] if '.py' not in file]:
        if '__' in folder: continue
        commands.append(f'python3 -m gcc {folder}/{file} all fast -c')
        print(f'python3 -m gcc {folder}/{file} all fast -c')


[system(command) for command in commands]
quit()

# run the commands in parallel
with ThreadPoolExecutor() as executor:
    executor.map(run_cmd, commands)

