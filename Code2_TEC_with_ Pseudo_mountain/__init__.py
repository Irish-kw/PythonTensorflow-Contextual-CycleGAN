import numpy as np
import glob
import os

def divPrime(num):
    lt = []
    print(num, '=', end=' ')
    while num != 1:
        for i in range(2, int(num+1)):
            if num % i == 0:
                lt.append(i)
                num = num / i
                break
    for i in range(0, len(lt)-1):
        print(lt[i], '*', end=' ')
 
    print(lt[-1])

def glob_all(dir_path):
    file_list = glob.glob(os.path.join(dir_path,'*.h5'))
    inside = os.listdir(dir_path)
    for dir_name in inside:
        if os.path.isdir(os.path.join(dir_path,dir_name)):
            file_list.extend(glob_all(os.path.join(dir_path,dir_name)))
    return file_list