# -*- coding:utf-8 -*-
import random

char_list = ['0', '1'] #['*', '#', 'P', '.']
with open('_new.csv', '+a') as buffer:
    buffer.write('Fail' + ','*10)
    buffer.write('\n')
    for i in range(11):
        for j in range(11):
            if i == 5 and j == 5:
                k = random.randint(0, 1)
                buffer.write(char_list[k])
            else:
                k = random.randint(0, 1)
                buffer.write(char_list[k])
            if j == 10:
                buffer.write('\n')
            else:
                buffer.write(',')
