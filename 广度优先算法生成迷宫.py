import random
import numpy as np
from turtle import *
import time

#初始化参数
num_rows = 20 # 生成迷宫的行数8
num_cols = 20 # 生成迷宫的列数

M = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)
# 阵列M将保存每个单元的阵列信息。
# 前四个坐标告诉墙壁在那些边上是否存在
# 和第五个指示在搜索中是否已访问该单元格。
# M【上，右，下，左，是否被访问】

# 我们先把第一个单元格最和后一个墙打开。
M[0, 0, 0] = 1
M[num_rows-1, num_cols-1, 2] = 1

#如下是turtle模块的初始化
tracer(0)# 最快生成图
ht()# 隐藏画笔
pensize(1)#画笔大小设为1

def pengoto(x, y):
    up()
    goto(x, y)
    down()


def drawing(r, c, M):
    x = 20*c-200
    y = 200-20*r
    pengoto(x, y)
    for i in range(4):
        if M[i] == 1:
            pencolor('blue')
            fd(1)
            pencolor('white')
            fd(19)
            right(90)
        else:
            pencolor('blue')
            fd(20)
            right(90)

# 设置开始的行和列
r = 0
c = 0
history = [(r, c)]  # 这个是历史访问的单元格列表。
n = 0  # 砸墙的数量。
while history:
    # 随机选择一个可以敲墙的单元格
    r, c = random.choice(history)
    M[r, c, 4] = 1  # 把单元设成以访问
    history.remove((r, c))
    check = []
#如果随机选择的单元格具有多个边
#将其连接到现有的迷宫，
    if c > 0:
        if M[r, c-1, 4] == 1:
            check.append('L')
        elif M[r, c-1, 4] == 0:
            history.append((r, c-1))
            M[r, c-1, 4] = 2
    if r > 0:
        if M[r-1, c, 4] == 1:
            check.append('U')
        elif M[r-1, c, 4] == 0:
            history.append((r-1, c))
            M[r-1, c, 4] = 2
    if c < num_cols-1:
        if M[r, c+1, 4] == 1:
            check.append('R')
        elif M[r, c+1, 4] == 0:
            history.append((r, c+1))
            M[r, c+1, 4] = 2
    if r < num_rows-1:
        if M[r+1, c, 4] == 1:
            check.append('D')
        elif M[r+1, c, 4] == 0:
            history.append((r+1, c))
            M[r+1, c, 4] = 2
# 随机前往一个边界墙.
    if len(check):
        n+=1
        move = random.choice(check)
        if move == 'L':  # [上,右，下，左，1]
            M[r, c, 3] = 1
            c = c-1
            M[r, c, 1] = 1
        if move == 'U':
            M[r, c, 0] = 1
            r = r-1
            M[r, c, 2] = 1
        if move == 'R':
            M[r, c, 1] = 1
            c = c+1
            M[r, c, 3] = 1
        if move == 'D':
            M[r, c, 2] = 1
            r = r+1
            M[r, c, 0] = 1

    # 红色方格显示当前单元格子位置
    clear()#清理下，不然内存会一直被占用。
    fillcolor("red")
    begin_fill()
    drawing(r, c, M[r, c])
    end_fill()
    # 调用turtle画图显示当前整个地图状态。
    for i in range(num_rows):
        for j in range(num_cols):
            drawing(i, j, M[i, j])
            update()#需要跟新下，不然会卡死
    time.sleep(0.5)
    if n == num_cols*num_rows-1:  # 当砸墙的数量等于单元格子的数量-1时结束循环。
        break

done()
