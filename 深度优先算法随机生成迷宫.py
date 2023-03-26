import random
import numpy as np
from turtle import *
import time

#初始化参数
num_rows = 20 # 生成迷宫的行数
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
tracer(10)# 最快生成图
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
# 从初始单元个一路开墙并进入下一个单元格，如果无路可走则返回。
# 我们使用while循环来执行此操作，重复该循环直到n=所有的单元格数-1.说明所有的单元格都是通的了。
while history:
    M[r, c, 4] = 1  # 将此位置指定为已访问

    # 检查相邻单元格是否可移动去，注意上下左右的边界。
    check = []
    if c > 0 and M[r, c - 1, 4] == 0:
        check.append('L')
    if r > 0 and M[r - 1, c, 4] == 0:
        check.append('U')
    if c < num_cols - 1 and M[r, c + 1, 4] == 0:
        check.append('R')
    if r < num_rows - 1 and M[r + 1, c, 4] == 0:
        check.append('D')

    if len(check):  # 如果有单元可以去
        history.append([r, c])
        n += 1
        move = random.choice(check)  # 随机打开一堵墙
        # 注意数组[上, 右，下，左，1]
        if move == 'L':
            M[r, c, 3] = 1
            c = c - 1
            M[r, c, 1] = 1
        if move == 'U':
            M[r, c, 0] = 1
            r = r - 1
            M[r, c, 2] = 1
        if move == 'R':
            M[r, c, 1] = 1
            c = c + 1
            M[r, c, 3] = 1
        if move == 'D':
            M[r, c, 2] = 1
            r = r + 1
            M[r, c, 0] = 1
    else:  # 如果发现没有下个单元格可以去，我们要回溯。
        r, c = history.pop()
    # 红色显示当前单元格子
    clear()  # 清理下，不然内存会被占用
    fillcolor("red")
    begin_fill()
    drawing(r, c, M[r, c])
    end_fill()
    # 调用turtle画图显示当前整个地图状态。
    for i in range(num_rows):
        for j in range(num_cols):
            drawing(i, j, M[i, j])
            update()  # 更新下，不然turtle会卡死
    time.sleep(0.5)
    if n == num_cols * num_rows - 1:  # 当砸墙的数量等于单元格子的数量-1时结束循环。
        break

done()
