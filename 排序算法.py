def mao_pao(a):
    num_len = len(a)
    # 控制循环的次数
    for j in range(num_len):
        # 添加标记位 用于优化(如果没有交换表示有序,结束循环)
        sign = False
        # 内循环每次将最大值放在最右边
        for i in range(num_len - 1 - j):
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                sign = True

    return (a)


if __name__ == '__main__':
    a = [1, 3, 4, 2, 6, 9, 12, 3, 22]
    mao_pao(a)
    print(a)
