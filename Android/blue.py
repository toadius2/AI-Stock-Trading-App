import numpy
bad = [1,1,1,0,0,0]
# bad = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
# bad = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]


stop_chance = bad.count(1)/len(bad)

def reward(N, B):
    count = 0
    for k in range(1, len(bad)+1):
        if bad[k-1]==1:
            count += (1/N)*(-1*B)
        else:
            count += (1/N)*(k)

    print(count)
    return count

def T(N,B):
    arr = []
    for k in range(1, len(bad) + 1):
        if bad[k-1]==1:
            arr.append(((1/N)*(-1*B),k))
        else:
            arr.append(((1/N)*(k),k))
    return arr

def bellman(N, B, ep=.001):
    d = 0
    count = []
    for i in range(10000):
        V = B + max(reward(N,B), 0)
        # if B < potential reward:
        #     stop game
        if B==V:
            print('Stop')
            break
        else:
            print('Roll')
            B += max(T(N,B))[1]
        count.append(V)
        # d = max(count[len(count)-1] - count[len(count)-2], d)
    return count


print(bellman(6, 0)[-10:])
