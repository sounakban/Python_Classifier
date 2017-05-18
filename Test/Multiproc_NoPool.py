import multiprocessing

def F(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)

def G(n, st):
    print 'Fibbonacci of {}: {}'.format(n, F(n)), st

processes = []
for i in range(25, 35):
    processes.append(multiprocessing.Process(target=G, args=(i, "this")))

for pro in processes:
    pro.start()
