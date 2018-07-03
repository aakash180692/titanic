import multiprocessing as mp

def cube(x):
    return x**3

def myfunc(x):
    return [i for i in range(x)]

A = []
def mycallback(x):
    print('mycallback is called with {}'.format(x))
    A.extend(x)

if __name__ == "__main__":
    pool = mp.Pool()
    results = []
    for x in (1,2):
        r = pool.apply_async(myfunc, (x,), callback=mycallback)
        results.append(r)
    for r in results:
        r.wait()
    
if __name__ == "__main__":
    pool    = mp.Pool(processes=2)
    results = [pool.apply_async(cube, args=(x,)) for x in range(1,7)]
    print("bahgsda")
    print([result.get() for result in results])