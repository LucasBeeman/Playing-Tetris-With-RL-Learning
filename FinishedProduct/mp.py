from multiprocessing.dummy import freeze_support
import time
import multiprocessing

start = time.perf_counter()

def do_something():
    print('sleeping')
    time.sleep(1)
    print('WAKE UP')


p1 = multiprocessing.Process(target=do_something)
p2 = multiprocessing.Process(target=do_something)
if __name__ == '__main__':
    freeze_support()
    p1.start()
    p2.start()
    p1.join()
    p2.join()

finsih = time.perf_counter()

print(f'Finished in {round(finsih-start, 2)} second(s)')