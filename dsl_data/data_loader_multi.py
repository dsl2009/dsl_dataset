from multiprocessing import Process,Queue
def get_thread(gen, thread_num):
    def new_get_data(quene):
        while True:
            try:
                r = next(gen)
                quene.put(r)
            except:
                print('err')

    numT = thread_num
    q = Queue(maxsize=10)
    ps = []
    for p in range(numT):
        ps.append(Process(target=new_get_data, args=(q,)))
    for pd in ps:
        pd.start()
    return q




