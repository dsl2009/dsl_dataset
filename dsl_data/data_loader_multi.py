from multiprocessing import Process,Queue
def get_thread(gen, thread_num):
    def new_get_data(quene):
        while True:
            r = next(gen)
            quene.put(r)

    numT = thread_num
    q = Queue(numT)
    ps = []
    for p in range(numT):
        ps.append(Process(target=new_get_data, args=(q,)))
    for pd in ps:
        pd.start()
    return q




