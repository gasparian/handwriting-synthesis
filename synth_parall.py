import os
import argparse
from multiprocessing import Queue, Process

class synthWorker(Process):
    def __init__(self, gpuid, queue):
        Process.__init__(self, name='ModelProcessor')
        self._gpuid = gpuid
        self._queue = queue

    def run(self):

        #set enviornment
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpuid)
        hand = Hand(path=path, length=words_count)

        print('net init done', self._gpuid)

        while True:
            word = self._queue.get()
            if word == None:
                self._queue.put(None)
                break

            words_count = len(words)*len(biases)*len(styles)*len(stroke_colors)*len(stroke_widths)
            for line in tqdm(words, desc='words'):
                for style in styles: 
                    for bias in biases:
                        hand.write(
                            filename='%s_b%s_s%s' % (line, bias, style),
                            lines=[line],
                            biases=[bias],
                            styles=[style],
                            stroke_colors=stroke_colors,
                            stroke_widths=stroke_widths)
            print('woker', self._gpuid, ' word ', word)

        print('net done ', self._gpuid)

class Scheduler:
    def __init__(self, gpuids):
        self._queue = Queue()
        self._gpuids = gpuids

        self.__init_workers()

    def __init_workers(self):
        self._workers = list()
        for gpuid in self._gpuids:
            self._workers.append(synthWorker(gpuid, self._queue))


    def start(self, xfilelst):

        # put all of files into queue
        for xfile in xfilelst:
            self._queue.put(xfile)

        #add a None into queue to indicate the end of task
        self._queue.put(None)

        #start the workers
        for worker in self._workers:
            worker.start()

        # wait all fo workers finish
        for worker in self._workers:
            worker.join()
        print "all of workers have been done"

                

def run(img_path, gpuids):
    #scan all files under img_path
    xlist = list()
    for xfile in os.listdir(img_path):
        xlist.append(os.path.join(img_path, xfile))
    
    #init scheduler
    x = Scheduler(gpuids)
    
    #start processing and wait for complete 
    x.start(xlist)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--imgpath", help="path to your images to be proceed")
#     parser.add_argument("--gpuids",  type=str, help="gpu ids to run" )

#     args = parser.parse_args()

#     gpuids = [int(x) for x in args.gpuids.strip().split(',')]

#     print args.imgpath
#     print gpuids

#     run(args.imgpath, gpuids)
    