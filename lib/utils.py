
import time,datetime

class TimedLog():
    def __init__(self, log, title):
        self.logger = log
        self.title = title
        self.tictime = time.time()

    def tic(self, cstr):
        return self.log(cstr, True)

    def toc(self, cstr):
        return self.log(cstr, False)

    def log(self, cstr = "", reset = False):
        difftime = time.time() - self.tictime
        self.logger.log("[" + self.title +  " @ " + str(datetime.timedelta(seconds = difftime)) + "] " + cstr)
        if reset:
            self.tictime = time.time()
        return difftime

import sys
#   Basic logging class to gather everything at the same place
class Log:
    def __init__(self, path, keepifexists = False):
        self.logfile = open(path, "a" if keepifexists else "w")
        self.stdout = sys.stdout

    def __del__(self):
        self.logfile.close() 

    def log(self, str):
        today = datetime.datetime.now()
        date_time = today.strftime("%d/%m/%Y, %H:%M:%S")
        self.logfile.write("[" + date_time + "] " + str + "\n")
        self.stdout.write("[" + date_time + "] " + str + "\n")

    def close(self):
        self.logfile.close()

    def tic(self, title, cstr = ""):
        nlog = TimedLog(self, title)
        nlog.tic(cstr)

        return nlog

    def hijack_std(self):
        sys.stdout = self

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.log(line.rstrip())

    def flush(self):
        pass
