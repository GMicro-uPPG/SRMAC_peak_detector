from datetime import datetime


class time:
    def getTimestamp():
        now = datetime.now()
        return datetime.timestamp(now)
    #end-def

    def getTime():
        timestamp = datetime.fromtimestamp(time.getTimestamp())
        return timestamp
    #end-def
#end-class

class log:
    dataFile = None

    def openLogFile():
        log.dataFile = open('log.txt', 'w+')
    #end-def

    def printWriteLog(data):
        print(data)
        log.dataFile.write("%s\n" % data)
    #end-def

    def closeLogFile():
        log.dataFile.close()
    #end-def
#end-class