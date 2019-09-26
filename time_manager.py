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
