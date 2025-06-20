import sys
import time
import os
import shutil
def help(error):
    if error == 0:
        print("Script should be given 5 arguments:" \
        "1. Path to the source directory" \
        "2. Path to the destination directory" \
        "3. Interval between synchronizations" \
        "4. Amount of synchronizations" \
        "5. Path to log file")

def validateDirectory(dirpath):
    try:
        os.chdir(dirpath)
        return os.path.abspath
    except:
        raise ValueError("Source and Destination must be a valid path to directory")
    
def validateInterval(interval):
    try:
        return float(interval)
    except: 
        raise ValueError

def validateAmount(amount):
    try:
        return int(amount)
    except: 
        raise ValueError("Amount of synchronizations must be an integer")

class Synchronizer:
    def __init__(self, arguments):
        self.cur = os.path.abspath
        self.source = validateDirectory(arguments[0])
        self.destination = validateDirectory(arguments[1])
        self.interval  = validateInterval(arguments[2])
        self.amount = validateAmount(arguments[3])
        self.logs = arguments[4]
    
    def synchronize(self):
        os.chdir(self.source)
        filesS = [f for f in os.listdir(self.source) if os.path.isfile(os.path.join(self.source, f))]
        os.chdir(self.destination)
        filesD = [f for f in os.listdir(self.destination) if os.path.isfile(os.path.join(self.destination, f))]
        os.chdir(self.source)
        toUpdate = [f for f in filesD if f in filesS]
        toDelete = [f for f in filesD if f not in filesS]
        toAdd = [f for f in filesS if f not in filesD]
        with open(self.logs, "a+") as file:
            for f in filesS:
                for f in toUpdate:
                    shutil.copy(f, self.destination + "/" + f)
                    file.write(f"Updated file {f}")

                for f in toAdd:
                    shutil.copy(f, self.destination + "/" + f)
                    file.write(f"Started tracking file {f}")

                for f in toDelete:
                    os.remove(self.destination + "/" + f)

            
    def start(self):
        for x in range(self.amount):
            time.sleep(self.interval)

def main():
    n = len(sys.argv)
    print(sys.argv)
    if n != 6:
        help(0)
        return 
    arguments = sys.argv[1:]
    synchro = Synchronizer(arguments)
    synchro.start()

if __name__ == "__main__":
    main()