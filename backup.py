import sys
import time
import os
import shutil
import datetime

def validateDirectory(dirpath):
    if not dirpath:
        raise ValueError("Directory path cannot be empty")
    
    if not os.path.exists(dirpath):
        raise ValueError(f"Directory does not exist: {dirpath}")
    
    abs_path = os.path.abspath(dirpath)
    
    if not os.path.isdir(abs_path):
        raise ValueError(f"Path exists but is not a directory: {dirpath}")
    
    if not os.access(abs_path, os.R_OK):
        raise ValueError(f"Directory is not accessible: {dirpath}")

    return abs_path
    
def validateInterval(interval):
    try:
        value = float(interval)
    except: 
        raise ValueError
    
    if not (value > 0 and value != float('inf')):
        if value <= 0:
            raise ValueError(f"Interval must be positive, got: {value}")
        elif value == float('inf'):
            raise ValueError("Interval cannot be infinity")
        elif value != value: 
            raise ValueError("Interval cannot be NaN")
        
    return value

def validateAmount(amount):
    try:
        value = int(amount)
    except: 
        raise ValueError("Amount of synchronizations must be an integer")
    
    if value <= 0:
        raise ValueError(f"Amount must be positive, got: {value}")
    
    return value


def validateFile(filepath):
    if not filepath or not filepath.strip():
        raise ValueError("File path cannot be empty")
    abs_path = os.path.abspath(filepath.strip())
    
    if os.path.exists(abs_path):
        if os.path.isdir(abs_path):
            raise ValueError(f"Path exists but is a directory, not a file: {filepath}")
        if not os.access(abs_path, os.W_OK):
            raise ValueError(f"File exists but is not writable: {filepath}")
    else:
        try:
            parent_dir = os.path.dirname(abs_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            with open(abs_path, 'a'):
                pass
                
        except OSError as e:
            raise ValueError(f"Cannot create file '{filepath}': {e}")
        except PermissionError:
            raise ValueError(f"Permission denied when creating file: {filepath}")
    
    return abs_path

class Synchronizer:
    def __init__(self, arguments):
        self.source = validateDirectory(arguments[0])
        self.destination = validateDirectory(arguments[1])
        self.interval  = validateInterval(arguments[2])
        self.amount = validateAmount(arguments[3])
        self.logs = validateFile(arguments[4])
    
    def synchronize(self):
        with open(self.logs, "a+") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"\n--- Sync started at {timestamp} ---\n")
        
        try:
            source_items = set(os.listdir(self.source))
            dest_items = set(os.listdir(self.destination))
            
            to_update = source_items & dest_items  # Items in both
            to_add = source_items - dest_items     # Items only in source
            to_delete = dest_items - source_items  # Items only in destination
            
            for item in to_update:
                source_path = os.path.join(self.source, item)
                dest_path = os.path.join(self.destination, item)
                
                try:
                    if os.path.isdir(source_path):
                        if os.path.isfile(dest_path):
                            os.remove(dest_path)
                            shutil.copytree(source_path, dest_path)
                            with open(self.logs, "a+") as file:
                                file.write(f"Replaced file with directory: {item}\n")
                        else:
                            shutil.rmtree(dest_path)
                            shutil.copytree(source_path, dest_path)
                            with open(self.logs, "a+") as file:
                                file.write(f"Updated directory: {item}\n")
                    else:
                        if os.path.isdir(dest_path):
                            shutil.rmtree(dest_path)
                            shutil.copy2(source_path, dest_path)
                            with open(self.logs, "a+") as file:
                                file.write(f"Replaced directory with file: {item}\n")
                        else:
                            shutil.copy2(source_path, dest_path)
                            with open(self.logs, "a+") as file:
                                file.write(f"Updated file: {item}\n")
                except Exception as e:
                    with open(self.logs, "a+") as file:
                        file.write(f"Error updating {item}: {e}\n")
            
            for item in to_add:
                source_path = os.path.join(self.source, item)
                dest_path = os.path.join(self.destination, item)
                
                try:
                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, dest_path)
                        with open(self.logs, "a+") as file:
                            file.write(f"Added directory: {item}\n")
                    else:
                        shutil.copy2(source_path, dest_path)
                        with open(self.logs, "a+") as file:
                            file.write(f"Added file: {item}\n")
                except Exception as e:
                    with open(self.logs, "a+") as file:
                        file.write(f"Error adding {item}: {e}\n")
            
            for item in to_delete:
                dest_path = os.path.join(self.destination, item)
                
                try:
                    if os.path.isdir(dest_path):
                        shutil.rmtree(dest_path)
                        with open(self.logs, "a+") as file:
                            file.write(f"Removed directory: {item}\n")
                    else:
                        os.remove(dest_path)
                        with open(self.logs, "a+") as file:
                            file.write(f"Removed file: {item}\n")
                except Exception as e:
                    with open(self.logs, "a+") as file:
                        file.write(f"Error removing {item}: {e}\n")
            
            with open(self.logs, "a+") as file:
                file.write(f"--- Sync completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                
        except Exception as e:
            with open(self.logs, "a+") as file:
                file.write(f"Critical error during sync: {e}\n")
            
    def start(self):
        with open(self.logs, "a+") as file:
            file.write(f"\nStarting {self.amount} synchronizations with {self.interval}s intervals...")
    
        for x in range(self.amount):
            with open(self.logs, "a+") as file:
                file.write(f"\n--- Synchronization {x+1}/{self.amount} ---")
            try:
                start_time = time.time()
                self.synchronize()
                end_time = time.time()
                with open(self.logs, "a+") as file:
                    file.write(f"Sync completed in {end_time - start_time:.2f} seconds")
            
                if x < self.amount - 1:
                    time.sleep(self.interval)
                
            except KeyboardInterrupt:
                with open(self.logs, "a+") as file:
                    file.write("\nSynchronization interrupted by user")
                break
            except Exception as e:
                with open(self.logs, "a+") as file:
                    file.write(f"ERROR in sync {x+1}: {e}\n")
                continue

def main():
    n = len(sys.argv)
    if n != 6:
        print("Script should be given 5 arguments:" \
        "1. Path to the source directory" \
        "2. Path to the destination directory" \
        "3. Interval between synchronizations" \
        "4. Amount of synchronizations" \
        "5. Path to log file")
        return 
    arguments = sys.argv[1:]
    synchro = Synchronizer(arguments)
    synchro.start()

if __name__ == "__main__":
    main()