from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
import csv
import time

from psutil import cpu_percent, virtual_memory, cpu_count


class UtilizationLogger:
    """
    This class is responsible for logging CPU and memory utilization to a csv file.
    
    TODO: Log GPU utilization as well.
    """
    def __init__(self, csv_path: Path, period: float = 0.2, start = True):
        self.period = period
        self.writer = UtilizationLogger._prepare_csv(csv_path)
        
        # This ensures that there's no race condition
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.logging = start
        
    @staticmethod
    def _prepare_csv(path: Path):
        """
        Prepares the CSV file by creating it and writing the header dependigng on the number of CPUs.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = csv.writer(path.open('w'))
        num_cpus = cpu_count()
        writer.writerow(
            chain(['time', 'memory', 'cpu'], (f'cpu{i}' for i in range(num_cpus)))
        )
        
        return writer
        
    def log(self):
        """
        Logs the current CPU and memory utilization.
        """
        timestamp = time.time()
        memory = virtual_memory().percent
        cpu_avg = cpu_percent()
        cpus = cpu_percent(percpu=True)
        
        row = chain([timestamp, memory, cpu_avg], cpus)
        self.writer.writerow(row)
        
        return row
    
    def start(self):
        """
        Starts the logging process.
        """
        self.logging = True
        self.pool.submit(self._log_periodically)
        
    def stop(self):
        """
        Stops the logging process.
        """
        self.logging = False
    
    def _log_periodically(self):
        """
        Logs the utilization periodically by submitting the log method to the thread pool,
        then sleeping for the period.
        
        This blocks, so it should be run in a separate thread.
        This is done in `start`, which submits it to the thread pool.
        """
        while self.logging:
            self.pool.submit(self.log)
            time.sleep(self.period)


if __name__ == '__main__':
    logger = UtilizationLogger(Path('utilization.csv'))
    logger.start()
    time.sleep(10)
    logger.stop()
    
    print('Done!')
