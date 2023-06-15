import subprocess
import random
import numpy as np
import warnings
import os, stat, signal
import math

class Plopper:
    def __init__(self,sourcefile,outputdir):
        # Initilizing global variables
        self.sourcefile = sourcefile
        self.outputdir = outputdir+"/tmp_files"
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    #Creating a dictionary using parameter label and value
    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            if type(v) is np.ndarray and v.shape == ():
                v = v.tolist()
            dictVal[p] = v
        return(dictVal)

    #Replace the Markers in the source file with the corresponding values
    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value and value != 'None': #For empty string options
                            modify_line = modify_line.replace('#'+key, str(value))
                if modify_line != line:
                    f2.write(modify_line)
                else:
                    #To avoid writing the Marker
                    f2.write(line)
        # Ensure the script is executable!
        os.chmod(outputfile,
                 stat.S_IRWXU |
                 stat.S_IRGRP | stat.S_IXGRP |
                 stat.S_IROTH | stat.S_IXOTH)

    # Function to find the execution time of the interim file, and return the execution time as cost to the search module
    def findRuntime(self, x, params, workerID, app_timeout, mpi_ranks, ranks_per_node, n_repeats=1):
        #print(workerID, x, params)
        dictVal = self.createDict(x, params)
        #print(workerID, dictVal)

        # Generate intermediate file
        counter = random.randint(1, 10001) # Reduce collision at greater sampling intervals
        interimfile = f"{self.outputdir}/{counter}.sh"
        self.plotValues(dictVal, self.sourcefile, interimfile)

        kernel_dir = os.path.dirname(self.sourcefile)
        #print(workerID, cmd2)

        #Find the execution metric
        # POLARIS
        #cmd = f"mpiexec -n {mpi_ranks} --ppn {ranks_per_node} --depth {dictVal['P9']} sh ./set_affinity_gpu_polaris.sh {interimfile}"
        
        # THETA KNL
        # Divide and promote instead of truncate
        j = math.ceil(dictVal['P9'] / 64)
        cmd = f"aprun -n {mpi_ranks} -N {ranks_per_node} -cc depth -d {dictVal['P9']} -j {j} sh {interimfile}"
        print(workerID,cmd)

        results = []
        for attempt in range(n_repeats):
            this_log = f"{self.outputdir}/{counter}_{attempt}.log"
            with open(this_log,"w") as logfile:
                execution_status = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=logfile)
                child_pid = execution_status.pid
                try:
                    execution_status.communicate(timeout=app_timeout)
                except subprocess.TimeoutExpired:
                    results.append(1.0)
                    os.kill(child_pid, signal.SIGTERM)
                    continue
            if execution_status.returncode != 0:
                results.append(2. + execution_status.returncode/1000)
                warnings.warn(f"{workerID} evaluation had bad return code {results[-1]}")
                continue
            try:
                with open(this_log,"r") as logged:
                    lines = [_.rstrip() for _ in logged.readlines()]
                    for line in lines:
                        if "Performance: " in line:
                            split = [_ for _ in line.split(' ') if len(_) > 0]
                            results.append(-1 * float(split[1]))
                            break
            except Exception as e:
                warnings.warn(f"Evaluation raised {e.__class__.__name__}: {e.args}")
        usable_results = [_ for _ in results if _ < 0]
        if len(usable_results) > 0:
            # Get the best result we observed (unstable measurements)
            return min(usable_results)
        else:
            # Ensure the gravest error is reported
            return max(results)

