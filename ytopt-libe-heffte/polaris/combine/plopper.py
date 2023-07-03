import subprocess
import random
import numpy as np
import warnings
import os, stat, signal
import math

class Plopper:
    def __init__(self,sourcefile,outputdir,formatSTR=None):
        # Initilizing global variables
        self.sourcefile = sourcefile
        self.outputdir = outputdir+"/tmp_files"
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)
        if formatSTR is None:
            self.cmd_template = "mpiexec -n {mpi_ranks} --ppn {ranks_per_node} --depth {depth} sh ./set_affinity_gpu_polaris.sh {interimfile}"
        else:
            self.cmd_template = formatSTR

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

    def read_logs(self, this_log, workerID):
        result = None
        try:
            with open(this_log,"r") as logged:
                lines = [_.rstrip() for _ in logged.readlines()]
                for line in lines:
                    if "Performance: " in line:
                        split = [_ for _ in line.split(' ') if len(_) > 0]
                        result = -1 * float(split[1])
                        print(f"[worker {workerID} - plopper] evaluation OK: {result}")
                        break
        except Exception as e:
            eval_warning = f"Evaluation raised {e.__class__.__name__}: {e.args}"
            warnings.warn(eval_warning)
            # Evaluation code 3 reserved for Python log processing error
            return 3.0
        return result

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
        # Divide and promote instead of truncate
        j = math.ceil(ranks_per_node * dictVal['P9'] / 64)
        # Command template set up in ytopt_obj.py -- this separates the need to look at what system we're on out of the plopper
        cmd = self.cmd_template.format(mpi_ranks=mpi_ranks, ranks_per_node=ranks_per_node, depth=dictVal['P9'], j=j, interimfile=interimfile)
        print(f"[worker {workerID} - plopper] runs: {cmd}")

        results = []
        for attempt in range(n_repeats):
            this_log = f"{self.outputdir}/{counter}_{attempt}.log"
            logged = False
            with open(this_log,"w") as logfile:
                execution_status = subprocess.Popen(cmd, shell=True, stdout=logfile, stderr=logfile)
                child_pid = execution_status.pid
                try:
                    execution_status.communicate(timeout=app_timeout)
                except subprocess.TimeoutExpired:
                    print(f"[worker {workerID} - plopper] triggers TIMEOUT ({app_timeout} s) on {interimfile} (procID: {child_pid})")
                    execution_status.kill()
                    #kill_cmd = ['kill', '-s', '9', str(child_pid)]
                    #kill_status = subprocess.run(kill_cmd, shell=True)
                    #os.kill(child_pid, signal.SIGTERM)
                    check_cmd = ['ps', '--pid', str(child_pid)]
                    check_status = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE)
                    if len(check_status.stdout.decode('utf-8').rstrip('\n').split('\n')) != 1:
                        print(f"FAILED TO KILL PROCESS {child_pid}")
                else:
                    logged = True
            if logged and execution_status.returncode != 0:
                # Evaluation code 2.* reserved for Execution errors
                results.append(2. + execution_status.returncode/1000)
                eval_error_warning = f"{workerID} evaluation had bad return code {results[-1]} on {interimfile}"
                warnings.warn(eval_error_warning)
                print(f"[worker {workerID} - plopper] receives: ERROR {execution_status.returncode} from {interimfile}")
                continue
            elif logged:
                logged = self.read_logs(this_log, workerID)
                if logged is not None:
                    results.append(logged)
                else:
                    bad_logs_warning = f"{workerID} failed to read logs of a successful evaluation of {interimfile}"
                    warnings.warn(bad_logs_warning)
                    results.append(1.1)
            else:
                # Timed out evaluations MAY be recoverable if the log is readable
                to_result = self.read_logs(this_log, workerID)
                if to_result is None:
                    # Evaluation code 1.0 reserved for unrecoverable Timeout
                    results.append(1.0)
                    print(f"[worker {workerID} - plopper] evaluation TIMED OUT: {results[-1]} from {interimfile}; non-recoverable")
                else:
                    results.append(to_result)

        usable_results = [_ for _ in results if _ < 0]
        if len(usable_results) > 0:
            # Get the best result we observed (unstable measurements)
            return min(usable_results)
        else:
            # Ensure the gravest error is reported
            return max(results)

