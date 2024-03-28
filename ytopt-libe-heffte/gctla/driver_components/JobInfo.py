import shlex
import subprocess

# Local imports
from .PythonEval import PythonEval

# Format the shell command based on own attributes and track # of occupied nodes by the job
class JobInfo():
    basic_job_string = ""
    prelaunch_tasks = []
    postlaunch_tasks = []
    finalize_tasks = []
    def __str__(self):
        return self.basic_job_string.format(**self.__dict__)
    def shortName(self):
        other_attrs = dict()
        for (k,v) in self.__dict__.items():
            if k.startswith('_') or k in ['spec','prelaunch_tasks','postlaunch_tasks','finalize_tasks']:
                continue
            other_attrs[k] = str(v)
        return str(self.__class__.__name__)+str(other_attrs)
    def specName(self):
        return self.spec
    @property
    def nodes(self):
        pass
    def process_cmd_queue(self, queue, identifier):
        n_queued = len(queue)
        if n_queued == 0:
            print("\t"+f"{identifier} -- Nothing to do!")
        for idx, cmd in enumerate(queue):
            if type(cmd) is str:
                command_split = shlex.split(cmd.format(**self.__dict__))
                print("\t"+f"{identifier} command {idx+1}/{n_queued}: SHELL_COMMAND")
                print("\t"+" ".join(command_split))
                new_sub = subprocess.Popen(command_split)
                retcode = new_sub.wait()
                if retcode != 0:
                    raise ValueError("Command failed")
            elif isinstance(cmd, PythonEval):
                print("\t"+f"{identifier} Python command {idx+1}/{n_queued}: ", end="")
                cmd(self)
    def prelaunch(self):
        self.process_cmd_queue(self.prelaunch_tasks, "Pre-launch")
    def format(self):
        return shlex.split(str(self))
    def postlaunch(self):
        self.process_cmd_queue(self.postlaunch_tasks, "Post-launch")
    def finalize(self):
        self.process_cmd_queue(self.finalize_tasks, "Finalize")
    def update_tasks(self, prelaunch, postlaunch, finalize):
        if prelaunch is not None:
            self.prelaunch_tasks = prelaunch
        if postlaunch is not None:
            self.postlaunch_tasks = postlaunch
        if finalize is not None:
            self.finalize_tasks = finalize
    def mutate_basic_job_string(self, _callable, *args, **kwargs):
        self.basic_job_string = _callable(*args, **kwargs)

