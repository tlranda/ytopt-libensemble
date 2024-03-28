import numpy as np

# This class emulates a polling object for debug purposes (run real LibeJobs without executing them)
class PretendSubprocess:
    def __init__(self, command, max_polls = None):
        self.poll_count = 0
        self.max_polls = max_polls if max_polls is not None else np.random.randint(1,4)
        self.command = command
        self.returncode = 0
    def poll(self):
        self.poll_count += 1
        if self.poll_count >= self.max_polls:
            return 0
        else:
            return None
    def __str__(self):
        return f"<Pretend Popen max_polls: {self.max_polls} returncode: {self.returncode} args: {self.command}>"


