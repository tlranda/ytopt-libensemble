# Local import
from .JobInfo import JobInfo

class SleepJobInfo(JobInfo):
    basic_job_string = "sleep {duration}"
    def __init__(self, spec, duration):
        self.spec = spec
        self.duration = duration
    @property
    def nodes(self):
        return self.duration

