
class MyLogger(object):
    def __init__(self, hparams):
        self.is_log_beside = hparams.is_log_beside
        self.stdout_level = hparams.stdout_level    #1
        self.log_level = hparams.log_level
        if self.is_log_beside:
            self.f = open(hparams.log_path, 'w')

    def log(self, *args, sep=' ', end='\n', level=2):
        if level >= self.stdout_level:
            print(*args, sep=sep, end=end)
        if self.is_log_beside and self.f:
            print(*args, sep=sep, end=end, file=self.f)

    def close(self):
        if self.is_log_beside and self.f:
            self.f.close()


