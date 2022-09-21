class Writer:

    def __init__(self, args):
        self.args = args

    def write(self, msg):
        if self.args.local_rank == 0:
            print(msg)
