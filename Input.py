import pymonntorch as pymo

class Input(pymo.Behavior):
    def initialize(self, neural_group):
        self.input = self.parameter("matrix", default=pymo.np.ndarray, required=True)
        self.iterationCount = 1

        for i in range(self.input.shape[0]): # type: ignore
            for j in range(self.input.shape[1]): # type: ignore
                if self.input[i, j] == 0: # type: ignore
                    self.input[i, j] = False # type: ignore
                else:
                    self.input[i, j] = True # type: ignore

        self.timewidth = self.input.shape[1] # type: ignore
        neural_group.spike = self.input[:,1] # type: ignore
        return

    def forward(self, neural_group):
        if self.timewidth > self.iterationCount:
            neural_group.spike = self.input[:,self.iterationCount] # type: ignore
            #print("in time ", self.iterationCount, " set neurons to ", neural_group.spike)
            self.iterationCount += 1
        return
    