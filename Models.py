import pymonntorch as pymo

class LIF (pymo.Behavior):
    def initialize(self, neural_group):
        self.R = self.parameter("R", default=None, required=True)
        self.tau = self.parameter("tau", default=None, required=True)
        self.u_rest = self.parameter("u_rest", default=-67, required=True)
        self.u_reset = self.parameter("u_reset", default=-75, required=True)
        self.threshold = self.parameter("threshold", default=-37, required=True)
        self.ratio = self.parameter("ration", default=1.1, required=False)
        self.refactory_period = self.parameter("refractory", default=False, required=False)

        neural_group.u =((neural_group.vector("uniform") #returns between 0-1
                        * (self.threshold - self.u_reset) #returns the length of values LIF can get # type: ignore
                        * self.ratio) #sets a ratio to boost the starting potential with
                        + self.u_reset) #sets the min value to be u_reset

        neural_group.spike = neural_group.u >= self.threshold
        neural_group.u[neural_group.spike] = self.u_reset
        return

    def forward(self, neural_group):
        leakage = -(neural_group.u - self.u_rest)
        input_u = self.R * neural_group.I
        if self.refactory_period:
            neural_group.is_refactory = (neural_group.u + 5) < self.u_rest 
            input_u[neural_group.is_refactory] = 0
        neural_group.u += (leakage + input_u) / self.tau * neural_group.network.dt

        neural_group.spike = neural_group.u >= self.threshold
        neural_group.u[neural_group.spike] = self.u_reset #resets u of neurons with spikes
        return
    

class ELIF (pymo.Behavior):
    def initialize(self, neural_group):
        self.R = self.parameter("R", default=None, required=True)
        self.tau = self.parameter("tau", default=None, required=True)
        self.u_rest = self.parameter("u_rest", default=-67, required=True)
        self.u_reset = self.parameter("u_reset", default=-75, required=True)
        self.threshold = self.parameter("threshold", default=-37, required=True)
        self.ratio = self.parameter("ration", default=1.1, required=False)
        self.sharpness = self.parameter("sharpness", default=None, required=True)
        self.firing_threshold = self.parameter("phi", default=None, required=True)
        self.refactory_period = self.parameter("refractory", default=False, required=False)

        neural_group.u =((neural_group.vector("uniform") #returns between 0-1
                        * (self.threshold - self.u_reset) #returns the length of values LIF can get # type: ignore
                        * self.ratio) #sets a ratio to boost the starting potential with
                        + self.u_reset) #sets the min value to be u_reset

        neural_group.spike = neural_group.u >= self.threshold
        neural_group.u[neural_group.spike] = self.u_reset
        return

    def forward(self, neural_group):
        leakage = - (neural_group.u - self.u_rest) + (self.sharpness * pymo.torch.exp((neural_group.u - self.firing_threshold)/self.sharpness))
        input_u = self.R * neural_group.I
        if self.refactory_period:
            neural_group.is_refactory = (neural_group.u + 5) < self.u_rest
            input_u[neural_group.is_refactory] = 0
        neural_group.u += (leakage + input_u) / self.tau * neural_group.network.dt

        neural_group.spike = neural_group.u >= self.threshold
        neural_group.u[neural_group.spike] = self.u_reset #resets u of neurons with spikes
        return


class AELIF (pymo.Behavior):
    def initialize(self, neural_group):
        self.R = self.parameter("R", default=None, required=True)
        self.tau = self.parameter("tau", default=None, required=True)
        self.u_rest = self.parameter("u_rest", default=-67, required=True)
        self.u_reset = self.parameter("u_reset", default=-75, required=True)
        self.threshold = self.parameter("threshold", default=-37, required=True)
        self.ratio = self.parameter("ration", default=1.1, required=False)
        self.sharpness = self.parameter("sharpness", default=None, required=True)
        self.firing_threshold = self.parameter("phi", default=None, required=True)
        self.A_param = self.parameter("A", default=None, required=True)
        self.B_param = self.parameter("B", default=None, required=True)
        self.tau_w = self.parameter("tau_w", default=None, required=True)
        self.refactory_period = self.parameter("refractory", default=False, required=False)

        neural_group.adaptation = neural_group.vector(mode="zeros")
        neural_group.u =((neural_group.vector("uniform") #returns between 0-1
                        * (self.threshold - self.u_reset) #returns the length of values LIF can get # type: ignore
                        * self.ratio) #sets a ratio to boost the starting potential with
                        + self.u_reset) #sets the min value to be u_reset

        neural_group.spike = neural_group.u >= self.threshold
        neural_group.u[neural_group.spike] = self.u_reset
        return

    def forward(self, neural_group):
        leakage = - (neural_group.u - self.u_rest) + (self.sharpness * pymo.torch.exp((neural_group.u - self.firing_threshold)/self.sharpness)) - (self.R * neural_group.adaptation)
        input_u = self.R * neural_group.I
        if self.refactory_period:
            neural_group.is_refactory = (neural_group.u + 5) < self.u_rest
            input_u[neural_group.is_refactory] = 0
        neural_group.u += ((leakage + input_u) / self.tau) * neural_group.network.dt

        neural_group.spike = neural_group.u >= self.threshold

        memory = (self.A_param * (neural_group.u - self.u_rest) - neural_group.adaptation) / self.tau_w
        effect = self.B_param
        neural_group.adaptation += (memory) * neural_group.network.dt
        neural_group.adaptation[neural_group.spike] += (effect) * neural_group.network.dt

        neural_group.u[neural_group.spike] = self.u_reset #resets u of neurons with spikes
        return

