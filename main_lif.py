import pymonntorch as pymo
import torch
import Models
import TimeResolution
import Currents
import Synapse
import Dendrite
import numpy as np

torch.manual_seed(43)

network = pymo.Network(device="cpu", synapse_mode="SxD", dtype=torch.float64, behavior={
    2: TimeResolution.TimeResolution(dt=1)
}, tag="main_net")

ng1 = pymo.NeuronGroup(net=network, size=1, behavior={
    10: Currents.ConstantCurrent(
        current = 12,
        noise = False
    ),
    #10: Currents.StepCurrent(
    #    initial_current = 10,
    #    step_threshold = 50,
    #    final_current = 50,
    #    noise = False
    #),
    #10: Currents.MultiStepCurrent(
    #    initial_current = 10,
    #    final_current = 50,
    #    step = 1,
    #    noise = False
    #),
    #10: Currents.SineCurrent(
    #    initial_current = 60,
    #    frequency_modifier = 0.25,
    #    current_modifier = 30,
    #    noise = False
    #),
    #20: Models.LIF(
    #    R = 10,
    #    threshold = -37,
    #    u_rest = -67,
    #    u_reset = -75,
    #    tau = 10,
    #    refractory = False
    #),
    17: Dendrite.Dendrite(

    ),
    #20: Models.ELIF(
    #    R = 5,
    #    threshold = -37,
    #    u_rest = -67,
    #    u_reset = -75,
    #    tau = 10,
    #    sharpness = 100,
    #    phi = -37,
    #    refractory = False
    #),
    20: Models.AELIF(
        R = 5,
        threshold = -37,
        u_rest = -67,
        u_reset = -75,
        tau = 10,
        sharpness = 2,
        phi = -39,
        A = 1,
        B = 10,
        tau_w = 100,
        refractory = False
    ),
    100: pymo.Recorder(
        variables = ["torch.sum(I)", "u", "I"],
        tag = "ng1_recorder"
    ),
    101: pymo.EventRecorder(
        variables = ["spike"],
        tag = "ng1_eventrecorder"
    )
}, tag="ng1")

syn = pymo.SynapseGroup(net=network, src=ng1, dst=ng1, behavior={
    15: Synapse.all_to_all_connection(
        coef = 5
    )
}, tag="synapse1")

network.initialize()

#network.simulate_iteration()
network.simulate_iterations(100)



import matplotlib.pyplot as plt

plt.plot(network["ng1_recorder", 0].variables["u"][:,:1])
plt.xlabel("time")
plt.ylabel("u")
plt.title("u-t")
plt.show()

plt.plot(network["ng1_recorder", 0].variables["I"][:,:1])
plt.xlabel("time")
plt.ylabel("I(current)")
plt.title("I-t")
plt.show()

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
colors = np.random.uniform(15, 80, (network["ng1_eventrecorder", 0].variables["spike"]).size()[0])
plt.scatter(x, np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]), c = colors)
plt.xlabel('time')
plt.title('spike time pattern')
plt.show()

x_new = [x[i+1]-x[i] for i in range(len(x)-1)]
plt.scatter(x_new, np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]-1), c = colors[:-1])
plt.xlabel('time')
plt.title('Spike time interval')
plt.show()

dot_size = []
x_new_sorted = sorted(x_new)
for i in range(len(x_new_sorted)):
    if i == 0 or x_new_sorted[i-1]!=x_new_sorted[i]:
        dot_size.append(10)
    else:
        dot_size.append(dot_size[-1]+10)
plt.scatter(x_new_sorted, np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]-1), c = colors[:-1], s = dot_size)
plt.xlabel('time')
plt.title('Cumulative spike time interval')
plt.show()


print((network["ng1_eventrecorder", 0].variables["spike"]).size())