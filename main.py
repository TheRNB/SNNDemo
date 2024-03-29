import pymonntorch as pymo
import numpy as np
import Models
import TimeResolution
import Currents
import Synapse
import Dendrite

pymo.torch.manual_seed(43)

network = pymo.Network(device="cpu", synapse_mode="SxD", dtype=pymo.torch.float64, behavior={
    2: TimeResolution.TimeResolution(dt=1)
}, tag="main_net")

ng1 = pymo.NeuronGroup(net=network, size=80, behavior={
    10: Currents.ConstantCurrent(
        current = 7,
        noise = True
    ),
    17: Dendrite.Dendrite(
    ),
    20: Models.LIF(
        R = 5,
        threshold = -37,
        u_rest = -67,
        u_reset = -75,
        tau = 10,
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

ng2 = pymo.NeuronGroup(net=network, size=20, behavior={
    10: Currents.ConstantCurrent(
        current = 7,
        noise = True
    ),
    17: Dendrite.Dendrite(
    ),
    20: Models.LIF(
        R = 5,
        threshold = -37,
        u_rest = -67,
        u_reset = -75,
        tau = 10,
        refractory = False
    ),
    100: pymo.Recorder(
        variables = ["torch.sum(I)", "u", "I"],
        tag = "ng2_recorder"
    ),
    101: pymo.EventRecorder(
        variables = ["spike"],
        tag = "ng2_eventrecorder"
    )
}, tag="ng2")

syn1 = pymo.SynapseGroup(net=network, src=ng1, dst=ng1, behavior={
    #15: Synapse.all_to_all_connection(
    #    J0 = 100,
    #    std = 100,
    #    pre_excitate = True
    #)
    #15: Synapse.random_fixed_prob_connection(
    #    J0 = 20,
    #    prob = 0.1,
    #    std = 0.01
    #)
    15: Synapse.random_fixed_edgeCount_connection(
        J0 = 30,
        C = 5,
        std = 0.01,
        pre_excitate = False
    )
}, tag="syn_ng1_ng1")

syn2 = pymo.SynapseGroup(net=network, src=ng2, dst=ng2, behavior={
    15: Synapse.random_fixed_edgeCount_connection(
        J0 = 30,
        C = 5,
        std = 0.01,
        pre_excitate = False
    )
}, tag="syn_ng2_ng2")

syn3 = pymo.SynapseGroup(net=network, src=ng1, dst=ng2, behavior={
    15: Synapse.random_fixed_edgeCount_connection(
        J0 = 30,
        C = 5,
        std = 0.01,
        pre_excitate = False
    )
}, tag="syn_ng1_ng2")

syn4 = pymo.SynapseGroup(net=network, src=ng2, dst=ng1, behavior={
    15: Synapse.random_fixed_edgeCount_connection(
        J0 = 30,
        C = 5,
        std = 0.01,
        pre_excitate = False
    )
}, tag="syn_ng2_ng1")



network.initialize()

#network.simulate_iteration()
network.simulate_iterations(100)



import matplotlib.pyplot as plt

def clear_plot(save_location):
    plt.savefig(save_location)
    plt.clf()
    return
def que_plot(variables, xlabel, ylabel, title):
    plt.plot(variables)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return

def que_scatter(variableX, variableY, xlabel, title, colors=None, size=None, ylabel=None):
    plt.scatter(variableX, variableY, c = colors, s=size)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return

que_plot(variables=network["ng1_recorder", 0].variables["u"][:,:], 
         xlabel="time", ylabel="u", title="u-t for population #1")
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_ut.jpg")

que_plot(variables=network["ng1_recorder", 0].variables["I"][:,:], 
         xlabel="time", ylabel="I(current)", title="I-t for population #1")
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_It.jpg")

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
colors = np.random.uniform(15, 80, (network["ng1_eventrecorder", 0].variables["spike"]).size()[0])
que_scatter(variableX=x, variableY=np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]),
            colors=colors,
            xlabel="time", title='spike time pattern for population #1')
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_stp.jpg")

x_new = [x[i+1]-x[i] for i in range(len(x)-1)]
que_scatter(variableX=x_new, variableY=np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]-1),
            colors=colors[:-1],
            xlabel='time', title='Spike time interval for population #1')
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_sti.jpg")

dot_size = []
x_new_sorted = sorted(x_new)
for i in range(len(x_new_sorted)):
    if i == 0 or x_new_sorted[i-1]!=x_new_sorted[i]:
        dot_size.append(10)
    else:
        dot_size.append(dot_size[-1]+10)
que_scatter(variableX=x_new_sorted, variableY=np.zeros((network["ng1_eventrecorder", 0].variables["spike"]).size()[0]-1),
            colors=colors[:-1], size=dot_size,
            xlabel='time', title='Cumulative spike time interval for population #1')
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_csti.jpg")

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
y = [0 for _ in range(network["ng1_recorder", 0].variables["u"][:,0].shape[0]+1)]
for num in x:
    y[num] += 1
y = np.array(y)
que_plot(variables=y/ng1.size,
         xlabel="time", ylabel="Activity", title="Activity for population #1")
clear_plot(save_location="/Users/aaron/Downloads/Figure_1_A.jpg")



que_plot(variables=network["ng2_recorder", 0].variables["u"][:,:], 
         xlabel="time", ylabel="u", title="u-t for population #2",)
clear_plot(save_location="/Users/aaron/Downloads/Figure_2_ut.jpg")

que_plot(variables=network["ng2_recorder", 0].variables["I"][:,:], 
         xlabel="time", ylabel="I(current)", title="I-t for population #2")
clear_plot(save_location="/Users/aaron/Downloads/Figure_2_It.jpg")

x = (network["ng2_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
colors = np.random.uniform(15, 80, (network["ng2_eventrecorder", 0].variables["spike"]).size()[0])
que_scatter(variableX=x, variableY=np.zeros((network["ng2_eventrecorder", 0].variables["spike"]).size()[0]),
            colors=colors,
            xlabel="time", title='spike time pattern for population #2')
clear_plot(save_location="/Users/aaron/Downloads/Figure_2_stp.jpg")

x_new = [x[i+1]-x[i] for i in range(len(x)-1)]
que_scatter(variableX=x_new, variableY=np.zeros((network["ng2_eventrecorder", 0].variables["spike"]).size()[0]-1),
            colors=colors[:-1],
            xlabel='time', title='Spike time interval for population #2')
clear_plot(save_location="/Users/aaron/Downloads/Figure_2_sti.jpg")

dot_size = []
x_new_sorted = sorted(x_new)
for i in range(len(x_new_sorted)):
    if i == 0 or x_new_sorted[i-1]!=x_new_sorted[i]:
        dot_size.append(10)
    else:
        dot_size.append(dot_size[-1]+10)
que_scatter(variableX=x_new_sorted, variableY=np.zeros((network["ng2_eventrecorder", 0].variables["spike"]).size()[0]-1),
            colors=colors[:-1], size=dot_size,
            xlabel='time', title='Cumulative spike time interval for population #2')
clear_plot(save_location="/Users/aaron/Downloads/Figure_2_csti.jpg")

x = (network["ng2_eventrecorder", 0].variables["spike"])[:,0].cpu().numpy()
y = [0 for _ in range(network["ng2_recorder", 0].variables["u"][:,0].shape[0]+1)]
for num in x:
    y[num] += 1
y = np.array(y)
que_plot(variables=y/ng1.size,
         xlabel="time", ylabel="Activity", title="Activity for population #2")
clear_plot(save_location="/Users/aaron/Downloads/Figure_2_A.jpg")


x = (network["ng2_eventrecorder", 0].variables["spike"])[:,:].cpu().numpy()
x_raster, y_raster = [], []
for time, id in x:
    x_raster.append(time)
    y_raster.append(id+ng1.size)
x_raster, y_raster = np.array(x_raster), np.array(y_raster)
que_scatter(variableX=x_raster, variableY=y_raster,
            colors="orange", size=5,
            xlabel=None, ylabel=None, title=None)

x = (network["ng1_eventrecorder", 0].variables["spike"])[:,:].cpu().numpy()
x_raster, y_raster = [], []
for time, id in x:
    x_raster.append(time)
    y_raster.append(id)
x_raster, y_raster = np.array(x_raster), np.array(y_raster)
que_scatter(variableX=x_raster, variableY=y_raster,
            colors="blue", size=5,
            xlabel='time', ylabel='Neurons', title="Raster Plot (Blue: pop #1, Orange: pop #2)")

plt.ylim(-1, ng1.size + ng2.size)
clear_plot(save_location="/Users/aaron/Downloads/Figure_raster.jpg")



print((network["ng1_eventrecorder", 0].variables["spike"]).size())