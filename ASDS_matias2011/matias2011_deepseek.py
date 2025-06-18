#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:59:22 2025

@author: chin0xff
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class HodgkinHuxley:
    """Hodgkin-Huxley neuron model implementation"""
    def __init__(self, I=0):
        # Constants
        self.C_m = 9 * np.pi  # Membrane capacitance (µF)
        self.g_Na = 1080 * np.pi  # Max Na conductance (mS)
        self.g_K = 324 * np.pi  # Max K conductance (mS)
        self.g_m = 2.7 * np.pi  # Leak conductance (mS)
        self.E_Na = 115  # Na reversal potential (mV)
        self.E_K = -12  # K reversal potential (mV)
        self.V_rest = 10.6  # Resting potential (mV)
        self.I = I  # Input current (pA)
        
    def alpha_n(self, V):
        return (10 - V) / (100 * (np.exp((10 - V)/10) - 1))
    
    def beta_n(self, V):
        return 0.125 * np.exp(-V/80)
    
    def alpha_m(self, V):
       return (25 - V) / (10 * (np.exp((25 - V)/10) - 1))
    
    def beta_m(self, V):
        return 4 * np.exp(-V/18)
    
    def alpha_h(self, V):
        return 0.07 * np.exp(-V/20)
    
    def beta_h(self, V):
        return 1 / (np.exp((30 - V)/10) + 1)
    
    def derivatives(self, state, t, I_syn=0):
        V, n, m, h = state
        
        # Ionic currents
        I_Na = self.g_Na * m**3 * h * (self.E_Na - V)
        I_K = self.g_K * n**4 * (self.E_K - V)
        I_L = self.g_m * (self.V_rest - V)
        
        # Voltage equation
        dVdt = (I_Na + I_K + I_L + self.I + I_syn) / self.C_m
        
        # Gating variables
        dndt = self.alpha_n(V)*(1-n) - self.beta_n(V)*n
        dmdt = self.alpha_m(V)*(1-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1-h) - self.beta_h(V)*h
        
        return [dVdt, dndt, dmdt, dhdt]
    
    def simulate(self, t, I_syn=0):
        # Initial conditions
        V0 = self.V_rest
        n0 = self.alpha_n(V0) / (self.alpha_n(V0) + self.beta_n(V0))
        m0 = self.alpha_m(V0) / (self.alpha_m(V0) + self.beta_m(V0))
        h0 = self.alpha_h(V0) / (self.alpha_h(V0) + self.beta_h(V0))
        
        # Solve ODE
        solution = odeint(self.derivatives, [V0, n0, m0, h0], t, args=(I_syn,))
        return solution

class Synapse:
    """Chemical synapse model"""
    def __init__(self, synapse_type, alpha, beta, g_max, E_syn):
        self.type = synapse_type  # 'AMPA' or 'GABA'
        self.alpha = alpha  # Binding rate (mM^-1 ms^-1)
        self.beta = beta  # Unbinding rate (ms^-1)
        self.g_max = g_max  # Max conductance (nS)
        self.E_syn = E_syn  # Reversal potential (mV)
        self.r = 0  # Fraction of bound receptors
        
    def update(self, V_pre, dt):
        """Update synaptic state"""
        # Neurotransmitter concentration (simplified)
        T = 1 / (1 + np.exp(-(V_pre - 62)/5))  # Sigmoid function
        
        # Receptor dynamics
        drdt = self.alpha * T * (1 - self.r) - self.beta * self.r
        self.r += drdt * dt
        return self.r
    
    def current(self, V_post):
        """Compute synaptic current"""
        return self.g_max * self.r * (self.E_syn - V_post)

def simulate_MSI():
    """Simulate Master-Slave-Interneuron motif"""
    # Time parameters
    dt = 0.05  # ms
    t_max = 1000  # ms
    t = np.arange(0, t_max, dt)
    
    # Create neurons
    I_master = 280  # pA (tonically spiking)
    master = HodgkinHuxley(I_master)
    slave = HodgkinHuxley(I_master)
    interneuron = HodgkinHuxley(I_master)
    
    # Create synapses (parameters from Matias 2011 Table I)
    # M->S (excitatory AMPA)
    MS_syn = Synapse('AMPA', alpha=1.1, beta=0.19, g_max=10, E_syn=60)
    # S->I (excitatory AMPA)
    SI_syn = Synapse('AMPA', alpha=1.1, beta=0.19, g_max=10, E_syn=60)
    # I->S (inhibitory GABA)
    IS_syn = Synapse('GABA', alpha=5.0, beta=0.3, g_max=40, E_syn=-20)  # g_G is varied
    
    # Simulation arrays
    V_master = np.zeros_like(t)
    V_slave = np.zeros_like(t)
    V_inter = np.zeros_like(t)
    
    # Initial conditions
    state_master = [master.V_rest, 0, 0, 0]
    state_slave = [slave.V_rest, 0, 0, 0]
    state_inter = [interneuron.V_rest, 0, 0, 0]
    
    for i in range(1, len(t)):
        # Update synapses
        r_MS = MS_syn.update(V_master[i-1], dt)
        r_SI = SI_syn.update(V_slave[i-1], dt)
        r_IS = IS_syn.update(V_inter[i-1], dt)

        # Compute synaptic currents
        I_MS = MS_syn.current(V_slave[i-1])
        I_SI = SI_syn.current(V_inter[i-1])
        I_IS = IS_syn.current(V_slave[i-1])

        # Update neurons
        state_master = master.simulate(t[i-1:i+1], I_syn=0)[-1]
        state_slave = slave.simulate(t[i-1:i+1], I_syn=I_MS - I_IS)[-1]
        state_inter = interneuron.simulate(t[i-1:i+1], I_syn=I_SI)[-1]

        # Store voltages
        V_master[i] = state_master[0]
        V_slave[i] = state_slave[0]
        V_inter[i] = state_inter[0]

    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(t, V_master, label='Master')
    plt.plot(t, V_slave, label='Slave')
    plt.plot(t, V_inter, label='Interneuron')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('MSI Motif Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    simulate_MSI()
