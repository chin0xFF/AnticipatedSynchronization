#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:44:37 2025

@author: chin0xff
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo Hodgkin-Huxley
g_Na, g_K, g_L = 120.0, 36.0, 0.3  # Conductancias máximas (mS/cm²)
E_Na, E_K, E_L = 50.0, -77.0, -54.4  # Potenciales de equilibrio (mV)
C_m = 1.0  # Capacitancia de la membrana (uF/cm²)

# Parámetros de las sinapsis
E_AMPA, E_GABA, E_NMDA = 0.0, -70.0, 0.0  # Potenciales de reversión (mV)
g_AMPA, g_GABA, g_NMDA = 0.1, 0.1, 0.05  # Conductancias máximas (mS/cm²)
alpha_AMPA, beta_AMPA = 1.1, 0.19
alpha_GABA, beta_GABA = 5.0, 0.30
alpha_NMDA, beta_NMDA = 0.072, 0.0066

# Función de activación de neurotransmisores
def dr_dt(r, V_pre, alpha, beta):
    T = 1 / (1 + np.exp(-(V_pre - 20) / 2))  # Liberación de neurotransmisores
    return alpha * T * (1 - r) - beta * r

# Función para el bloqueo dependiente de voltaje de NMDA
def B_NMDA(V):
    return 1 / (1 + np.exp(-0.062 * V) * (1 / 3.57))  # Bloqueo por Mg2+

# Número de variables en el sistema
N = 25  # Asegurar que coincide con dmsi_network

# Modelo de red neuronal DMSI
def dmsi_network(t, y, I_driver, I_master, I_slave, I_interneuron):
    (V_d, m_d, h_d, n_d, r_NMDA_dm, r_NMDA_ds, r_NMDA_di, 
     V_m, m_m, h_m, n_m, r_AMPA_m, r_NMDA_m, 
     V_s, m_s, h_s, n_s, r_AMPA_s, r_GABA_s, 
     V_i, m_i, h_i, n_i, r_AMPA_i, r_GABA_i) = y  # Se esperan 25 variables

    # Dinámica sináptica
    dr_NMDA_dm_dt = dr_dt(r_NMDA_dm, V_d, alpha_NMDA, beta_NMDA)  
    dr_NMDA_ds_dt = dr_dt(r_NMDA_ds, V_d, alpha_NMDA, beta_NMDA)  
    dr_NMDA_di_dt = dr_dt(r_NMDA_di, V_d, alpha_NMDA, beta_NMDA)  
    dr_AMPA_m_dt = dr_dt(r_AMPA_m, V_d, alpha_AMPA, beta_AMPA)  
    dr_NMDA_m_dt = dr_dt(r_NMDA_m, V_m, alpha_NMDA, beta_NMDA)  
    dr_AMPA_s_dt = dr_dt(r_AMPA_s, V_m, alpha_AMPA, beta_AMPA)  
    dr_AMPA_i_dt = dr_dt(r_AMPA_i, V_s, alpha_AMPA, beta_AMPA)  
    dr_GABA_s_dt = dr_dt(r_GABA_s, V_i, alpha_GABA, beta_GABA)  
    dr_GABA_i_dt = dr_dt(r_GABA_i, V_i, alpha_GABA, beta_GABA)  

    # Función dinámica de cada neurona
    def neuron_dynamics(V, m, h, n, I_ext):
        dmdt = alpha_AMPA * (1 - m) - beta_AMPA * m
        dhdt = alpha_GABA * (1 - h) - beta_GABA * h
        dndt = alpha_NMDA * (1 - n) - beta_NMDA * n
        return dmdt, dhdt, dndt

    # Ecuaciones diferenciales
    dm_d, dh_d, dn_d = neuron_dynamics(V_d, m_d, h_d, n_d, I_driver)
    dm_m, dh_m, dn_m = neuron_dynamics(V_m, m_m, h_m, n_m, I_master)
    dm_s, dh_s, dn_s = neuron_dynamics(V_s, m_s, h_s, n_s, I_slave)
    dm_i, dh_i, dn_i = neuron_dynamics(V_i, m_i, h_i, n_i, I_interneuron)

    return [
        V_d, dm_d, dh_d, dn_d, dr_NMDA_dm_dt, dr_NMDA_ds_dt, dr_NMDA_di_dt,
        V_m, dm_m, dh_m, dn_m, dr_AMPA_m_dt, dr_NMDA_m_dt,
        V_s, dm_s, dh_s, dn_s, dr_AMPA_s_dt, dr_GABA_s_dt,
        V_i, dm_i, dh_i, dn_i, dr_AMPA_i_dt, dr_GABA_i_dt
    ]

# Simulación
t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)
y0 = np.full(N, -65.0)  # Tamaño corregido a 25 variables
sol = solve_ivp(dmsi_network, t_span, y0, args=(10.0, 5.0, 0.0, 0.0), t_eval=t_eval, method='RK45')

# Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[7], label='Master (V_m)')
plt.plot(sol.t, sol.y[14], label='Slave (V_s)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de membrana (mV)')
plt.title('Sincronización en DMSI')
plt.legend()
plt.show()
