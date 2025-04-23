#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:28:51 2025

@author: chin0xff
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo Hodgkin-Huxley
C_m = 9.0 * np.pi  # Capacitancia de la membrana (uF/cm²)
E_Na, E_K, E_L = 115.0, -12.0, 10.6  # Potenciales de equilibrio (mV)
G_Na, G_K, G_L = 1080.0 * np.pi, 324.0 * np.pi, 2.7 * np.pi  # Conductancias máximas (mS/cm²)

# Parámetros de las sinapsis
E_AMPA, E_GABA, E_NMDA = 60.0, -20.0, 60.0  # Potenciales de reversión (mV)
alpha_AMPA, beta_AMPA = 1.1, 0.19
alpha_GABA, beta_GABA = 5.0, 0.30
alpha_NMDA, beta_NMDA = 0.072, 0.0066

# Función de activación de neurotransmisores
def dr_dt(r, V_pre, alpha, beta):
    T_max = 1.0  # Concentración máxima de neurotransmisor
    V_p = 62.0  # Potencial de media activación
    K_p = 5.0  # Pendiente de la función sigmoide
    T = T_max / (1 + np.exp(-(V_pre - V_p) / K_p))  # Liberación de neurotransmisores
    return alpha * T * (1 - r) - beta * r

# Función para el bloqueo dependiente de voltaje de NMDA
def B_NMDA(V):
    Mg_ext = 1.0  # Concentración extracelular de Mg2+
    return 1 / (1 + np.exp(-0.062 * V) * (Mg_ext / 3.57))  # Bloqueo por Mg2+

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
        alpha_n = 0.1 * (10 - V) / (np.exp((10 - V) / 10) - 1)
        beta_n = 0.125 * np.exp(-V / 80)
        alpha_m = 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)
        beta_m = 4 * np.exp(-V / 18)
        alpha_h = 0.07 * np.exp(-V / 20)
        beta_h = 1 / (np.exp((30 - V) / 10) + 1)

        dmdt = alpha_m * (1 - m) - beta_m * m
        dhdt = alpha_h * (1 - h) - beta_h * h
        dndt = alpha_n * (1 - n) - beta_n * n
        return dmdt, dhdt, dndt

    # Ecuaciones diferenciales
    dm_d, dh_d, dn_d = neuron_dynamics(V_d, m_d, h_d, n_d, I_driver)
    dm_m, dh_m, dn_m = neuron_dynamics(V_m, m_m, h_m, n_m, I_master)
    dm_s, dh_s, dn_s = neuron_dynamics(V_s, m_s, h_s, n_s, I_slave)
    dm_i, dh_i, dn_i = neuron_dynamics(V_i, m_i, h_i, n_i, I_interneuron)

    # Conductancias máximas de las sinapsis (nS)
    g_AMPA = 10.0  # Conductancia máxima para sinapsis AMPA
    g_NMDA = 10.0  # Conductancia máxima para sinapsis NMDA
    g_GABA = 40.0  # Conductancia máxima para sinapsis GABA


    # Ecuaciones de voltaje
    I_syn_d = g_AMPA * r_AMPA_m * (E_AMPA - V_d) + g_NMDA * B_NMDA(V_d) * r_NMDA_dm * (E_NMDA - V_d)
    I_syn_m = g_AMPA * r_AMPA_s * (E_AMPA - V_m) + g_NMDA * B_NMDA(V_m) * r_NMDA_m * (E_NMDA - V_m)
    I_syn_s = g_AMPA * r_AMPA_i * (E_AMPA - V_s) + g_GABA * r_GABA_s * (E_GABA - V_s)
    I_syn_i = g_AMPA * r_AMPA_s * (E_AMPA - V_i) + g_GABA * r_GABA_i * (E_GABA - V_i)

    dV_d_dt = (G_Na * m_d**3 * h_d * (E_Na - V_d) + G_K * n_d**4 * (E_K - V_d) + G_L * (E_L - V_d) + I_driver + I_syn_d) / C_m
    dV_m_dt = (G_Na * m_m**3 * h_m * (E_Na - V_m) + G_K * n_m**4 * (E_K - V_m) + G_L * (E_L - V_m) + I_master + I_syn_m) / C_m
    dV_s_dt = (G_Na * m_s**3 * h_s * (E_Na - V_s) + G_K * n_s**4 * (E_K - V_s) + G_L * (E_L - V_s) + I_slave + I_syn_s) / C_m
    dV_i_dt = (G_Na * m_i**3 * h_i * (E_Na - V_i) + G_K * n_i**4 * (E_K - V_i) + G_L * (E_L - V_i) + I_interneuron + I_syn_i) / C_m

    return [
        dV_d_dt, dm_d, dh_d, dn_d, dr_NMDA_dm_dt, dr_NMDA_ds_dt, dr_NMDA_di_dt,
        dV_m_dt, dm_m, dh_m, dn_m, dr_AMPA_m_dt, dr_NMDA_m_dt,
        dV_s_dt, dm_s, dh_s, dn_s, dr_AMPA_s_dt, dr_GABA_s_dt,
        dV_i_dt, dm_i, dh_i, dn_i, dr_AMPA_i_dt, dr_GABA_i_dt
    ]

# Simulación
t_span = (0, 100)
t_eval = np.linspace(*t_span, 1000)
y0 = np.full(25, -65.0)  # Condiciones iniciales para 25 variables
sol = solve_ivp(dmsi_network, t_span, y0, args=(10.0, 5.0, 0.0, 0.0), t_eval=t_eval, method='RK45')

# Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Driver (V_d)')
plt.plot(sol.t, sol.y[7], label='Master (V_m)')
plt.plot(sol.t, sol.y[14], label='Slave (V_s)')
plt.plot(sol.t, sol.y[21], label='Interneuron (V_i)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de membrana (mV)')
plt.title('Sincronización en DMSI')
plt.legend()
plt.show()

