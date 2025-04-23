import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del modelo Hodgkin-Huxley
g_Na = 120.0  # Conductancia máxima del canal de sodio (mS/cm^2)
g_K = 36.0    # Conductancia máxima del canal de potasio (mS/cm^2)
g_L = 0.3     # Conductancia de fuga (mS/cm^2)
E_Na = 50.0   # Potencial de equilibrio del sodio (mV)
E_K = -77.0   # Potencial de equilibrio del potasio (mV)
E_L = -54.4   # Potencial de fuga (mV)
C_m = 1.0     # Capacitancia de la membrana (uF/cm^2)

# Parámetros de la sinapsis
E_AMPA = 0.0   # Potencial de reversión de AMPA (mV)
E_GABA = -70.0 # Potencial de reversión de GABA (mV)
g_AMPA = 0.1   # Conductancia máxima AMPA (mS/cm^2)
g_GABA = 0.1   # Conductancia máxima GABA (mS/cm^2)
alpha_AMPA = 1.1  # Tasa de activación de AMPA (ms^-1)
beta_AMPA = 0.19  # Tasa de desactivación de AMPA (ms^-1)
alpha_GABA = 5.0  # Tasa de activación de GABA (ms^-1)
beta_GABA = 0.30  # Tasa de desactivación de GABA (ms^-1)

# Funciones de las variables de compuerta (alpha y beta)
def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    return 4.0 * np.exp(-0.0556 * (V + 65))

def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65))

def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

# Dinámica de los neurotransmisores
def dr_dt(r, V_pre, alpha, beta):
    T = 1 / (1 + np.exp(-(V_pre - 20) / 2))  # Liberación de neurotransmisores
    return alpha * T * (1 - r) - beta * r

# Función que define el sistema de ecuaciones diferenciales
def hodgkin_huxley(t, y, I_ext, V_pre_AMPA, V_pre_GABA):
    V, m, h, n, r_AMPA, r_GABA = y
    
    # Cálculo de las compuertas
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    
    # Corrientes iónicas
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # Dinámica de las sinapsis
    dr_AMPA_dt = dr_dt(r_AMPA, V_pre_AMPA, alpha_AMPA, beta_AMPA)
    dr_GABA_dt = dr_dt(r_GABA, V_pre_GABA, alpha_GABA, beta_GABA)
    
    I_AMPA = g_AMPA * r_AMPA * (V - E_AMPA)
    I_GABA = g_GABA * r_GABA * (V - E_GABA)
    
    # Ecuación diferencial para el potencial de membrana
    dVdt = (I_ext - I_Na - I_K - I_L - I_AMPA - I_GABA) / C_m
    
    return [dVdt, dmdt, dhdt, dndt, dr_AMPA_dt, dr_GABA_dt]

# Condiciones iniciales
V0 = -65.0  # Potencial inicial (mV)
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
r_AMPA0 = 0.0
r_GABA0 = 0.0
y0 = [V0, m0, h0, n0, r_AMPA0, r_GABA0]

# Parámetros de la simulación
t_span = (0, 50)  # Tiempo de simulación en ms
t_eval = np.linspace(*t_span, 1000)  # Tiempo de evaluación
I_ext = 10.0  # Corriente externa constante (uA/cm^2)
V_pre_AMPA = 0.0  # Potencial presináptico ficticio (debe ser actualizado en una red)
V_pre_GABA = -70.0  # Potencial presináptico ficticio

# Resolver el sistema de ecuaciones
tol = 1e-6
sol = solve_ivp(hodgkin_huxley, t_span, y0, args=(I_ext, V_pre_AMPA, V_pre_GABA), t_eval=t_eval, method='RK45', atol=tol, rtol=tol)

# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='V (mV)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Potencial de membrana (mV)')
plt.title('Modelo de Hodgkin-Huxley con Sinapsis')
plt.legend()
plt.show()
