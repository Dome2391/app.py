import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1) COSTRUZIONE GEOMETRIA E CALCOLO CARICHI (autopeso + sovraccarico)
# --------------------------------------------------------------------

def build_arch_with_surcharge(R, t, N, gamma, q_surcharge):
    """
    Costruisce la discretizzazione di un arco a tutto sesto e calcola:
      - intrados_joints[i], extrados_joints[i] : coordinate intradosso/estradosso del giunto i
      - arch_weights[i], arch_centroids[i]     : peso del blocco i e relativo baricentro
      - surch_weights[i], surch_centroids[i]   : sovraccarico lineare sul blocco i (convertito in forza) e suo baricentro

    Parametri:
      R          : raggio intradosso (m)
      t          : spessore (m)
      N          : numero di conci
      gamma      : peso specifico muratura (N/m^3)
      q_surcharge: sovraccarico distribuito in kN/m (line load), 
                   da convertire in N/m (1 kN = 1000 N).

    Si assume un "spessore fuori piano" = 1 m. 
    """
    # Angoli di discretizzazione
    theta_j = np.linspace(0, np.pi, N+1)
    
    # Giunti intradosso
    xi_in = R * np.cos(theta_j)
    yi_in = R * np.sin(theta_j)
    intrados_joints = np.column_stack((xi_in, yi_in))
    
    # Giunti estradosso
    xi_ex = (R + t) * np.cos(theta_j)
    yi_ex = (R + t) * np.sin(theta_j)
    extrados_joints = np.column_stack((xi_ex, yi_ex))
    
    # Prepara array per peso blocchi e baricentri
    arch_weights = []
    arch_centroids = []
    
    # Sovraccarico: convertiamo q_surcharge da kN/m a N/m
    q_surcharge_n = q_surcharge * 1000.0
    
    surch_weights = []
    surch_centroids = []
    
    thickness_z = 1.0  # spessore fuori piano (ipotesi)
    
    for i in range(N):
        dtheta = theta_j[i+1] - theta_j[i]
        
        # -----------------------
        # 1. Autopeso blocco i
        # -----------------------
        # Area del settore di corona circolare
        area_i = 0.5 * dtheta * ((R + t)**2 - R**2)
        W_i = area_i * thickness_z * gamma
        arch_weights.append(W_i)
        
        # Baricentro blocco (approssimazione: r_m = R + t/2)
        theta_m = 0.5*(theta_j[i] + theta_j[i+1])
        r_m = R + t/2
        xG = r_m * np.cos(theta_m)
        yG = r_m * np.sin(theta_m)
        arch_centroids.append([xG, yG])
        
        # -----------------------
        # 2. Sovraccarico lineare sul blocco i
        # -----------------------
        # Se la lunghezza dell'estradosso "locale" = (R+t)*dtheta
        # Forza = q_surcharge_n * lunghezza * spessore_z
        length_i = (R + t) * dtheta
        S_i = q_surcharge_n * length_i * thickness_z
        
        # Baricentro del sovraccarico (lo poniamo a raggio = R+t, 
        # e angolo medio = theta_m)
        xS = (R + t)*np.cos(theta_m)
        yS = (R + t)*np.sin(theta_m)
        
        surch_weights.append(S_i)
        surch_centroids.append([xS, yS])
    
    return (intrados_joints,
            extrados_joints,
            np.array(arch_weights),
            np.array(arch_centroids),
            np.array(surch_weights),
            np.array(surch_centroids))

# ------------------------------------------------------------------------------------------
# 2) RISOLUZIONE DELL'EQUILIBRIO BLOCCO-BLOCCO (H, V0 INCERTE) con linea di spinta "interna"
# ------------------------------------------------------------------------------------------

def solve_arch(
    intrados_joints, extrados_joints,
    arch_weights, arch_centroids,
    surch_weights, surch_centroids,
    H, V0
):
    """
    Dato:
      - Reazione iniziale appoggio sinistro = (H, V0)
      - Per ciascun blocco i: 
         * archivio di autopeso (arch_weights[i]) e baricentro (arch_centroids[i])
         * sovraccarico (surch_weights[i]) e baricentro (surch_centroids[i])
      - In ogni giunto i, la reazione R[i], e in giunto i+1 la reazione R[i+1].
      - La posizione del centro di pressione su giunto i+1 è incognita,
        costretta a stare tra intrados_joints[i+1] e extrados_joints[i+1].

    Restituisce:
      - line_of_thrust: array (N+1, 2) = coordinate del centro di pressione per i giunti 0..N
      - Rres: array (N+1, 2) = reazione in ciascun giunto
      - inside: bool, True se la linea di spinta è rimasta entro [intrados, extrados] in tutti i giunti
    """
    N = len(arch_weights)
    
    # Inizializziamo array di reazioni e punti di spinta
    Rres = np.zeros((N+1, 2))
    line_of_thrust = np.zeros((N+1, 2))
    
    # Appoggio sinistro
    Rres[0,0] = H
    Rres[0,1] = V0
    
    # Centro di pressione al giunto 0 -> ipotesi a metà spessore
    x0 = 0.5*(intrados_joints[0,0] + extrados_joints[0,0])
    y0 = 0.5*(intrados_joints[0,1] + extrados_joints[0,1])
    line_of_thrust[0] = [x0, y0]
    
    inside = True
    
    for i in range(N):
        # Carico verticale del blocco: autopeso + sovraccarico
        W_block = arch_weights[i]
        S_block = surch_weights[i]
        
        # Forze verticali negative (verso il basso)
        Wx = 0.0
        Wy = - (W_block + S_block)
        
        # Baricentro autopeso
        xGa, yGa = arch_centroids[i]
        # Baricentro sovraccarico
        xGs, yGs = surch_centroids[i]
        
        # Reazione al giunto i
        Rx_i, Ry_i = Rres[i]
        
        # Punto di pressione al giunto i
        xA, yA = line_of_thrust[i]
        
        # Giunto i+1: definito da intrados e extrados
        xi_in2, yi_in2 = intrados_joints[i+1]
        xi_ex2, yi_ex2 = extrados_joints[i+1]
        
        # Reazione al giunto i+1 => R[i+1] = - ( R[i] + W )
        #   ma W = (0, - (W_block+S_block)).
        def next_reaction():
            Rx_next = - (Rx_i + Wx)
            Ry_next = - (Ry_i + Wy)
            return Rx_next, Ry_next
        
        # Momento rispetto al punto (xB, yB) (incognito),
        #  cross(A->B, R[i]) + cross(Ga->B, W_block) + cross(Gs->B, W_surcharge) = 0
        # (dove W_surcharge = (0, - S_block))
        def residual_moment(lmbd):
            # Parametrizzazione (xB, yB) in base a lmbd in [0,1]
            xB = xi_in2 + lmbd * (xi_ex2 - xi_in2)
            yB = yi_in2 + lmbd * (yi_ex2 - yi_in2)
            
            # Forza R[i]
            dxA = xA - xB
            dyA = yA - yB
            M_Ri = dxA*Ry_i - dyA*Rx_i
            
            # Forza autopeso blocco
            dxGa = xGa - xB
            dyGa = yGa - yB
            M_Wa = dxGa*(-W_block) - dyGa*(0.0)
            # cross((dx,dy), (0, -W_block)) = dx*(-W_block) - dy*0 => dx*(-W)
            
            # Forza sovraccarico
            dxGs = xGs - xB
            dyGs = yGs - yB
            M_Ws = dxGs*(-S_block) - dyGs*(0.0)
            
            return (M_Ri + M_Wa + M_Ws)
        
        # Cerchiamo lmbd in [0,1] che annulla residual_moment
        lam_min, lam_max = 0.0, 1.0
        m_min = residual_moment(lam_min)
        m_max = residual_moment(lam_max)
        
        if m_min*m_max > 0:
            # Non c'è cambio di segno -> non c'è soluzione nel [0,1]
            inside = False
            # fallback: scegli lam_min o lam_max a seconda del valore assoluto minore
            if abs(m_min) < abs(m_max):
                lam_star = 0.0
            else:
                lam_star = 1.0
        else:
            # Bisezione su lam
            for _ in range(30):
                lam_mid = 0.5*(lam_min + lam_max)
                m_mid = residual_moment(lam_mid)
                if abs(m_mid) < 1e-8:
                    lam_star = lam_mid
                    break
                if m_mid*m_min > 0:
                    lam_min = lam_mid
                    m_min = m_mid
                else:
                    lam_max = lam_mid
                    m_max = m_mid
            lam_star = 0.5*(lam_min + lam_max)
        
        if lam_star < 0.0 or lam_star > 1.0:
            inside = False
        
        # Calcoliamo (xB, yB)
        xB = xi_in2 + lam_star*(xi_ex2 - xi_in2)
        yB = yi_in2 + lam_star*(yi_ex2 - yi_in2)
        
        # Aggiorniamo reazione al giunto i+1
        Rx_next, Ry_next = next_reaction()
        
        # Salviamo
        line_of_thrust[i+1] = [xB, yB]
        Rres[i+1,0] = Rx_next
        Rres[i+1,1] = Ry_next
    
    return line_of_thrust, Rres, inside


def total_load(arch_weights, surch_weights):
    """
    Somma totale dei carichi verticali (autopeso + sovraccarico).
    """
    return np.sum(arch_weights) + np.sum(surch_weights)

# ------------------------------------------------------------------------------------------
# 3) DOPPIA BISEZIONE SU (H, V0)
# ------------------------------------------------------------------------------------------

def solve_with_bisection(
    intrados_joints, extrados_joints,
    arch_weights, arch_centroids,
    surch_weights, surch_centroids,
    H_min=0, H_max=1e6,
    V0_min=0, V0_max=None,
    tol=1e-2, max_iter=40
):
    """
    Cerca (H, V0) tali che:
      1) sum reazioni verticali = sum carichi verticali
      2) reazione orizzontale a destra = H
      3) la linea di spinta resti 'inside' in tutti i giunti.

    Procedura:
      - Bisezione su H
      - Per ogni H, bisezione su V0 (per equilibrare i carichi verticali).
      - Controllo inside e differenza R[N,0] - H.
    """
    W_tot = total_load(arch_weights, surch_weights)
    if V0_max is None:
        V0_max = 1.2 * W_tot  # un tetto ragionevole
    
    best_line = None
    best_H = None
    best_V0 = None
    success = False
    
    for _ in range(max_iter):
        H_mid = 0.5*(H_min + H_max)
        
        # Bisezione su V0
        v_low, v_high = V0_min, V0_max
        found_V0 = None
        line_sol = None
        R_sol = None
        inside_sol = False
        
        for _ in range(20):
            V0_try = 0.5*(v_low + v_high)
            
            line_thrust, Rres, inside_flag = solve_arch(
                intrados_joints, extrados_joints,
                arch_weights, arch_centroids,
                surch_weights, surch_centroids,
                H_mid, V0_try
            )
            
            # f(V0) = [R[0,1] + R[N,1]] - W_tot
            # se f(V0)=0 => sum reazioni verticali = peso totale
            sumV = Rres[0,1] + Rres[-1,1]
            fV = sumV - W_tot
            
            if inside_flag:
                # salviamo la linea di spinta e reazioni
                found_V0 = V0_try
                line_sol = line_thrust
                R_sol = Rres
                inside_sol = True
            
            if abs(fV) < tol:  # se la differenza è molto piccola
                break
            
            if fV > 0:
                # reazioni verticali > W_tot => abbassiamo V0
                v_high = V0_try
            else:
                v_low = V0_try
        
        if (found_V0 is None) or (not inside_sol):
            # non inside => in genere serve spinta H più grande
            H_min = H_mid
            continue
        
        # Ora verifichiamo differenza orizzontale
        dH = R_sol[-1,0] - H_mid
        if abs(dH) < tol:
            # Trovata soluzione
            success = True
            best_line = line_sol
            best_H = H_mid
            best_V0 = found_V0
            break
        
        if dH > 0:
            H_min = H_mid
        else:
            H_max = H_mid
        
        success = True
        best_line = line_sol
        best_H = H_mid
        best_V0 = found_V0
    
    return best_H, best_V0, best_line, success

# ------------------------------------------------------------------------------------------
# 4) STREAMLIT APP
# ------------------------------------------------------------------------------------------

def main():
    st.title("Analisi di un Arco in Muratura con Sovraccarico Distribuito")
    st.write("""
    - **Arco a tutto sesto**, discretizzato in N blocchi.
    - **Muratura** senza resistenza a trazione (Heyman).
    - **Sovraccarico** lineare (kN/m) applicato lungo l'estradosso.
    - **Doppia bisezione** su spinta orizzontale (H) e reazione verticale sinistra (V0).
    - Si impone che la **linea di spinta** resti entro lo spessore (intradosso-estradosso) ad ogni giunto.
    
    La soluzione fornisce:
    1) Spinta orizzontale H
    2) Reazione verticale sinistra V0
    3) Verifica che la reazione verticale totale corrisponda al peso (autopeso + sovraccarico).
    """)

    st.sidebar.header("Parametri Arco & Carichi")
    R = st.sidebar.slider("Raggio intradosso (m)", 1.0, 20.0, 5.0, 0.5)
    t = st.sidebar.slider("Spessore (m)", 0.1, 3.0, 0.8, 0.1)
    N = st.sidebar.slider("Numero di conci", 2, 40, 10, 1)
    gamma_kN = st.sidebar.slider("Peso specifico muratura (kN/m³)", 15.0, 30.0, 20.0, 1.0)
    gamma = gamma_kN * 1000.0  # da kN/m³ a N/m³
    
    q_surcharge = st.sidebar.slider("Sovraccarico lineare (kN/m) su estradosso", 0.0, 50.0, 5.0, 1.0)
    
    if st.button("Esegui Calcolo"):
        # Costruzione geometry & carichi
        (intrados_joints, extrados_joints,
         arch_weights, arch_centroids,
         surch_weights, surch_centroids) = build_arch_with_surcharge(
            R, t, N, gamma, q_surcharge
        )
        
        # Risoluzione con doppia bisezione
        H_opt, V0_opt, line_thrust, success = solve_with_bisection(
            intrados_joints, extrados_joints,
            arch_weights, arch_centroids,
            surch_weights, surch_centroids,
            H_min=0, H_max=1e6,
            V0_min=0, V0_max=None,
            tol=1e-2,
            max_iter=40
        )
        
        if not success or (line_thrust is None):
            st.error("Impossibile trovare una linea di spinta interna con i parametri scelti.")
        else:
            st.success("Soluzione trovata!")
            st.write(f"**Spinta orizzontale** H ≈ {H_opt:,.2f} N ({H_opt/1000:.2f} kN)")
            st.write(f"**Reazione verticale sinistra** V0 ≈ {V0_opt:,.2f} N ({V0_opt/1000:.2f} kN)")
            
            # Peso totale
            W_tot = total_load(arch_weights, surch_weights)
            st.write(f"**Peso totale** (muratura + sovraccarico) = {W_tot:,.2f} N ({W_tot/1000:.2f} kN)")

            # Plot
            theta_plot = np.linspace(0, np.pi, 200)
            x_in = R*np.cos(theta_plot)
            y_in = R*np.sin(theta_plot)
            x_out = (R+t)*np.cos(theta_plot)
            y_out = (R+t)*np.sin(theta_plot)
            
            fig, ax = plt.subplots(figsize=(6,5))
            ax.plot(x_in, y_in, 'b', label="Intradosso")
            ax.plot(x_out, y_out, 'b', label="Estradosso")
            
            ax.plot(line_thrust[:,0], line_thrust[:,1], 'ro-', label="Linea di spinta")
            ax.set_aspect('equal', 'box')
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Arco con sovraccarico - Linea di spinta interna")
            ax.legend()
            st.pyplot(fig)

            st.write("""
            **Osservazioni**:
            - L'aggiunta del sovraccarico fa aumentare il peso totale e può
              richiedere una spinta orizzontale maggiore per restare in compressione pura.
            - Se i valori (R, t) e q_surcharge sono tali da rendere l'arco troppo
              "esile", potrebbe non esistere una linea di spinta completamente interna.
            - Con un arco simmetrico e carichi simmetrici, ci si aspetta V0_opt
              vicino a ~metà del peso totale. Ma dipende dalla geometria e dall'eccentricità.
            """)


if __name__ == "__main__":
    main()
