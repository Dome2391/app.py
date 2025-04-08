import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# 1) COSTRUZIONE DELLA GEOMETRIA E DEI CARICHI
# ----------------------------------------------------------------------------------------

def build_arch_with_surcharge(R, t, N, gamma, q_surcharge):
    """
    Costruisce la discretizzazione di un arco a tutto sesto e calcola:
      - intrados_joints[i], extrados_joints[i]: coordinate dei giunti intradosso/estradosso
      - arch_weights[i], arch_centroids[i]: peso e baricentro di ciascun blocco di muratura
      - surch_weights[i], surch_centroids[i]: carico dovuto al sovraccarico lineare (kN/m)
        e suo baricentro

    Parametri:
      R : raggio intradosso (m)
      t : spessore (m)
      N : numero di conci (blocchi)
      gamma : peso specifico muratura (N/m^3) [ricorda: 1 kN/m^3 = 1000 N/m^3]
      q_surcharge : sovraccarico lineare in kN/m, da convertire in N/m (q_surcharge * 1000)

    Ritorna: (intrados_joints, extrados_joints,
              arch_weights, arch_centroids,
              surch_weights, surch_centroids)
    """
    # suddivisione angolare (da 0 a pi in N blocchi)
    theta_j = np.linspace(0, np.pi, N+1)

    # Giunti intradosso
    xi_in = R * np.cos(theta_j)
    yi_in = R * np.sin(theta_j)
    intrados_joints = np.column_stack((xi_in, yi_in))

    # Giunti estradosso
    xi_ex = (R + t) * np.cos(theta_j)
    yi_ex = (R + t) * np.sin(theta_j)
    extrados_joints = np.column_stack((xi_ex, yi_ex))

    # Array per pesi e baricentri (muratura)
    arch_weights = []
    arch_centroids = []

    # Sovraccarico (da kN/m a N/m)
    q_surcharge_n = q_surcharge * 1000.0
    surch_weights = []
    surch_centroids = []

    # ipotizziamo spessore costante fuori piano (1 m)
    thickness_z = 1.0

    for i in range(N):
        dtheta = theta_j[i+1] - theta_j[i]

        # 1) Autopeso del blocco i (muratura)
        area_i = 0.5 * dtheta * ((R + t)**2 - R**2)
        W_i = area_i * thickness_z * gamma
        arch_weights.append(W_i)

        # baricentro del blocco: raggio medio R + t/2, angolo medio
        theta_m = 0.5*(theta_j[i] + theta_j[i+1])
        r_m = R + t/2
        xGa = r_m * np.cos(theta_m)
        yGa = r_m * np.sin(theta_m)
        arch_centroids.append([xGa, yGa])

        # 2) Sovraccarico
        # lunghezza estradosso in questo blocco = (R+t)*dtheta
        length_i = (R + t)*dtheta
        S_i = q_surcharge_n * length_i * thickness_z

        # baricentro del sovraccarico: raggio = R + t, angolo medio
        xGs = (R + t) * np.cos(theta_m)
        yGs = (R + t) * np.sin(theta_m)

        surch_weights.append(S_i)
        surch_centroids.append([xGs, yGs])

    return (intrados_joints, extrados_joints,
            np.array(arch_weights),
            np.array(arch_centroids),
            np.array(surch_weights),
            np.array(surch_centroids))

# ----------------------------------------------------------------------------------------
# 2) FUNZIONE PER RISOLVERE L'EQUILIBRIO DI UN BLOCCO (DATI H, V0)
# ----------------------------------------------------------------------------------------

def solve_arch(
    intrados_joints, extrados_joints,
    arch_weights, arch_centroids,
    surch_weights, surch_centroids,
    H, V0
):
    """
    Dato un valore di H e V0 (reazioni all'appoggio sinistro),
    calcola la linea di spinta blocco per blocco, imponendo:
    - sum(Momenti) = 0 con centro di pressione tra intradosso ed estradosso (lambda in [0,1])
    - sum(Forze) = 0 su ogni blocco

    Ritorna:
      - line_of_thrust: array (N+1, 2)
      - Rres: array (N+1, 2) reazioni a ogni giunto
      - inside: bool (True se la linea di spinta non esce dallo spessore in nessun blocco)
    """
    N = len(arch_weights)  # numero blocchi

    Rres = np.zeros((N+1, 2))       # reazioni a ogni giunto
    line_of_thrust = np.zeros((N+1, 2))  # punto di pressione a ogni giunto

    # Impostiamo la reazione all'appoggio sinistro
    Rres[0,0] = H
    Rres[0,1] = V0

    # Centro di pressione al giunto 0 (ipotizziamo a metà spessore)
    x0 = 0.5*(intrados_joints[0,0] + extrados_joints[0,0])
    y0 = 0.5*(intrados_joints[0,1] + extrados_joints[0,1])
    line_of_thrust[0] = [x0, y0]

    inside = True

    for i in range(N):
        # Carichi verticali (autopeso + sovraccarico)
        W_block = arch_weights[i]
        S_block = surch_weights[i]
        Wx = 0.0
        Wy = -(W_block + S_block)  # negativo = verso il basso

        # Baricentri
        xGa, yGa = arch_centroids[i]
        xGs, yGs = surch_centroids[i]

        # Reazione al giunto i
        Rx_i, Ry_i = Rres[i]
        # Punto di pressione giunto i
        xA, yA = line_of_thrust[i]

        # Giunto i+1
        xi_in2, yi_in2 = intrados_joints[i+1]
        xi_ex2, yi_ex2 = extrados_joints[i+1]

        # Reazione al giunto i+1 si ricava come:
        # R[i+1] = - (R[i] + W)
        Rx_next = - (Rx_i + Wx)
        Ry_next = - (Ry_i + Wy)

        # Funzione di momento residuo rispetto a (xB,yB) incognito
        def residual_moment(lam):
            # Parametrizzazione del punto di pressione sul giunto i+1
            xB = xi_in2 + lam*(xi_ex2 - xi_in2)
            yB = yi_in2 + lam*(yi_ex2 - yi_in2)

            # Momento generato da R[i], W_block, S_block
            dxA = xA - xB
            dyA = yA - yB
            M_Ri = dxA*Ry_i - dyA*Rx_i

            dxGa = xGa - xB
            dyGa = yGa - yB
            M_Wa = dxGa * (-W_block)  # cross((dxGa, dyGa),(0,-W_block))

            dxGs = xGs - xB
            dyGs = yGs - yB
            M_Ws = dxGs * (-S_block)  # cross((dxGs, dyGs),(0,-S_block))

            return M_Ri + M_Wa + M_Ws

        # Cerchiamo lam in [0,1] via bisezione
        lam_min, lam_max = 0.0, 1.0
        m_min = residual_moment(lam_min)
        m_max = residual_moment(lam_max)

        if m_min*m_max > 0:
            # Non c'è cambio di segno => non esiste lam in [0,1]
            inside = False
            # fallback: lam = 0 o 1, a seconda di chi è più vicino allo zero
            if abs(m_min) < abs(m_max):
                lam_star = 0.0
            else:
                lam_star = 1.0
        else:
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

        xB = xi_in2 + lam_star*(xi_ex2 - xi_in2)
        yB = yi_in2 + lam_star*(yi_ex2 - yi_in2)

        # Assegniamo reazione e punto di pressione
        Rres[i+1,0] = Rx_next
        Rres[i+1,1] = Ry_next

        line_of_thrust[i+1] = [xB, yB]

    return line_of_thrust, Rres, inside


def total_load(arch_weights, surch_weights):
    return np.sum(arch_weights) + np.sum(surch_weights)

# ----------------------------------------------------------------------------------------
# 3) DOPPIA BISEZIONE SU (H, V0)
# ----------------------------------------------------------------------------------------

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
      - sum reazioni verticali = sum dei carichi (autopeso + sovraccarico)
      - la reazione orizzontale a destra = H
      - la linea di spinta rimanga 'inside' su tutti i giunti.

    Ritorna:
      H_opt, V0_opt, line_of_thrust, success
    """
    W_tot = total_load(arch_weights, surch_weights)
    if V0_max is None:
        V0_max = 1.2 * W_tot  # limite di sicurezza per V0

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

            sumV = Rres[0,1] + Rres[-1,1]  # reazione verticale tot (sinistra + destra)
            fV = sumV - W_tot

            if inside_flag:
                found_V0 = V0_try
                line_sol = line_thrust
                R_sol = Rres
                inside_sol = True

            if abs(fV) < tol:
                # Siamo abbastanza vicini alla condizione sumV = W_tot
                break

            # Bisezione su V0
            if fV > 0:
                v_high = V0_try
            else:
                v_low = V0_try

        # Se non inside => non c'è soluzione con H_mid => aumenta H_min
        if (found_V0 is None) or (not inside_sol):
            H_min = H_mid
            continue

        # dH = differenza di spinta orizzontale a destra rispetto a H_mid
        dH = R_sol[-1,0] - H_mid
        if abs(dH) < tol:
            success = True
            best_line = line_sol
            best_H = H_mid
            best_V0 = found_V0
            break

        if dH > 0:
            # reazione orizzontale a dx > H_mid => serve + spinta
            H_min = H_mid
        else:
            # reazione orizzontale a dx < H_mid => troppa spinta
            H_max = H_mid

        # Salviamo i migliori
        success = True
        best_line = line_sol
        best_H = H_mid
        best_V0 = found_V0

    return best_H, best_V0, best_line, success

# ----------------------------------------------------------------------------------------
# 4) STREAMLIT APP
# ----------------------------------------------------------------------------------------

def main():
    st.title("Analisi Arco in Muratura - Doppia Bisezione con Sovraccarico")
    st.write("""
    Esempio di app Streamlit per calcolare la linea di spinta in un arco a tutto sesto,
    con bisezione su spinta orizzontale (H) e reazione verticale (V0), e sovraccarico lineare.
    """)

    st.sidebar.header("Parametri")
    # Raggio intradosso
    R = st.sidebar.slider("Raggio intradosso (m)", 1.0, 20.0, 5.0, 0.5)
    # Spessore
    t = st.sidebar.slider("Spessore (m)", 0.1, 3.0, 1.0, 0.1)
    # Numero conci
    N = st.sidebar.slider("Numero conci", 2, 40, 10, 1)
    # Peso specifico (kN/m^3)
    gamma_kN = st.sidebar.slider("Peso specifico muratura (kN/m³)", 15.0, 30.0, 20.0, 1.0)
    gamma = gamma_kN * 1000.0  # conversione in N/m^3

    # Sovraccarico lineare in kN/m
    q_surcharge = st.sidebar.slider("Sovraccarico (kN/m)", 0.0, 50.0, 10.0, 1.0)

    if st.button("Esegui Calcolo"):
        # Costruzione geometria e carichi
        (intrados, extrados,
         arch_w, arch_c,
         surch_w, surch_c) = build_arch_with_surcharge(R, t, N, gamma, q_surcharge)

        # Doppia bisezione per trovare H, V0
        H_opt, V0_opt, line_thrust, success = solve_with_bisection(
            intrados, extrados,
            arch_w, arch_c,
            surch_w, surch_c,
            H_min=0, H_max=1e6,
            tol=1e-2, max_iter=40
        )

        if not success or (line_thrust is None):
            st.error("Impossibile trovare una linea di spinta interna con i parametri scelti. "
                     "Prova a incrementare lo spessore o ridurre il sovraccarico.")
        else:
            st.success("Soluzione trovata!")
            st.write(f"**Spinta orizzontale** H ≈ {H_opt:.2f} N ({H_opt/1000:.2f} kN)")
            st.write(f"**Reazione verticale sinistra** V0 ≈ {V0_opt:.2f} N ({V0_opt/1000:.2f} kN)")
            W_tot = np.sum(arch_w) + np.sum(surch_w)
            st.write(f"**Peso totale** = {W_tot:.2f} N ({W_tot/1000:.2f} kN)")

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
            ax.set_title("Arco - Linea di spinta interna")
            ax.legend()
            st.pyplot(fig)

            st.write("""
            _Nota_: Se l'arco è troppo sottile e/o il sovraccarico è elevato, 
            può essere fisicamente impossibile mantenere la linea di spinta 
            entro l'intradosso e l'estradosso.
            """)


if __name__ == "__main__":
    main()

