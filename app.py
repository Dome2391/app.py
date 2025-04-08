import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------
# 1) COSTRUZIONE GEOMETRIA + CARICHI
# ----------------------------------------------------------------------------------------

def build_arch_with_surcharge(R, t, N, gamma, q_surcharge):
    """
    Discretizza un arco a tutto sesto:
      - intrados_joints[i], extrados_joints[i]
      - arch_weights[i], arch_centroids[i]: autopeso blocco
      - surch_weights[i], surch_centroids[i]: sovraccarico blocco (lineare kN/m)
    """
    theta_j = np.linspace(0, np.pi, N+1)

    # Intradosso
    xi_in = R * np.cos(theta_j)
    yi_in = R * np.sin(theta_j)
    intrados_joints = np.column_stack((xi_in, yi_in))

    # Estradosso
    xi_ex = (R + t) * np.cos(theta_j)
    yi_ex = (R + t) * np.sin(theta_j)
    extrados_joints = np.column_stack((xi_ex, yi_ex))

    # Converti q_surcharge in N/m
    q_surcharge_n = q_surcharge * 1000.0

    arch_weights = []
    arch_centroids = []
    surch_weights = []
    surch_centroids = []

    thickness_z = 1.0  # spessore fuori piano

    for i in range(N):
        dtheta = theta_j[i+1] - theta_j[i]

        # Autopeso
        area_i = 0.5 * dtheta * ((R + t)**2 - R**2)
        W_i = area_i * thickness_z * gamma
        arch_weights.append(W_i)

        # Baricentro blocco
        theta_m = 0.5*(theta_j[i] + theta_j[i+1])
        r_m = R + t/2
        xGa = r_m * np.cos(theta_m)
        yGa = r_m * np.sin(theta_m)
        arch_centroids.append([xGa, yGa])

        # Sovraccarico
        length_i = (R + t)*dtheta
        S_i = q_surcharge_n * length_i * thickness_z
        xGs = (R + t)*np.cos(theta_m)
        yGs = (R + t)*np.sin(theta_m)
        surch_weights.append(S_i)
        surch_centroids.append([xGs, yGs])

    return (intrados_joints,
            extrados_joints,
            np.array(arch_weights),
            np.array(arch_centroids),
            np.array(surch_weights),
            np.array(surch_centroids))

# ----------------------------------------------------------------------------------------
# 2) FUNZIONE CHE RISOLVE L'EQUILIBRIO SU UN BLOCCO (DATI H, V0)
# ----------------------------------------------------------------------------------------

def solve_arch(
    intrados_joints, extrados_joints,
    arch_weights, arch_centroids,
    surch_weights, surch_centroids,
    H, V0
):
    """
    Dato (H, V0) all'appoggio sinistro, calcola la linea di spinta
    blocco per blocco. Se in ogni giunto la posizione del centro di pressione
    è tra intradosso ed estradosso, inside = True. Altrimenti False.
    """
    N = len(arch_weights)
    Rres = np.zeros((N+1, 2))
    line_of_thrust = np.zeros((N+1, 2))

    # Reazione appoggio sinistro
    Rres[0,0] = H
    Rres[0,1] = V0

    # Punto di pressione giunto 0: ipotesi a metà spessore
    x0 = 0.5*(intrados_joints[0,0] + extrados_joints[0,0])
    y0 = 0.5*(intrados_joints[0,1] + extrados_joints[0,1])
    line_of_thrust[0] = [x0, y0]

    inside = True

    for i in range(N):
        Wx = 0.0
        Wy = -(arch_weights[i] + surch_weights[i])  # forza discendente

        xGa, yGa = arch_centroids[i]
        xGs, yGs = surch_centroids[i]

        # Reazione al giunto i
        Rx_i, Ry_i = Rres[i]
        xA, yA = line_of_thrust[i]

        # Giunto i+1
        xi_in2, yi_in2 = intrados_joints[i+1]
        xi_ex2, yi_ex2 = extrados_joints[i+1]

        # Reazione giunto i+1 = -(R[i] + W)
        Rx_next = - (Rx_i + Wx)
        Ry_next = - (Ry_i + Wy)

        # Funzione di momento rispetto (xB, yB)
        def residual_moment(lam):
            xB = xi_in2 + lam*(xi_ex2 - xi_in2)
            yB = yi_in2 + lam*(yi_ex2 - yi_in2)

            # Momenti
            dxA = xA - xB
            dyA = yA - yB
            M_Ri = dxA*Ry_i - dyA*Rx_i

            dxGa = xGa - xB
            dyGa = yGa - yB
            M_Wa = dxGa*(-arch_weights[i])

            dxGs = xGs - xB
            dyGs = yGs - yB
            M_Ws = dxGs*(-surch_weights[i])

            return (M_Ri + M_Wa + M_Ws)

        # Bisezione su lam in [0,1]
        lam_min, lam_max = 0.0, 1.0
        m_min = residual_moment(lam_min)
        m_max = residual_moment(lam_max)

        if m_min*m_max > 0:
            # Nessuna soluzione in [0,1]
            inside = False
            # fallback: lam=0.0 o lam=1.0
            lam_star = 0.0 if abs(m_min)<abs(m_max) else 1.0
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

        if lam_star < 0 or lam_star > 1:
            inside = False

        xB = xi_in2 + lam_star*(xi_ex2 - xi_in2)
        yB = yi_in2 + lam_star*(yi_ex2 - yi_in2)

        Rres[i+1,0] = Rx_next
        Rres[i+1,1] = Ry_next
        line_of_thrust[i+1] = [xB, yB]

    return line_of_thrust, Rres, inside

def total_load(arch_weights, surch_weights):
    return np.sum(arch_weights) + np.sum(surch_weights)

# ----------------------------------------------------------------------------------------
# 3) DOPPIA BISEZIONE (H, V0) - SENZA LIMITE ALLA REAZIONE VERT
# ----------------------------------------------------------------------------------------

def solve_with_bisection(
    intrados_joints, extrados_joints,
    arch_weights, arch_centroids,
    surch_weights, surch_centroids,
    H_min=0, H_max=1e8,      # range ampio per la spinta
    V0_min=0, V0_max=1e8,   # range ampio per la reaz. verticale
    tol=1e-2,
    max_iter=50
):
    """
    Cerca (H, V0) in [H_min,H_max] x [V0_min,V0_max]
    tali che:
      - sum reazioni verticali (sinistra+destra) = sum carichi (autopeso + sovraccarico)
      - la reazione orizzontale a dx = H
      - inside = True in tutti i blocchi
    """
    W_tot = total_load(arch_weights, surch_weights)

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

        # 1) Cerchiamo V0 in [v_low, v_high] che soddisfi sumV ~ W_tot e inside=True
        for _ in range(25):
            V0_try = 0.5*(v_low + v_high)
            line_thrust, Rres, inside_flag = solve_arch(
                intrados_joints, extrados_joints,
                arch_weights, arch_centroids,
                surch_weights, surch_centroids,
                H_mid, V0_try
            )

            sumV = Rres[0,1] + Rres[-1,1]  # reazione vert tot
            fV = sumV - W_tot

            if inside_flag:
                found_V0 = V0_try
                line_sol = line_thrust
                R_sol = Rres
                inside_sol = True

            if abs(fV) < tol:
                # trovato un V0 che equilibra
                break

            if fV > 0:
                v_high = V0_try
            else:
                v_low = V0_try

        if not inside_sol or found_V0 is None:
            # Non abbiamo trovato un V0 in quell'intervallo che mantenga inside => serve + spinta
            H_min = H_mid
            continue

        # 2) Controlliamo differenza orizzontale
        dH = R_sol[-1,0] - H_mid
        if abs(dH) < tol:
            success = True
            best_line = line_sol
            best_H = H_mid
            best_V0 = found_V0
            break

        if dH > 0:
            # reazione orizzontale a dx > H_mid => alziamo H
            H_min = H_mid
        else:
            # reazione orizzontale a dx < H_mid => abbassiamo H
            H_max = H_mid

        # salviamo l'ultima configurazione "inside"
        success = True
        best_line = line_sol
        best_H = H_mid
        best_V0 = found_V0

    return best_H, best_V0, best_line, success

# ----------------------------------------------------------------------------------------
# 4) STREAMLIT APP
# ----------------------------------------------------------------------------------------

def main():
    st.title("Arco in Muratura (Senza limite alla reazione verticale)")
    st.write("""
    - Metodo doppia bisezione su H e V0
    - Nessun limite su reazione verticale (V0).
    - Range ampio di H, V0 = 0..1e8
    """)

    st.sidebar.header("Parametri Arco")
    R = st.sidebar.slider("Raggio intradosso (m)", 1.0, 20.0, 5.0, 0.5)
    t = st.sidebar.slider("Spessore (m)", 0.1, 5.0, 1.0, 0.1)
    N = st.sidebar.slider("Numero conci", 2, 40, 10, 1)
    gamma_kN = st.sidebar.slider("Peso specifico muratura (kN/m³)", 15.0, 30.0, 20.0, 1.0)
    gamma = gamma_kN * 1000.0
    q_surcharge = st.sidebar.slider("Sovraccarico (kN/m)", 0.0, 50.0, 10.0, 1.0)

    if st.button("Esegui Calcolo"):
        (intrados, extrados,
         arch_w, arch_c,
         surch_w, surch_c) = build_arch_with_surcharge(R, t, N, gamma, q_surcharge)

        H_opt, V0_opt, line_thrust, success = solve_with_bisection(
            intrados, extrados,
            arch_w, arch_c,
            surch_w, surch_c,
            H_min=0, H_max=1e8,      # spinta orizzontale
            V0_min=0, V0_max=1e8,   # reaz. verticale
            tol=1e-2, max_iter=50
        )

        if not success or line_thrust is None:
            st.error("Impossibile trovare una linea di spinta interna con questi parametri, "
                     "anche con spessore elevato e sovraccarico ridotto. "
                     "Verifica la geometria o possibile bug.")
        else:
            st.success("Soluzione trovata!")
            st.write(f"**Spinta orizzontale** H ≈ {H_opt:.2f} N  ({H_opt/1000:.2f} kN)")
            st.write(f"**Reazione verticale sinistra** V0 ≈ {V0_opt:.2f} N  ({V0_opt/1000:.2f} kN)")

            W_tot = np.sum(arch_w) + np.sum(surch_w)
            st.write(f"**Carico totale** = {W_tot:.2f} N  ({W_tot/1000:.2f} kN)")

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
            ax.set_title("Arco - Linea di spinta interna (no V0 limit)")
            ax.legend()
            st.pyplot(fig)

            st.write("""
            _Se anche così non trovi soluzione, potrebbero esserci errori di calcolo, 
            oppure la geometria scelta è incompatibile con l'equilibrio a compressione pura._
            """)

if __name__ == "__main__":
    main()
