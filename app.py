import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# A) Build arch geometry + weights (come prima, invariato)
# -------------------------------------------------------------------------
def build_arch_with_surcharge(R, t, N, gamma, q_surcharge):
    # (Codice invariato: calcolo di intrados_joints, extrados_joints, arch_weights, surch_weights, ecc.)
    # ...
    pass  # Sostituisci con la tua implementazione

# -------------------------------------------------------------------------
# B) Risoluzione di un singolo set (H, V0) => solve_arch
# -------------------------------------------------------------------------
def solve_arch(...):
    # (Codice invariato: calcolo della linea di spinta blocco-blocco)
    # ...
    pass  # Sostituisci con la tua implementazione

def total_load(arch_weights, surch_weights):
    return np.sum(arch_weights) + np.sum(surch_weights)

# -------------------------------------------------------------------------
# C) Doppia bisezione su (H, V0) con controllo dei motivi di fallimento
# -------------------------------------------------------------------------
def solve_with_bisection(
    intrados_joints, extrados_joints,
    arch_weights, arch_centroids,
    surch_weights, surch_centroids,
    H_min=0, H_max=1e6,
    V0_min=0, V0_max=None,
    tol=1e-2, max_iter=40
):
    """
    Ritorna:
      - (H_opt, V0_opt, line_of_thrust, success, error_message)
    """
    W_tot = total_load(arch_weights, surch_weights)
    if V0_max is None:
        V0_max = 1.2 * W_tot

    best_line = None
    best_H = None
    best_V0 = None
    success = False
    error_message = "Motivo non identificato."

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

            # Forza verticale totale
            sumV = Rres[0,1] + Rres[-1,1]
            fV = sumV - W_tot

            if inside_flag:
                found_V0 = V0_try
                line_sol = line_thrust
                R_sol = Rres
                inside_sol = True

            # Bisezione su fV
            if abs(fV) < tol:
                break
            if fV > 0:
                v_high = V0_try
            else:
                v_low = V0_try

        if (found_V0 is None) or (not inside_sol):
            # Non abbiamo trovato un V0 con cui la linea di spinta fosse inside
            # => in genere serve spinta H più grande => H_min = H_mid
            # Controlliamo se H_mid è già molto vicino a H_max:
            if abs(H_max - H_mid) < 1e-3:
                error_message = (
                    "Spinta orizzontale massima raggiunta. "
                    "Forse l'arco è troppo sottile o il sovraccarico è troppo elevato. "
                    "Prova ad aumentare lo spessore o ridurre il carico."
                )
            H_min = H_mid
            continue

        # Se siamo qui, inside_sol = True => verifichiamo l'equilibrio orizzontale
        dH = R_sol[-1,0] - H_mid
        if abs(dH) < tol:
            # Successo
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

    if not success:
        # Se siamo usciti dal loop e success = False
        # Forse abbiamo saturato anche la reazione verticale
        # Verifichiamo se V0 ha toccato V0_max
        if abs(V0_max - 1.2*W_tot) < 1e-3:
            error_message = (
                "La reazione verticale sinistra ha raggiunto il massimo (1.2 * peso totale). "
                "Possibile che il carico sia troppo elevato o la configurazione geometrica non regga "
                "senza tiranti. Prova ad aumentare lo spessore o ridurre il sovraccarico."
            )
        else:
            error_message = (
                "Non è stato possibile trovare una linea di spinta interna. "
                "Prova a modificare i parametri (maggiore spessore, minore sovraccarico, ecc.)."
            )

    return best_H, best_V0, best_line, success, error_message


# -------------------------------------------------------------------------
# D) STREAMLIT APP
# -------------------------------------------------------------------------
def main():
    st.title("Arco in Muratura - con spiegazione dell’errore")
    st.write("""
    Se non si trova la linea di spinta, forniamo suggerimenti specifici
    sui parametri da cambiare.
    """)

    # Sidebar
    R = st.sidebar.slider("Raggio intradosso (m)", 1.0, 20.0, 5.0, 0.5)
    t = st.sidebar.slider("Spessore (m)", 0.1, 3.0, 1.0, 0.1)
    # ... e così via per gamma, q_surcharge, N

    if st.button("Esegui Calcolo"):
        # 1) build_arch_with_surcharge
        # intrados, extrados, arch_w, arch_c, surch_w, surch_c = build_arch_with_surcharge(...)

        # 2) solve_with_bisection
        H_opt, V0_opt, line_thrust, success, err_msg = solve_with_bisection(
            intrados, extrados, 
            arch_w, arch_c,
            surch_w, surch_c,
            # parametri di default
        )

        if success and line_thrust is not None:
            st.success(f"Soluzione trovata: H = {H_opt:.2f} N, V0 = {V0_opt:.2f} N")
            # Mostra grafico ...
        else:
            # Error e suggerimento
            st.error(err_msg)

if __name__ == "__main__":
    main()

