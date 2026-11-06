"""
Diagnostic for how CircAdapt handles beat-by-beat timing. It does not run the pacemaker model.
Run it from the Modeling Project folder and paste the whole output back:

    python circadapt_timing_diagnostic.py
"""
import traceback

import numpy as np
from circadapt import VanOsta2024

MMHG = 133.322


def beat_summary(m):
    """Stroke volume (mL) and peak LV pressure (mmHg) for the most recent cycle."""
    t = np.asarray(m["Solver"]["t"], float)
    q = np.asarray(m["Valve"]["q"][:, "LvSyArt"], float)
    p = np.asarray(m["Cavity"]["p"][:, "cLv"], float) / MMHG
    dt = np.diff(t)
    sv = float(np.sum(0.5 * (q[1:] + q[:-1]) * dt)) * 1e6
    return sv, float(p.max())


def fresh():
    m = VanOsta2024()
    m["General"]["t_cycle"] = 0.8
    m["Timings"]["law_tau_av"] = 0
    m["Timings"]["tau_av"] = 0.12
    m["PFC"]["is_active"] = True
    m["PFC"]["q0"] = 5.0 / 60e3
    m.run(stable=True)
    m["PFC"]["is_active"] = False
    return m


def section(title, fn):
    print(f"\n=== {title} ===")
    try:
        fn()
    except Exception:
        traceback.print_exc(limit=2)


def a_names():
    m = fresh()
    print("Timings parameters:", list(m["Timings"].parameters))
    print("Timings signals:   ", list(m["Timings"].signals))
    print("Timings objects:   ", list(m["Timings"].objects))
    print("General parameters:", list(m["General"].parameters))
    print("Patch objects:     ", list(m["Patch"].objects))


def b_activation():
    m = fresh()
    print("Activation time of each patch, relative to Solver t at the start of the cycle")
    for tau in (0.12, 0.25):
        m["Timings"]["tau_av"] = tau
        m.run(1)
        act = np.asarray(m["Patch"]["moment_of_activation"], float)
        t0 = float(np.asarray(m["Solver"]["t"], float)[0])
        print(f"tau_av={tau}: first={np.round(act[0], 4)} last={np.round(act[-1], 4)} t0={t0:.4f}")


def c_time_act():
    m = fresh()
    print("Default laws on:")
    for T in (0.6, 0.8, 1.0):
        m["General"]["t_cycle"] = T
        m.run(1)
        print(f"  t_cycle={T}: time_act={np.round(np.ravel(m['Patch']['time_act'][:]), 4)}")
    m["General"]["t_cycle"] = 0.8
    m["Timings"]["law_ta"] = 0
    m["Timings"]["law_tv"] = 0
    print("Laws off, writing ventricular time_act, 5 beats each:")
    for value in (0.25, 0.425, 0.60):
        m["Patch"]["time_act"][2:] = value
        for _ in range(5):
            m.run(1)
        sv, peak = beat_summary(m)
        print(f"  wrote {value}: read {np.round(np.ravel(m['Patch']['time_act'][:]), 4)} "
              f"SV={sv:.1f} LVpeak={peak:.0f}")


def d_tau_sweep():
    print("Where does a late ventricular activation stop working? (5 beats at each point)")
    print(f"{'t_cycle':>8}{'tau_av':>8}{'SV mL':>9}{'LV peak':>9}{'status':>10}")
    for T in (0.8, 1.2):
        for frac in (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
            tau = round(T * frac, 3)
            m = fresh()
            m["General"]["t_cycle"] = T
            m["Timings"]["tau_av"] = tau
            try:
                for _ in range(5):
                    m.run(1)
                sv, peak = beat_summary(m)
                print(f"{T:>8}{tau:>8}{sv:>9.1f}{peak:>9.0f}{'ok':>10}")
            except Exception as e:
                print(f"{T:>8}{tau:>8}{'':>18}{type(e).__name__:>10}")


def e_patch_dt():
    print("Can Patch dt place a late atrial contraction while the ventricle stays early?")
    for shift in (0.0, 0.2, 0.4, 0.6, 0.7):
        m = fresh()
        m["Patch"]["dt"][:2] = shift
        for _ in range(5):
            m.run(1)
        sv, peak = beat_summary(m)
        la = np.asarray(m["Cavity"]["p"][:, "La"], float) / MMHG
        t = np.asarray(m["Solver"]["t"], float)
        print(f"atrial dt={shift}: SV={sv:.1f} LVpeak={peak:.0f} "
              f"LApeak={la.max():.1f} at t={t[int(la.argmax())] - t[0]:.3f}s")


def f_late_next_cycle():
    print("Does a contraction started late in one cycle carry over when the next cycle is normal?")
    m = fresh()
    for _ in range(5):
        m.run(1)
    print("baseline: SV=%.1f LVpeak=%.0f" % beat_summary(m))
    m["Timings"]["tau_av"] = 0.70
    m.run(1)
    print("cycle with tau_av=0.70: SV=%.1f LVpeak=%.0f" % beat_summary(m))
    m["Timings"]["tau_av"] = 0.12
    m.run(1)
    print("next cycle tau_av=0.12: SV=%.1f LVpeak=%.0f" % beat_summary(m))


if __name__ == "__main__":
    section("A. Parameter names", a_names)
    section("B. Activation times", b_activation)
    section("C. Contraction duration", c_time_act)
    section("D. Late ventricular activation", d_tau_sweep)
    section("E. Delaying the atria with Patch dt", e_patch_dt)
    section("F. Carry-over between cycles", f_late_next_cycle)
