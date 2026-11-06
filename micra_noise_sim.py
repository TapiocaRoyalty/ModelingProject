"""
micra_noise_sim.py -- Micra AV-style leadless pacing in mechanical noise, driving CircAdapt.

Question: when accelerometer noise changes what the pacemaker does (false atrial
sensing, lost atrial sensing, or a switch from VDD tracking to sensor-driven
pacing), what happens to hemodynamics?

How the file is organized (search for "STEP" to jump between sections)
    STEP 1  Settings            pacemaker, experiment and noise settings
    STEP 2  Signals             true atrial beats and what the accelerometer sees
    STEP 3  Pacemaker logic     decides when each ventricular beat is paced
    STEP 4  Timing conversion   turns beat times into CircAdapt cycles
    STEP 5  Heart model         runs CircAdapt one cycle at a time and reads the results
    STEP 6  Full simulation     sends one rhythm through the heart model
    STEP 7  Metrics             summarizes rhythm and hemodynamics
    STEP 8  Experiment          repeats noise vs. control across seeds
    STEP 9  Output              prints the table and saves the plot
    CHECK   Preflight           tests that CircAdapt behaves as this approach requires
    START   main                reads the command line options and runs everything

Usage
    python micra_noise_sim.py --preflight        # run this first, with CircAdapt installed
    python micra_noise_sim.py --noise-rms 1.5 --seeds 5

Scope and assumptions
    * Patient: complete AV block, no retrograde VA conduction, AV conduction mode switch off.
    * This is not Medtronic's firmware. Its noise handling is not public. See the SOURCE
      tags in DeviceConfig: [published] came from the literature, [assumed] is a placeholder
      that you should replace or sweep.
    * No baroreflex and no sinus-rate response, so results show the unbuffered, acute effect
      of timing. Sinus rate is an input and is not changed by the hemodynamics.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import circadapt
import matplotlib.pyplot as plt
import numpy as np
from circadapt import VanOsta2024

MMHG = 133.322  # Pa per mmHg


# ===========================================================================
# STEP 1: SETTINGS
# What the pacemaker is programmed to do, and what experiment to run.
# ===========================================================================
# 1a. Pacemaker settings
@dataclass
class DeviceConfig:
    # [published] programmable values / nominals (Garweg 2024; Micra AV spec sheet)
    lower_rate: float = 50.0          # bpm
    upper_tracking_rate: float = 105.0  # bpm; 115 needs PVAB 500 ms
    am_vp: float = 0.020              # s, sensed AV delay, nominal 20 ms
    pvab: float = 0.550               # s, nominal 550 ms
    a3_window_end: float = 0.700      # s, suggested start 700-800 ms
    rate_smoothing_delta: float = 0.100  # s, nominal 100 ms
    activity_margin: float = 20.0     # bpm, sensor must exceed VDD rate by this much
    # [published] A4 occurs about 100 ms after the P wave
    em_delay: float = 0.100           # s

    # [assumed] patient- and setup-specific; sweep these
    a4_amplitude: float = 1.6         # m/s^2
    a4_threshold: float = 1.0         # m/s^2 (programmable 0.7-8)
    a3_threshold: float = 3.0         # m/s^2 (programmable 1-10)

    # [assumed] how noise reaches the rate-response sensor (ADL range 60-160 bpm)
    activity_mode_switch: bool = True
    adl_rate: float = 95.0            # bpm
    upper_sensor_rate: float = 120.0  # bpm
    sensor_gain: float = 45.0         # bpm per m/s^2 RMS above deadband
    sensor_deadband: float = 0.10     # m/s^2 RMS
    sensor_tau_up: float = 30.0       # s
    sensor_tau_down: float = 150.0    # s
    return_hysteresis: float = 5.0    # bpm below ADL rate before returning to VDD

    # [assumed] crossing rate of band-limited noise (Rice formula)
    noise_crossing_rate: float = 10.0  # Hz


# 1b. Experiment settings
@dataclass
class ScenarioConfig:
    duration: float = 420.0
    noise_on: float = 60.0
    noise_off: float = 240.0
    noise_rms: float = 1.5            # m/s^2
    sinus_rate: float = 75.0          # bpm
    sinus_jitter_sd: float = 0.015    # s
    dyssynchrony: float = 0.0         # s, RV-paced LV lateral delay; 0 turns multipatch off


# 1c. Noise level at any moment in time
@dataclass
class NoiseProfile:
    """Noise RMS over time. Subclass or replace `rms` to use a measured vibration trace."""
    t_on: float
    t_off: float
    level: float

    def rms(self, t: float) -> float:
        return self.level if self.t_on <= t < self.t_off else 0.0


# ===========================================================================
# STEP 2: SIGNALS
# Creates the true atrial beats and everything the accelerometer could detect.
# ===========================================================================
# 2a. Container for what the pacemaker did; filled in during STEP 3
@dataclass
class DeviceLog:
    a_times: np.ndarray               # true atrial activations (P waves)
    v_times: np.ndarray               # paced ventricular activations
    v_modes: list                     # 'VDD' or 'VVIR' for each V
    v_causes: list                    # 'am_true', 'am_false', 'escape', 'sensor'
    switches: list                    # (time, new_mode)


# 2b. True atrial beats (P waves): sinus rhythm with small random variation
def sinus_times(duration: float, rate: float, jitter: float, rng) -> np.ndarray:
    base = 60.0 / rate
    n = int(duration / base) + 3
    intervals = np.maximum(base + rng.normal(0.0, jitter, n), 0.25)
    t = np.cumsum(intervals)
    return t[t < duration]


# 2c. Accelerometer signals: real A4 signals plus false peaks caused by noise
def am_candidates(a_times, noise, duration, cfg: DeviceConfig, rng):
    """All accelerometer events that could register as an AM: (time, peak, is_true)."""
    # Real A4 signals: noise is added to their size, so some may fall below threshold.
    events = []
    for a in a_times:
        t = a + cfg.em_delay
        events.append((t, cfg.a4_amplitude + rng.normal(0.0, noise.rms(t)), True))

    # False events: Poisson threshold crossings at the Rice rate, with Rice-distributed
    # peak heights, so a fixed A4 threshold gives a noise-dependent false-event rate.
    dt = 0.05
    for t in np.arange(0.0, duration, dt):
        sigma = noise.rms(t)
        if sigma <= 0:
            continue
        rate = cfg.noise_crossing_rate * np.exp(-cfg.a4_threshold ** 2 / (2 * sigma ** 2))
        if rng.random() < rate * dt:
            peak = np.sqrt(cfg.a4_threshold ** 2 + 2 * sigma ** 2 * rng.exponential())
            events.append((t + rng.uniform(0, dt), peak, False))
    events.sort(key=lambda e: e[0])
    return events


# ===========================================================================
# STEP 3: PACEMAKER LOGIC
# Decides when each ventricular beat is paced, and when the device changes mode.
# ===========================================================================
def simulate_device(cfg: DeviceConfig, scn: ScenarioConfig, noise, rng) -> DeviceLog:
    # 3a. Get the signals from STEP 2.
    a_times = sinus_times(scn.duration, scn.sinus_rate, scn.sinus_jitter_sd, rng)
    events = am_candidates(a_times, noise, scn.duration, cfg, rng)

    # 3b. Set the starting state: VDD tracking, with the sensor at the lower rate.
    lr_int = 60.0 / cfg.lower_rate
    url_int = 60.0 / cfg.upper_tracking_rate
    v_times, v_modes, v_causes, switches = [], [], [], []

    last_v = 0.0
    prev_vv = 60.0 / scn.sinus_rate
    sensor = cfg.lower_rate
    mode = "VDD"
    vdd_rate = scn.sinus_rate
    idx = 0

    # Each pass through this loop produces one ventricular beat.
    while True:
        # 3c. Decide when the next beat happens.
        if mode == "VDD":
            # First event past blanking that clears the threshold for its window.
            # If none arrives in time, pace at the rate-smoothing escape time.
            found = None
            escape = last_v + min(prev_vv + cfg.rate_smoothing_delta, lr_int)
            while idx < len(events) and events[idx][0] < last_v + cfg.pvab:
                idx += 1
            j = idx
            while j < len(events) and events[j][0] + cfg.am_vp < escape:
                t, peak, is_true = events[j]
                thr = cfg.a3_threshold if t < last_v + cfg.a3_window_end else cfg.a4_threshold
                if peak >= thr:
                    found = (t, is_true)
                    break
                j += 1
            if found:
                next_v = max(found[0] + cfg.am_vp, last_v + url_int)
                cause = "am_true" if found[1] else "am_false"
            else:
                next_v, cause = escape, "escape"
        else:
            # VVIR mode: ignore atrial signals and pace at the sensor rate.
            next_v, cause = last_v + 60.0 / max(sensor, cfg.lower_rate), "sensor"

        if next_v >= scn.duration:
            break

        # 3d. Record the beat.
        vv = next_v - last_v
        v_times.append(next_v)
        v_modes.append(mode)
        v_causes.append(cause)
        prev_vv, last_v = vv, next_v

        # 3e. Update the activity sensor. Rate-response sensor: first-order response to noise-driven activity.
        target = np.clip(
            cfg.lower_rate + cfg.sensor_gain * max(noise.rms(next_v) - cfg.sensor_deadband, 0.0),
            cfg.lower_rate, cfg.upper_sensor_rate)
        tau = cfg.sensor_tau_up if target > sensor else cfg.sensor_tau_down
        sensor += (target - sensor) * (1 - np.exp(-vv / tau))

        # 3f. Check whether the device should switch mode.
        if mode == "VDD":
            vdd_rate = 0.9 * vdd_rate + 0.1 * 60.0 / vv
            if (cfg.activity_mode_switch and sensor >= cfg.adl_rate
                    and sensor >= vdd_rate + cfg.activity_margin):
                mode = "VVIR"
                switches.append((next_v, mode))
        elif sensor < cfg.adl_rate - cfg.return_hysteresis:
            mode = "VDD"
            prev_vv = vv
            switches.append((next_v, mode))

    return DeviceLog(a_times, np.array(v_times), v_modes, v_causes, switches)


# ===========================================================================
# STEP 4: TIMING CONVERSION
# Turns beat times into the settings CircAdapt needs for each cycle.
# ===========================================================================
TAU_BASE = 0.12   # s, fixed tau_av; every cycle's ventricular activation sits here
BOUNDARY_TOL = 1e-6  # s


@dataclass
class Frame:
    """One CircAdapt cycle, anchored to the ventricle: V at tau_av, optional A at atrial_dt."""
    start: float
    t_cycle: float      # normally the V-V interval, so the default laws scale contraction
    tau_av: float
    atrial_dt: float
    atrial_on: bool
    t_v: float


def late_zone(t_cycle):
    """How close to the end of a cycle an activation can be before CircAdapt loses it."""
    return max(0.12, 0.15 * t_cycle)


def build_frames(a_times, v_times, tau_base=TAU_BASE):
    """
    Each cycle normally runs from (V - tau_base) to (next V - tau_base). The atrial activation
    in that span is placed with Patch dt, so it can land anywhere, including during systole.

    An activation very late in a cycle is lost by CircAdapt, so if an A would fall in that
    late zone, the cycle boundary moves back to the A. It then fires at dt = 0 of the next
    cycle with a slightly longer tau_av. Event times stay exact, but t_cycle differs from V-V
    for those two cycles, which slightly changes their contraction duration.
      * several A's in a cycle -> the last is kept and the others are dropped. This is the
        approximation: check that dropped_atrial stays small.
      * no A in a cycle -> the atria are silenced. This is the correct rhythm when the
        ventricle is paced faster than sinus, not an error.
    """
    v_times = np.asarray(v_times)
    a_times = np.asarray(a_times)
    # 4a. Place cycle boundaries, moving any that would leave an A in the late zone.
    starts = v_times - tau_base
    moved = 0
    for j in range(1, len(starts)):
        zone = late_zone(starts[j] - starts[j - 1])
        late = a_times[(a_times >= starts[j] - zone) & (a_times < starts[j] - BOUNDARY_TOL)]
        if late.size:
            starts[j] = late[-1]
            moved += 1

    # 4b. Build each cycle and place its atrial beat, if it has one.
    frames, dropped, silent = [], 0, 0
    for j in range(len(v_times) - 1):
        s, e = starts[j], starts[j + 1]
        # Tracked beats land exactly on a boundary, so rounding must not decide their cycle.
        inside = a_times[(a_times >= s - BOUNDARY_TOL) & (a_times < e - BOUNDARY_TOL)]
        dropped += max(inside.size - 1, 0)
        silent += inside.size == 0
        dt = max(float(inside[-1] - s), 0.0) if inside.size else 0.0
        frames.append(Frame(float(s), float(e - s), float(v_times[j] - s), dt,
                            bool(inside.size), float(v_times[j])))
    return frames, {"dropped_atrial": dropped, "silenced_atrial": int(silent),
                    "moved_boundaries": moved, "n_frames": len(frames)}


# ===========================================================================
# STEP 5: HEART MODEL
# Runs CircAdapt one cycle at a time and reads the results of each beat.
# ===========================================================================
class CircAdaptBackend:
    # 5a. Create and stabilize the heart.
    def __init__(self, sinus_rate=75.0, dyssynchrony=0.0, target_co=5.0):
        self.m = m = VanOsta2024()
        if dyssynchrony > 0:
            # Multipatch pattern from the CircAdapt tutorial; RV pacing activates septum
            # then LV lateral wall late. Slices: La, Ra, Lv x12, Sv x6, Rv.
            m["Wall"]["n_patch"][2:4] = [12, 6]
            m["Patch"]["dt"][2:14] = np.linspace(dyssynchrony / 2, dyssynchrony, 12)
            m["Patch"]["dt"][14:20] = np.linspace(0.0, dyssynchrony / 2, 6)

        # The default contraction laws stay on: t_cycle now equals V-V, so ventricular
        # duration scales correctly. Atrial duration also follows V-V, a small approximation.
        m["General"]["t_cycle"] = 60.0 / sinus_rate
        m["Timings"]["law_tau_av"] = 0
        m["Timings"]["tau_av"] = TAU_BASE
        m["PFC"]["is_active"] = True
        m["PFC"]["q0"] = target_co / 60e3
        m.run(stable=True)

        # Then freeze regulation and take beat-by-beat control.
        m["PFC"]["is_active"] = False
        self.atrial_sf = np.array(m["Patch"]["Sf_act"][:2], dtype=float)

    # 5b. Apply one cycle's settings from STEP 4 and run one beat.
    def run_frame(self, f: Frame) -> dict:
        m = self.m
        m["General"]["t_cycle"] = f.t_cycle
        m["Timings"]["tau_av"] = f.tau_av
        m["Patch"]["dt"][:2] = f.atrial_dt
        m["Patch"]["Sf_act"][:2] = self.atrial_sf if f.atrial_on else 0.0
        m.run(1)
        return self._read()

    # 5c. Read pressures and flows, then calculate this beat's results.
    def _read(self) -> dict:
        m = self.m
        t = np.asarray(m["Solver"]["t"], dtype=float)
        span = t[-1] - t[0]
        p = m["Cavity"]["p"]
        q_ao = np.asarray(m["Valve"]["q"][:, "LvSyArt"], dtype=float)
        q_mv = np.asarray(m["Valve"]["q"][:, "LaLv"], dtype=float)
        q_pv = np.asarray(m["Valve"]["q"][:, "PuVenLa"], dtype=float)
        p_ao = np.asarray(p[:, "SyArt"], dtype=float) / MMHG
        p_la = np.asarray(p[:, "La"], dtype=float) / MMHG
        p_ra = np.asarray(p[:, "Ra"], dtype=float) / MMHG
        p_lv = np.asarray(p[:, "cLv"], dtype=float) / MMHG
        mean = lambda y: float(np.trapezoid(y, t) / span) if hasattr(np, "trapezoid") \
            else float(np.trapz(y, t) / span)
        integ = lambda y: mean(y) * span
        return {
            "map": mean(p_ao), "sbp": float(p_ao.max()), "dbp": float(p_ao.min()),
            "mlap": mean(p_la), "peak_lap": float(p_la.max()),
            "peak_rap": float(p_ra.max()), "lv_peak": float(p_lv.max()),
            "stroke_ml": integ(q_ao) * 1e6,
            "mv_back_ml": -integ(np.minimum(q_mv, 0)) * 1e6,
            "pv_back_ml": -integ(np.minimum(q_pv, 0)) * 1e6,
        }


# ===========================================================================
# STEP 6: FULL SIMULATION
# Sends one complete rhythm through the heart model, beat by beat.
# ===========================================================================
BEAT_KEYS = ["map", "sbp", "dbp", "mlap", "peak_lap", "peak_rap", "lv_peak",
             "stroke_ml", "mv_back_ml", "pv_back_ml"]


# 6a. Save CircAdapt's own plot of the current beat
def snapshot_model_plot(backend, label, path):
    """CircAdapt's own plot of the most recent beat, titled and saved straight away."""
    try:
        backend.m.plot()
        fig = plt.gcf()
        fig.suptitle(label)
        fig.savefig(path, dpi=160)
        print(f"Saved {path}")
    except Exception as e:
        print(f"model.plot() failed at '{label}': {type(e).__name__}: {e}")


# 6b. Convert the rhythm (STEP 4), then run every cycle (STEP 5)
def run_scenario(backend, log: DeviceLog, snapshots=None) -> dict:
    pending = sorted(snapshots or [], key=lambda s: s[0])
    frames, fidelity = build_frames(log.a_times, log.v_times)
    rows, times, crashed = [], [], None
    crash_type = circadapt.error.ModelCrashed
    for f in frames:
        try:
            rows.append(backend.run_frame(f))
            times.append(f.t_v)
            if pending and f.t_v >= pending[0][0]:
                _, label, path = pending.pop(0)
                mode = log.v_modes[len(times) - 1]
                snapshot_model_plot(backend, f"{label}: {mode}, t = {f.t_v:.1f} s", path)
        except crash_type:
            crashed = f.t_v
            break
    out = {k: np.array([r[k] for r in rows]) for k in BEAT_KEYS}
    out.update(t=np.array(times), log=log, fidelity=fidelity, crashed_at=crashed,
               frames=frames[:len(rows)])
    return out


# ===========================================================================
# STEP 7: METRICS
# Summarizes the rhythm and hemodynamics for each phase of the experiment.
# ===========================================================================
# 7a. Rhythm: rate, AV synchrony, false signals, escape beats and time in VVIR
def rhythm_metrics(log: DeviceLog, t0: float, t1: float) -> dict:
    """AV synchrony uses the 300 ms P-to-V criterion from the MARVEL 2 protocol."""
    v = log.v_times
    sel = (v >= t0) & (v < t1)
    vs = v[sel]
    if vs.size == 0:
        return {}
    causes = np.array(log.v_causes)[sel]
    pv, va = [], []
    for x in vs:
        before, after = log.a_times[log.a_times <= x], log.a_times[log.a_times > x]
        pv.append(x - before[-1] if before.size else np.inf)
        va.append(after[0] - x if after.size else np.inf)
    return {
        "v_rate_bpm": 60.0 / np.mean(np.diff(vs)) if vs.size > 1 else np.nan,
        "av_sync_pct": 100.0 * np.mean(np.array(pv) <= 0.300),
        "atria_in_systole_pct": 100.0 * np.mean(np.array(va) < 0.300),
        "false_am_pct": 100.0 * np.mean(causes == "am_false"),
        "escape_pct": 100.0 * np.mean(causes == "escape"),
        "vvir_pct": 100.0 * np.mean(np.array(log.v_modes)[sel] == "VVIR"),
    }


# 7b. Hemodynamics: cardiac output, arterial pressure, LA and RA pressure, backflow
def phase_metrics(res: dict, t0: float, t1: float) -> dict:
    sel = (res["t"] >= t0) & (res["t"] < t1)
    if not sel.any():
        return {}
    out = {
        "co_L_min": res["stroke_ml"][sel].sum() / 1000.0 / ((t1 - t0) / 60.0),
        "map": res["map"][sel].mean(), "map_min": res["map"][sel].min(),
        "mlap": res["mlap"][sel].mean(), "peak_lap_max": res["peak_lap"][sel].max(),
        "peak_rap_max": res["peak_rap"][sel].max(),
        "pv_back_ml": res["pv_back_ml"][sel].mean(),
    }
    out.update(rhythm_metrics(res["log"], t0, t1))
    return out


# 7c. Pressure changes in the 10 s before and after each mode switch
def switch_transients(res: dict, window=10.0) -> list:
    """MAP and LAP change around each mode switch, per direction."""
    out = []
    t, m, l = res["t"], res["map"], res["mlap"]
    for ts, mode in res["log"].switches:
        pre = (t >= ts - window) & (t < ts)
        post = (t >= ts) & (t < ts + window)
        if pre.any() and post.any():
            out.append({"t": ts, "to": mode, "map_pre": m[pre].mean(), "map_nadir": m[post].min(),
                        "d_map": m[post].min() - m[pre].mean(),
                        "d_mlap_peak": l[post].max() - l[pre].mean()})
    return out


# ===========================================================================
# STEP 8: EXPERIMENT
# Runs STEPS 2 to 7 for the noise and control arms, once per seed.
# ===========================================================================
def run_experiment(dev: DeviceConfig, scn: ScenarioConfig, seeds=3, plot_path=None):
    phases = {"before": (10.0, scn.noise_on), "noise": (scn.noise_on, scn.noise_off),
              "after": (scn.noise_off, scn.duration)}
    rows, examples = [], None
    # Repeat the whole comparison for each seed.
    for seed in range(seeds):
        pair = {}
        for arm, level in (("noise", scn.noise_rms), ("control", 0.0)):
            rng = np.random.default_rng(seed)  # same seed gives a matched sinus sequence
            # STEPS 2-3: generate the rhythm.
            noise = NoiseProfile(scn.noise_on, scn.noise_off, level)
            log = simulate_device(dev, scn, noise, rng)
            # Plan the CircAdapt plot snapshots (first seed, noise arm only).
            shots = None
            if plot_path and seed == 0 and arm == "noise":
                shots = [(scn.noise_on / 2, "Before noise", "model_plot_before.png"),
                         ((scn.noise_on + scn.noise_off) / 2, "During noise", "model_plot_during.png"),
                         (scn.duration - 10.0, "After noise", "model_plot_after.png")]
            # STEPS 4-6: run the rhythm through a freshly stabilized heart.
            res = run_scenario(CircAdaptBackend(scn.sinus_rate, scn.dyssynchrony), log, shots)
            # STEP 7: summarize each phase.
            pair[arm] = res
            for name, (t0, t1) in phases.items():
                rows.append(dict(seed=seed, arm=arm, phase=name, **phase_metrics(res, t0, t1)))
            rows[-1]["crashed_at"] = res["crashed_at"]
        if examples is None:
            examples = pair
    # STEP 9: report the results.
    print_summary(rows, examples)
    if plot_path:
        plot_example(examples, scn, plot_path)
    return rows, examples


# ===========================================================================
# STEP 9: OUTPUT
# Prints the results table and saves the main plot.
# ===========================================================================
# 9a. Results table and mode switch summary
def print_summary(rows, examples):
    keys = ["v_rate_bpm", "av_sync_pct", "vvir_pct", "false_am_pct", "escape_pct",
            "atria_in_systole_pct", "co_L_min", "map", "map_min", "mlap",
            "peak_lap_max", "peak_rap_max", "pv_back_ml"]
    head = "".join(f"{a[:4]}/{p[:3]:<7}" for a in ("control", "noise")
                   for p in ("before", "noise", "after"))
    print(f"{'mean across seeds':<22}{head}")
    for k in keys:
        line = f"{k:<22}"
        for arm in ("control", "noise"):
            for phase in ("before", "noise", "after"):
                vals = [r[k] for r in rows if r["arm"] == arm and r["phase"] == phase and k in r]
                line += f"{np.mean(vals):<12.1f}" if vals else f"{'-':<12}"
        print(line)
    res = examples["noise"]
    print("\nCycle mapping (first seed, noise arm):", res["fidelity"])
    if res["crashed_at"]:
        print(f"MODEL CRASHED at t = {res['crashed_at']:.1f} s")
    for s in switch_transients(res):
        print(f"  t={s['t']:6.1f}s -> {s['to']:<4}  MAP {s['map_pre']:.1f} -> nadir "
              f"{s['map_nadir']:.1f} ({s['d_map']:+.1f}); peak mLAP rise {s['d_mlap_peak']:+.1f} mmHg")


# 9b. Four-panel plot of the first seed: noise vs. control
def plot_example(examples, scn, path):
    n, c = examples["noise"], examples["control"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    vv = lambda r: 60.0 / np.diff(r["t"], prepend=r["t"][0] - 0.8)
    panels = [(vv, "Ventricular rate (bpm)"), (lambda r: r["map"], "Mean arterial (mmHg)"),
              (lambda r: r["mlap"], "Mean LA (mmHg)"), (lambda r: r["stroke_ml"], "Stroke volume (mL)")]
    for ax, (fn, label) in zip(axes, panels):
        ax.plot(c["t"], fn(c), color="0.65", lw=1, label="No noise")
        ax.plot(n["t"], fn(n), color="#b5412b", lw=1, label="Noise")
        ax.axvspan(scn.noise_on, scn.noise_off, color="0.93", zorder=0)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        for ts, _ in n["log"].switches:
            ax.axvline(ts, color="k", lw=0.6, ls=":")
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Time (s); shaded = noise, dotted = mode switch")
    fig.suptitle(f"{scn.noise_rms} m/s² RMS noise")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"Saved {path}")


# ===========================================================================
# CHECK: PREFLIGHT (only runs with --preflight)
# Tests the assumptions the beat-by-beat approach makes about CircAdapt.
# ===========================================================================
def preflight() -> bool:
    """Tests what the ventricle-anchored approach relies on. Trust results only if all pass."""
    results = []

    def check(name, ok, detail):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def frame(t_cycle=0.8, dt=0.0, atrial=True, tau=TAU_BASE):
        return Frame(0.0, t_cycle, tau, dt, atrial, 0.0)

    def steady(n=5, **kw):
        b = CircAdaptBackend()
        for _ in range(n):
            r = b.run_frame(frame(**kw))
        return r

    # 1. run(1) stores exactly one cycle, of the new length.
    b = CircAdaptBackend()
    m = b.m
    for T in (0.8, 0.6, 1.1):
        b.run_frame(frame(T))
        t = np.asarray(m["Solver"]["t"], float)
        span = t[-1] - t[0]
        check(f"run(1) spans one cycle at t_cycle={T}", abs(span - T) < 0.02 * T, f"span={span:.3f}s")

    # 2. The default law shortens ventricular contraction at short cycles.
    durations = []
    for T in (0.6, 1.0):
        b.run_frame(frame(T))
        durations.append(float(np.ravel(m["Patch"]["time_act"][:])[2]))
    check("ventricular duration follows t_cycle", durations[0] < durations[1],
          f"{durations[0]:.3f} s at 0.6 s, {durations[1]:.3f} s at 1.0 s")

    # 3. Same A-to-V interval by two routes gives the same heart: T + tau_base - dt = tau.
    for dt, tau in ((0.60, 0.32), (0.68, 0.24)):
        via_dt, via_tau = steady(dt=dt), steady(tau=tau)
        check(f"Patch dt={dt} matches tau_av={tau}", abs(via_dt["stroke_ml"] - via_tau["stroke_ml"]) < 2.0,
              f"SV {via_dt['stroke_ml']:.1f} vs {via_tau['stroke_ml']:.1f} mL")

    # 3b. Well-timed atrial beats must add filling at every A-to-V interval, including those that
    # would otherwise fall in the late zone of a cycle.
    for T in (0.8, 1.2):
        v = np.arange(1.0, 1.0 + 12 * T, T)
        none, _ = build_frames(np.array([]), v)
        b = CircAdaptBackend(60.0 / T)
        for f in none:
            silent = b.run_frame(f)
        for interval in (0.13, 0.18, 0.24, 0.30):
            frames, info = build_frames(v - interval, v)
            b = CircAdaptBackend(60.0 / T)
            for f in frames:
                r = b.run_frame(f)
            check(f"atrial beat counts at A-V={interval}, t_cycle={T}",
                  r["stroke_ml"] > silent["stroke_ml"] + 1.0,
                  f"SV {r['stroke_ml']:.1f} vs {silent['stroke_ml']:.1f} mL with no atrial beat "
                  f"(tau_av {frames[-1].tau_av:.2f}, dt {frames[-1].atrial_dt:.2f})")

    # 4. An atrial contraction during systole gives an LA pressure rise and the heart still ejects.
    base = steady()
    cannon = steady(dt=0.40)
    check("atrial contraction in systole raises peak LA pressure",
          cannon["peak_lap"] > base["peak_lap"] + 3,
          f"{base['peak_lap']:.1f} -> {cannon['peak_lap']:.1f} mmHg")
    early = steady(dt=0.20)
    check("heart still ejects with atrial activation just after V",
          early["stroke_ml"] > 40 and early["lv_peak"] > 80,
          f"SV={early['stroke_ml']:.1f} mL, LV peak={early['lv_peak']:.0f} mmHg")

    # 5. Atria can be silenced and restored.
    b = CircAdaptBackend()
    for _ in range(5):
        on = b.run_frame(frame())
    for _ in range(5):
        off = b.run_frame(frame(atrial=False))
    for _ in range(10):
        back = b.run_frame(frame())
    check("atrial silencing lowers stroke volume", off["stroke_ml"] < on["stroke_ml"],
          f"{on['stroke_ml']:.1f} -> {off['stroke_ml']:.1f} mL")
    check("atrial contraction restored", abs(back["stroke_ml"] - on["stroke_ml"]) < 0.05 * on["stroke_ml"],
          f"{back['stroke_ml']:.1f} mL")

    # 6. Moving dt abruptly does not break pressure continuity, even with a contraction in progress.
    b = CircAdaptBackend()
    b.run_frame(frame(dt=0.70))
    end = float(np.asarray(b.m["Cavity"]["p"][:, "La"], float)[-1])
    b.run_frame(frame(atrial=False))
    start = float(np.asarray(b.m["Cavity"]["p"][:, "La"], float)[0])
    jump = abs(end - start) / MMHG
    check("LA pressure continuous when atria change between beats", jump < 2.0, f"jump={jump:.2f} mmHg")

    # 7. A real dissociated rhythm: V at 100 bpm, sinus at 75 bpm.
    frames, _ = build_frames(np.arange(0.5, 40.0, 0.8), np.arange(1.0, 40.0, 0.6))
    b = CircAdaptBackend()
    rows = [b.run_frame(f) for f in frames]
    sv = np.array([r["stroke_ml"] for r in rows[10:]])
    lv = np.array([r["lv_peak"] for r in rows[10:]])
    check("every beat ejects during V-A dissociation", sv.min() > 20 and lv.min() > 60,
          f"SV {sv.min():.1f}-{sv.max():.1f} mL, LV peak {lv.min():.0f}-{lv.max():.0f} mmHg")

    ok = all(results)
    print("\nAll checks passed." if ok else "\nDo NOT trust results yet: see the failures above.")
    return ok


# ===========================================================================
# START: PROGRAM ENTRY POINT
# Reads the command line options, then runs either the preflight or the experiment.
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--noise-rms", type=float, default=1.5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dyssynchrony-ms", type=float, default=0.0)
    ap.add_argument("--no-activity-switch", action="store_true")
    ap.add_argument("--plot", default="micra_noise_result.png")
    ap.add_argument("--no-show", action="store_true", help="save figures without opening them")
    a = ap.parse_args(argv)
    if a.preflight:
        return 0 if preflight() else 1
    dev = DeviceConfig(activity_mode_switch=not a.no_activity_switch)
    scn = ScenarioConfig(noise_rms=a.noise_rms, dyssynchrony=a.dyssynchrony_ms / 1000)
    run_experiment(dev, scn, a.seeds, a.plot)
    if not a.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
