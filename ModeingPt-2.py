"""Tests for device logic, cycle mapping and the CircAdapt adapter (using a fake model)."""
import sys
import types

import numpy as np
import pytest

# Lets the tests run on a machine without CircAdapt; the real package is used if present.
try:
    import circadapt  # noqa: F401
except ImportError:
    _stub = types.ModuleType("circadapt")
    _stub.VanOsta2024 = None
    _stub.error = types.SimpleNamespace(ModelCrashed=type("ModelCrashed", (Exception,), {}))
    sys.modules["circadapt"] = _stub

import matplotlib
matplotlib.use("Agg")

import micra_noise_sim as s


def run_dev(level=0.0, dev=None, scn=None, seed=0):
    scn = scn or s.ScenarioConfig(noise_rms=level)
    dev = dev or s.DeviceConfig()
    noise = s.NoiseProfile(scn.noise_on, scn.noise_off, level)
    return s.simulate_device(dev, scn, noise, np.random.default_rng(seed)), scn


def test_quiet_tracks_sinus_with_expected_pv():
    log, _ = run_dev(0.0)
    v = log.v_times[5:]
    assert not log.switches
    assert set(log.v_modes) == {"VDD"}
    assert np.mean(np.array(log.v_causes)[5:] == "am_true") == 1.0
    pv = [x - log.a_times[log.a_times <= x][-1] for x in v]
    assert np.allclose(pv, 0.120, atol=1e-6)   # EM delay + AM-VP
    assert abs(60 / np.mean(np.diff(v)) - 75) < 1


def test_high_noise_switches_then_returns():
    log, scn = run_dev(1.5)
    assert [m for _, m in log.switches][:2] == ["VVIR", "VDD"]
    t_on, t_off = log.switches[0][0], log.switches[1][0]
    assert scn.noise_on < t_on < scn.noise_off
    assert t_off > scn.noise_off


def test_switch_can_be_disabled():
    log, _ = run_dev(1.5, dev=s.DeviceConfig(activity_mode_switch=False))
    assert not log.switches


def test_noise_degrades_sensing_without_switch():
    dev = s.DeviceConfig(activity_mode_switch=False)
    log, scn = run_dev(1.5, dev=dev)
    m = s.rhythm_metrics(log, scn.noise_on, scn.noise_off)
    assert m["false_am_pct"] > 5
    assert m["av_sync_pct"] < 90


def test_ventricular_rate_limits_respected():
    dev = s.DeviceConfig(activity_mode_switch=False)
    log, _ = run_dev(3.0, dev=dev)
    vv = np.diff(log.v_times)
    assert vv.min() >= 60 / dev.upper_tracking_rate - 1e-9
    assert vv.max() <= 60 / dev.lower_rate + 1e-9


@pytest.mark.parametrize("level", [0.0, 0.5, 1.5, 3.0])
def test_frames_reconstruct_events(level):
    log, _ = run_dev(level)
    frames, fid = s.build_frames(log.a_times, log.v_times)
    for j, f in enumerate(frames):
        assert f.t_cycle > 0.2
        assert 0 < f.tau_av < f.t_cycle
        assert f.start + f.tau_av == pytest.approx(log.v_times[j], abs=0.0011)  # tau_av floor is 1 ms
    for a, b in zip(frames, frames[1:]):
        assert a.start + a.t_cycle == pytest.approx(b.start)
    assert fid["n_frames"] == len(log.v_times) - 1


def test_frames_handle_dropped_and_silent_atria():
    a = np.array([0.1, 0.5, 2.9])
    v = np.array([0.7, 1.5, 2.3, 3.0])
    frames, fid = s.build_frames(a, v)
    assert fid["dropped_atrial"] == 1 and fid["silenced_atrial"] == 2
    assert frames[0].atrial_on and not frames[1].atrial_on


# --- fake CircAdapt exposing the access patterns shown in the tutorials ------------
class Sig:
    def __init__(self, arr, names):
        self.arr, self.names = np.asarray(arr, float), names

    def __getitem__(self, key):
        r, c = key
        return self.arr[r, self.names.index(c) if isinstance(c, str) else c]


class Comp(dict):
    pass


class FakeModel:
    def __init__(self):
        cav = ["SyArt", "SyVen", "PuArt", "PuVen", "La", "Ra", "cLv", "cRv"]
        val = ["SyVenRa", "RaRv", "RvPuArt", "PuVenLa", "LaLv", "LvSyArt"]
        self.cav, self.val = cav, val
        self.c = {
            "General": {"t_cycle": 0.8},
            "PFC": {"is_active": True, "q0": 0},
            "Timings": {"law_tau_av": 1, "tau_av": 0.1, "law_ta": 1, "law_tv": 1,
                        "c_ta_rest": 0.15, "c_ta_tcycle": 0.1, "c_tv_rest": 0.1,
                        "c_tv_tcycle": 0.3, "t_cycle_rest": 0.85, "time_fac": 1.0},
            "Patch": {"Sf_act": np.full(5, 120e3), "time_act": np.full(5, 0.3)},
            "Wall": {},
            "Solver": {}, "Cavity": {}, "Valve": {},
        }
        self.calls = []

    def __getitem__(self, k):
        return self.c[k]

    def run(self, n=1, stable=False):
        T = self.c["General"]["t_cycle"]
        self.calls.append((T, self.c["Timings"]["tau_av"],
                           self.c["Patch"]["Sf_act"][:2].copy(),
                           self.c["Patch"]["time_act"].copy()))
        t = np.linspace(0, T, 200)
        self.c["Solver"]["t"] = t
        self.c["Cavity"]["p"] = Sig(np.full((200, 8), 90 * s.MMHG), self.cav)
        q = np.full((200, 6), 70e-6 / T)
        self.c["Valve"]["q"] = Sig(q, self.val)
        act = np.zeros((200, 5))
        act[:, 2:] = self.c["Timings"]["tau_av"]
        self.c["Patch"]["moment_of_activation"] = act


def test_adapter_against_fake(monkeypatch):
    monkeypatch.setattr(s, "VanOsta2024", FakeModel)
    b = s.CircAdaptBackend()
    assert b.m["PFC"]["is_active"] is False
    assert b.m["Timings"]["law_tau_av"] == 0 and b.m["Timings"]["law_ta"] == 0

    r = b.run_frame(s.Frame(0, 0.7, 0.2, False, 1.0, 0.2))
    T, tau, sf, ta = b.m.calls[-1]
    assert T == 0.7 and tau == 0.2 and np.all(sf == 0)
    assert ta[2] == pytest.approx(0.1 * 0.85 + 0.3 * 1.0)   # scaled by VV, not t_cycle
    assert r["map"] == pytest.approx(90) and r["stroke_ml"] == pytest.approx(70, rel=0.01)

    b.run_frame(s.Frame(0, 0.7, 0.2, True, 1.0, 0.2))
    assert np.all(b.m.calls[-1][2] == 120e3)                # atrial force restored


def test_preflight_runs_against_fake(monkeypatch, capsys):
    monkeypatch.setattr(s, "VanOsta2024", FakeModel)
    s.preflight()
    out = capsys.readouterr().out
    assert out.count("[PASS]") + out.count("[FAIL]") == 10
    assert "run(1) spans one cycle at t_cycle=1.1" in out


def test_full_pipeline_against_fake(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "VanOsta2024", FakeModel)
    png = tmp_path / "x.png"
    rows, ex = s.run_experiment(s.DeviceConfig(), s.ScenarioConfig(), 2, png)
    assert png.exists() and rows
    assert ex["noise"]["log"].switches and not ex["control"]["log"].switches