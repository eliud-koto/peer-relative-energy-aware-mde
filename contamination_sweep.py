"""Contamination robustness sweep for MDE-based energy inefficiency detection.

Evaluates anomaly detection performance (ROC-AUC) across contamination rates
from 1% to 99% over 10 random seeds. Compares the proposed MDE embedding
displacement score against residual baselines (physics, RF, LR, Huber) and
unsupervised feature baselines (IsoForest, LOF, GMM, Autoencoder).

QUICK START (no private data needed)
-------------------------------------
A pre-generated synthetic base dataset is included in this repository:

    python contamination_sweep.py

DATA REQUIREMENT (to regenerate synthetic data from scratch)
-------------------------------------------------------------
The synthetic base dataset was derived from a real-world telecom operator
dataset ('data_extended.pkl') that is not publicly available due to data-
sharing restrictions. To regenerate it from your own data, supply the path:

    python contamination_sweep.py --data path/to/data_extended.pkl

Your dataset must contain: meter_kwh, num_total_cells, total_non_ran_equipment,
has_shared_ran, ran_vendor_type, mast_type, ps_traffic_mb.

    python contamination_sweep.py --data path/to/data.pkl --generate-only

Use --generate-only to write data/synthetic_base.csv without running the sweep.

Outputs (written to ./data/):
    contamination_sweep_results.pkl
    contamination_sweep_residual.{png,pdf}
    contamination_sweep_unsupervised.{png,pdf}
    contamination_sweep_feat.{png,pdf}

Runtime: approximately 60-90 minutes on CPU (10 seeds x 14 contamination rates).
Requirements: Python 3.12, pymde 0.3.0, torch 2.10.0, scikit-learn 1.8.0
"""
import argparse
import os
import pickle
from warnings import filterwarnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymde
import torch
import torch.nn as _nn
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

filterwarnings("ignore", category=UserWarning, module="sklearn")

np.random.seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_SAMPLES          = 5000
N_SEEDS            = 10
K_STRUCT_NEIGHBORS = 50    # structural peer set for displacement scoring
K_BASELINE         = 10    # tight local peer set for delta_node energy comparison
K_GRAPH            = 300   # kNN graph size for MDE
BETA               = 35    # repulsion strength; pushes high-delta_node nodes outward
EMBEDDING_DIM      = 4
N_DISSIMILAR_MULT  = 4     # dissimilar edges = N_DISSIMILAR_MULT x similar edges
NEG_WEIGHT         = -2.0  # weight on dissimilar edges in PushAndPull
TRAFFIC_WEIGHT     = 0.05  # down-weight traffic column in graph construction
DELTA_PERCENTILE   = 35    # percentile of neighbour energies used as local baseline

# Lognormal noise scale per mast type (sigma_log = std of log-energy residual)
SIGMA_LOG_MAP = {
    'Tower': 0.04, 'Disguised': 0.035,
    'Rooftop': 0.03, 'Pole': 0.02, 'Other': 0.0325,
}

CONTAMINATION_RATES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25,
                       0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99]

_MAST_BUCKET_MAP = {
    'Lattice': 'Tower',          'Monopole': 'Tower',        'Concrete Tower': 'Tower',
    'Spine': 'Tower',            'Mono / Lattice': 'Tower',  'Mono Lattice': 'Tower',
    'Lattice on Roof': 'Tower',  'Temp_Lattice': 'Tower',    'Temp_Spine': 'Tower',
    'Tree': 'Disguised',         'Palm Tree': 'Disguised',   'Pine Tree': 'Disguised',
    'Camouflage': 'Disguised',   'Anna Tree': 'Disguised',   'Cypress Tree': 'Disguised',
    'Palm / Cocus': 'Disguised', 'Yellow Wood': 'Disguised',
    'FlagPole': 'Disguised',     'Signage tower': 'Disguised',
    'Building': 'Rooftop',       'Rooftop': 'Rooftop',       'Indoor': 'Rooftop',
    'DAS': 'Rooftop',            'ULCS': 'Rooftop',
    'Pole': 'Pole',              'LampPost': 'Pole',         'Billboard': 'Pole',
    'Street Light Pole': 'Pole', 'CameraPole': 'Pole',
}

_COOLING_RANGE = {
    'Tower': (200, 400), 'Disguised': (150, 350),
    'Rooftop': (80, 200), 'Pole': (100, 250), 'Other': (100, 200),
}

_VENDOR_DISPLAY = {'HUA': 'Vendor B', 'NOK': 'Vendor A'}

_CONFIG_COLS = ['num_total_cells', 'total_non_ran_equipment',
                'has_shared_ran', 'ran_vendor_type', 'mast_group']


# ── Physics model ──────────────────────────────────────────────────────────────

def _fit_physics_model(df):
    """Fit per-group positive linear regression: kWh ~ num_cells + non_ran."""
    models = []
    for (_shared, _vendor, _mast), _g in df.groupby(
            ['has_shared_ran', 'ran_vendor_type', 'mast_group']):
        if len(_g) < 3:
            continue
        _m = LinearRegression(positive=True, fit_intercept=True)
        _m.fit(_g[['num_total_cells', 'total_non_ran_equipment']].values,
               _g['meter_kwh'].values)
        models.append({
            'has_shared_ran': _shared, 'ran_vendor_type': _vendor, 'mast_group': _mast,
            'base_load': _m.intercept_, 'alpha_cells': _m.coef_[0],
            'beta_non_ran': _m.coef_[1],
        })
    physics_df = pd.DataFrame(models).round(2)
    # HUA-shared group has too few samples for reliable regression;
    # coefficients are overridden from domain knowledge.
    _mask = (physics_df['has_shared_ran'] == 1) & (physics_df['ran_vendor_type'] == 'HUA')
    physics_df.loc[_mask, 'base_load']    = 480.0
    physics_df.loc[_mask, 'beta_non_ran'] = 220.0
    return physics_df


def _predict_kwh(num_cells, num_non_ran, shared, vendor, mast, physics_df):
    row = physics_df[(physics_df['has_shared_ran'] == shared) &
                     (physics_df['ran_vendor_type'] == vendor) &
                     (physics_df['mast_group'] == mast)]
    if len(row) == 0:
        row = physics_df[(physics_df['has_shared_ran'] == shared) &
                         (physics_df['ran_vendor_type'] == vendor)]
    r = row.iloc[0]
    return r['base_load'] + r['alpha_cells'] * num_cells + r['beta_non_ran'] * num_non_ran


# ── Synthetic data generation ──────────────────────────────────────────────────

def generate_synthetic_base(df_real):
    """Generate the synthetic base dataset from real operator data.

    Samples N_SAMPLES configurations with probability proportional to empirical
    frequency (probability-weighted), assigns traffic from per-configuration
    pools, computes physics-model expected energy, and applies lognormal
    multiplicative noise. Returns a DataFrame with no anomaly labels.

    Vendor names are anonymised: HUA -> Vendor B, NOK -> Vendor A.
    """
    physics_df = _fit_physics_model(df_real)

    _joint = (df_real[_CONFIG_COLS]
              .value_counts(normalize=True)
              .reset_index(name='probability'))
    df_base = (_joint
               .sample(n=N_SAMPLES, replace=True, weights='probability', random_state=42)
               [_CONFIG_COLS]
               .reset_index(drop=True))

    # Sample traffic from per-configuration empirical pools with small Gaussian jitter.
    # A fixed secondary seed (99) separates traffic sampling from the injection RNG.
    _traffic_lookup     = df_real.groupby(_CONFIG_COLS)['ps_traffic_mb'].apply(list).to_dict()
    _traffic_global_med = df_real['ps_traffic_mb'].median()
    _traffic_global_std = df_real['ps_traffic_mb'].std()
    _rng_t = np.random.default_rng(99)
    _ps_vals = []
    for _, _row in df_base.iterrows():
        _key  = tuple(_row[c] for c in _CONFIG_COLS)
        _pool = _traffic_lookup.get(_key)
        if _pool:
            _val = float(_rng_t.choice(_pool))
            _val = max(_val + _rng_t.normal(0, _traffic_global_std * 0.05), 0.0)
        else:
            _val = _traffic_global_med
        _ps_vals.append(_val)
    df_base['ps_traffic_mb'] = np.array(_ps_vals)

    df_base['kwh_expected'] = df_base.apply(
        lambda r: _predict_kwh(r['num_total_cells'], r['total_non_ran_equipment'],
                               r['has_shared_ran'], r['ran_vendor_type'], r['mast_group'],
                               physics_df), axis=1)

    np.random.seed(42)
    _sigma_log = df_base['mast_group'].map(SIGMA_LOG_MAP).values
    _log_eps   = np.random.normal(0, _sigma_log)
    df_base['noise_std']          = _sigma_log * df_base['kwh_expected'].values
    df_base['kwh_expected_noise'] = df_base['kwh_expected'].values * np.exp(_log_eps)

    df_base['ran_vendor_type'] = df_base['ran_vendor_type'].map(
        lambda v: _VENDOR_DISPLAY.get(v, v))

    return df_base


# ── Anomaly injection ──────────────────────────────────────────────────────────

def inject_anomalies(df_base, contamination_rate, rng_seed):
    """Inject synthetic inefficiency anomalies into the base dataset.

    Four anomaly types:
        Type 1 - Multiplicative overload : meter *= U(1.2, 1.8)
        Type 2 - Cooling failure         : meter += mast-specific U(lo, hi)
        Type 3 - Idle RF overhead        : meter += max(cells^2 * idle * U(0.5,1.5), 2*sigma)
        Type 4 - Auxiliary parasitic     : meter += max((non_ran+1)^2 * U(20,50), 2*sigma)

    Returns:
        energy (np.ndarray): simulated meter readings, shape (N_SAMPLES,)
        y_true (np.ndarray): binary anomaly labels,   shape (N_SAMPLES,)
    """
    n_bad      = int(contamination_rate * N_SAMPLES)
    n_per_type = n_bad // 4
    rng        = np.random.default_rng(rng_seed)

    _traf_med = df_base['ps_traffic_mb'].median()
    _traf_med = _traf_med if _traf_med > 0 else df_base['ps_traffic_mb'].mean()

    counts = [0, 0, 0, 0]
    itype  = np.zeros(N_SAMPLES, dtype=int)
    for _i in rng.permutation(N_SAMPLES).tolist():
        if sum(counts) >= n_bad:
            break
        opts = [t for t in range(4) if counts[t] < n_per_type]
        if not opts:
            break
        t = int(rng.choice(opts))
        itype[_i] = t + 1
        counts[t] += 1

    meter = df_base['kwh_expected_noise'].values.copy()
    for i in np.where(itype == 1)[0]:
        meter[i] *= rng.uniform(1.2, 1.8)
    for i in np.where(itype == 2)[0]:
        lo, hi = _COOLING_RANGE.get(df_base.iloc[i]['mast_group'], (100, 200))
        meter[i] += rng.uniform(lo, hi)
    for i in np.where(itype == 3)[0]:
        cells      = df_base.iloc[i]['num_total_cells']
        traffic    = df_base.iloc[i]['ps_traffic_mb']
        idle       = max(1.0 - traffic / _traf_med, 0.1)
        site_noise = df_base.iloc[i]['noise_std']
        overhead   = max(cells, 5) ** 2 * idle * rng.uniform(0.5, 1.5)
        meter[i]  += max(overhead, 2.0 * site_noise)
    for i in np.where(itype == 4)[0]:
        nran       = df_base.iloc[i]['total_non_ran_equipment']
        site_noise = df_base.iloc[i]['noise_std']
        overhead   = (nran + 1) ** 2 * rng.uniform(20, 50)
        meter[i]  += max(overhead, 2.0 * site_noise)

    return np.round(meter, 2), (itype > 0).astype(int)


# ── Autoencoder ────────────────────────────────────────────────────────────────

class _SimpleAE(_nn.Module):
    """Lightweight symmetric autoencoder for unsupervised anomaly scoring."""

    def __init__(self, in_dim, hidden, bottleneck):
        super().__init__()
        enc, dec = [], []
        prev = in_dim
        for h in hidden:
            enc += [_nn.Linear(prev, h), _nn.ReLU()]; prev = h
        enc.append(_nn.Linear(prev, bottleneck))
        prev = bottleneck
        for h in reversed(hidden):
            dec += [_nn.Linear(prev, h), _nn.ReLU()]; prev = h
        dec.append(_nn.Linear(prev, in_dim))
        self.enc = _nn.Sequential(*enc)
        self.dec = _nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(self.enc(x))


def _train_ae(X_np, hidden, bottleneck, epochs, seed):
    """Train autoencoder; return per-sample mean squared reconstruction error.

    Args:
        X_np:       scaled input matrix, shape (n, d)
        hidden:     encoder hidden layer sizes, e.g. [32, 16]
        bottleneck: bottleneck dimension
        epochs:     training epochs
        seed:       torch seed for reproducibility
    """
    torch.manual_seed(seed)
    X_t = torch.tensor(X_np, dtype=torch.float32)
    ae  = _SimpleAE(X_np.shape[1], hidden, bottleneck)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    dl  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t), batch_size=256, shuffle=True)
    ae.train()
    for _ in range(epochs):
        for (xb,) in dl:
            loss = _nn.functional.mse_loss(ae(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    with torch.no_grad():
        recon = ae(X_t)
    return ((X_t - recon) ** 2).mean(dim=1).numpy()


# ── MDE pipeline helpers ───────────────────────────────────────────────────────

def _build_structural_graph(df_base):
    """Build kNN graph and structural neighbour index from df_base.

    Both are energy-independent and can be reused across all seeds.

    Returns:
        graph:           pymde graph (edges + weights)
        nn_idx:          global neighbour indices, shape (N_SAMPLES, K_FIT), -1 for padding
        X_struct:        raw structural feature matrix, shape (N_SAMPLES, n_feat)
        X_struct_scaled: scaled version (traffic down-weighted) used for graph
    """
    _vendor_bin  = (df_base['ran_vendor_type'] == 'Vendor A').astype(float).values
    _mast_dums   = pd.get_dummies(df_base['mast_group'], prefix='mast').astype(float)
    _log_traffic = np.log1p(df_base['ps_traffic_mb'].values)

    X_struct = np.column_stack([
        df_base['num_total_cells'].values,
        df_base['total_non_ran_equipment'].values,
        df_base['has_shared_ran'].values,
        _vendor_bin,
        _log_traffic,       # column index 4 — down-weighted below
        _mast_dums.values,
    ])

    X_struct_scaled = StandardScaler().fit_transform(X_struct)
    X_struct_scaled[:, 4] *= TRAFFIC_WEIGHT

    graph = pymde.preprocess.k_nearest_neighbors(
        csr_matrix(X_struct_scaled), k=K_GRAPH, verbose=False)

    # Structural neighbour index, stratified by (vendor, sharing, mast)
    _K_fit = max(K_BASELINE, K_STRUCT_NEIGHBORS)
    nn_idx = np.full((N_SAMPLES, _K_fit), -1, dtype=np.int64)
    for _, _grp_idx in df_base.groupby(
            ['ran_vendor_type', 'has_shared_ran', 'mast_group']).groups.items():
        _grp_idx = _grp_idx.to_numpy()
        _k       = min(_K_fit + 1, len(_grp_idx))
        _nbrs    = NearestNeighbors(n_neighbors=_k).fit(X_struct_scaled[_grp_idx])
        _, _nn   = _nbrs.kneighbors(X_struct_scaled[_grp_idx])
        _nn      = _nn[:, 1:]  # drop self
        nn_idx[_grp_idx, :_nn.shape[1]] = _grp_idx[_nn]

    return graph, nn_idx, X_struct, X_struct_scaled


def _compute_delta_node(energy, df_base, nn_idx):
    """Compute delta_node = log(energy / P35 of K_BASELINE structural neighbours)."""
    delta = np.zeros(N_SAMPLES)
    for _, _grp_idx in df_base.groupby(
            ['ran_vendor_type', 'has_shared_ran', 'mast_group']).groups.items():
        _grp_idx = _grp_idx.to_numpy()
        _nn_base = nn_idx[_grp_idx, :K_BASELINE]
        _valid   = _nn_base >= 0
        _nn_safe = np.where(_valid, _nn_base, 0)
        _e_nn    = energy[_nn_safe].astype(float)
        _e_nn[~_valid] = np.nan
        _med     = np.nanpercentile(_e_nn, DELTA_PERCENTILE, axis=1)
        _med     = np.maximum(_med, 1.0)
        delta[_grp_idx] = np.log((energy[_grp_idx] + 1e-6) / (_med + 1e-6))
    return delta


def _run_mde_seed(energy, df_base, graph, nn_idx, seed):
    """Run MDE pipeline for one (energy, seed) pair.

    Args:
        energy:  simulated kWh values, shape (N_SAMPLES,)
        df_base: base synthetic dataset (structural features, no anomaly info)
        graph:   pre-built kNN graph on structural features
        nn_idx:  pre-built structural neighbour index, shape (N_SAMPLES, K_FIT)
        seed:    used for torch.manual_seed before edge sampling and embed

    Returns:
        X_emb:              embedding coordinates, shape (N_SAMPLES, EMBEDDING_DIM)
        emb_rel_dist_gated: energy-gated displacement score, shape (N_SAMPLES,)
    """
    delta_node = _compute_delta_node(energy, df_base, nn_idx)

    _edges2   = graph.edges.numpy()
    _weights2 = graph.weights.numpy()
    _rep      = np.maximum(
        np.maximum(delta_node[_edges2[:, 0]], 0),
        np.maximum(delta_node[_edges2[:, 1]], 0),
    )
    weights = _weights2 - BETA * _rep

    _similar   = graph.edges
    _n_similar = _similar.shape[0]
    torch.manual_seed(seed)
    _dissimilar = pymde.preprocess.dissimilar_edges(
        N_SAMPLES, num_edges=N_DISSIMILAR_MULT * _n_similar, similar_edges=_similar)
    _new_edges   = torch.cat([_similar, _dissimilar])
    _new_weights = torch.cat([
        torch.tensor(weights, dtype=torch.float32),
        NEG_WEIGHT * torch.ones(_dissimilar.shape[0]),
    ])

    _f = pymde.penalties.PushAndPull(
        weights=_new_weights,
        attractive_penalty=pymde.penalties.Log1p,
        repulsive_penalty=pymde.penalties.Log,
    )
    mde = pymde.MDE(
        n_items=N_SAMPLES, embedding_dim=EMBEDDING_DIM,
        edges=_new_edges, distortion_function=_f,
    )
    torch.manual_seed(seed)
    embedding = mde.embed(verbose=False)
    X_emb = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)

    _scores = np.zeros(N_SAMPLES)
    for _i in range(N_SAMPLES):
        _neigh = nn_idx[_i]
        _neigh = _neigh[_neigh >= 0]
        if len(_neigh) == 0:
            continue
        _neigh_pos     = X_emb[_neigh]
        _dist_to_neigh = np.linalg.norm(X_emb[_i] - _neigh_pos, axis=1).mean()
        _within_spread = pdist(_neigh_pos).mean() + 1e-8
        _scores[_i]    = _dist_to_neigh / _within_spread
    emb_rel_dist_gated = _scores * np.maximum(1 + delta_node, 1)

    return X_emb, emb_rel_dist_gated


# ── Plot helper ────────────────────────────────────────────────────────────────

def _draw_plot(ax, rates_pct, all_roc, methods_spec, legend_loc="upper right"):
    for label, key, color, ls, marker in methods_spec:
        means = all_roc[key].mean(axis=1)
        lo    = np.percentile(all_roc[key], 2.5,  axis=1)
        hi    = np.percentile(all_roc[key], 97.5, axis=1)
        ax.plot(rates_pct, means, color=color, ls=ls, marker=marker,
                ms=5, label=label, lw=1.8)
        ax.fill_between(rates_pct, lo, hi, alpha=0.15, color=color)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Random baseline")
    ax.set_xlabel("Contamination rate (%)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=8.5, loc=legend_loc, framealpha=0.85,
              borderpad=0.5, labelspacing=0.3, handlelength=1.8)
    ax.grid(alpha=0.3)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Contamination robustness sweep for MDE energy inefficiency detection.")
    parser.add_argument(
        "--data", default=None,
        help="Path to data_extended.pkl. If omitted, loads pre-generated "
             "data/synthetic_base.csv.")
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Write data/synthetic_base.csv from --data then exit.")
    args = parser.parse_args()

    _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(_data_dir, exist_ok=True)
    _csv_path = os.path.join(_data_dir, "synthetic_base.csv")

    if args.data:
        print(f"Loading real data from {args.data} ...")
        df_real = pd.read_pickle(args.data)
        df_real['mast_group'] = df_real['mast_type'].map(_MAST_BUCKET_MAP).fillna('Other')
        print("Generating synthetic base dataset (probability-weighted sampling) ...")
        df_base = generate_synthetic_base(df_real)
        df_base.to_csv(_csv_path, index=False)
        print(f"Saved {len(df_base)} rows -> {_csv_path}")
        if args.generate_only:
            return
    else:
        if not os.path.exists(_csv_path):
            raise FileNotFoundError(
                f"synthetic_base.csv not found at {_csv_path}.\n"
                "Run with --data path/to/data_extended.pkl to generate it.")
        print(f"Loading synthetic base dataset from {_csv_path} ...")
        df_base = pd.read_csv(_csv_path)

    print(f"Dataset: {len(df_base)} sites  "
          f"vendor={df_base['ran_vendor_type'].value_counts().to_dict()}\n")

    print("Building structural kNN graph (pre-computed once) ...")
    graph, nn_idx, X_struct, _ = _build_structural_graph(df_base)
    print(f"  {graph.edges.shape[0]} edges  neighbour index {nn_idx.shape}\n")

    METHODS = ['phys', 'mde_erd_gated', 'rf', 'lr', 'huber', 'raw_energy',
               'iso_feat', 'iso_mde', 'gmm_feat', 'gmm_mde',
               'lof_feat', 'lof_mde', 'ae_feat', 'ae_mde']
    all_roc = {m: np.full((len(CONTAMINATION_RATES), N_SEEDS), np.nan) for m in METHODS}

    print(f"Sweep: {N_SEEDS} seeds x {len(CONTAMINATION_RATES)} contamination rates\n")

    for ri, rate in enumerate(CONTAMINATION_RATES):
        print(f"Rate {rate:.0%}  ", end="", flush=True)
        for seed in range(N_SEEDS):
            energy, y_true = inject_anomalies(df_base, rate, rng_seed=seed)
            _two_classes   = len(np.unique(y_true)) > 1

            def _roc(s):
                return roc_auc_score(y_true, s) if _two_classes else float("nan")

            X_emb, mde_gated = _run_mde_seed(energy, df_base, graph, nn_idx, seed)
            all_roc['mde_erd_gated'][ri, seed] = _roc(mde_gated)

            X_feat_sc = StandardScaler().fit_transform(np.column_stack([X_struct, energy]))
            X_mde_sc  = StandardScaler().fit_transform(X_emb)

            all_roc['raw_energy'][ri, seed] = _roc(energy)
            all_roc['phys'][ri, seed]       = _roc(
                energy / (df_base['kwh_expected'].values + 1e-6))

            _lr_r = energy - LinearRegression().fit(X_struct, energy).predict(X_struct)
            all_roc['lr'][ri, seed] = _roc(np.maximum(_lr_r, 0))

            _X_hub = StandardScaler().fit_transform(X_struct)
            _hub_r = energy - HuberRegressor(epsilon=1.35, max_iter=300).fit(
                _X_hub, energy).predict(_X_hub)
            all_roc['huber'][ri, seed] = _roc(np.maximum(_hub_r, 0))

            _rf = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=-1)
            _rf.fit(X_struct, energy)
            all_roc['rf'][ri, seed] = _roc(np.maximum(energy - _rf.predict(X_struct), 0))

            for _key, _X in [('iso_feat', X_feat_sc), ('iso_mde', X_mde_sc)]:
                _iso = IsolationForest(n_estimators=100, contamination='auto',
                                       random_state=seed)
                all_roc[_key][ri, seed] = _roc(-_iso.fit(_X).score_samples(_X))

            for _key, _X in [('gmm_feat', X_feat_sc), ('gmm_mde', X_mde_sc)]:
                _gmm = GaussianMixture(n_components=3, covariance_type='full',
                                       random_state=seed)
                all_roc[_key][ri, seed] = _roc(-_gmm.fit(_X).score_samples(_X))

            for _key, _X in [('lof_feat', X_feat_sc), ('lof_mde', X_mde_sc)]:
                _lof = LocalOutlierFactor(
                    n_neighbors=K_STRUCT_NEIGHBORS, contamination='auto')
                _lof.fit(_X)
                all_roc[_key][ri, seed] = _roc(-_lof.negative_outlier_factor_)

            all_roc['ae_feat'][ri, seed] = _roc(
                _train_ae(X_feat_sc, hidden=[32, 16], bottleneck=3, epochs=100, seed=seed))
            all_roc['ae_mde'][ri, seed]  = _roc(
                _train_ae(X_mde_sc, hidden=[16], bottleneck=1, epochs=100, seed=seed))

            print(".", end="", flush=True)

        def _fmt(m):
            return f"{all_roc[m][ri].mean():.4f}+/-{all_roc[m][ri].std():.4f}"
        print(f"\n  mde={_fmt('mde_erd_gated')}  iso_feat={_fmt('iso_feat')}"
              f"  lof_feat={_fmt('lof_feat')}  ae_feat={_fmt('ae_feat')}\n",
              flush=True)

    print("Done.\n")

    _col = 12
    _hdr = (f"{'Rate':>6} | {'MDE(proposed)':>{_col}} | {'IsoFeat':>{_col}}"
            f" | {'GmmFeat':>{_col}} | {'LofFeat':>{_col}} | {'AEFeat':>{_col}}"
            f" | {'RF':>{_col}} | {'Physics':>{_col}}")
    print(f"Summary -- mean ROC-AUC  ({N_SEEDS} seeds)")
    print(_hdr)
    print("-" * len(_hdr))
    for ri, rate in enumerate(CONTAMINATION_RATES):
        def _f(m): return f"{all_roc[m][ri].mean():.4f}"
        print(f"{rate:>6.0%} | {_f('mde_erd_gated'):>{_col}} | {_f('iso_feat'):>{_col}}"
              f" | {_f('gmm_feat'):>{_col}} | {_f('lof_feat'):>{_col}}"
              f" | {_f('ae_feat'):>{_col}} | {_f('rf'):>{_col}} | {_f('phys'):>{_col}}")

    _pkl_path = os.path.join(_data_dir, "contamination_sweep_results.pkl")
    with open(_pkl_path, "wb") as _fh:
        pickle.dump({
            'all_roc': all_roc, 'rates': CONTAMINATION_RATES,
            'n_seeds': N_SEEDS, 'methods': METHODS,
            'params': {
                'BETA': BETA, 'K_GRAPH': K_GRAPH, 'K_BASELINE': K_BASELINE,
                'K_STRUCT_NEIGHBORS': K_STRUCT_NEIGHBORS, 'EMBEDDING_DIM': EMBEDDING_DIM,
                'N_DISSIMILAR_MULT': N_DISSIMILAR_MULT, 'NEG_WEIGHT': NEG_WEIGHT,
                'TRAFFIC_WEIGHT': TRAFFIC_WEIGHT, 'DELTA_PERCENTILE': DELTA_PERCENTILE,
                'sampling': 'probability_weighted', 'noise_model': 'lognormal_multiplicative',
                'injection_types': [1, 2, 3, 4],
            },
        }, _fh)
    print(f"\nResults saved -> {_pkl_path}")

    _rates_pct = [r * 100 for r in CONTAMINATION_RATES]
    _plot_specs = {
        'residual': (
            [("Physics residual (upper bound)",    "phys",          "#aec7e8", ":",  "x"),
             ("MDE rel. displacement (proposed)",  "mde_erd_gated", "#e67e22", "-",  "^"),
             ("Random Forest residual",            "rf",            "#2ca02c", "--", "s"),
             ("LR residual",                       "lr",            "#9467bd", "--", "P"),
             ("Huber residual",                    "huber",         "#e377c2", "--", "v")],
            "lower left"),
        'unsupervised': (
            [("MDE rel. displacement (proposed)",  "mde_erd_gated", "#e67e22", "-",  "^"),
             ("IsoForest (MDE emb)",               "iso_mde",       "#17becf", "--", "D"),
             ("GMM (MDE emb)",                     "gmm_mde",       "#d62728", "--", "s"),
             ("AE (MDE emb)",                      "ae_mde",        "#2ca02c", "--", "v")],
            "lower left"),
        'feat': (
            [("MDE rel. displacement (proposed)",  "mde_erd_gated", "#e67e22", "-",  "^"),
             ("IsoForest (features)",              "iso_feat",      "#17becf", "--", "D"),
             ("LOF (features)",                    "lof_feat",      "#8c564b", "--", "o"),
             ("GMM (features)",                    "gmm_feat",      "#d62728", "--", "s"),
             ("AE (features)",                     "ae_feat",       "#2ca02c", "--", "v")],
            "upper right"),
    }

    for _name, (_spec, _loc) in _plot_specs.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        _draw_plot(ax, _rates_pct, all_roc, _spec, _loc)
        fig.tight_layout()
        fig.savefig(os.path.join(_data_dir, f"contamination_sweep_{_name}.png"),
                    dpi=150, bbox_inches="tight")
        fig.savefig(os.path.join(_data_dir, f"contamination_sweep_{_name}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved -> {_data_dir}/contamination_sweep_{{residual,unsupervised,feat}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
