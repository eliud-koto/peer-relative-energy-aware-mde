"""Energy inefficiency detection in mobile networks using MDE.

Plain-Python equivalent of the marimo notebook pipeline. Runs end-to-end:
  physics model -> synthetic data -> MDE embedding -> scoring -> baseline comparison.

QUICK START (no private data needed)
-------------------------------------
    python energy_inefficiency_mde_pipeline_script.py

DATA REQUIREMENT (to regenerate synthetic data from scratch)
-------------------------------------------------------------
The synthetic base dataset was derived from a real-world telecom operator
dataset ('data_extended.pkl') that is not publicly available due to data-
sharing restrictions. To regenerate it from your own data:

    python energy_inefficiency_mde_pipeline_script.py --data path/to/data_extended.pkl

Your dataset must contain: meter_kwh, num_total_cells, total_non_ran_equipment,
has_shared_ran, ran_vendor_type, mast_type, ps_traffic_mb.

Optional flags:
    --eda       Also produce EDA diagnostic figures (S1-S4) in ./data/
    --out DIR   Directory for output figures (default: ./data/)

Requirements: Python 3.12, pymde 0.3.0, torch 2.10.0, scikit-learn 1.8.0
"""
import argparse
import os
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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Hyperparameters ────────────────────────────────────────────────────────────
K_GRAPH            = 300
K_BASELINE         = 10     # tight local peer set for delta_node energy comparison
K_STRUCT_NEIGHBORS = 50     # wider peer set for displacement scoring and LOF
TRAFFIC_WEIGHT     = 0.05   # down-weight traffic in graph construction
N_SAMPLES          = 5000
INEFF_PCT          = 0.10   # contamination rate for single-run evaluation
BETA               = 35     # repulsion strength
N_DISSIMILAR_MULT  = 4      # dissimilar edges = N_DISSIMILAR_MULT x similar edges
NEG_WEIGHT         = -2.0
EMBEDDING_DIM      = 4

SIGMA_LOG_MAP = {
    'Tower': 0.04, 'Disguised': 0.035,
    'Rooftop': 0.03, 'Pole': 0.02, 'Other': 0.0325,
}

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
_CONFIG_COLS    = ['num_total_cells', 'total_non_ran_equipment',
                   'has_shared_ran', 'ran_vendor_type', 'mast_group']


# ── 1. Physics model ───────────────────────────────────────────────────────────

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


def predict_kwh_physics(num_cells, num_non_ran, shared, vendor, mast, physics_df):
    """Return physics-model expected kWh for a single site configuration."""
    row = physics_df[(physics_df['has_shared_ran'] == shared) &
                     (physics_df['ran_vendor_type'] == vendor) &
                     (physics_df['mast_group'] == mast)]
    if len(row) == 0:
        row = physics_df[(physics_df['has_shared_ran'] == shared) &
                         (physics_df['ran_vendor_type'] == vendor)]
    r = row.iloc[0]
    return r['base_load'] + r['alpha_cells'] * num_cells + r['beta_non_ran'] * num_non_ran


# ── 2. Synthetic data generation ───────────────────────────────────────────────

def generate_synthetic_base(df_real):
    """Generate the synthetic base dataset from real operator data.

    Samples N_SAMPLES configurations with probability proportional to empirical
    frequency, assigns traffic from per-configuration pools, computes physics-
    model expected energy, and applies lognormal multiplicative noise.

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
        lambda r: predict_kwh_physics(r['num_total_cells'], r['total_non_ran_equipment'],
                                      r['has_shared_ran'], r['ran_vendor_type'],
                                      r['mast_group'], physics_df), axis=1)

    np.random.seed(42)
    _sigma_log = df_base['mast_group'].map(SIGMA_LOG_MAP).values
    _log_eps   = np.random.normal(0, _sigma_log)
    df_base['noise_std']          = _sigma_log * df_base['kwh_expected'].values
    df_base['kwh_expected_noise'] = df_base['kwh_expected'].values * np.exp(_log_eps)

    df_base['ran_vendor_type'] = df_base['ran_vendor_type'].map(
        lambda v: _VENDOR_DISPLAY.get(v, v))

    return df_base


def inject_anomalies(df_base, ineff_pct, rng_seed):
    """Inject anomalies and return full augmented dataframe.

    Four types:
        1 - Multiplicative overload : meter *= U(1.2, 1.8)
        2 - Cooling failure         : meter += mast-specific U(lo, hi)
        3 - Idle RF overhead        : meter += max(cells^2 * idle * U(0.5,1.5), 2*sigma)
        4 - Auxiliary parasitic     : meter += max((non_ran+1)^2 * U(20,50), 2*sigma)
    """
    n_bad      = int(ineff_pct * N_SAMPLES)
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

    df_mock = df_base.copy()
    df_mock['meter_kwh_sim']     = np.round(meter, 2)
    df_mock['inefficiency_type'] = itype
    df_mock['is_inefficient']    = (itype > 0).astype(int)
    return df_mock


# ── 3. EDA figures (optional) ──────────────────────────────────────────────────

def plot_eda(df_mock, out_dir):
    """Produce S1-S4 diagnostic figures and save to out_dir."""

    # S1 – Structural configuration characteristics
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    _vendor_display = {'HUA': 'Vendor B', 'NOK': 'Vendor A'}
    _vc = df_mock['ran_vendor_type'].value_counts()
    _vc.index = _vc.index.map(lambda v: _vendor_display.get(v, v))
    axes[0, 0].bar(_vc.index, _vc.values, color=['#1f77b4', '#ff7f0e'][:len(_vc)],
                   edgecolor='white')
    axes[0, 0].set_ylabel("Count")
    for p, v in zip(axes[0, 0].patches, _vc.values):
        axes[0, 0].text(p.get_x() + p.get_width() / 2, p.get_height() + 15,
                        f'{v}\n({100*v/len(df_mock):.0f}%)', ha='center', fontsize=9)
    _mg = df_mock['mast_group'].value_counts()
    axes[0, 1].bar(_mg.index, _mg.values, color='#2ca02c', edgecolor='white')
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].set_ylabel("Count")
    axes[1, 0].hist(df_mock['num_total_cells'],
                    bins=range(1, int(df_mock['num_total_cells'].max()) + 2),
                    color='#1f77b4', edgecolor='white', align='left')
    axes[1, 0].set_xlabel("num_total_cells"); axes[1, 0].set_ylabel("Count")
    axes[1, 1].hist(df_mock['total_non_ran_equipment'],
                    bins=range(0, int(df_mock['total_non_ran_equipment'].max()) + 2),
                    color='#ff7f0e', edgecolor='white', align='left')
    axes[1, 1].set_xlabel("total_non_ran_equipment"); axes[1, 1].set_ylabel("Count")
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_structural.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("  eda_structural.pdf")

    # S2 – Expected energy vs structure
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(df_mock['kwh_expected'], bins=30, color='#2ca02c', edgecolor='white')
    axes[0].axvline(df_mock['kwh_expected'].median(), color='red', linestyle='--',
                    lw=1.5, label=f"Median: {df_mock['kwh_expected'].median():.0f} kWh")
    axes[0].set_xlabel("kwh_expected (kWh)"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    _grp = (df_mock.groupby(['has_shared_ran', 'ran_vendor_type'])['kwh_expected']
            .agg(['median', 'std', 'count'])
            .rename(columns={'median': 'median_kwh', 'std': 'std_kwh', 'count': 'n'}))
    _glabels = [f"{'Shared' if s else 'Solo'}\n{_vendor_display.get(v, v)}"
                for (s, v) in _grp.index]
    _bars = axes[1].bar(_glabels, _grp['median_kwh'].values, yerr=_grp['std_kwh'].values,
                        capsize=6, color=['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78'],
                        edgecolor='white', error_kw=dict(elinewidth=1.5, ecolor='black'))
    for b, n in zip(_bars, _grp['n'].values):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + _grp['std_kwh'].max() * 0.05,
                     f"n={n}", ha='center', va='bottom', fontsize=9)
    axes[1].set_ylabel("Expected Energy (kWh)")
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_energy_structure.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("  eda_energy_structure.pdf")

    # S3 – Noise characteristics
    from scipy import stats as _stats
    _residual = (df_mock['kwh_expected_noise'] - df_mock['kwh_expected']).values
    _ratio    = (df_mock['kwh_expected_noise'] / df_mock['kwh_expected'].clip(lower=1)).values
    _p5, _p95 = np.percentile(_ratio, [5, 95])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].hist(_ratio, bins=60, color='#ff7f0e', edgecolor='white', alpha=0.8)
    axes[0].axvline(1.0,  color='black',   ls='--', lw=1.5, label='ratio = 1')
    axes[0].axvline(_p5,  color='#2ca02c', ls=':',  lw=1.5, label=f'P5 = {_p5:.3f}')
    axes[0].axvline(_p95, color='#2ca02c', ls=':',  lw=1.5, label=f'P95 = {_p95:.3f}')
    axes[0].set_xlabel("Noisy energy / expected energy"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    _xs = np.linspace(df_mock['kwh_expected'].min(), df_mock['kwh_expected'].max(), 200)
    _sigma_log_map = SIGMA_LOG_MAP
    _colors_mast = {'Tower': '#d62728', 'Disguised': '#ff7f0e', 'Rooftop': '#2ca02c',
                    'Pole': '#1f77b4', 'Other': '#9467bd'}
    axes[1].scatter(df_mock['kwh_expected'], np.abs(_residual), s=4, alpha=0.3,
                    color='#9467bd', linewidths=0)
    for _mg, _sl in _sigma_log_map.items():
        axes[1].plot(_xs, _sl * _xs, lw=1.5, color=_colors_mast[_mg],
                     label=f'{_mg} sigma={int(_sl*100)}%')
    axes[1].set_xlabel("Expected energy (kWh)"); axes[1].set_ylabel("|noise residual| (kWh)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_noise.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("  eda_noise.pdf")

    # S4 – Injection counts
    _type_labels = {0: 'Efficient', 1: 'Type 1\n(Overload)', 2: 'Type 2\n(Cooling)',
                    3: 'Type 3\n(Idle RF)', 4: 'Type 4\n(Non-RAN)'}
    _bar_colors  = ['#2ca02c', '#d62728', '#d62728', '#d62728', '#d62728']
    _tc = df_mock['inefficiency_type'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar([_type_labels.get(t, str(t)) for t in _tc.index], _tc.values,
                  color=[_bar_colors[t] for t in _tc.index], edgecolor='white')
    for b, v in zip(bars, _tc.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15,
                str(v), ha='center', fontsize=9)
    ax.set_ylabel("Count"); ax.tick_params(axis='x', labelsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_injection_counts.pdf"), bbox_inches='tight')
    plt.close(fig)

    # S4 – Injection KDE panels
    from scipy.stats import gaussian_kde as _kde
    _eff_mask   = df_mock['is_inefficient'] == 0
    _ineff_mask = df_mock['is_inefficient'] == 1
    _log_eff    = np.log1p(df_mock.loc[_eff_mask,   'meter_kwh_sim'].values)
    _log_ineff  = np.log1p(df_mock.loc[_ineff_mask, 'meter_kwh_sim'].values)
    _ratio_all  = (df_mock['meter_kwh_sim'] / df_mock['kwh_expected_noise'].clip(lower=1)).values
    _ratio_eff  = _ratio_all[_eff_mask]
    _ratio_ineff = _ratio_all[_ineff_mask]
    _xs_e  = np.linspace(_log_eff.min(),    _log_eff.max(),   300)
    _xs_i  = np.linspace(_log_ineff.min(),  _log_ineff.max(), 300)
    _xr    = np.linspace(0, min(np.percentile(_ratio_ineff, 99.5), 6), 300)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].fill_between(_xs_e, _kde(_log_eff)(_xs_e),   alpha=0.45,
                         color='#1f77b4', label='Efficient')
    axes[0].fill_between(_xs_i, _kde(_log_ineff)(_xs_i), alpha=0.45,
                         color='#d62728', label='Inefficient')
    axes[0].set_xlabel("log(1 + meter_kwh_sim)"); axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=9)
    axes[1].fill_between(_xr, _kde(_ratio_eff)(_xr),   alpha=0.45,
                         color='#1f77b4', label='Efficient')
    axes[1].fill_between(_xr, _kde(_ratio_ineff)(_xr), alpha=0.45,
                         color='#d62728', label='Inefficient')
    axes[1].axvline(1.0, color='black', ls='--', lw=1.5, label='ratio = 1')
    axes[1].set_xlabel("meter_kwh_sim / kwh_expected_noise"); axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "eda_injection.pdf"), bbox_inches='tight')
    plt.close(fig)
    print("  eda_injection_counts.pdf  eda_injection.pdf")


# ── 4. Feature matrix ──────────────────────────────────────────────────────────

def build_feature_matrix(df_mock):
    """Build structural feature matrix X (no energy column)."""
    _vendor_bin  = (df_mock['ran_vendor_type'] == 'Vendor A').astype(int).values
    _mast_dums   = pd.get_dummies(df_mock['mast_group'], prefix='mast').astype(float)
    _log_traffic = np.log1p(df_mock['ps_traffic_mb'].values)

    X = np.column_stack([
        df_mock['num_total_cells'].values,
        df_mock['total_non_ran_equipment'].values,
        df_mock['has_shared_ran'].values,
        _vendor_bin,
        _log_traffic,       # column index 4
        _mast_dums.values,
    ])
    return X


# ── 5. MDE pipeline ────────────────────────────────────────────────────────────

def run_mde_pipeline(df_mock, X_struct):
    """Run the full MDE pipeline and return embedding + scores.

    Returns:
        embedding:          torch tensor, shape (N_SAMPLES, EMBEDDING_DIM)
        emb_rel_dist_gated: energy-gated displacement score, shape (N_SAMPLES,)
        delta_node:         log-energy deviation from structural peers, shape (N_SAMPLES,)
        nn_idx_structural:  neighbour indices, shape (N_SAMPLES, K_FIT)
    """
    print("Building kNN graph ...")
    _sc = StandardScaler()
    X_scaled = _sc.fit_transform(X_struct)
    X_scaled[:, 4] *= TRAFFIC_WEIGHT  # down-weight traffic

    graph = pymde.preprocess.k_nearest_neighbors(
        csr_matrix(X_scaled), k=K_GRAPH, verbose=False)
    print(f"  {graph.edges.shape[0]} edges")

    # Structural neighbour index, stratified by (vendor, sharing, mast)
    print("Computing structural neighbour index ...")
    _K_fit = max(K_BASELINE, K_STRUCT_NEIGHBORS)
    nn_idx = np.full((N_SAMPLES, _K_fit), -1, dtype=np.int64)
    _energy = df_mock['meter_kwh_sim'].values
    _delta  = np.zeros(N_SAMPLES)
    for _, _grp_idx in df_mock.groupby(
            ['ran_vendor_type', 'has_shared_ran', 'mast_group']).groups.items():
        _grp_idx = _grp_idx.to_numpy()
        _k       = min(_K_fit + 1, len(_grp_idx))
        _nbrs    = NearestNeighbors(n_neighbors=_k).fit(X_scaled[_grp_idx])
        _, _nn   = _nbrs.kneighbors(X_scaled[_grp_idx])
        _nn      = _nn[:, 1:]  # drop self
        nn_idx[_grp_idx, :_nn.shape[1]] = _grp_idx[_nn]
        # delta_node: log(energy / P35 of K_BASELINE nearest structural neighbours)
        _e_grp   = _energy[_grp_idx]
        _nn_base = _nn[:, :min(K_BASELINE, _nn.shape[1])]
        _med     = np.maximum(np.percentile(_e_grp[_nn_base], 35, axis=1), 1.0)
        _delta[_grp_idx] = np.log((_e_grp + 1e-6) / (_med + 1e-6))
    delta_node = _delta

    # Modify edge weights: repel high-energy nodes from structural neighbours
    print("Modifying edge weights (beta={}) ...".format(BETA))
    _edges2   = graph.edges.numpy()
    _weights2 = graph.weights.numpy()
    _rep      = np.maximum(
        np.maximum(delta_node[_edges2[:, 0]], 0),
        np.maximum(delta_node[_edges2[:, 1]], 0),
    )
    weights = _weights2 - BETA * _rep

    # Build MDE problem with PushAndPull penalty
    _similar   = graph.edges
    _n_similar = _similar.shape[0]
    torch.manual_seed(42)
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

    print("Embedding ...")
    torch.manual_seed(42)
    embedding = mde.embed(verbose=True)
    X_emb = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)

    # Teacher score: embedding relative displacement, energy-gated
    print("Computing teacher score ...")
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

    df_mock['emb_rel_dist_gated'] = emb_rel_dist_gated

    return embedding, emb_rel_dist_gated, delta_node, nn_idx


# ── 6. Autoencoder helper ──────────────────────────────────────────────────────

class _SimpleAE(_nn.Module):
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


def _train_ae(X_np, hidden, bottleneck, epochs, seed=0):
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


# ── 7. Baseline comparison ─────────────────────────────────────────────────────

def run_baseline_comparison(df_mock, X_struct, embedding, emb_rel_dist_gated):
    """Compute all baselines and print results table."""
    energy   = df_mock['meter_kwh_sim'].values
    y_true   = df_mock['is_inefficient'].values
    top_n    = int(INEFF_PCT * N_SAMPLES)
    pct      = int(INEFF_PCT * 100)
    X_emb_np = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)

    _vendor_bin  = (df_mock['ran_vendor_type'] == 'Vendor A').astype(int).values
    _mast_dums   = pd.get_dummies(df_mock['mast_group'], prefix='mast').astype(float).values
    _struct_full = np.column_stack([
        df_mock['num_total_cells'].values, df_mock['total_non_ran_equipment'].values,
        df_mock['has_shared_ran'].values, _vendor_bin, _mast_dums,
        np.log1p(df_mock['ps_traffic_mb'].values),
    ])

    _ss = StandardScaler()
    X_all_sc    = _ss.fit_transform(np.column_stack([_struct_full, energy]))
    X_struct_sc = StandardScaler().fit_transform(_struct_full)
    X_mde_sc    = StandardScaler().fit_transform(X_emb_np)

    def _met(s):
        pr   = average_precision_score(y_true, s)
        roc  = roc_auc_score(y_true, s)
        idx  = np.argsort(s)[::-1][:top_n]
        prec = y_true[idx].mean()
        rec  = y_true[idx].sum() / max(y_true.sum(), 1)
        return pr, roc, prec, rec

    results = {}

    results['Raw energy'] = _met(energy)

    _lr_r = energy - LinearRegression().fit(X_struct, energy).predict(X_struct)
    results['Linear regression residual'] = _met(np.maximum(_lr_r, 0))

    _X_hub = StandardScaler().fit_transform(X_struct)
    _hub_r = energy - HuberRegressor(epsilon=1.35, max_iter=300).fit(
        _X_hub, energy).predict(_X_hub)
    results['Huber regression residual'] = _met(np.maximum(_hub_r, 0))

    _rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    _rf.fit(X_struct, energy)
    results['Random Forest residual'] = _met(np.maximum(energy - _rf.predict(X_struct), 0))

    # Physics residual is privileged (anomalies were generated relative to kwh_expected)
    _phys = energy / (df_mock['kwh_expected'].values + 1e-6)
    results['Physics residual (privileged upper bound)'] = _met(_phys)

    for _name, _X in [('features', X_all_sc), ('MDE emb', X_mde_sc)]:
        _iso = IsolationForest(n_estimators=100, contamination=INEFF_PCT, random_state=42)
        results[f'IsoForest ({_name})'] = _met(-_iso.fit(_X).score_samples(_X))

    for _name, _X in [('features', X_all_sc), ('MDE emb', X_mde_sc)]:
        _gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        results[f'GMM ({_name})'] = _met(-_gmm.fit(_X).score_samples(_X))

    for _name, _X, _k in [('features', X_all_sc, K_STRUCT_NEIGHBORS),
                            ('MDE emb',  X_mde_sc, K_STRUCT_NEIGHBORS)]:
        _lof = LocalOutlierFactor(n_neighbors=_k, contamination=INEFF_PCT)
        _lof.fit(_X)
        results[f'LOF ({_name})'] = _met(-_lof.negative_outlier_factor_)

    results['MDE rel. displacement (proposed)'] = _met(emb_rel_dist_gated)

    print("Training AE (features) ...", flush=True)
    results['AE (features)'] = _met(_train_ae(X_all_sc, [32, 16], 3, 200))
    print("Training AE (MDE emb) ...", flush=True)
    results['AE (MDE emb)']  = _met(_train_ae(X_mde_sc, [16], 1, 150))

    hdr = (f"\n{'Method':<46} {'ROC-AUC':>9} {'PR-AUC':>8}"
           f" {'Prec@{pct}%':>9} {'Rec@{pct}%':>9}")
    print("\n-- Baseline comparison " + "-" * 64)
    print(hdr)
    print("-" * 86)
    for name, (pr, roc, prec, rec) in sorted(results.items(), key=lambda x: -x[1][1]):
        tag = " <--" if name == 'MDE rel. displacement (proposed)' else ""
        print(f"{name:<46} {roc:>9.4f} {pr:>8.4f} {prec:>9.4f} {rec:>9.4f}{tag}")

    return results


# ── 8. Embedding visualisation ─────────────────────────────────────────────────

def plot_embedding(df_mock, embedding, X_scaled, out_dir):
    """Save static embedding scatter and PCA comparison figure."""
    X_emb_np = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)
    is_ineff  = df_mock['is_inefficient'].values.astype(bool)

    # 2D projection of first two embedding dims
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_emb_np[is_ineff,  0], X_emb_np[is_ineff,  1],
               c='#e41a1c', s=12, alpha=0.6, linewidths=0, label='Inefficient')
    ax.scatter(X_emb_np[~is_ineff, 0], X_emb_np[~is_ineff, 1],
               c='#000080', s=5,  alpha=0.9, linewidths=0, label='Efficient')
    ax.set_title("MDE embedding (dims 1-2)")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedding_scatter.pdf"), bbox_inches='tight')
    plt.close(fig)

    # MDE vs PCA comparison
    pca_emb = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (title, emb) in zip(axes, [("MDE (energy-aware)", X_emb_np[:, :2]),
                                        ("PCA", pca_emb)]):
        ax.scatter(emb[is_ineff,  0], emb[is_ineff,  1],
                   c='#e41a1c', label='Inefficient', s=12, alpha=0.6, linewidths=0)
        ax.scatter(emb[~is_ineff, 0], emb[~is_ineff, 1],
                   c='#000080', label='Efficient',   s=5,  alpha=0.9, linewidths=0)
        ax.set_title(title, fontsize=13)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedding_comparison.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"  embedding_scatter.pdf  embedding_comparison.pdf")


# ── 9. Evaluate MDE ───────────────────────────────────────────────────────────

def evaluate_mde(df_mock, embedding, X_structural, score_col="emb_rel_dist_gated",
                 k_neighbors=50, top_k_frac=0.05):
    """Compute embedding quality and detection metrics."""
    from sklearn.manifold import trustworthiness
    from scipy.stats import ttest_ind

    y_true = df_mock['is_inefficient'].values
    scores = df_mock[score_col].fillna(0).values
    X_emb  = embedding.cpu().numpy() if hasattr(embedding, 'cpu') else np.array(embedding)
    top_k  = int(top_k_frac * len(df_mock))

    res = {}
    res['roc_auc'] = roc_auc_score(y_true, scores)
    res['pr_auc']  = average_precision_score(y_true, scores)
    _idx = np.argsort(scores)[::-1][:top_k]
    res['precision_at_k'] = y_true[_idx].mean()
    res['recall_at_k']    = y_true[_idx].sum() / max(y_true.sum(), 1)
    res['trustworthiness'] = trustworthiness(X_structural, X_emb, n_neighbors=k_neighbors)

    _nbrs_e = NearestNeighbors(n_neighbors=k_neighbors).fit(X_emb)
    _, _idx_e = _nbrs_e.kneighbors(X_emb)
    _nbrs_s = NearestNeighbors(n_neighbors=k_neighbors).fit(X_structural)
    _, _idx_s = _nbrs_s.kneighbors(X_structural)
    res['knn_overlap_mean'] = np.mean([
        len(set(_idx_s[i]) & set(_idx_e[i])) / k_neighbors for i in range(len(df_mock))])

    _centroid_dist = [
        np.linalg.norm(X_emb[i] - X_emb[_idx_e[i]].mean(axis=0))
        for i in range(len(df_mock))]
    df_mock = df_mock.copy()
    df_mock['centroid_dist'] = _centroid_dist
    eff   = df_mock.loc[df_mock['is_inefficient'] == 0, 'centroid_dist']
    ineff = df_mock.loc[df_mock['is_inefficient'] == 1, 'centroid_dist']
    res['centroid_dist_mean_eff']   = eff.mean()
    res['centroid_dist_mean_ineff'] = ineff.mean()
    _pstd = np.sqrt((eff.std() ** 2 + ineff.std() ** 2) / 2)
    res['cohens_d'] = (ineff.mean() - eff.mean()) / (_pstd + 1e-6)
    res['t_stat'], res['p_value'] = ttest_ind(eff, ineff, equal_var=False)

    return res


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MDE energy inefficiency detection pipeline.")
    parser.add_argument(
        "--data", default=None,
        help="Path to data_extended.pkl. If omitted, loads pre-generated "
             "data/synthetic_base.csv.")
    parser.add_argument(
        "--eda", action="store_true",
        help="Produce EDA diagnostic figures (S1-S4).")
    parser.add_argument(
        "--out", default=None,
        help="Output directory for figures (default: ./data/).")
    args = parser.parse_args()

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _data_dir   = args.out if args.out else os.path.join(_script_dir, "data")
    _csv_path   = os.path.join(_script_dir, "data", "synthetic_base.csv")
    os.makedirs(_data_dir, exist_ok=True)

    # Load / generate base dataset
    if args.data:
        print(f"Loading real data from {args.data} ...")
        df_real = pd.read_pickle(args.data)
        df_real['mast_group'] = df_real['mast_type'].map(_MAST_BUCKET_MAP).fillna('Other')
        print("Generating synthetic base dataset (probability-weighted sampling) ...")
        df_base = generate_synthetic_base(df_real)
        df_base.to_csv(_csv_path, index=False)
        print(f"Saved {len(df_base)} rows -> {_csv_path}")
    else:
        if not os.path.exists(_csv_path):
            raise FileNotFoundError(
                f"synthetic_base.csv not found at {_csv_path}.\n"
                "Run with --data path/to/data_extended.pkl to generate it.")
        print(f"Loading synthetic base dataset from {_csv_path} ...")
        df_base = pd.read_csv(_csv_path)

    print(f"\nDataset: {len(df_base)} sites  "
          f"vendor={df_base['ran_vendor_type'].value_counts().to_dict()}\n")

    print(f"Injecting anomalies at {INEFF_PCT:.0%} contamination (seed 42) ...")
    df_mock = inject_anomalies(df_base, INEFF_PCT, rng_seed=42)
    _tc = df_mock['inefficiency_type'].value_counts().sort_index()
    print(f"  " + "  ".join([f"Type{t}:{n}" for t, n in _tc.items() if t > 0]) +
          f"  Total:{df_mock['is_inefficient'].sum()}/{N_SAMPLES}\n")

    if args.eda:
        print("Generating EDA figures ...")
        plot_eda(df_mock, _data_dir)
        print()

    X_struct = build_feature_matrix(df_mock)

    embedding, emb_rel_dist_gated, delta_node, nn_idx = run_mde_pipeline(df_mock, X_struct)

    print("\n")
    run_baseline_comparison(df_mock, X_struct, embedding, emb_rel_dist_gated)

    print("\n-- MDE evaluation " + "-" * 68)
    _X_scaled = StandardScaler().fit_transform(X_struct)
    eval_res  = evaluate_mde(df_mock, embedding, _X_scaled,
                              score_col='emb_rel_dist_gated',
                              k_neighbors=K_STRUCT_NEIGHBORS,
                              top_k_frac=INEFF_PCT)
    pct = int(INEFF_PCT * 100)
    print(f"  ROC-AUC                         : {eval_res['roc_auc']:.4f}")
    print(f"  PR-AUC                          : {eval_res['pr_auc']:.4f}")
    print(f"  Precision@{pct}%                  : {eval_res['precision_at_k']:.4f}")
    print(f"  Recall@{pct}%                     : {eval_res['recall_at_k']:.4f}")
    print(f"  Trustworthiness (k={K_STRUCT_NEIGHBORS})         : {eval_res['trustworthiness']:.4f}")
    print(f"  kNN overlap (struct vs emb)     : {eval_res['knn_overlap_mean']:.4f}")
    print(f"  Centroid dist (efficient)       : {eval_res['centroid_dist_mean_eff']:.4f}")
    print(f"  Centroid dist (inefficient)     : {eval_res['centroid_dist_mean_ineff']:.4f}")
    print(f"  Cohen's d                       : {eval_res['cohens_d']:.4f}")

    print("\nSaving embedding figures ...")
    _X_scaled_for_plot = StandardScaler().fit_transform(X_struct)
    plot_embedding(df_mock, embedding, _X_scaled_for_plot, _data_dir)


if __name__ == "__main__":
    main()
