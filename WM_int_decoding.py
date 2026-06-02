# LDA on anaimal raising, but not looking at accuracy but underlying encoding structure
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = ""
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import mne
import json
from glob import glob
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from scipy.linalg import subspace_angles
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp

N_REPEATS = 10 #how many random runs of CV
N_SPLITS = 5 #of folds
# subject_ids = ['10811','10748','10770','10808','10763','10787','10769','10764',
# '10840','10824','10736','10797','10758','10784','10844','10849','10834','10843',
# '10767','10792','10825','10836','10841','10809','10831','10762','10771','10851',
# '10814','10747','10730','10842','10835','10083','10263','10761','10785','10845',
# '10819','10813']

with open("/mnt/nfs/lss/lss_kahwang_hpc/scripts/mind_mosaic/eeg/qc_check/EEG_clean_subject_ids.txt") as f:
    txt = f.read()
subject_ids = np.array( [x.strip() for x in txt.split(",") if x.strip()], dtype=str )

ROOT = '/mnt/nfs/lss/lss_kahwang_hpc'
EVENT_CONFIG_FILE = os.path.join(ROOT, 'scripts/mind_mosaic/eeg/preprocessing/event_config.json')
session = 'testing'
EPOCH_FOLDER = 'epochs'
PREPROCESS_DATA = 'preprocessed_decoding_hp01_lp50_interp8'
DESIGNS = ['Int', 'NoInt']


# ============================================================
# use LDA to extract neural patterns that discriminate color or ori
# ============================================================
def compute_patterns_multiclass(X, y):
    # --------------------------------------------------------
    # X shape:
    #   (n_trials, n_channels)
    #
    # y shape:
    #   (n_trials,)
    #
    # Goal:
    #   Train multiclass LDA decoders and convert decoder
    #   weights into activation patterns.
    #
    #   Raw LDA weights are not directly interpretable because
    #   discriminative decoders absorb covariance structure and
    #   noise suppression.
    #
    #   pattern = covariance @ decoder_weights
    #
    # This approximately maps decoder weights back into
    # sensor space and gives a more interpretable estimate of
    # the underlying neural activity pattern for each class.
    #
    # Eventually, it is possible to project single trial data on these these
    # patterns to get single trial pattern estiamte.
    #
    #
    # --------------------------------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_patterns = {c: [] for c in np.unique(y_enc)}

    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)

        for train_idx, _ in skf.split(X, y_enc):

            X_train = X[train_idx]
            y_train = y_enc[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X_train_scaled, y_train)

            Ws = lda.coef_
            cov = np.cov(X_train_scaled, rowvar=False)

            for c_idx, w in enumerate(Ws):

                pattern = cov @ w
                norm = np.linalg.norm(pattern)

                if norm > 0:
                    pattern = pattern / norm

                class_patterns[c_idx].append(pattern)

    patterns_avg = []
    for c_idx in sorted(class_patterns.keys()):
        p = np.mean(class_patterns[c_idx], axis=0)
        norm = np.linalg.norm(p)
        if norm > 0:
            p = p / norm
        patterns_avg.append(p)

    return np.array(patterns_avg)

    # output:
    # shape: (n_classes, n_channels)
    # each row = neural representation pattern of a class


# ============================================================
# COSINE SIMILARITY
# ============================================================
def cosine_metric(A, B):

    # Compare alignment of individual representation vectors
    # high cosine:   patterns point in similar directions
    # low cosine:   patterns use different channel configurations
    #


    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    return np.mean(cosine_similarity(A, B))


# ============================================================
# SUBSPACE ANGLE
# ============================================================
def subspace_angle_metric(A, B):

    # Compare representational spaces
    #
    # A and B spaces are(n_classes, n_channels)
    #
    # treat rows as spanning a subspace in channel space.
    # So 0 degree will mean overlap, 90 degree would be orthogonal
    # small angle:   shared / overlapping representational geometry
    #
    # large angle:   orthogonal / factorized geometry

    angles = subspace_angles(A.T, B.T)

    # mean principal angle
    return np.mean(angles)


def compute_means(X, y):

    classes = np.unique(y)

    means = {
        c: X[y == c].mean(axis=0)
        for c in classes
    }

    return means


def compute_noise_cov(X, y):

    resid = []
    for c in np.unique(y):
        Xc = X[y == c]
        mean = Xc.mean(axis=0)
        resid.append(Xc - mean)

    resid = np.vstack(resid)
    cov = np.cov(resid, rowvar=False)

    return cov


# ============================================================
# SUBSPACE CROSSNOBIS
# ============================================================

def subspace_crossnobis_metric(X, y, patterns):
    # This is the distance-based analogue of subspace decoding
    #
    # Step 1:
    #   project EEG data into another representational space for example
    #   project ORI into color subspace
    #
    # Step 2:
    #   ask whether ori information is still separable
    #   inside that projected space
    # high value: ori survives projection into color space shared / integrated representation
    # low value:ori disappears after projection factorized representation
   
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    all_distances = []

    for rep in range(N_REPEATS):

        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        distances = []

        for train_idx, test_idx in skf.split(X, y_enc):

            X_train = X[train_idx]
            X_test  = X[test_idx]

            y_train = y_enc[train_idx]
            y_test  = y_enc[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            X_train_proj = X_train @ patterns.T
            X_test_proj  = X_test @ patterns.T

            means_train = compute_means(X_train_proj, y_train)
            means_test  = compute_means(X_test_proj, y_test)

            cov = compute_noise_cov(X_train_proj, y_train)
            cov += np.eye(cov.shape[0]) * 1e-6

            inv_cov = np.linalg.pinv(cov)

            classes = np.unique(y_train)
            fold_dists = []

            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    c1, c2 = classes[i], classes[j]

                    d1 = means_train[c1] - means_train[c2]
                    d2 = means_test[c1]  - means_test[c2]

                    d = d1 @ inv_cov @ d2
                    fold_dists.append(d)

            distances.append(np.mean(fold_dists))

        all_distances.append(np.mean(distances))

    return np.mean(all_distances)


def get_feature_maps(design):

    if design == 'Int':

        color_map = {
            'Af1':360,'Am1':300,'Am2':360,'Af2':300,
            'Bf2':120,'Bm1':60,'Bm2':120,'Bf1':60
        }

        ori_map = {
            'Af1':195,'Am1':195,'Am2':225,'Af2':225,
            'Bf2':15,'Bm1':15,'Bm2':45,'Bf1':45
        }

    else:

        color_map = {
            "Caf1":360,"Caf2":300,"Cam1":360,"Cam2":300,
            "Cbf1":120,"Cbf2":60,"Cbm1":120,"Cbm2":60,
            "Daf1":360,"Daf2":360,"Dam1":300,"Dam2":300,
            "Dbf1":120,"Dbf2":120,"Dbm1":60,"Dbm2":60
        }

        ori_map = {
            "Caf1":225,"Caf2":225,"Cam1":195,"Cam2":195,
            "Cbf1":15,"Cbf2":15,"Cbm1":45,"Cbm2":45,
            "Daf1":225,"Daf2":195,"Dam1":225,"Dam2":195,
            "Dbf1":15,"Dbf2":45,"Dbm1":15,"Dbm2":45
        }

    return color_map, ori_map


###### distance metrics:

def euclidean_pattern_metric(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    D = pairwise_distances(A, B, metric='euclidean')
    return np.mean(D)



def correlation_similarity_metric(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    D = 1 - pairwise_distances(A, B, metric='correlation')
    return np.mean(D)



def correlation_distance_metric(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    D = pairwise_distances(A, B, metric='correlation')
    return np.mean(D)


def process_subject(subject_id):

    results = {}
    for design in DESIGNS:

        files = glob(os.path.join(
            ROOT,
            f"data/MindMosaic_EEG/sub-{subject_id}",
            PREPROCESS_DATA,
            EPOCH_FOLDER,
            f"sub-{subject_id}_design-{design}_session-{session}_cue*epo*.fif.gz"
        ))

        if not files:
            continue

        epochs = mne.read_epochs( files[0], preload=True, verbose=False )

        epochs.apply_baseline((-0.5, 0))
        X = epochs.copy().pick('eeg').get_data()
        metadata = epochs.metadata.copy()
        times = epochs.times

        with open(EVENT_CONFIG_FILE) as f:
            event_config = json.load(f)[design][session]

        rev = {
            v:k[5:]
            for k,v in event_config['probe_event_id'].items()
        }

        metadata['combos'] = metadata['probe_trigger'].map(rev)
        color_map, ori_map = get_feature_maps(design)
        metadata['color'] = metadata['combos'].map(color_map)
        metadata['ori']   = metadata['combos'].map(ori_map)
        valid = ~metadata['response_trigger'].isna()

        X = X[valid]
        metadata = metadata[valid]

        tc = {
            m:[] for m in [
                'cosine',
                'subspace_angle',
                'subspace_crossnobis',
                'euclidean',
                'corr_sim',
                'corr_dist'
            ]
        }

        for t in range(len(times)):

            try:

                Xt = X[:,:,t]
                p_color = compute_patterns_multiclass( Xt, metadata['color'] )

                p_ori = compute_patterns_multiclass( Xt, metadata['ori'] )

                tc['cosine'].append( cosine_metric(p_color, p_ori) )
                tc['subspace_angle'].append( subspace_angle_metric(p_color, p_ori) )
                tc['subspace_crossnobis'].append( subspace_crossnobis_metric( Xt, metadata['ori'], p_color ) )
                tc['euclidean'].append( euclidean_pattern_metric(p_color, p_ori) )
                tc['corr_sim'].append( correlation_similarity_metric(p_color, p_ori) )
                tc['corr_dist'].append( correlation_distance_metric(p_color, p_ori) )

            except:
                for k in tc:
                    tc[k].append(np.nan)

        results[design] = {
            'times': times,
            'metrics': {
                k: np.array(v)
                for k,v in tc.items()
            }
        }

    return results


all_results = Parallel(n_jobs=40)( delayed(process_subject)(sid) for sid in subject_ids )

metrics = [
    'cosine',
    'subspace_angle',
    'subspace_crossnobis',
    'euclidean',
    'corr_sim',
    'corr_dist'
]

group_data = { m: {'Int':[], 'NoInt':[]} for m in metrics }

for r in all_results:
    if isinstance(r,dict) and 'Int' in r and 'NoInt' in r:
        for m in metrics:
            a = r['Int']['metrics'][m]
            b = r['NoInt']['metrics'][m]
            if a.shape == b.shape:
                group_data[m]['Int'].append(a)
                group_data[m]['NoInt'].append(b)



################################
#### t stats
############################
from statsmodels.stats.multitest import fdrcorrection

for m in metrics:
    tc_a = np.stack(group_data[m]['Int'])
    tc_b = np.stack(group_data[m]['NoInt'])
    diff = tc_a - tc_b 
    times = all_results[0]['Int']['times']
    t_obs, p_vals = ttest_1samp(diff, 0, axis=0)
    # fdr
    reject, p_fdr = fdrcorrection(p_vals, alpha=0.05)

    plt.figure()
    plt.plot(times, t_obs, label='t-value')
    plt.axhline(0)
    plt.scatter(
        times[reject],
        t_obs[reject],
        color='red',
        s=15,
        label='FDR q < 0.05'
    )

    plt.title(f"{m} (Int - NoInt t-values, FDR corrected)")
    plt.legend()
    plt.show()


############################
#### here is cluster statistics
############################
# find clusters (contiguous supra-threshold points)
def find_clusters(t_vals, threshold):

    clusters = []
    current = []

    for i, t in enumerate(t_vals):
        if np.abs(t) > threshold:
            current.append(i)
        else:
            if current:
                clusters.append(current)
                current = []

    if current:
        clusters.append(current)

    return clusters

# cluster statistic = sum of t-values
def cluster_stat(t_vals, clusters):
    return [np.sum(t_vals[c]) for c in clusters]

# one permutation (sign flip)
def permute_once(diff):
    n_subj = diff.shape[0]
    signs = np.random.choice([1, -1], size=(n_subj, 1))
    permuted = diff * signs
    t_perm, _ = ttest_1samp(permuted, 0, axis=0)
    return t_perm


n_perm = 1000
threshold = 2.0   # ~p < .05 for ~40 subjects
for m in metrics:

    print(f"\nRunning cluster test for: {m}")
    tc_a = np.stack(group_data[m]['Int'])
    tc_b = np.stack(group_data[m]['NoInt'])
    diff = tc_a - tc_b
    times = all_results[0]['Int']['times']
    t_obs, _ = ttest_1samp(diff, 0, axis=0)
    clusters = find_clusters(t_obs, threshold)
    cluster_stats_obs = cluster_stat(t_obs, clusters)

    perm_t = Parallel(n_jobs=40)( delayed(permute_once)(diff) for _ in range(n_perm) )
    max_cluster_dist = []

    for t_perm in perm_t:

        clust = find_clusters(t_perm, threshold)
        if len(clust) == 0:
            max_cluster_dist.append(0)
        else:
            stats = cluster_stat(t_perm, clust)
            max_cluster_dist.append(np.max(stats))

    max_cluster_dist = np.array(max_cluster_dist)
    cluster_pvals = []

    for stat in cluster_stats_obs:
        p = np.mean(max_cluster_dist >= stat)
        cluster_pvals.append(p)

    plt.figure()
    plt.plot(times, t_obs, label='t-values')
    plt.axhline(0)

    # highlight significant clusters
    for c, p in zip(clusters, cluster_pvals):
        if p < 0.05:
            plt.plot(times[c], t_obs[c], linewidth=4)

    plt.title(f"{m} cluster-corrected (p<.05)")
    plt.legend()
    plt.show()

    print("Clusters:")
    for c, p in zip(clusters, cluster_pvals):
        print(f"  time {times[c[0]]:.3f}–{times[c[-1]]:.3f} | p={p:.4f}")

# MDS
def run_mds_visualization(X, labels, title='MDS'):

    classes = np.unique(labels)
    means = [ X[labels == c].mean(axis=0) for c in classes ]
    means = np.array(means)
    # pairwise condition distances
    D = pairwise_distances( means, metric='euclidean' )

    mds = MDS( n_components=2, dissimilarity='precomputed', random_state=0 )

    coords = mds.fit_transform(D)
    plt.figure()

    for i, c in enumerate(classes):
        plt.scatter(coords[i,0], coords[i,1])
        plt.text(coords[i,0], coords[i,1], str(c))
    plt.title(title)
    plt.show()
