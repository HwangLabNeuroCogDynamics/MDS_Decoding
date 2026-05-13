# LDA on anaimal raising, but not looking at accuracy but underlying encoding structure
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
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
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp, t


subject_ids = ['10811','10748','10770','10808','10763','10787','10769','10764',
'10840','10824','10736','10797','10758','10784','10844','10849','10834','10843',
'10767','10792','10825','10836','10841','10809','10831','10762','10771','10851',
'10814','10747','10730','10842','10835','10083','10263','10761','10785','10845',
'10819','10813']

ROOT = '/mnt/nfs/lss/lss_kahwang_hpc'
EVENT_CONFIG_FILE = os.path.join(ROOT, 'scripts/mind_mosaic/eeg/preprocessing/event_config.json')

session = 'testing'
EPOCH_FOLDER = 'epochs'
PREPROCESS_DATA = 'preprocessed_cue_decoding'
DESIGNS = ['Int', 'NoInt']

# LDA 

# ============================================================
# use LDA to extract neural patterns that discrimate color or ori
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
    #   weights into activation patterns 
    #   Raw decoder weights are difficult to interpret because
    #   discriminative models are influenced by covariance
    #   structure and noise suppression.
    #
    #   therefore we get pattern = covariance @ decoder_weights
    #
    #   approximately maps decoder weights back into signal /
    #   sensor space, then compute similarity later
    # --------------------------------------------------------


    # --------------------------------------------------------
    # First convert labels into integer values
    # --------------------------------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)


    
    # 5 fold CV
    skf = StratifiedKFold( 5, shuffle=True, random_state=0 )

    class_patterns = {
        c: [] for c in np.unique(y_enc)
    }

    # cross-validation loop
    for train_idx, test_idx in skf.split(X, y_enc):

        # ----------------------------------------------------
        # extract training data for this fold
        #
        # X_train shape:
        #   (n_train_trials, n_channels)
        # ----------------------------------------------------
        X_train = X[train_idx]
        y_train = y_enc[train_idx]
        
        # z-score each channel
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)


        # ----------------------------------------------------
        # train multiclass LDA
        lda = LinearDiscriminantAnalysis( solver='lsqr', shrinkage='auto')
        lda.fit(X_train_scaled, y_train)

        # ----------------------------------------------------
        # decoder weights
        #
        # shape:
        #   (n_classes, n_channels)
        #
        # each row is one discriminant vector
        # ----------------------------------------------------
        Ws = lda.coef_

        # ----------------------------------------------------
        # compute feature covariance matrix
        #
        # shape:
        #   (n_channels, n_channels)
        #
        # rowvar=False:
        #   columns = variables/channels
        # ----------------------------------------------------
        cov = np.cov( X_train_scaled, rowvar=False )


        # ====================================================
        # compute patterns for each class or trial type
        # ====================================================
        for c_idx, w in enumerate(Ws):


            # ------------------------------------------------
            # transform decoding coeficients into comaprable scale 
            # pattern = covariance @ weights
            # converts discriminative weights into activation
            # patterns that better reflect underlying neural topographies
            # ------------------------------------------------
            pattern = cov @ w


            # ------------------------------------------------
            # normalize vector length to unit norm
            # this removes arbitrary magnitude scaling and
            # allows comparisons across folds/classes
            # ------------------------------------------------
            norm = np.linalg.norm(pattern)

            if norm > 0:
                pattern = pattern / norm
            class_patterns[c_idx].append(pattern)

    patterns_avg = []
    for c_idx in sorted(class_patterns.keys()):
        # average across folds 
        p = np.mean( class_patterns[c_idx], axis=0 )

        # renormalize after averaging
        norm = np.linalg.norm(p)

        if norm > 0:
            p = p / norm

        patterns_avg.append(p)

    return np.array(patterns_avg)

#################################
### below is the similarity matrices for comparing whether the pattern encoding color are similar to those for orientation
# it is possible that joint v selective will have different encoding geometry?
##########################################
def cosine_metric(A, B):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    return np.mean(cosine_similarity(A, B)) #cosine similarity of the pattern

# from XT
def get_feature_maps(design):
    if design == 'Int':
        color_map = {'Af1':360,'Am1':300,'Am2':360,'Af2':300,'Bf2':120,'Bm1':60,'Bm2':120,'Bf1':60}
        ori_map   = {'Af1':195,'Am1':195,'Am2':225,'Af2':225,'Bf2':15,'Bm1':15,'Bm2':45,'Bf1':45}
    else:
        color_map = {"Caf1":360,"Caf2":300,"Cam1":360,"Cam2":300,"Cbf1":120,"Cbf2":60,"Cbm1":120,"Cbm2":60,
                     "Daf1":360,"Daf2":360,"Dam1":300,"Dam2":300,"Dbf1":120,"Dbf2":120,"Dbm1":60,"Dbm2":60}
        ori_map   = {"Caf1":225,"Caf2":225,"Cam1":195,"Cam2":195,"Cbf1":15,"Cbf2":15,"Cbm1":45,"Cbm2":45,
                     "Daf1":225,"Daf2":195,"Dam1":225,"Dam2":195,"Dbf1":15,"Dbf2":45,"Dbm1":15,"Dbm2":45}

    return color_map, ori_map

####################################################
# below is cross validated distance
##########################################

def compute_means(X, y):
    classes = np.unique(y)
    means = {c: X[y == c].mean(axis=0) for c in classes}
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

def crossnobis_metric(X, y):

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    skf = StratifiedKFold(5, shuffle=True, random_state=0)

    distances = []

    for train_idx, test_idx in skf.split(X, y_enc):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # standardize (important!)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        means_train = compute_means(X_train, y_train)
        means_test  = compute_means(X_test, y_test)

        cov = compute_noise_cov(X_train, y_train)
        inv_cov = np.linalg.pinv(cov)

        classes = np.unique(y_train)

        # compute all pairwise distances
        fold_dists = []

        for i in range(len(classes)):
            for j in range(i+1, len(classes)):

                c1, c2 = classes[i], classes[j]

                d1 = means_train[c1] - means_train[c2]
                d2 = means_test[c1]  - means_test[c2]

                d = d1 @ inv_cov @ d2
                fold_dists.append(d)

        distances.append(np.mean(fold_dists))

    return np.mean(distances)


def process_subject(subject_id):

    results = {}
    for design in DESIGNS:
        files = glob(os.path.join( ROOT, f"data/MindMosaic_EEG/sub-{subject_id}", PREPROCESS_DATA, EPOCH_FOLDER, f"sub-{subject_id}_design-{design}_session-{session}_cue*epo*.fif.gz" ))

        if not files:
            continue

        epochs = mne.read_epochs(files[0], preload=True, verbose=False)
        epochs.apply_baseline((-0.5, 0)) # is this necessary?

        X = epochs.copy().pick('eeg').get_data()
        metadata = epochs.metadata.copy()
        times = epochs.times

        with open(EVENT_CONFIG_FILE) as f:
            event_config = json.load(f)[design][session]

        rev = {v:k[5:] for k,v in event_config['probe_event_id'].items()}
        metadata['combos'] = metadata['probe_trigger'].map(rev)
        color_map, ori_map = get_feature_maps(design)
        metadata['color'] = metadata['combos'].map(color_map)
        metadata['ori'] = metadata['combos'].map(ori_map)

        valid = ~metadata['response_trigger'].isna()
        X = X[valid]
        metadata = metadata[valid]

        # time course of subspace similarity
        tc = {m:[] for m in ['cosine', 'crossnobis']}

        for t in range(len(times)):
            try:
                p_color = compute_patterns_multiclass(X[:,:,t], metadata['color'])
                p_ori   = compute_patterns_multiclass(X[:,:,t], metadata['ori'])
                tc['cosine'].append(cosine_metric(p_color, p_ori))

                cn_color = crossnobis_metric(X[:,:,t], metadata['color'])
                cn_ori   = crossnobis_metric(X[:,:,t], metadata['ori'])

                # store average (you could also store separately if you want)
                tc['crossnobis'].append((cn_color + cn_ori) / 2)

            except:
                for k in tc:
                    tc[k].append(np.nan)

        results[design] = {
            'times': times,
            'metrics': {k: np.array(v) for k,v in tc.items()}
        }

    return results


# RUN 
all_results = Parallel(n_jobs=40)( delayed(process_subject)(sid) for sid in subject_ids )

metrics = ['cosine', 'crossnobis']
group_data = {m: {'Int':[], 'NoInt':[]} for m in metrics}

for r in all_results:
    if isinstance(r,dict) and 'Int' in r and 'NoInt' in r:
        for m in metrics:
            a = r['Int']['metrics'][m]
            b = r['NoInt']['metrics'][m]
            if a.shape == b.shape:
                group_data[m]['Int'].append(a)
                group_data[m]['NoInt'].append(b)

for m in metrics:

    tc_a = np.stack(group_data[m]['Int'])
    tc_b = np.stack(group_data[m]['NoInt'])

    diff = tc_a - tc_b
    mean_diff = diff.mean(0)

    times = r['Int']['times']

    # compute t-values then plot
    t_obs, _ = ttest_1samp(diff, 0, axis=0)

    plt.figure()
    plt.plot(times, t_obs, label='t-value')
    plt.axhline(0)

    plt.title(f"{m} (t-values)")
    plt.legend()
    plt.show()
