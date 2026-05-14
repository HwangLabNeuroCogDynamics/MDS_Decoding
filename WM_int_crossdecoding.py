
# cross decoding version
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import mne
import json
from glob import glob
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

N_REPEATS = 1 # no CV anymore
N_SPLITS = 5 #of folds
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


def cross_decode_feature(X, y_feature, y_context):

    le_feat = LabelEncoder()
    y_feat = le_feat.fit_transform(y_feature)

    contexts = np.unique(y_context)
    scores = []

    for rep in range(N_REPEATS):
        for i in range(len(contexts)):
            for j in range(len(contexts)):
                if i == j:
                    continue

                train_mask = (y_context == contexts[i])
                test_mask  = (y_context == contexts[j])

                if np.sum(train_mask) < 5 or np.sum(test_mask) < 5:
                    continue

                X_train = X[train_mask]
                X_test  = X[test_mask]

                y_train = y_feat[train_mask]
                y_test  = y_feat[test_mask]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test  = scaler.transform(X_test)

                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lda.fit(X_train, y_train)

                y_pred = lda.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                scores.append(acc)

    return np.mean(scores) if len(scores) > 0 else np.nan

def get_feature_maps(design):

    if design == 'Int':
        color_map = {'Af1':360,'Am1':300,'Am2':360,'Af2':300,'Bf2':120,'Bm1':60,'Bm2':120,'Bf1':60}
        ori_map   = {'Af1':195,'Am1':195,'Am2':225,'Af2':225,'Bf2':15,'Bm1':15,'Bm2':45,'Bf1':45}
    else:
        color_map = {"Caf1":360,"Caf2":300,"Cam1":360,"Cam2":300,
                     "Cbf1":120,"Cbf2":60,"Cbm1":120,"Cbm2":60,
                     "Daf1":360,"Daf2":360,"Dam1":300,"Dam2":300,
                     "Dbf1":120,"Dbf2":120,"Dbm1":60,"Dbm2":60}
        ori_map   = {"Caf1":225,"Caf2":225,"Cam1":195,"Cam2":195,
                     "Cbf1":15,"Cbf2":15,"Cbm1":45,"Cbm2":45,
                     "Daf1":225,"Daf2":195,"Dam1":225,"Dam2":195,
                     "Dbf1":15,"Dbf2":45,"Dbm1":15,"Dbm2":45}

    return color_map, ori_map


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

        epochs = mne.read_epochs(files[0], preload=True, verbose=False)
        epochs.apply_baseline((-0.5, 0))

        X = epochs.copy().pick('eeg').get_data()
        metadata = epochs.metadata.copy()
        times = epochs.times

        with open(EVENT_CONFIG_FILE) as f:
            event_config = json.load(f)[design][session]

        rev = {v:k[5:] for k,v in event_config['probe_event_id'].items()}
        metadata['combos'] = metadata['probe_trigger'].map(rev)

        color_map, ori_map = get_feature_maps(design)
        metadata['color'] = metadata['combos'].map(color_map)
        metadata['ori']   = metadata['combos'].map(ori_map)

        valid = ~metadata['response_trigger'].isna()
        X = X[valid]
        metadata = metadata[valid]

        tc_color = []
        tc_ori = []

        for t in range(len(times)):
            try:
                Xt = X[:,:,t]

                color_cd = cross_decode_feature(
                    Xt,
                    metadata['color'].values,
                    metadata['ori'].values
                )

                ori_cd = cross_decode_feature(
                    Xt,
                    metadata['ori'].values,
                    metadata['color'].values
                )

                tc_color.append(color_cd)
                tc_ori.append(ori_cd)

            except:
                tc_color.append(np.nan)
                tc_ori.append(np.nan)

        results[design] = {
            'times': times,
            'color_crossdec': np.array(tc_color),
            'ori_crossdec': np.array(tc_ori)
        }

    return results


all_results = Parallel(n_jobs=40)( delayed(process_subject)(sid) for sid in subject_ids )

def run_group_analysis(metric_name):

    data_int = []
    data_no = []

    for r in all_results:
        if isinstance(r, dict) and 'Int' in r and 'NoInt' in r:
            data_int.append(r['Int'][metric_name])
            data_no.append(r['NoInt'][metric_name])

    data_int = np.stack(data_int)
    data_no = np.stack(data_no)

    diff = data_int - data_no
    times = all_results[0]['Int']['times']

    t_obs, p_vals = ttest_1samp(diff, 0, axis=0)
    reject, _ = fdrcorrection(p_vals, alpha=0.05)

    plt.figure()
    plt.plot(times, t_obs)
    plt.scatter(times[reject], t_obs[reject])
    plt.axhline(0)
    plt.title(f"{metric_name} (Int - NoInt)")
    plt.show()


run_group_analysis('color_crossdec')
run_group_analysis('ori_crossdec')
