"""Training-only feature integrity and recording-shift diagnostic."""
import json
from pathlib import Path
import tomllib
import numpy as np
from eeg_project.dataset import sha256_file
from eeg_project.evaluation import split_masks
from eeg_project.intensity import window_amplitude_uv

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/'reports/training_feature_quality.json'
    if output.exists():
        raise ValueError('Report exists; preserve previous results')
    config_path = ROOT/'configs/user_notes_logistic.toml'
    config = tomllib.loads(config_path.read_text())
    archive = (config_path.parent/config['archive']).resolve()
    with np.load(archive, allow_pickle=False) as data:
        mask = split_masks(data['recording'],data['subject'],config['splits'])['train']
        X,y,recording = data['X'][mask],data['y'][mask],data['recording'][mask]
        starts = data['start_seconds'][mask]
        names = data['feature_names'].tolist()
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite training features')
    means = [i for i,n in enumerate(names) if n.endswith(':mean')]
    std = [i for i,n in enumerate(names) if n.endswith(':std')]
    variance = [i for i,n in enumerate(names) if n.endswith(':variance')]
    power = [i for i,n in enumerate(names) if n.endswith('_mean_psd')]
    if len(means) != 22 or len(std) != 22 or len(variance) != 22:
        raise ValueError('Expected approved 22-channel feature schema')
    np.testing.assert_allclose(X[:,std]**2, X[:,variance], rtol=1e-10, atol=1e-30)
    if (X[:,std+variance+power] < 0).any():
        raise ValueError('Negative magnitude or PSD feature')
    amplitude = window_amplitude_uv(X,names)
    folds=[]
    effects=[]
    for name in sorted(set(recording)):
        held=recording==name
        reference=~held
        center=np.median(X[reference],axis=0)
        scale=X[reference].std(axis=0)
        valid=scale>0
        shift=np.zeros(X.shape[1])
        shift[valid]=(np.median(X[held],axis=0)[valid]-center[valid])/scale[valid]
        order=np.argsort(np.abs(shift))[::-1][:8]
        # Bounds derived only from the other training recordings. Flags are not artifact labels.
        q1,q3=np.quantile(X[reference],[.25,.75],axis=0)
        iqr=q3-q1
        flagged=((X[held]<q1-3*iqr)|(X[held]>q3+3*iqr)) & (iqr>0)
        classes={}
        for label in (0,1):
            selected=held & (y==label)
            if selected.any():
                classes[str(label)]={'windows':int(selected.sum()),'median_channel_rms_uv_p10_p50_p90':np.quantile(amplitude[selected],[.1,.5,.9]).tolist()}
        if len(classes)==2:
            effect=np.zeros(X.shape[1])
            delta=np.median(X[held & (y==1)],axis=0)-np.median(X[held & (y==0)],axis=0)
            effect[valid]=delta[valid]/scale[valid]
            effects.append(effect)
        largest=np.argsort(amplitude[held])[-3:][::-1]
        folds.append({'recording':name,'classes':classes,
                      'zero_variance_channel_windows':int((X[held][:,variance]==0).sum()),
                      'fraction_windows_any_extreme_feature':float(flagged.any(axis=1).mean()),
                      'median_fraction_features_flagged_per_window':float(np.median(flagged.mean(axis=1))),
                      'largest_median_shifts_in_other_recordings_std':[{'feature':names[i],'shift':float(shift[i])} for i in order],
                      'largest_amplitude_windows':[{'start_seconds':float(starts[held][i]),'median_channel_rms_uv':float(amplitude[held][i])} for i in largest]})
    effects=np.asarray(effects)
    # Direction consistency is descriptive only; no significance or independent-sample claim.
    consistent=np.all(effects>0,axis=0)|np.all(effects<0,axis=0)
    strength=np.min(np.abs(effects),axis=0)
    ranked=[int(i) for i in np.argsort(strength)[::-1] if consistent[i]][:10]
    report={'scope':'Configured training recordings only; no model fit or validation/test scores',
            'configuration':config,'archive_sha256':sha256_file(archive),'script_sha256':sha256_file(Path(__file__)),
            'windows':len(y),'features':len(names),'finite':True,'std_squared_matches_variance':True,
            'constant_features':[names[i] for i in np.flatnonzero(np.ptp(X,axis=0)==0)],
            'folds':folds,'seizure_recordings_compared':len(effects),
            'same_direction_feature_count':int(consistent.sum()),
            'strongest_consistent_median_class_differences':[{'feature':names[i],'standardized_differences':effects[:,i].tolist()} for i in ranked],
            'interpretation':'3-IQR flags are statistical screening, not verified EEG artifacts. Correlated features and overlapping windows are not independent evidence. No features or rows removed.'}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('windows','features','finite','constant_features','same_direction_feature_count','strongest_consistent_median_class_differences')},indent=2))
    for fold in folds:
        print(json.dumps(fold))


if __name__=='__main__':
    main()
