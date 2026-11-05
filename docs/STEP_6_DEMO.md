# Step 6: EEG Recording Explorer

Run from the repository root with the virtual environment active:

```bash
python scripts/train_baseline.py
python scripts/evaluate_intensity.py
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The first two commands prepare local artifacts. Skip them if the model,
amplitude reference, and sample already exist. Open the local address printed
by Streamlit.

Select the held-out CHB01 recording or upload a compatible EDF (256 Hz, same
bipolar channel order after duplicate handling, at most 50 MiB). Use the window
slider and channel selector to inspect the filtered trace. The timeline shows
the model score and median channel RMS relative to training background. Scores
are produced at the end of each complete 10-second window. Playback is manual
and uses a recorded file.

The app uses the same causal filtering and feature extraction as training.
It displays exploratory scores without alerts. Keep the local Joblib model
artifact from this project's training script. EDF uploads are processed locally.
