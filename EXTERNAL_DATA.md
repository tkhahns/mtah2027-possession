# External training data

The frozen models use only public matches other than the organiser-provided challenge match. The
raw external files are not copied into this repository: they total more than 3 GB, are unnecessary
for inference, and remain governed by their source licences. Reproducible download scripts and all
loaders/training code are included.

## Metrica Sports sample data

- Source: [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data)
- Publisher/annotator: Metrica Sports
- Inventory: Sample Games 1, 2 and 3; tracking plus the provider's timed event records
- Local destination: `external/metrica/`
- Fetch command: `python src/training/fetch_metrica.py`

The event records are converted algorithmically into dead/live and controlled-possession targets.
No labels from the challenge match are used.

## IDSSE

- Source: [pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data)
- Licence: CC BY 4.0
- Publisher/annotations: the public provider dataset, including observed tracking, ball status and
  ball-possession fields
- Inventory: `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9`
- Local destination: `external/idsse/`
- Fetch command: `python src/training/fetch_idsse.py`

Short bracketed touch reversals are merged during target construction so that supervision reflects
sustained control rather than every instantaneous contact.

## Retraining entry points

```bash
python src/training/possession_models/train_restart_state.py --include-idsse
python src/training/possession_models/train_spell_candidate.py
python src/training/possession_models/train_set_temporal.py --include-idsse
```

Training and model selection use whole-match grouping. Synthetic hidden-ball and messy-tracklet
views are generated in code from these external matches.
