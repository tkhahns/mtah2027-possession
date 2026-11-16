# How we work

## Setup (once)

1. Register for the hackathon yourself, with the same team name as everyone else.
2. Download the match data and unzip it into `data/`. Git ignores that folder.
3. Clone the repo, follow **Regenerate the submission** in the README, and check that the
   validator says the file is valid.
4. `pip install pre-commit && pre-commit install`. It strips notebook outputs and blocks large
   files before every commit.

## Every piece of work

```bash
git checkout main && git pull
git checkout -b <your-name>/<topic>        # e.g. thao/dead-ball-rules
# ... work, commit ...
git push -u origin <your-name>/<topic>     # then open a pull request on GitHub
```

- `main` must always build a valid submission. Nothing goes in without a pull request and one
  teammate's OK.
- Keep pull requests small. Pull `main` into your branch often.
- Try ideas in `experiments/<your-name>/` (no review needed). Move what works into
  `src/possession/`.
- After every scored run, add a row to `docs/EXPERIMENTS.md`.

## Don't fool ourselves

- The public answers cover only 5 minutes (17 possessions). Use them to check that the code is
  correct, not to choose between two versions.
- Always note the full-match possession count. A version predicting 600 possessions is wrong,
  whatever its score.

## Keep the data safe

- No data, cached arrays or notebook outputs in git. The repo becomes public at submission.
- Don't paste data into hosted AI tools, and don't let coding assistants open files in `data/`.

## Before submitting (week 4)

Someone who didn't write the code clones the repo fresh, runs `src/make_submission.py`, and
checks the file is identical to the one we upload.
