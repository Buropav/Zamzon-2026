# Old pipeline (reference only, does not run)

This folder is the source of the retired `entity_resolution.ipynb` pipeline. It is kept only as a
reference for porting individual, benchmarked ideas into the friend's pipeline (`erk_*.ipynb`), e.g.
address-number alignment (`src/extra_feats.py`), group context (`src/groups.py`), learned native-script
map (`src/translit.py`), char-n-gram / reverse / exact-key blocking (`src/two_stage.py`).

It no longer runs: its static lexicon (`src/resources/`, `src/er_lexicon.py`) was built with an external
LLM from test-set word counts and has been deleted, because the competition forbids external data
augmentation. Nothing from that lexicon may be copied into a submission. Full history: git.
