# COMP90042 A3 Fact-Checking System

This repository is organized around the final Colab submission and the report.

## Top-Level Layout

```text
assets/                 Shared static assets and versioned experiment configs.
data/                   Local course data and generated outputs; ignored by Git.
docs/                   Report, course spec mirror, meeting notes, and research archives.
model/                  Local checkpoints; ignored by Git.
src/                    Reusable code, experiment scripts, tests, and helper tools.
submissions/            Final submission zip and the exact source files used to build it.
```

Agent memory and local automation state live under hidden directories such as
`.agent/` and are not part of the submission.

## Submission

The canonical code submission archive is:

```text
submissions/COMP90042_teamname_resource.zip
```

It contains exactly three root-level files:

```text
GroupID_COMP90042_Project_2026.ipynb
requirements.txt
readme.md
```

The source files for that archive live in `submissions/resource/`.

## Report

The ACL-style report project is under `submissions/report/latex/`.

Useful report artifacts:

```text
submissions/report/latex/report.tex
submissions/report/latex/report.pdf
submissions/report/latex/overleaf_acl_report.zip
```

## Verification

Use these checks after structural changes:

```bash
pytest -q
unzip -t submissions/COMP90042_teamname_resource.zip
zipinfo -1 submissions/COMP90042_teamname_resource.zip
```
