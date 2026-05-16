# Submission Artifacts

`COMP90042_teamname_resource.zip` is the canonical code submission archive. It
contains only:

- `GroupID_COMP90042_Project_2026.ipynb`
- `requirements.txt`
- `readme.md`

The source files used to build that zip are under `submissions/resource/`.
Build the zip from that directory with an explicit file list:

```bash
cd submissions/resource
zip -X ../COMP90042_teamname_resource.zip \
  GroupID_COMP90042_Project_2026.ipynb \
  requirements.txt \
  readme.md
```

The PDF report is tracked under `docs/report/latex/report.pdf`; report source
and Overleaf packaging are also under `docs/report/latex/`.
