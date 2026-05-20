# Submission Artifacts

`COMP90042_Group_131_resource.zip` is the canonical code submission archive. It
contains only:

- `Group_131_COMP90042_Project_2026.ipynb`
- `requirements.txt`
- `readme.md`

The source files used to build that zip are under `submissions/resource/`.
Build the zip from that directory with an explicit file list:

```bash
cd submissions/resource
zip -X ../COMP90042_Group_131_resource.zip \
  Group_131_COMP90042_Project_2026.ipynb \
  requirements.txt \
  readme.md
```

The final report PDF is tracked as `submissions/COMP90042_Group_131.pdf`.
Report source and Overleaf packaging are under `submissions/report/latex/`.
