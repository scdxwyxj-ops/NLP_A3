# Round18 Run Manifest Schema

Every Round18 exploratory or strict run must produce a `run_manifest.json` with these fields.

```json
{
  "round": "round18",
  "run_id": "",
  "status": "strict-candidate | diagnostic-only | rejected",
  "mode": "STRICT | DIAGNOSTIC",
  "stage": "",
  "agent_id": "",
  "agent_role": "",
  "timestamp_utc": "",
  "git_commit": "",
  "git_status_short": "",
  "command": "",
  "working_directory": "",
  "config_path": "",
  "config_hash": "",
  "random_seed": 0,
  "cv_seed": 0,
  "fold_policy": "",
  "model_names": [],
  "input_files": [
    {
      "path": "",
      "sha256": "",
      "split": "train | dev | test | evidence | code | config | current_run_artifact",
      "labels_used": false
    }
  ],
  "forbidden_input_scan": {
    "passed": false,
    "notes": ""
  },
  "output_files": [],
  "metrics": {},
  "runtime": {
    "wall_seconds": null,
    "peak_memory_mb": null,
    "device": ""
  },
  "data_flow_summary": "",
  "split_isolation_summary": "",
  "leakage_risk": "low | medium | high",
  "reproducibility_risk": "low | medium | high",
  "notes": ""
}
```

Missing manifest fields are a review failure unless the agent explains why the field is not applicable.

