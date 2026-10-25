# artifacts/

Outputs of the lesson experiments. Everything here except this file is git-ignored and can be regenerated.

```
artifacts/<chapter>/<lesson>/<name>.png    figure
artifacts/<chapter>/<lesson>/<name>.json   summary numbers of the same run
```

Paths come from `rlcore.artifacts` (`artifact_path`, `save_figure`, `save_json`), so a script writes to the same place whatever folder it is started from.
