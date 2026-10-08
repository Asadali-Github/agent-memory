# Moved — 7 August 2026

The July-20 mock run backup that lived here has been archived to:

    _archive/10_mock_runs_backup_2026-07-20/

Contents: `custom_ds.jsonl` (1,290 mock records) and `manifest.json`.

It is referenced twice in `results/audit_2026_08_02.md` (§ Mode A data, and the
1,290-shared-key comparison in item 203) as the pre-fix comparison baseline used to
verify the provenance fix. Nothing was altered — the files are byte-identical to
what was here, only relocated. Restore with:

    mv _archive/10_mock_runs_backup_2026-07-20/* results/runs_mock_backup_jul20/

This is mock data. It must never be used as an experimental result.
