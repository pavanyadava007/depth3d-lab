# paper

arXiv-style technical report for depth3d-lab. Every table and every number in the prose is generated from
`results/*.json`; nothing is typed by hand.

## Build

```bash
# from the repository root
.venv/bin/python paper/make_tables.py     # -> paper/tables/*.tex and paper/numbers.tex (macros used in the prose)
.venv/bin/python paper/make_figs.py       # -> paper/figs/*.pdf
cd paper && SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt ~/.local/bin/tectonic main.tex   # -> paper/main.pdf
```

or `make -C paper` (targets: `tables`, `figs`, `pdf`, `check`, `clean`).

## Files

- `main.tex` - the report; includes `numbers.tex` and `tables/*.tex`, figures from `figs/` and `../docs/img/`
- `refs.bib` - bibliography (natbib, plainnat)
- `make_tables.py` - reads `results/*.json` the same way `scripts/make_results.py` does; optional files
  (`mono_depth_moge2*.json`, `gs_*_s<k>.json`, `synthesis_nextframe_s<k>.json`, `nextframe_train_*_s<k>.json`)
  add rows / seed columns when present and print "not measured" when absent
- `make_figs.py` - scale factor per camera and AbsRel per depth bin (matplotlib, PDF)

## Notes

- Tectonic (XeTeX based) fetches packages on demand; the first build needs network access.
- The warning "TeX rerun seems needed, but stopping at 6 passes" comes from Tectonic's bbl change check
  with natbib and is harmless; `make check` verifies that no citation is unresolved.
- No em or en dashes: page ranges in `refs.bib` are written as "pages 1 to 17" because plainnat would turn
  a hyphenated range into an en dash.
- All timings are from one NVIDIA L4. nuScenes images in the figures: CC BY-NC-SA 4.0.
