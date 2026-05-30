# Demo Assets

`videos.json` is the stable manifest for the frontend sample library. Add new demo
videos there instead of relying on filesystem directory listing.

Generate TrafficDB demo clips from local AVI files with:

```bash
python3 backend/scripts/convert_trafficdb_demo_videos.py
```

The script randomly samples 10 AVI files from `assets/video/`, converts them to
browser-friendly MP4 files in `videos/`, and rewrites `videos.json`. The default
seed is fixed so repeated runs are reproducible; pass `--seed` to rotate the
sample set.

The older synthetic clip generator is still available for fallback assets:

```bash
python3 backend/scripts/generate_demo_videos.py
```
