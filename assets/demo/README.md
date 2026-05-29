# Demo Assets

The videos in `videos/` are synthetic clips generated for the demo. They are not
copied from traffic-camera footage or any third-party dataset, so they are safe to
redistribute with this repository.

`videos.json` is the stable manifest for a frontend random-rotator control. Add new
demo videos there instead of relying on filesystem directory listing.

Regenerate the clips with:

```bash
python3 scripts/generate_demo_videos.py
```
