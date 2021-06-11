# Apex Tracker

## What it does
### Intended Goal
The whole goal is more granular Apex tracking.
- Per-match, per-gun accuracy counting
- Map hotspots for kills and deaths
- Weapon, legend, armor, etc. winrate
- Basically all the good Halo 3 stats from back in the day

### Currently
- Interprets current weapon being used based on HUD image
- Attempts to get current ammo count based on HUD (gets 0's confused for 8's because of line through)
- Logs weapon and ammo to csv file for selected clip

## Setup
Install 'requirements.txt' (may be missing some packages, feel free to update)
Download sample clips from [here](https://drive.google.com/drive/folders/1j5LEU8r1QddmbkUHCgA62Sre8BHXb5cT?usp=sharing) and place in folder called 'test_clips'

## Using
Run `python detection_test.py` and wait for 100%