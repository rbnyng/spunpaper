# Raw Sentinel acquisition and preprocessing

This is the stage that turns sample coordinates into the per-tile Sentinel stacks (`bands.npy`, `masks.npy`, `doys.npy`, `sar_ascending.npy`, `sar_descending.npy`). This is only necessary for reproducing the results from Sentinel raw data, used with the Tessera v1 checkpoint available at https://drive.google.com/drive/folders/18RPptbUkCIgUfw1aMdMeOrFML_ZVMszn

## Contents

| File | Desc |
|---|---|
| `s2_fast_processor_small_patches.py` | Sentinel-2 L2A acquisition and patch assembly. Queries the Microsoft Planetary Computer STAC API. |
| `s1_fast_processor.py` | Sentinel-1 RTC acquisition and ROI mosaicing. |
| `spun_to_tiff_patches.py` | Turns sample coordinates into the patch AOIs the processors consume. |
| `create_mask_tiff_from_raw_data.py` | Builds validity/cloud mask rasters. |
| `check_fail_patch.py` | QA pass over completed patches and feeds `reprocess_failed_s2_patches.sh`. |
| `process_s2_patches.sh`, `process_s1_patches.sh` | Parallel over the patch list. |
| `reprocess_failed_s2_patches.sh` | Retries flagged patches. |

## Dependencies

`pystac_client`, `planetary_computer`, `stackstac`, `rasterio`, `rioxarray`,
`xarray`, `psutil`. 
