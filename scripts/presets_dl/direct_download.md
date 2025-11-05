## Notes:
kvraudio have anti bot measures. To download it, you need to create `kvr_cookies.txt` file in this subfolder with your session cookies. Make sure to remove cloudlfare's cookies from it. (`cf_clearance` line). If no file is found, the script will skip kvraudio downloads.
> Note: sometimes, kvraudio is blocked by cloudflare even with cookies, so the downloads may fail.

presetshare also needs cookies to download presets. Create `presetshare_cookies.txt` file in this subfolder with your session cookies.


## Downloaded and extracted automatically:
> by running `scripts/scripting/scrapping.ipynb`

[newloops](https://newloops.com/pages/free-surge-presets-new-loops): 25 presets. ([ddl](https://demos.newloops.com/New_Loops-Surge_Presets.zip))

[kvraudio](https://www.kvraudio.com/product/free-surge-presets-vol-2-by-damon-armani): 36 presets ([ddl](https://damon-armani.com/wp-content/uploads/Damon-Armani-Surge-Presets-Vol-2.zip))

[rekkerd](https://rekkerd.org/patches/plug-in/surge/): 6 fichiers zips de presets. ([0](https://rekkerd.org/bin/presets/inigo_kennedy_03.zip) [1](https://rekkerd.org/bin/presets/inigo_kennedy_02.zip) [2](https://rekkerd.org/bin/presets/inigo_kennedy_01.zip) [3](https://rekkerd.org/bin/presets/NICK_MORITZ_Surge_Bank_v.1.rar) [4](https://rekkerd.org/bin/presets/Bronto_Scorpio_Surge_2.zip) [5](https://rekkerd.org/bin/presets/Bronto_Scorpio_Surge.zip))

[github](https://github.com/surge-synthesizer/surge-synthesizer.github.io/wiki/Additional-Content): plusieurs packs de presets. ([0](https://raw.githubusercontent.com/surge-synthesizer/surge-extra-content/main/Website/wiki/Additional%20Content/Philippe%20Favre%20Patches%202024.zip) [1](https://raw.githubusercontent.com/surge-synthesizer/surge-extra-content/main/Website/wiki/Additional%20Content/Psiome-Album.7z) 

## Works only with cookies:
kvraudio - https://www.kvraudio.com/product/surge-by-surge-synth-team/downloads
presetshare - https://www.presetshare.com/presets?query=&instrument=7&page=1

## Notes:
patool uses system's archive tools (unzip, unrar, 7z, tar, etc). Make sure you have what is needed for common archive formats (here .zip, .rar, .7z, .tar.gz).