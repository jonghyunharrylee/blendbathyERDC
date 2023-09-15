# blendbathyERDC

The bathymetry blending class and dataset for our upcoming paper "Blending Bathymetry: Combination of Parametric Beach Tool and Celerity data sets for real-time bathymetry estimation"

## Usage 

```Python
import bathyblendingERDC as bathyinv
params = {'use_testdata':0} # define additional parameters use 032717 dataset 
DuckBathy = bathyinv.BathyBlending(cbathy_dir='./cbathy', pbt_dir='./pbt', survey_dirr='./survey', params=params)
blended_bathy = DuckBathy.blend()
```
