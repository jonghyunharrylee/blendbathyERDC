# blendbathyERDC

The bathymetry blending class and dataset for our upcoming paper "Blending Bathymetry: Combination of Parametric Beach Tool and Celerity data sets for real-time bathymetry estimation"

## Usage 

```Python
import bathyblendingERDC as bathyinv

# 1. Initialize 
DuckBathy = bathyinv.BathyBlending(cbathy_dir='./cbathy', pbt_dir='./pbt', survey_dirr='./survey', params=params)
# 2. Perform blending
blended_bathy = DuckBathy.blend()

# 3. Post processing 
DuckBathy.plot_bathy()
DuckBathy.plot_bathy_error()
DuckBathy.plot_bathy_fitting()
DuckBathy.plot_bathy_fitting_all()
DuckBathy.plot_transects([np.where(DuckBathy.y[0,:]==600)[0][0],np.where(DuckBathy.y[0,:]==950)[0][0]])
DuckBathy.plot_obs_fitting()

#DuckBathy.plot_obslocs()
#DuckBathy.plot_cbathy_errors()
```
