# blendbathyERDC

The bathymetry blending class and dataset for our upcoming paper "Blending Bathymetry: Combination of image-derived parametric approximations and celerity data sets for nearshore bathymetry estimation". You can read a preprint from arXiv: https://arxiv.org/abs/2311.01085

We also added an example notebook (see BlendingExampleFRF.ipynb above) that can reproduce all the results and figures from the paper. You can test it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg 'Open In Colab')](https://colab.research.google.com/github/jonghyunharrylee/blendbathyERDC/blob/main/BlendingExampleFRF.ipynb)

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
