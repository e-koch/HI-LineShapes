# HI-LineShapes
Which model best describes HI line shapes in M31 and M33?

This repo has the code for Koch et al. (2021) (ADD PAPER LINK) for fitting a single opaque model
(see [Braun et al. 2009](https://ui.adsabs.harvard.edu/abs/2009ApJ...695..937B/abstract)) versus
a multi-component Gaussian model for 21-cm HI spectra in M31 and M33 on ~100 pc scales.


Of particular note is the autonomous Gaussian decomposition from [Lindner et al 2015](https://ui.adsabs.harvard.edu/abs/2015AJ....149..138L/abstract) (gausspy, https://github.com/gausspy/gausspy)
and [Riener et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...628A..78R/abstract) (gausspyplus, https://github.com/mriener/gausspyplus).
Our version of this is adapted from these works in `AGD_decomposer.py` with a few crucial changes for our
HI emission fitting:

* We are not using the learning step to find the smoothing scales. Instead, we loop over a range of smoothing scales and look for additional components from the residuals while smoothing progressively larger scales.
* From the final fit of the decomposition, we include a component significance test where we remove estimated components whose integrated intensity is less than 5-sigma.
* Additionally, we use a model selection test to ensure the decomposition estimates use the least number of components. We remove the least significant component and determine if the delta BIC from the initial and updated fits changes by more than 10. We iteratively remove components until the final fit converges in delta BIC.

While our method performed well for the Local Group HI data, caution should be used when applying this technique to other data! This method has not been extensively tested on different data sets. Consider first using gausspy and gausspyplus which have been applied to a wider range of data.

