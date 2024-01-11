# Repository for moment analysis of two-(pseudo)scalar meson systems #

Currently, the repository contains three test cases:

* **Diffractive scattering** (`testMomentsDiffractive.py`): For this process, the intensity distribution is a function of the decay angles $\Omega = (\theta, \phi)$ of one of the daughters particles and is described by a set of moments $H(L, M)$.
* **Photoproduction with linearly polarized photon beam** (`testMomentsPhotoProd.py`): For this process, the intensity distribution is a function of $\Omega$ and of the azimuthal angle $\Phi$ of the beam-photon polarization with respect to the production plane. The intensity is described by three sets of moments $H_{0, 1, 2}(L, M)$.
* **Photoproduction with linearly polarized photon beam and sideband subtraction** (`testMomentsPhotoProdWeighted.py`): In addition to the signal sample that is identical to the above case, a background sample is added that has a different angular distribution and is subtracted using side bands in a hypothetical discriminating variable. The signal follows a Gaussian in the discriminating variable, whereas the background is uniformly distributed.
