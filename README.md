# Repository for moment analysis of two-(pseudo)scalar meson systems #

Currently, the repository contains 2 analyses that are based on real data:

* **Unpolarized photoproduction of pi+ pi-**: Unpolarized moments of the $\pi^+ \pi^-$ final state are calculated from GlueX low-energy data in the mass range from 0.4 to 1.4 GeV and compared to results from a CLAS analysis or from a mass-independent PWA.
* **Polarized photoproduction of pi+ pi-**: Moments for linearly polarized photoproduction of the $\pi^+ \pi^-$ final state are calculated from real data and compared to the moments calculated from the partial-wave amplitudes obtained from a mass-independent PWA.
* Both analyses are implemented in a set of 6 scripts
  * `photoProdCalcMoments.py` calculates the moments and writes them to disk.
  * `photoProdMomentsPwa.py` calculates the moments from the result of a mass-independent PWA and writes the to disk.
  * `photoProdPlotMoments.py` reads moments from disk and plots all moments.
  * `photoProdWeightData.py` weights accepted phase-space data with intensity distribution defined by moment values.
  * `photoProdCombineMoments.py` combines moment values from independent data sets, e.g. the 4 beam-polarization orientations.
  * `overlayMoments.py` creates overlay plots of moments.

There are three test cases with contrived acceptance:

* **Diffractive scattering** (`testMomentsDiffractive.py`): For this process, the intensity distribution is a function of the decay angles $\Omega = (\theta, \phi)$ of one of the daughters particles and is described by a set of moments $H(L, M)$.
* **Photoproduction with linearly polarized photon beam** (`testMomentsPhotoProd.py`): For this process, the intensity distribution is a function of $\Omega$ and of the azimuthal angle $\Phi$ of the beam-photon polarization with respect to the production plane. The intensity is described by three sets of moments $H_{0, 1, 2}(L, M)$.
* **Photoproduction with linearly polarized photon beam and sideband subtraction** (`testMomentsPhotoProdWeighted.py`): In addition to the signal sample that is identical to the above case, a background sample is added that has a different angular distribution and is subtracted using side bands in a contrived discriminating variable. The signal follows a Gaussian in the discriminating variable, whereas the background is uniformly distributed.

There are two test cases with realistic GlueX acceptance:

* **Polarized photoproduction of rho**  (`testMomentsPhotoProdRho.py`): Moments for linearly polarized photoproduction of $\rho(770)$ are extracted from signal Monte Carlo data generated using the assumptions of natural-parity exchange and s-channel helicity conservation. No background subtraction is performed. No event weighting is performed; background from RF sidebands is removed using Chi^2-ranking approach.
* **Polarized photoproduction of eta pi0**  (`testMomentsPhotoProdEtaPi0.py`): Moments for linearly polarized photoproduction of the $\eta \pi^0$ final state are calculated from partial-wave amplitudes. The amplitudes were obtained by performing a partial-wave analysis of signal Monte Carlo data generated using model with 7 partial waves, where the $S_0^-$ and $S_0^+$ waves contain a Breit-Wigner amplitude for the $a_0(980)$ and the $D_{-1, 0, +1}^-$ and $D_{-2, +2}^+$ waves contain a Breit-Wigner amplitude for the $a_2(1320)$.  No background was simulated. Event weighting is performed to remove background from RF sidebands.
