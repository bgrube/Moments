#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <omp.h>

#include "Math/SpecFuncMathMore.h"
#include "TMath.h"


// need typedef because templates are not allowed in TFormula expressions
// see https://root-forum.cern.ch/t/trying-to-define-imaginary-error-function/50032/10
typedef std::complex<double> complexT;


const std::complex<double> I = std::complex<double>(0, 1);


// test OpenMP compilation and thread spawning
void
testOpenMp()
{
	// Fork a team of threads giving them their own copies of variables
	int nmbThreads, threadId;
	#pragma omp parallel private(nmbThreads, threadId)
	{
		// Obtain thread number
		threadId = omp_get_thread_num();
		#pragma omp critical
		std::cout << "Hello World from thread = " <<  threadId << std::endl;
		// Only master thread does this
		if (threadId == 0) {
			nmbThreads = omp_get_num_threads();
			#pragma omp critical
			std::cout << "Number of threads = " <<  nmbThreads << std::endl;
		}
	}
	// All threads join master thread and disband
}


// returns number of threads used by OpenMP
// number of threads can be controlled by setting the environment variable OMP_NUM_THREADS
int
getNmbOpenMpThreads()
{
	int nmbThreads = 1;
	#pragma omp parallel
	{
		#pragma omp single
		nmbThreads = omp_get_num_threads();
	}
	return nmbThreads;
}


// function that calculates (-1)^n
inline
int
powMinusOne(const int exponent)
{
	if (exponent & 0x1)  // exponent is odd
		return -1;
	else                 // exponent is even
		return +1;
}


// Wigner D-function D^J_{M1 M2}^*(phi, theta, 0) in canonical basis
//!NOTE! quantum numbers J, M1, and M2 are given in units of hbar/2
std::complex<double>
wignerD(
	const int    twoJ,
	const int    twoM1,
	const int    twoM2,
	const double phi,   // [rad]
	const double theta  // [rad]
) {
	// swap spin projections for negative angle
	int    _twoM1    = twoM1;
	int    _twoM2    = twoM2;
	double thetaHalf = theta / 2;
	if (theta < 0) {
		thetaHalf = std::abs(thetaHalf);
		std::swap(_twoM1, _twoM2);
	}

	const double cosThetaHalf = std::cos(thetaHalf);
	const double sinThetaHalf = std::sin(thetaHalf);

	// calculate value of small-d function d^J_{M1 M2}(theta) in canonical basis
	const int jpm = (twoJ + _twoM1) / 2;
	const int jpn = (twoJ + _twoM2) / 2;
	const int jmm = (twoJ - _twoM1) / 2;
	const int jmn = (twoJ - _twoM2) / 2;
	const double kk  =   TMath::Factorial(jpm)
	                   * TMath::Factorial(jmm)
	                   * TMath::Factorial(jpn)
	                   * TMath::Factorial(jmn);
	const double constTerm = powMinusOne(jpm) * std::sqrt(kk);

	double sumTerm = 0;
	const int mpn  = (_twoM1 + _twoM2) / 2;
	const int kMin = std::max(0,   mpn);
	const int kMax = std::min(jpm, jpn);
	for (int k = kMin; k <= kMax; ++k) {
		const int kmn1 = 2 * k - (_twoM1 + _twoM2) / 2;
		const int jmnk = twoJ + (_twoM1 + _twoM2) / 2 - 2 * k;
		const int jmk  = (twoJ + _twoM1) / 2 - k;
		const int jnk  = (twoJ + _twoM2) / 2 - k;
		const int kmn2 = k - (_twoM1 + _twoM2) / 2;
		const double factor = (  TMath::Factorial(k   )
		                       * TMath::Factorial(jmk )
		                       * TMath::Factorial(jnk )
		                       * TMath::Factorial(kmn2)) / powMinusOne(k);
		sumTerm += std::pow(cosThetaHalf, kmn1) * std::pow(sinThetaHalf, jmnk) / factor;
	}
	const double dFcnVal = constTerm * sumTerm;

	// calculate value of D function D^J_{M1 M2}(phi, theta, 0) in canonical basis
	const double               arg      = ((double)twoM1 / 2) * phi;
	const std::complex<double> DFcnVal = std::exp(std::complex<double>(0, -arg)) * dFcnVal;

	return DFcnVal;
}


// complex conjugated Wigner D-function refl^D^J_{M1 M2}^*(phi, theta, 0) in reflectivity basis
//!NOTE! quantum numbers J, M1, and M2 are given in units of hbar/2
std::complex<double>
wignerDReflConj(
	const int    twoJ,
	const int    twoM1,
	const int    twoM2,
	const int    P,
	const int    refl,
	const double phi,   // [rad]
	const double theta  // [rad]
) {
	std::complex<double> DFcnVal;
	const int reflFactor = refl * P * powMinusOne((twoJ - twoM1) / 2);
	if (twoM1 == 0) {
		if (reflFactor == +1) {
			DFcnVal = 0;
		} else {
			DFcnVal = wignerD(twoJ, 0, twoM2, phi, theta);
		}
	} else {
		DFcnVal = (1 / std::sqrt(2))
		           * (                       wignerD(twoJ, +twoM1, twoM2, phi, theta)
		              - (double)reflFactor * wignerD(twoJ, -twoM1, twoM2, phi, theta));
	}

	return std::conj(DFcnVal);
}


// spherical harmonics; theta-dependent part
// corresponds to Wigner d-function d^l_{m 0}(theta) (see Eq. (12) in https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3)
double
ylm(
	const int    l,
	const int    m,
	const double theta  // [rad]
) {
	//!NOTE! ROOT::Math::sph_legendre works only for non-negative m values
	return ROOT::Math::sph_legendre(l, std::abs(m), theta) * ((m >= 0) ? 1 : powMinusOne(std::abs(m)));
}

// spherical harmonics
std::complex<double>
Ylm(
	const int    l,
	const int    m,
	const double theta,  // [rad]
	const double phi     // [rad]
) {
	// compare with Wigner D-function
	// const std::complex<double> delta = ylm(l, m, theta) * std::exp(std::complex<double>(0.0, 1.0) * (m * phi)) - std::sqrt((2 * l + 1) / (4 * TMath::Pi())) * std::conj(wignerD(2 * l, 2 * m, 0, phi, theta));
	// if (std::abs(delta) > 1e-15)
	// 	cout << "!!! " << delta << std::endl;
	return ylm(l, m, theta) * std::exp(std::complex<double>(0.0, 1.0) * (m * phi));
}

// real part of spherical harmonics
double
ReYlm(
	const int    l,
	const int    m,
	const double theta,  // [rad]
	const double phi     // [rad]
) {
  return ylm(l, m, theta) * std::cos(m * phi);
}

// imaginary part of spherical harmonics
double
ImYlm(
	const int    l,
	const int    m,
	const double theta,  // [rad]
	const double phi     // [rad]
) {
  return ylm(l, m, theta) * std::sin(m * phi);
}


// basis functions for (polarized) photoproduction moments
// equation numbers below refer to https://halldweb.jlab.org/doc-private/DocDB/ShowDocument?docid=6124&version=3
// see https://stackoverflow.com/a/58043564 for an example of implementing a NumPy function in C++

// basis functions for physical moments; Eq. (175)
// scalar version that calculates function value for a single event
std::complex<double>
f_phys(
	const int    momentIndex,  // 0, 1, or 2
	const int    L,
	const int    M,
	const double theta,  // [rad]
	const double phi,    // [rad]
	const double Phi,    // [rad]
	const double polarization
) {
	const double commonTerm = std::sqrt((2 * L + 1) / (4 * TMath::Pi())) * ((M == 0) ? 1 : 2) * ylm(L, M, theta);
	switch (momentIndex) {
	case 0:
		return     commonTerm * std::cos(M * phi);
	case 1:
		return     commonTerm * std::cos(M * phi) * polarization * std::cos(2 * Phi);
	case 2:
		return I * commonTerm * std::sin(M * phi) * polarization * std::sin(2 * Phi);
	default:
		throw std::domain_error("f_phys() unknown moment index.");
	}
}

// vector version that calculates basis-function value for each entry
// in the input vectors for a constant polarization value
std::vector<std::complex<double>>
f_phys(
	const int                  momentIndex,  // 0, 1, or 2
	const int                  L,
	const int                  M,
	const std::vector<double>& theta,  // [rad]
	const std::vector<double>& phi,    // [rad]
	const std::vector<double>& Phi,    // [rad]
	const double               polarization
) {
	// assume that theta, phi, and Phi have the same length
	const size_t nmbEvents = theta.size();
	std::vector<std::complex<double>> fcnValues(nmbEvents);
	// multi-threaded loop over events using OpenMP
	#pragma omp parallel for
	for (size_t i = 0; i < nmbEvents; ++i) {
		fcnValues[i] = f_phys(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}

// vector version that calculates basis-function value for each entry
// in the input vectors for event-dependent polarization values
std::vector<std::complex<double>>
f_phys(
	const int                  momentIndex,  // 0, 1, or 2
	const int                  L,
	const int                  M,
	const std::vector<double>& theta,  // [rad]
	const std::vector<double>& phi,    // [rad]
	const std::vector<double>& Phi,    // [rad]
	const std::vector<double>& polarizations
) {
	// assume that theta, phi, and Phi have the same length
	const size_t nmbEvents = theta.size();
	std::vector<std::complex<double>> fcnValues(nmbEvents);
	// multi-threaded loop over events using OpenMP
	#pragma omp parallel for
	for (size_t i = 0; i < nmbEvents; ++i) {
		fcnValues[i] = f_phys(momentIndex, L, M, theta[i], phi[i], Phi[i], polarizations[i]);
	}
	return fcnValues;
}


// basis functions for measured moments; Eq. (176)
// scalar version that calculates function value for a single event
std::complex<double>
f_meas(
	const int    momentIndex,  // 0, 1, or 2
	const int    L,
	const int    M,
	const double theta,  // [rad]
	const double phi,    // [rad]
	const double Phi,    // [rad]
	const double polarization
) {
	const std::complex<double> commonTerm = (1 / TMath::Pi()) * std::sqrt((4 * TMath::Pi()) / (2 * L + 1)) * std::conj(Ylm(L, M, theta, phi));
	switch (momentIndex) {
	case 0:
		return commonTerm / 2.0;
	case 1:
		return commonTerm * std::cos(2 * Phi) / polarization;
	case 2:
		return commonTerm * std::sin(2 * Phi) / polarization;
	default:
		throw std::domain_error("f_meas() unknown moment index.");
	}
}

// vector version that calculates basis-function value for each entry
// in the input vectors for a constant polarization value
std::vector<std::complex<double>>
f_meas(
	const int                  momentIndex,  // 0, 1, or 2
	const int                  L,
	const int                  M,
	const std::vector<double>& theta,  // [rad]
	const std::vector<double>& phi,    // [rad]
	const std::vector<double>& Phi,    // [rad]
	const double               polarization
) {
	// assume that theta, phi, and Phi have the same length
	const size_t nmbEvents = theta.size();
	std::vector<std::complex<double>> fcnValues(nmbEvents);
	// multi-threaded loop over events using OpenMP
	#pragma omp parallel for
	for (size_t i = 0; i < nmbEvents; ++i) {
		fcnValues[i] = f_meas(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}

// vector version that calculates basis-function value for each entry
// in the input vectors for event-dependent polarization values
std::vector<std::complex<double>>
f_meas(
	const int                  momentIndex,  // 0, 1, or 2
	const int                  L,
	const int                  M,
	const std::vector<double>& theta,  // [rad]
	const std::vector<double>& phi,    // [rad]
	const std::vector<double>& Phi,    // [rad]
	const std::vector<double>& polarizations
) {
	// assume that theta, phi, and Phi have the same length
	const size_t nmbEvents = theta.size();
	std::vector<std::complex<double>> fcnValues(nmbEvents);
	// multi-threaded loop over events using OpenMP
	#pragma omp parallel for
	for (size_t i = 0; i < nmbEvents; ++i) {
		fcnValues[i] = f_meas(momentIndex, L, M, theta[i], phi[i], Phi[i], polarizations[i]);
	}
	return fcnValues;
}
