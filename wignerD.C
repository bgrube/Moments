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


// optimized function that calculates (-1)^n
inline
int
powMinusOne(const int exponent)
{
	if (exponent & 0x1)  // exponent is odd
		return -1;
	else                 // exponent is even
		return +1;
}


// complex conjugated Wigner d-function refl^D^J_{M1 M2}^*(phi, theta, 0) in reflectivity basis
// Wigner D-function D^J_{M1 M2}^*(phi, theta, 0) in canonical basis
// !NOTE! quantum numbers J, M1, and M2 are given in units of hbar/2
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
	const double dFuncVal = constTerm * sumTerm;

	// calculate value of D function D^J_{M1 M2}(phi, theta, 0) in canonical basis
	const double               arg      = ((double)twoM1 / 2) * phi;
	const std::complex<double> DFuncVal = std::exp(std::complex<double>(0, -arg)) * dFuncVal;

	return DFuncVal;
}


// complex conjugated Wigner D-function refl^D^J_{M1 M2}^*(phi, theta, 0) in reflectivity basis
// !NOTE! quantum numbers J, M1, and M2 are given in units of hbar/2
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
	std::complex<double> DFuncVal;
	const int reflFactor = refl * P * powMinusOne((twoJ - twoM1) / 2);
	if (twoM1 == 0) {
		if (reflFactor == +1) {
			DFuncVal = 0;
		} else {
			DFuncVal = wignerD(twoJ, 0, twoM2, phi, theta);
		}
	} else {
		DFuncVal = (1 / std::sqrt(2))
		           * (                       wignerD(twoJ, +twoM1, twoM2, phi, theta)
		              - (double)reflFactor * wignerD(twoJ, -twoM1, twoM2, phi, theta));
	}

	return std::conj(DFuncVal);
}


double
ylm(
	const int    l,
	const int    m,
	const double theta  // [rad]
) {
	// !Note! ROOT::Math::sph_legendre works only for non-negative m values
	return ROOT::Math::sph_legendre(l, std::abs(m), theta) * ((m >= 0) ? 1 : powMinusOne(std::abs(m)));
}

std::complex<double>
Ylm(
	const int    l,
	const int    m,
	const double theta,  // [rad]
	const double phi     // [rad]
) {
	// compare with Wigner D-function
	// const complex<double> delta = ylm(l, m, theta) * std::exp(std::complex<double>(0.0, 1.0) * (m * phi)) - std::sqrt((2 * l + 1) / (4 * TMath::Pi())) * std::conj(wignerD(2 * l, 2 * m, 0, phi, theta));
	// if (std::abs(delta) > 1e-15)
	// 	cout << "!!! " << delta << std::endl;
	return ylm(l, m, theta) * std::exp(std::complex<double>(0.0, 1.0) * (m * phi));
}

double
ReYlm(
	const int    l,
	const int    m,
	const double theta,  // [rad]
	const double phi     // [rad]
) {
  return ylm(l, m, theta) * std::cos(m * phi);
}

double
ImYlm(
	const int    l,
	const int    m,
	const double theta,  // [rad]
	const double phi     // [rad]
) {
  return ylm(l, m, theta) * std::sin(m * phi);
}


// basis functions for physical moments; Eq. (175)
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
	const double norm = std::sqrt((2 * L + 1) / (4 * TMath::Pi())) * ((M == 0) ? 1 : 2) * ylm(L, M, theta);
	switch (momentIndex) {
	case 0:
		return norm * std::cos(M * phi);
	case 1:
		return norm * polarization * std::cos(M * phi) * std::cos(2 * Phi);
	case 2:
		return norm * I * polarization * std::sin(M * phi) * std::sin(2 * Phi);
	default:
		throw std::domain_error("f_phys() unknown moment index.");
	}
}


// vector version that calculates function value for each entry in the input vectors
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
	for (size_t i = 0; i < theta.size(); ++i) {
		fcnValues[i] = f_phys(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}


// vector version that calculates function value for each entry in the input vectors; OpenMP version
std::vector<std::complex<double>>
f_phys_omp(
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
	#pragma omp parallel for
	for (size_t i = 0; i < theta.size(); ++i) {
		fcnValues[i] = f_phys(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}


// basis functions for measured moments; Eq. (176)
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
	const std::complex<double> norm = (1 / TMath::Pi()) * std::sqrt((4 * TMath::Pi()) / (2 * L + 1)) * std::conj(Ylm(L, M, theta, phi));
	switch (momentIndex) {
	case 0:
		return norm / 2.0;
	case 1:
		return norm * std::cos(2 * Phi) / polarization;
	case 2:
		return norm * std::sin(2 * Phi) / polarization;
	default:
		throw std::domain_error("f_meas() unknown moment index.");
	}
}


// vector version that calculates function value for each entry in the input vectors
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
	for (size_t i = 0; i < theta.size(); ++i) {
		fcnValues[i] = f_meas(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}


// vector version that calculates function value for each entry in the input vectors; OpenMP version
std::vector<std::complex<double>>
f_meas_omp(
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
	#pragma omp parallel for
	for (size_t i = 0; i < theta.size(); ++i) {
		fcnValues[i] = f_meas(momentIndex, L, M, theta[i], phi[i], Phi[i], polarization);
	}
	return fcnValues;
}


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
