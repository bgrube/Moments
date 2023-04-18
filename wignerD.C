#include <complex>


std::complex<double>
wignerDConj(
  const double theta,
  const double phi,
  const size_t twoJ,
  const size_t twoM1,
  const size_t twoM2
) {
  return twoJ + twoM1 + twoM2;
}
