#include <cassert>
#include <cmath>
#include <iostream>

#include "TLorentzRotation.h"
#include "TLorentzVector.h"
#include "TMath.h"

using namespace std;


// define sin and cos functions that take degrees as arguments
#define sind(x) (sin(fmod((x), 360) * M_PI / 180))
#define cosd(x) (cos(fmod((x), 360) * M_PI / 180))


class MyFSMath{

	public:

	// for beam + target -> X + recoil and X -> a + b (see FSBasic/FSMath.h)
	//      D               R     C        R    A   B
	// azimuthal angle of particle a in X Gottfried-Jackson RF [rad]
	//   GJPHI(a; b; recoil; beam)
	static
	double
	gjphi(
		const double PxPA, const double PyPA, const double PzPA, const double EnPA,
		const double PxPB, const double PyPB, const double PzPB, const double EnPB,
		const double PxPC, const double PyPC, const double PzPC, const double EnPC,
		const double PxPD, const double PyPD, const double PzPD, const double EnPD
	) {
		TLorentzVector       pa    (PxPA, PyPA, PzPA, EnPA);
		const TLorentzVector pb    (PxPB, PyPB, PzPB, EnPB);
		TLorentzVector       recoil(PxPC, PyPC, PzPC, EnPC);
		TLorentzVector       beam  (PxPD, PyPD, PzPD, EnPD);
		const TLorentzVector resonance = pa + pb;
		// boost all needed four-vectors to the resonance rest frame
		const TVector3 boostVect = -resonance.BoostVector();
		pa.Boost    (boostVect);
		recoil.Boost(boostVect);
		beam.Boost  (boostVect);
		// rotate so beam is aligned along the z-axis
		pa.RotateZ(-beam.Phi());    recoil.RotateZ(-beam.Phi());
		pa.RotateY(-beam.Theta());  recoil.RotateY(-beam.Theta());
		// rotate so recoil is in the xz-plane
		pa.RotateZ(-recoil.Phi());
		return pa.Phi();
	}


	// for beam + target -> X + recoil and X -> a + b
	//      D               R     C        R    A   B
	// azimuthal angle of photon polarization vector in R rest frame [rad]
	// BIGPHI(a; b; recoil; beam)
	// code taken from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
	static
	double
	bigPhi(
		const double PxPC, const double PyPC, const double PzPC, const double EnPC,
		const double PxPD, const double PyPD, const double PzPD, const double EnPD,
		const double polAngle  // [deg]
	) {
		TLorentzVector recoil(PxPC, PyPC, PzPC, EnPC);
		TLorentzVector beam  (PxPD, PyPD, PzPD, EnPD);
		const TVector3 yAxis = (beam.Vect().Unit().Cross(-recoil.Vect().Unit())).Unit();  // normal of production plane in lab frame
		const TVector3 eps(1, 0, 0);  // reference beam polarization vector at 0 degrees in lab frame
		double Phi = polAngle * TMath::DegToRad() + atan2(yAxis.Dot(eps), beam.Vect().Unit().Dot(eps.Cross(yAxis)));  // angle in lab frame [rad]
		// ensure [-pi, +pi] range
		while (Phi > TMath::Pi()) {
			Phi -= TMath::TwoPi();
		}
		while (Phi < -TMath::Pi()) {
			Phi += TMath::TwoPi();
		}
		assert(polAngle == 45);
		return Phi;
	}

};
