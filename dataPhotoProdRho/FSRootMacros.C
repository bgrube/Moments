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

	// GJPHI(a; b; recoil; beam) returns azimuthal angle of particle a in X Gottfried-Jackson RF [rad]
	// for beam + target -> X + recoil and X -> a + b
	//     D                    C               A   B
	// copied from FSRoot/FSBasic/FSMath.{h,C}
	static
	double
	gjphi(
		const double PxPA, const double PyPA, const double PzPA, const double EnPA,
		const double PxPB, const double PyPB, const double PzPB, const double EnPB,
		const double PxPC, const double PyPC, const double PzPC, const double EnPC,
		const double PxPD, const double PyPD, const double PzPD, const double EnPD
	) {
		TLorentzVector       pA    (PxPA, PyPA, PzPA, EnPA);
		const TLorentzVector pB    (PxPB, PyPB, PzPB, EnPB);
		TLorentzVector       recoil(PxPC, PyPC, PzPC, EnPC);
		TLorentzVector       beam  (PxPD, PyPD, PzPD, EnPD);
		const TLorentzVector resonance = pA + pB;
		// boost all needed four-vectors to the resonance rest frame
		const TVector3 boostVect = -resonance.BoostVector();
		pA.Boost    (boostVect);
		recoil.Boost(boostVect);
		beam.Boost  (boostVect);
		// rotate so beam is aligned along the z-axis
		pA.RotateZ(-beam.Phi());    recoil.RotateZ(-beam.Phi());
		pA.RotateY(-beam.Theta());  recoil.RotateY(-beam.Theta());
		// rotate so recoil is in the xz-plane
		pA.RotateZ(-recoil.Phi());
		return pA.Phi();
	}


	// BIGPHI(recoil; beam; polarization angle) returns azimuthal angle of photon polarization vector in X rest frame [rad]
	// for beam + target -> X + recoil and X -> a + b
	//     D                    C
	// code taken from https://github.com/JeffersonLab/halld_sim/blob/538677ee1347891ccefa5780e01b158e035b49b1/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc#L94
	static
	double
	bigPhi(
		const double PxPC, const double PyPC, const double PzPC, const double EnPC,
		const double PxPD, const double PyPD, const double PzPD, const double EnPD,
		const double polAngle  // polarization angle [deg]
	) {
		assert(polAngle == 45);  // MC data was generated with this value
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
		return Phi;
	}

};
