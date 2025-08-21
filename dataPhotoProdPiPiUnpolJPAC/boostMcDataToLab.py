#!/usr/bin/env python3

# see https://halldweb.jlab.org/wiki/index.php/Guide_to_roll-your-own_python_hddm_transforms#example_5:_writing_Monte_Carlo_events


from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import os

import ROOT
import hddm_s

from makeMomentsInputTree import (
  lorentzVectors,
  readData,
)


class ParticleIDsGeant(IntEnum):
  """Particle IDs defined by GEANT"""
  Photon  = 1
  Proton  = 14
  PiPlus  = 8
  PiMinus = 9

class ParticleIDsPdg(IntEnum):
  """Particle IDs defined by PDG"""
  Photon  = 22
  Proton  = 2212
  PiPlus  = 211
  PiMinus = -211

PROTON_MASS = 0.938272  # [GeV]


@dataclass
class EventData:
  runNmb:       int
  eventNmb:     int
  beamMomLv:    ROOT.TLorentzVector
  recoilMomLv:  ROOT.TLorentzVector
  piPlusMomLv:  ROOT.TLorentzVector
  piMinusMomLv: ROOT.TLorentzVector
  vertexPos:    ROOT.TVector3


def fillHddmRecord(
  record:      hddm_s.HDDM,  # HDDM record to fill
  eventData:   EventData,    # event data to fill into HDDM record
  targetMomLv: ROOT.TLorentzVector = ROOT.TLorentzVector(0, 0, 0, PROTON_MASS),
) -> None:
  """Fill given HDDM record with a gamma + p -> pi+ pi- p physics event using the provided Lorentz-vectors and vertex position"""
  # create event and set event info
  physicsEvent = record.addPhysicsEvents(1)[0]
  physicsEvent.runNo = eventData.runNmb
  physicsEvent.eventNo = eventData.eventNmb

  # add reaction
  reaction = physicsEvent.addReactions(1)[0]
  reaction.weight = 1.0

  # add beam photon
  beam = reaction.addBeams(1)[0]
  beam.type = ParticleIDsGeant.Photon
  property = beam.addPropertiesList(1)[0]
  property.charge = 0
  property.mass = 0
  # polarization = beam.addPolarizations(1)[0]
  # polarization.Px = 0.0
  # polarization.Py = 0.0
  # polarization.Pz = 0.0

  # add target proton
  target = reaction.addTargets(1)[0]
  target.type = ParticleIDsGeant.Proton
  property = target.addPropertiesList(1)[0]
  property.charge = +1
  property.mass = targetMomLv.M()

  # add interaction vertex and its position
  vertex = reaction.addVertices(1)[0]
  origin = vertex.addOrigins(1)[0]
  origin.t  = 0.0  # [ns]
  origin.vx = eventData.vertexPos.X()
  origin.vy = eventData.vertexPos.Y()
  origin.vz = eventData.vertexPos.Z()

  # add recoil proton
  recoil = vertex.addProducts(1)[0]
  recoil.id = 1
  recoil.type = ParticleIDsGeant.Proton
  recoil.pdgtype = ParticleIDsPdg.Proton

  # add produced pi^+
  piPlus = vertex.addProducts(1)[0]
  piPlus.id = 2
  piPlus.type = ParticleIDsGeant.PiPlus
  piPlus.pdgtype = ParticleIDsPdg.PiPlus

  # add produced pi^-
  piMinus = vertex.addProducts(1)[0]
  piMinus.id = 3
  piMinus.type = ParticleIDsGeant.PiMinus
  piMinus.pdgtype = ParticleIDsPdg.PiMinus

  # set common attributes of final-state particles
  for particle in (recoil, piPlus, piMinus):
    particle.decayVertex = 0
    particle.mech = 0
    particle.parentid = 0

  # fill four-momenta of all particles
  momLvData = (
    (beam,    eventData.beamMomLv),
    (target,  targetMomLv),
    (recoil,  eventData.recoilMomLv),
    (piPlus,  eventData.piPlusMomLv),
    (piMinus, eventData.piMinusMomLv),
  )
  for particle, momLv in momLvData:
    momentum = particle.addMomenta(1)[0]
    momentum.E  = momLv.E()
    momentum.px = momLv.Px()
    momentum.py = momLv.Py()
    momentum.pz = momLv.Pz()


def defineDataFrameColumns(
  df:       ROOT.RDataFrame,
  lvBeam:   str,  # function-argument list with Lorentz-vector components of beam photon
  lvTarget: str,  # function-argument list with Lorentz-vector components of target proton
  lvRecoil: str,  # function-argument list with Lorentz-vector components of recoil proton
  lvPip:    str,  # function-argument list with Lorentz-vector components of pi^+
  lvPim:    str,  # function-argument list with Lorentz-vector components of pi^-
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns"""
  vertexPosFcn = """
    // define target geometry
    const double targetR    =  2;  // [cm]
    const double targetZmin = 52;  // [cm]
    const double targetZmax = 78;  // [cm]
    // throw random point in target volume
    const double r   = targetR * sqrt(gRandom->Uniform(0, 1));
    const double phi = gRandom->Uniform(0, TMath::TwoPi());
    const double z   = gRandom->Uniform(targetZmin, targetZmax);
    return TVector3(r * cos(phi), r * sin(phi), z);
  """
  boostEventFcn = """
    // define boost to lab frame
    const TVector3 labBoost = -lvTarget.BoostVector();
    // boost all particles into lab frame
    std::vector<TLorentzVector> lvParticlesLab;
    for (auto particle : {lvBeam, lvTarget, lvRecoil, lvPip, lvPim}) {
      particle.Boost(labBoost);
      lvParticlesLab.push_back(particle);
    }
    // rotate all particles such that beam is along z
    // beam in lab frame is Wigner rotated about y-axis with momentum components: x < 0, y = 0, and z > 0
    // anticlockwise rotation about polar angle undoes the Wigner rotation
    const double wignerAngle = lvParticlesLab[0].Theta();
    for (auto& particle : lvParticlesLab) {
      particle.RotateY(wignerAngle);
    }
    return lvParticlesLab;
  """
  ROOT.gInterpreter.GenerateDictionary("std::vector<TLorentzVector>", "vector;TLorentzVector.h")
  df = (
    df.Define("lvBeam",        f"TLorentzVector({lvBeam})")
      .Define("lvTarget",      f"TLorentzVector({lvTarget})")
      .Define("lvRecoil",      f"TLorentzVector({lvRecoil})")
      .Define("lvPip",         f"TLorentzVector({lvPip})")
      .Define("lvPim",         f"TLorentzVector({lvPim})")
      # boost events to target rest frame = lab frame
      .Define("lvParticlesLab", boostEventFcn)
      .Define("lvBeamLab",      "lvParticlesLab[0]")
      .Define("lvTargetLab",    "lvParticlesLab[1]")
      .Define("lvRecoilLab",    "lvParticlesLab[2]")
      .Define("lvPipLab",       "lvParticlesLab[3]")
      .Define("lvPimLab",       "lvParticlesLab[4]")
      .Define("lvBeamLabTheta", "lvBeamLab.Theta() * TMath::RadToDeg()")
      # generate vertex distribution
      .Define("vertexPos",      vertexPosFcn)
      .Define("vertexPosX",     "vertexPos.X()")
      .Define("vertexPosY",     "vertexPos.Y()")
  )
  return df


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)
  ROOT.gStyle.SetOptStat(111111)

  inputData: dict[str, str] = {  # mapping of t-bin labels to input file names
    "tbin_0.4_0.5" : "./mc/mc_full_model/mc0.4-0.5_ful.dat",
    # "tbin_0.5_0.6" : "./mc/mc_full_model/mc0.5-0.6_ful.dat",
    # "tbin_0.6_0.7" : "./mc/mc_full_model/mc0.6-0.7_ful.dat",
    # "tbin_0.7_0.8" : "./mc/mc_full_model/mc0.7-0.8_ful.dat",
    # "tbin_0.8_0.9" : "./mc/mc_full_model/mc0.8-0.9_ful.dat",
    # "tbin_0.9_1.0" : "./mc/mc_full_model/mc0.9-1.0_ful.dat",
  }
  outputDirName  = "mc_full"
  outputTreeName = "PiPi"
  outputColumns  = ("lvBeamLab", "lvTargetLab", "lvRecoilLab", "lvPipLab", "lvPimLab", "vertexPos")

  for tBinLabel, inputFileName in inputData.items():
    os.makedirs(f"{outputDirName}/{tBinLabel}", exist_ok = True)
    # outputFileName = f"{outputDirName}/{tBinLabel}/data_labFrame_flat.root"
    outputFileName = f"./data_labFrame_flat.root"

    df = defineDataFrameColumns(
      df = readData(inputFileName),
      **lorentzVectors(),
    # ).Snapshot(outputTreeName, outputFileName, outputColumns)
    ).Snapshot(outputTreeName, outputFileName)
    print(f"ROOT DataFrame columns: {list(df.GetColumnNames())}")
    print(f"ROOT DataFrame entries: {df.Count().GetValue()}")

    hist = df.Histo2D(ROOT.RDF.TH2DModel(f"vertexXY", ";x_{vert} [cm];y_{vert} [cm];", 100, -2, +2, 100, -2, +2), "vertexPosX", "vertexPosY").GetValue()
    canv = ROOT.TCanvas()
    hist.Draw("COLZ")
    canv.SaveAs(f"./{hist.GetName()}.pdf")
    hist = df.Histo1D(ROOT.RDF.TH1DModel(f"beamThetaLab", ";#theta_{beam} [deg];", 100, 0, 180), "lvBeamLabTheta").GetValue()
    canv = ROOT.TCanvas()
    hist.Draw()
    canv.SaveAs(f"./{hist.GetName()}.pdf")

  # eventData = EventData(
  #   runNmb       = 30731,
  #   eventNmb     = 1,
  #   beamMomLv    = ROOT.TLorentzVector( 0,          0,          8.58286,  8.58286),  # [GeV]
  #   recoilMomLv  = ROOT.TLorentzVector(-0.429367,  -0.00775752, 0.162113, 1.04453),  # [GeV]
  #   piPlusMomLv  = ROOT.TLorentzVector( 0.0960281, -0.382075,   4.61858,  4.63745),  # [GeV]
  #   piMinusMomLv = ROOT.TLorentzVector( 0.333339,   0.389832,   3.80217,  3.83915),  # [GeV]
  #   vertexPos    = ROOT.TVector3(0, 0, 0),  # [cm]
  # )

  # # create output stream
  # outFileName = "./test.hddm"
  # outStream = hddm_s.ostream(outFileName)

  # # create, fill, and write record
  # record = hddm_s.HDDM()
  # fillHddmRecord(record, eventData)
  # outStream.write(record)
