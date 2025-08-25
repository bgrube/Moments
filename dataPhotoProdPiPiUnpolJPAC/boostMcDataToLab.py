#!/usr/bin/env python3


from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import os

import ROOT
import hddm_s

from makeMomentsInputTree import (
  lorentzVectorsJpac,
  readDataJpac,
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
  runNmb:    int
  eventNmb:  int
  lvBeam:    ROOT.TLorentzVector
  lvTarget:  ROOT.TLorentzVector
  lvRecoil:  ROOT.TLorentzVector
  lvPip:     ROOT.TLorentzVector
  lvPim:     ROOT.TLorentzVector
  vertexPos: ROOT.TVector3


# see https://halldweb.jlab.org/wiki/index.php/Guide_to_roll-your-own_python_hddm_transforms#example_5:_writing_Monte_Carlo_events
def fillHddmRecord(eventData: EventData) -> hddm_s.HDDM:
  """Fills and returns given HDDM record with a gamma + p -> pi+ pi- p physics event using the provided event data"""
  # create HDDM record
  record = hddm_s.HDDM()

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
  property.mass = eventData.lvTarget.M()

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
  pip = vertex.addProducts(1)[0]
  pip.id = 2
  pip.type = ParticleIDsGeant.PiPlus
  pip.pdgtype = ParticleIDsPdg.PiPlus

  # add produced pi^-
  pim = vertex.addProducts(1)[0]
  pim.id = 3
  pim.type = ParticleIDsGeant.PiMinus
  pim.pdgtype = ParticleIDsPdg.PiMinus

  # set common attributes of final-state particles
  for particle in (recoil, pip, pim):
    particle.decayVertex = 0
    particle.mech = 0
    particle.parentid = 0

  # fill four-momenta of all particles
  momLvData = (
    (beam,   eventData.lvBeam),
    (target, eventData.lvTarget),
    (recoil, eventData.lvRecoil),
    (pip,    eventData.lvPip),
    (pim,    eventData.lvPim),
  )
  for particle, momLv in momLvData:
    momentum = particle.addMomenta(1)[0]
    momentum.E  = momLv.E()
    momentum.px = momLv.Px()
    momentum.py = momLv.Py()
    momentum.pz = momLv.Pz()

  return record


def defineDataFrameColumns(
  df:       ROOT.RDataFrame,
  lvBeam:   str,  # function-argument list with Lorentz-vector components of beam photon
  lvTarget: str,  # function-argument list with Lorentz-vector components of target proton
  lvRecoil: str,  # function-argument list with Lorentz-vector components of recoil proton
  lvPip:    str,  # function-argument list with Lorentz-vector components of pi^+
  lvPim:    str,  # function-argument list with Lorentz-vector components of pi^-
) -> ROOT.RDataFrame:
  """Returns RDataFrame with additional columns required for analysis"""
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
    // events where generated such that recoil proton is in x-z plane
    // hence the recoil-proton momentum has no y component
    // fix this by rotating all particles in the event by the same random azimuthal angle around the z (= beam) axis
    const double phi = gRandom->Uniform(0, TMath::TwoPi());
    for (auto& particle : lvParticlesLab) {
      particle.RotateZ(phi);
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
      # force GEANT to simulate vertex position
      # see https://github.com/JeffersonLab/gluex_MCwrapper/blob/bac5b0d6c868632ba898d1b9a09706031fea6752/Gcontrol.in#L26
      .Define("vertexPos",      "TVector3(0, 0, 0)")
  )
  return df


if __name__ == "__main__":
  ROOT.gROOT.SetBatch(True)

  inputData: dict[str, str] = {  # mapping of t-bin labels to input file names
    "tbin_0.4_0.5" : "./mc/mc_full_model/mc0.4-0.5_ful.dat",
    "tbin_0.5_0.6" : "./mc/mc_full_model/mc0.5-0.6_ful.dat",
    "tbin_0.6_0.7" : "./mc/mc_full_model/mc0.6-0.7_ful.dat",
    "tbin_0.7_0.8" : "./mc/mc_full_model/mc0.7-0.8_ful.dat",
    "tbin_0.8_0.9" : "./mc/mc_full_model/mc0.8-0.9_ful.dat",
    "tbin_0.9_1.0" : "./mc/mc_full_model/mc0.9-1.0_ful.dat",
  }
  outputDirName  = "mc_full"
  outputTreeName = "PiPi"
  outputColumns  = ("lvBeamLab", "lvTargetLab", "lvRecoilLab", "lvPipLab", "lvPimLab", "vertexPos")
  runNmb         = 1

  for tBinLabel, inputFileName in inputData.items():
    os.makedirs(f"{outputDirName}/{tBinLabel}", exist_ok = True)
    outputRootFileName = f"{outputDirName}/{tBinLabel}/data_labFrame_flat.root"
    outputHddmFileName = f"{outputDirName}/{tBinLabel}/data_labFrame.hddm"

    df = defineDataFrameColumns(
      df = readDataJpac(inputFileName),
      **lorentzVectorsJpac(),
    ).Snapshot(outputTreeName, outputRootFileName, outputColumns)
    # ).Snapshot(outputTreeName, outputFileName)
    print(f"ROOT DataFrame columns: {list(df.GetColumnNames())}")
    print(f"ROOT DataFrame entries: {df.Count().GetValue()}")

    # convert ROOT tree to HDDM file
    print(f"Converting tree '{outputTreeName}' in file '{outputRootFileName}' to HDDM file '{outputHddmFileName}'")
    # get input tree from ROOT file
    inFile = ROOT.TFile.Open(outputRootFileName, "READ")
    tree = inFile.Get(outputTreeName)
    assert tree != ROOT.nullptr, f"Failed to get input tree '{outputTreeName}' from ROOT file"
    # create output stream
    outStream = hddm_s.ostream(outputHddmFileName)
    outStream.compression = hddm_s.k_bz2_compression  # compress output with bzip2
    for eventIndex, event in enumerate(tree):  #TODO loop in Python is slow
      # get data
      eventData = EventData(
        runNmb    = runNmb,
        eventNmb  = eventIndex,
        lvBeam    = event.lvBeamLab,
        lvTarget  = event.lvTargetLab,
        lvRecoil  = event.lvRecoilLab,
        lvPip     = event.lvPipLab,
        lvPim     = event.lvPimLab,
        vertexPos = event.vertexPos,
      )
      # fill and write HDDM record
      outStream.write(fillHddmRecord(eventData))
    inFile.Close()
