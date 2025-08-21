#!/usr/bin/env python3

# see https://halldweb.jlab.org/wiki/index.php/Guide_to_roll-your-own_python_hddm_transforms#example_5:_writing_Monte_Carlo_events


from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import ROOT
import hddm_s


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
  eventData:   EventData,
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
  momentum = beam.addMomenta(1)[0]
  momentum.E  = eventData.beamMomLv.E()
  momentum.px = eventData.beamMomLv.Px()
  momentum.py = eventData.beamMomLv.Py()
  momentum.pz = eventData.beamMomLv.Pz()
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
  momentum = target.addMomenta(1)[0]
  momentum.E  = targetMomLv.E()
  momentum.px = targetMomLv.Px()
  momentum.py = targetMomLv.Py()
  momentum.pz = targetMomLv.Pz()
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
  recoil.decayVertex = 0
  recoil.id = 1
  recoil.mech = 0
  recoil.parentid = 0
  recoil.type = ParticleIDsGeant.Proton
  recoil.pdgtype = ParticleIDsPdg.Proton
  momentum = recoil.addMomenta(1)[0]
  momentum.E  = eventData.recoilMomLv.E()
  momentum.px = eventData.recoilMomLv.Px()
  momentum.py = eventData.recoilMomLv.Py()
  momentum.pz = eventData.recoilMomLv.Pz()

  # add produced pi^+
  piPlus = vertex.addProducts(1)[0]
  piPlus.decayVertex = 0
  piPlus.id = 2
  piPlus.mech = 0
  piPlus.parentid = 0
  piPlus.type = ParticleIDsGeant.PiPlus
  piPlus.pdgtype = ParticleIDsPdg.PiPlus
  momentum = piPlus.addMomenta(1)[0]
  momentum.E  = eventData.piPlusMomLv.E()
  momentum.px = eventData.piPlusMomLv.Px()
  momentum.py = eventData.piPlusMomLv.Py()
  momentum.pz = eventData.piPlusMomLv.Pz()

  # add produced pi^-
  piMinus = vertex.addProducts(1)[0]
  piMinus.decayVertex = 0
  piMinus.id = 3
  piMinus.mech = 0
  piMinus.parentid = 0
  piMinus.type = ParticleIDsGeant.PiMinus
  piMinus.pdgtype = ParticleIDsPdg.PiMinus
  momentum = piMinus.addMomenta(1)[0]
  momentum.E  = eventData.piMinusMomLv.E()
  momentum.px = eventData.piMinusMomLv.Px()
  momentum.py = eventData.piMinusMomLv.Py()
  momentum.pz = eventData.piMinusMomLv.Pz()


if __name__ == "__main__":
  eventData = EventData(
    runNmb       = 30731,
    eventNmb     = 1,
    beamMomLv    = ROOT.TLorentzVector( 0,          0,          8.58286,  8.58286),  # [GeV]
    recoilMomLv  = ROOT.TLorentzVector(-0.429367,  -0.00775752, 0.162113, 1.04453),  # [GeV]
    piPlusMomLv  = ROOT.TLorentzVector( 0.0960281, -0.382075,   4.61858,  4.63745),  # [GeV]
    piMinusMomLv = ROOT.TLorentzVector( 0.333339,   0.389832,   3.80217,  3.83915),  # [GeV]
    vertexPos    = ROOT.TVector3(0, 0, 0),  # [cm]
  )

  # create output stream
  outFileName = "./test.hddm"
  outStream = hddm_s.ostream(outFileName)

  # create, fill, and write record
  record = hddm_s.HDDM()
  fillHddmRecord(record, eventData)
  outStream.write(record)
