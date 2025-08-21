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
