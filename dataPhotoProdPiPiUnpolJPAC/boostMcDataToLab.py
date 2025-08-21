#!/usr/bin/env python3

# see https://halldweb.jlab.org/wiki/index.php/Guide_to_roll-your-own_python_hddm_transforms#example_5:_writing_Monte_Carlo_events


from enum import IntEnum

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


if __name__ == "__main__":
  # create output stream
  outFileName = "./test.hddm"
  outStream = hddm_s.ostream(outFileName)

  # create record and add event
  record = hddm_s.HDDM()
  physicsEvent = record.addPhysicsEvents(1)[0]

  # set event info
  physicsEvent.runNo = 30731
  physicsEvent.eventNo = 1

  # add reaction
  reaction = physicsEvent.addReactions(1)[0]
  reaction.weight = 1.0

  # beam photon
  beam = reaction.addBeams(1)[0]
  beam.type = ParticleIDsGeant.Photon
  momentum = beam.addMomenta(1)[0]
  momentum.E  = 8.58286  # [GeV]
  momentum.px = 0.0      # [GeV]
  momentum.py = 0.0      # [GeV]
  momentum.pz = 8.58286  # [GeV]
  property = beam.addPropertiesList(1)[0]
  property.charge = 0
  property.mass = 0
  # polarization = beam.addPolarizations(1)[0]
  # polarization.Px = 0.0
  # polarization.Py = 0.0
  # polarization.Pz = 0.0

  # target proton
  target = reaction.addTargets(1)[0]
  target.type = ParticleIDsGeant.Proton
  momentum = target.addMomenta(1)[0]
  momentum.E  = PROTON_MASS
  momentum.px = 0.0
  momentum.py = 0.0
  momentum.pz = 0.0
  property = target.addPropertiesList(1)[0]
  property.charge = +1
  property.mass = PROTON_MASS

  # interaction vertex and its position
  vertex = reaction.addVertices(1)[0]
  origin = vertex.addOrigins(1)[0]
  origin.t  = 0.0  # [ns]
  origin.vx = 0.0  # [cm]
  origin.vy = 0.0  # [cm]
  origin.vz = 0.0  # [cm]

  # recoil proton
  recoil = vertex.addProducts(1)[0]
  recoil.decayVertex = 0
  recoil.id = 1
  recoil.mech = 0
  recoil.parentid = 0
  recoil.type = ParticleIDsGeant.Proton
  recoil.pdgtype = ParticleIDsPdg.Proton
  momentum = recoil.addMomenta(1)[0]
  momentum.E  =  1.04453     # [GeV]
  momentum.px = -0.429367    # [GeV]
  momentum.py = -0.00775752  # [GeV]
  momentum.pz =  0.162113    # [GeV]

  # produced pi^+
  piPlus = vertex.addProducts(1)[0]
  piPlus.decayVertex = 0
  piPlus.id = 2
  piPlus.mech = 0
  piPlus.parentid = 0
  piPlus.type = ParticleIDsGeant.PiPlus
  piPlus.pdgtype = ParticleIDsPdg.PiPlus
  momentum = piPlus.addMomenta(1)[0]
  momentum.E  =  4.63745    # [GeV]
  momentum.px =  0.0960281  # [GeV]
  momentum.py = -0.382075   # [GeV]
  momentum.pz =  4.61858    # [GeV]

  # produced pi^-
  piMinus = vertex.addProducts(1)[0]
  piMinus.decayVertex = 0
  piMinus.id = 3
  piMinus.mech = 0
  piMinus.parentid = 0
  piMinus.type = ParticleIDsGeant.PiMinus
  piMinus.pdgtype = ParticleIDsPdg.PiMinus
  momentum = piMinus.addMomenta(1)[0]
  momentum.E  = 3.83915   # [GeV]
  momentum.px = 0.333339  # [GeV]
  momentum.py = 0.389832  # [GeV]
  momentum.pz = 3.80217   # [GeV]

  outStream.write(record)
