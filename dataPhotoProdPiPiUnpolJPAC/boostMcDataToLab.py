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
  physicsEvent = record.addPhysicsEvents(1)

  # set event info
  physicsEvent[0].runNo = 30731
  physicsEvent[0].eventNo = 1

  # add reaction
  reaction = physicsEvent[0].addReactions(1)
  reaction[0].weight = 1.0

  # beam photon
  beam = reaction[0].addBeams(1)
  beam[0].type = ParticleIDsGeant.Photon
  momentum = beam[0].addMomenta(1)
  momentum[0].E  = 8.58286  # [GeV]
  momentum[0].px = 0.0      # [GeV]
  momentum[0].py = 0.0      # [GeV]
  momentum[0].pz = 8.58286  # [GeV]
  properties = beam[0].addPropertiesList(1)
  properties[0].charge = 0
  properties[0].mass = 0
  # polarization = beam[0].addPolarizations(1)
  # polarization[0].Px = 0.0
  # polarization[0].Py = 0.0
  # polarization[0].Pz = 0.0

  # target proton
  target = reaction[0].addTargets(1)
  target[0].type = ParticleIDsGeant.Proton
  momentum = target[0].addMomenta(1)
  momentum[0].E  = PROTON_MASS
  momentum[0].px = 0.0
  momentum[0].py = 0.0
  momentum[0].pz = 0.0
  properties = target[0].addPropertiesList(1)
  properties[0].charge = +1
  properties[0].mass = PROTON_MASS

  # interaction vertex
  vertex = reaction[0].addVertices(1)
  origin = vertex[0].addOrigins(1)
  origin[0].t  = 0.0  # [ns]
  origin[0].vx = 0.0  # [cm]
  origin[0].vy = 0.0  # [cm]
  origin[0].vz = 0.0  # [cm]

  # recoil proton
  recoil = vertex[0].addProducts(1)
  recoil[0].decayVertex = 0
  recoil[0].id = 1
  recoil[0].mech = 0
  recoil[0].parentid = 0
  recoil[0].type = ParticleIDsGeant.Proton
  recoil[0].pdgtype = ParticleIDsPdg.Proton
  momentum = recoil[0].addMomenta(1)
  momentum[0].E  =  1.04453     # [GeV]
  momentum[0].px = -0.429367    # [GeV]
  momentum[0].py = -0.00775752  # [GeV]
  momentum[0].pz =  0.162113    # [GeV]

  # produced pi^+
  piPlus = vertex[0].addProducts(1)
  piPlus[0].decayVertex = 0
  piPlus[0].id = 2
  piPlus[0].mech = 0
  piPlus[0].parentid = 0
  piPlus[0].type = ParticleIDsGeant.PiPlus
  piPlus[0].pdgtype = ParticleIDsPdg.PiPlus
  momentum = piPlus[0].addMomenta(1)
  momentum[0].E  =  4.63745    # [GeV]
  momentum[0].px =  0.0960281  # [GeV]
  momentum[0].py = -0.382075   # [GeV]
  momentum[0].pz =  4.61858    # [GeV]

  # produced pi^-
  piMinus = vertex[0].addProducts(1)
  piMinus[0].decayVertex = 0
  piMinus[0].id = 3
  piMinus[0].mech = 0
  piMinus[0].parentid = 0
  piMinus[0].type = ParticleIDsGeant.PiMinus
  piMinus[0].pdgtype = ParticleIDsPdg.PiMinus
  momentum = piMinus[0].addMomenta(1)
  momentum[0].E  = 3.83915   # [GeV]
  momentum[0].px = 0.333339  # [GeV]
  momentum[0].py = 0.389832  # [GeV]
  momentum[0].pz = 3.80217   # [GeV]

  outStream.write(record)
