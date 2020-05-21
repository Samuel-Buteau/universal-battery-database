from enum import Enum

CHARGE = 'chg'
DISCHARGE = 'dchg'

POLARITIES = [(CHARGE, 'CHARGE'), (DISCHARGE, 'DISCHARGE')]

SINGLE_CRYSTAL = 'sc'
POLY_CRYSTAL = 'po'
MIXED_CRYSTAL = 'mx'
UNKNOWN_CRYSTAL = 'un'
CRYSTAL_TYPES = [
    (SINGLE_CRYSTAL, 'Single'),
    (POLY_CRYSTAL, 'Poly'),
    (MIXED_CRYSTAL, 'Mixed'),
]




SALT = 'sa'
ADDITIVE = 'ad'
SOLVENT = 'so'
ACTIVE_MATERIAL = 'am'
CONDUCTIVE_ADDITIVE = 'co'
BINDER = 'bi'
SEPARATOR_MATERIAL = 'se'

COMPONENT_TYPES = [
    (SALT, 'salt'),
    (ADDITIVE, 'additive'),
    (SOLVENT, 'solvent'),
    (ACTIVE_MATERIAL, 'active_material'),
    (CONDUCTIVE_ADDITIVE, 'conductive_additive'),
    (BINDER, 'binder'),
    (SEPARATOR_MATERIAL, 'separator_material'),
]

ELECTROLYTE = 'el'
CATHODE = 'ca'
ANODE = 'an'
SEPARATOR = 'se'
COMPOSITE_TYPES = [
    (ELECTROLYTE, 'electrolyte'),
    (CATHODE, 'cathode'),
    (ANODE, 'anode'),
    (SEPARATOR, 'separator'),
]

ELECTRODE = 'ed'
COMPOSITE_TYPES_2 = [
    (ELECTROLYTE, 'electrolyte'),
    (ELECTRODE, 'electrode'),
    (SEPARATOR, 'separator'),
]

class LotTypes(Enum):
    none = 0
    unknown = 1
    no_lot = 2
    lot = 3
