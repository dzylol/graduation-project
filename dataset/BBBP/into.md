About Dataset
The blood-brain barrier penetration (BBBP) dataset is designed for the
modeling and prediction of barrier permeability. As a membrane separating
circulating blood and brain extracellular fluid, the blood-brain barrier
blocks most drugs, hormones, and neurotransmitters. Thus penetration of the
barrier forms a long-standing issue in the development of drugs targeting
the central nervous system.
This dataset includes binary labels for over 2000 compounds on their
permeability properties.
Scaffold splitting is recommended for this dataset.
The raw data CSV file contains the columns below:

"name" - Name of the compound
"smiles" - SMILES representation of the molecular structure
"p_np" - Binary labels for penetration/non-penetration
References

.. [1] Martins, Ines Filipa, et al. "A Bayesian approach to in silico
blood-brain barrier penetration modeling." Journal of chemical
information and modeling 52.6 (2012): 1686-1697.