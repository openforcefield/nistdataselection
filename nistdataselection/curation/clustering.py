import logging

import numpy as np
from openeye import oechem, oegraphsim
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def cluster_similar_molecules(
    smiles, fingerprint_type=oegraphsim.OEFPType_Tree, eps=0.5, min_samples=2
):
    """The method attempts to cluster a sets of molecules based on their
    similarity using a Tanimoto distance metric and the `sklearn` DBSCAN
    clustering code.

    Notes
    -----
    This is based on the code by David Mobley:

    https://github.com/openforcefield/release-1-benchmarking/blob/master/QM_molecule_selection/divide_sets.ipynb

    Parameters
    ----------
    smiles: list of str
        The SMILES representations of the molecules to cluster.
    fingerprint_type
        The type of molecular fingerprint to use.
    eps: float
        The `eps` parameter to pass as an argument to DBSCAN while clustering.
    min_samples: int
        The `min_samples` parameter to pass as an argument to DBSCAN while
        clustering.

    Returns
    -------
    dict of str and list of str
        The clustered SMILES patterns.
    """
    assert isinstance(smiles, list)

    # Build fingerprints
    fingerprints = {}

    for smiles_pattern in smiles:

        oe_molecule = oechem.OEMol()
        oechem.OEParseSmiles(oe_molecule, smiles_pattern)

        fingerprint = oegraphsim.OEFingerPrint()
        oegraphsim.OEMakeFP(fingerprint, oe_molecule, fingerprint_type)

        fingerprints[smiles_pattern] = fingerprint

    # Build a similarity matrix
    distance_matrix = np.zeros((len(smiles), len(smiles)))

    for i, smiles_i in enumerate(smiles):

        for j, smiles_j in enumerate(smiles):

            if i == j:
                continue

            distance_matrix[i, j] = 1.0 - oegraphsim.OETanimoto(
                fingerprints[smiles_i], fingerprints[smiles_j]
            )

    # Cluster the data
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    clustered_smiles = clustering.fit(distance_matrix)

    labels = clustered_smiles.labels_

    smiles_by_cluster = {}

    for label in set(labels):

        smiles_by_cluster[label] = [
            smiles[i] for i, x in enumerate(labels) if x == label
        ]

    return smiles_by_cluster
