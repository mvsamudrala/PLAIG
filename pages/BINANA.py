import streamlit as st

st.markdown("<h1 style='text-align: center;'>BINANA Documentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 24px'>BINANA (BINding ANAlyzer) is a Python package "
            "used to identify the number and location of different intermolecular chemical interactions. PLAIG uses "
            "BINANA to calculate electrostatic energies, hydrogen bonds, halogen bonds, hydrophobic contacts, metal "
            "contacts, π-π stacking, t-stacking, salt bridges, and cation-π interactions. With the number and location "
            "of chemical interactions from BINANA, our graph representation algorithm matches the chemical interactions "
            "between atoms from the protein pocket and the ligand that are within a specified cutoff distance (3 Å). "
            "These features are embedded as edge features in the graph, which are fed into the GNN model to help "
            "with binding affinity predictions. This documentation page will take you deeper into the different kinds "
            "of intermolecular interactions that BINANA computes and how we use these interaction features in our graph "
            "representation of the protein-ligand complex.</p>", unsafe_allow_html=True)
st.image("BINANA.png")
st.markdown("<p style='text-align: left; font-size: 20px'><b>Electrostatic Energies:</b> This refers to the potential energy "
            "stored between two atoms due to the separation of charges. BINANA natively returns a list of the summed "
            "electrostatic energies between different atoms, not the individual electrostatic energy between each "
            "distinct pair of protein-ligand atoms. For example, BINANA returns C-C: -2567, C-N: -1888, O-C: -7278, "
            "instead of the electrostatic energy between all pairs of atoms that fulfill the cutoff distance. Because of "
            "this, we modified the existing BINANA package to return the individual electrostatic energies between "
            "each pair of atoms. We then included the normalized electrostatic energies into the edge feature vector "
            "between protein-ligand atom pairs for each graph.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Hydrogen Bonds:</b> BINANA considers amine, hydroxyl, and "
            "thiol groups as hydrogen bond donors. Oxygen, nitrogen, and sulfur atoms are considered hydrogen bond "
            "acceptors. BINANA returns the location (atom number) of donor-acceptor pairs within the cutoff distance. "
            "We converted this information into 3 quantities: the donor atom location, hydrogen atom location, and "
            "acceptor atom location. To store hydrogen bond information in each graph, we put a 1 in the edge feature "
            "vector between atoms that participated in a hydrogen bond and a 0 if no hydrogen bond is present.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Halogen Bonds:</b> Similar to hydrogen bonds, BINANA "
            "considers oxygen, nitrogen, and sulfur atoms as hydrogen bond acceptors. Halogen bond donors are any "
            "groups such as O-X, N-X, S-X, and C-X, where X can be iodine, bromine, chlorine, or fluorine. This "
            "information was converted into 3 quantities, similar to hydrogen bonds: donor, halogen, and acceptor. To "
            "store halogen bond information in each graph, we put a 1 in the edge feature vector between atoms that "
            "participated in a halogen bond and a 0 if no halogen bond is present.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Hydrophobic Contacts:</b> Hydrophobic contacts in BINANA "
            "are made simple. The Python package finds the location of any protein-ligand carbon atom pairs that are "
            "within the specified cutoff distance. BINANA categorizes these contacts based on the flexibility of the "
            "protein carbon atom, but we only use the location of a hydrophobic contact when constructing our graphs. To "
            "store hydrophobic contacts in each graph, we put a 1 in the edge feature vector between atoms that "
            "have a hydrophobic contact and a 0 if no hydrophobic contact is present.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Metal Contacts:</b> BINANA returns a metal contact anywhere "
            "that a nitrogen, oxygen, chlorine, fluorine, bromine, iodine, or sulfur is within the cutoff distance "
            "from a metal cation. The most common example is when one of these types of ligand atoms is close to a "
            "metal ion that is an enzyme cofactor. To store metal contacts in each graph, we put a 1 in the edge feature "
            "vector between atoms that have a metal contact and a 0 if no metal contact is present.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>π Interactions:</b> BINANA uses a variety of techniques to "
            "locate aromatic systems in both the protein and the ligand. You can read BINANA's documentation cited "
            "below for more information on these algorithms. After BINANA locates aromatic systems, it then determines "
            "the locations of <b>π-π stacking</b>. BINANA does this by finding the center of each aromatic ring, then "
            "comparing the distance of this center to the centers of aromatic rings on the opposite molecule. For "
            "example, it compares the center of an aromatic ring on a ligand to all the aromatic ring centers in the "
            "protein. Any aromatic centers that are within the cutoff distance are then checked for the angle between "
            "the two vectors normal to each aromatic ring plane. If the angle is within the π stacking angle tolerance "
            "degrees of being parallel, then each atom of the aromatic ring is directly projected onto the plane of "
            "the opposite ring. If any of the project points are within the ring disk of the opposite aromatic system, "
            "then this is considered a location for π-π stacking. The location of <b>t-stacking</b> is calculated in the exact "
            "same way as π-π stacking, except the angle between the aromatic rings is checked if it is within the "
            "t-stacking angle tolerance degrees of being parallel. In addition, a second t-stacking closest distance "
            "cutoff is checked on top of the primary cutoff distance check. Locations of <b>cation-π</b> interactions are "
            "calculated in the following manner. First, any charged functional groups that come into the cutoff distance "
            "to the center of any aromatic rings are selected. Then, the coordinate of the charged group is projected "
            "onto the plane of the aromatic ring. If the projected coordinate is within the ring disk of the aromatic "
            "system, then this signals a cation-π interaction. To store π-π stacking, t-stacking, and cation-π locations "
            "in each graph, we put 1s in the edge feature vector between atoms that perform π-π stacking, t-stacking, "
            "and cation-π interactions and 0s if none of these interactions are present.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Salt Bridges:</b> For a full description of how BINANA "
            "calculates salt bridges, please see the citation below. BINANA calculates possible salt bridges that bind "
            "the ligand to the receptor using metal ions as the center and other charged amine groups such as "
            "sp3-hybridized amines, quarternary ammoniums, imidamides, guanidino groups, and amines on lysine groups. "
            "If two protein-ligand atoms fufill the necessary salt bridge criteria set in place by BINANA and are "
            "opposite charges, then a salt bridge is detected between them. BINANA also categorizes the salt bridges "
            "by the secondary structure of the associated protein residue, but we do not take this into account. To "
            "store salt bridge locations in each graph, we put a 1 in the edge feature vector between atoms that "
            "participate in a salt bridge and a 0 if no salt bridge activity is present.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: left; font-size: 20px'><b>Citations:</b><br> Young, J.; Garikipati, N.; Durrant, J. D. BINANA 2: Characterizing Receptor/Ligand Interactions in Python and JavaScript. J. Chem. Inf. Model. 2022, 62 (4), 753–760. https://doi.org/10.1021/acs.jcim.1c01461.</p>", unsafe_allow_html=True)
