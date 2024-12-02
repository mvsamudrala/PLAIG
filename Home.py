import streamlit as st


st.set_page_config(page_title="Multipage App")
st.markdown("<h1 style='text-align: center;'>Welcome to PLAIG's Documentation Webpage</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 24px'>PLAIG is a GNN-based deep learning model for protein-ligand binding affinity "
            "prediction. This app provides documentation on how to use PLAIG and details how PLAIG generates "
            "graph representations to predict binding affinity. By clicking on the '>' in the top left and navigating to the side bar, "
            "you will be able to test PLAIG's binding affinity prediction model by submitting your own protein-ligand "
            "complex in .pdb and .pdbqt files. You can read the citation listed below for background information on "
            "PLAIG before navigating through this webpage.</p>", unsafe_allow_html=True)
st.image("GNN Model Framework.png")
st.markdown("<p style='text-align: left; font-size: 20px'><b>Citations:</b><br> In pre-publication stage.</p>", unsafe_allow_html=True)


