import time
import pickle
import os
import warnings
import streamlit as st
import PLAIG_Run
import matplotlib.pyplot as plt
import networkx as nx


current_directory = os.getcwd()
st.markdown("<h1 style='text-align: center;'>PLAIG Demo</h1>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center;"><a href="https://github.com/mvsamudrala/PLAIG/tree/main/refined_general_files" target="_blank">Click here '
            'to download files from the refined or general set for demo testing.</a>', unsafe_allow_html=True)
form = st.form(key="Options")
complex_files = form.file_uploader("Submit protein and ligand files in this order (1. xxxx_hydrogenated_pocket.pdb, 2. xxxx_pocket.pdbqt, 3. xxxx_hydrogenated_ligand.pdb, 4. xxxx_ligand.pdbqt)", accept_multiple_files=True)
complex_files_paths = []
count = 1
if complex_files:
    if len(complex_files) > 4:
        st.warning("You can only upload up to 4 files.")
    else:
        for file in complex_files:
            form.write(f"File name: {count}. {file.name}")
            new_file_path = os.path.join(current_directory, file.name)
            with open(new_file_path, 'wb') as f:
                f.write(file.getbuffer())
            complex_files_paths.append(new_file_path)
            count += 1
submitted = form.form_submit_button("Submit Files")
if submitted:
    complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
    prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
    node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
    plt.figure(figsize=(8, 6), dpi=600)
    nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
    caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
    plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=14, family='sans-serif')
    st.pyplot(plt)
    plt.clf()
    st.markdown(
        f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}.</p>",
        unsafe_allow_html=True)



