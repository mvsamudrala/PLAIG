import time
import pickle
import os
import warnings
import streamlit as st
import PLAIG_Run
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from io import StringIO


current_directory = os.getcwd()
st.markdown("<h1 style='text-align: center;'>PLAIG Testing</h1>", unsafe_allow_html=True)

# First expander (General vs Refined set)
with st.expander("General and Refined Set Demo"):
    general_index_file = "pages/pdb_key_general"
    with open(general_index_file, 'r') as general_file:
        general_text = general_file.read()
        general_text = general_text[general_text.find("3zzf"):]
        general_index_df = pd.read_csv(StringIO(general_text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
        general_index_df = general_index_df[general_index_df["Kd/Ki"].str.contains("Kd|Ki")].reset_index(drop=True)
        general_index_df["-logKd/Ki"] = pd.to_numeric(general_index_df["-logKd/Ki"], errors='coerce')
        print(general_index_df)

    refined_index_file = "pages/pdb_key_refined"
    with open(refined_index_file, 'r') as refined_file:
        refined_text = refined_file.read()
        refined_text = refined_text[refined_text.find("2r58"):]
        refined_index_df = pd.read_csv(StringIO(refined_text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
        refined_index_df["-logKd/Ki"] = pd.to_numeric(refined_index_df["-logKd/Ki"], errors='coerce')
        print(refined_index_df)

    st.markdown(f"<p style='text-align: center; font-size: 16px'>Would you like to demo protein-ligand complexes "
                f"from the PDBbindv.2020 general or refined set?</p>", unsafe_allow_html=True)
    dataset = st.radio("", ["General Set", "Refined Set"])
    st.markdown(f"<p style='text-align: center; font-size: 16px'>You selected the <b>{dataset}</b>.</p>", unsafe_allow_html=True)

    st.markdown('<div style="text-align: center; font-size: 16px"><a href="https://github.com/mvsamudrala/PLAIG/tree/main/refined_general_files" target="_blank">Click here '
                'to download files from the general or refined set for demo testing.</a>', unsafe_allow_html=True)

    st.markdown(f"<p style='text-align: center; font-size: 16px'>Submit general or refined set files in this "
                f"order:<br>1. xxxx_hydrogenated_pocket.pdb<br>2. xxxx_pocket.pdbqt<br>3. "
                f"xxxx_hydrogenated_ligand.pdb<br> 4. xxxx_ligand.pdbqt</p>", unsafe_allow_html=True)

    form1 = st.form(key="Options1")
    complex_files = form1.file_uploader("Choose your general or refined set files", accept_multiple_files=True)
    submitted1 = form1.form_submit_button("Submit Files")
    complex_files_paths = []
    count = 1
    if complex_files:
        if len(complex_files) > 4:
            st.warning("You can only upload up to 4 files.")
        elif all(file.name[:4] != complex_files[0].name[:4] for file in complex_files):
            st.warning("The files must come from the same complex! The files you submitted have different PDB codes.")
        else:
            for file in complex_files:
                form1.write(f"File name: {count}. {file.name}")
                new_file_path = os.path.join(current_directory, file.name)
                with open(new_file_path, 'wb') as f:
                    f.write(file.getbuffer())
                complex_files_paths.append(new_file_path)
                count += 1
            if submitted1:
                pdb_code = complex_files[0].name[:4]
                if dataset == "General Set":
                    try:
                        experimental_log_ba = general_index_df.loc[general_index_df['PDB Code'] == pdb_code, '-logKd/Ki'].iloc[0]
                        experimental_ba = 10 ** (-1 * experimental_log_ba) * (10 ** 6)
                        complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in
                                               range(0, len(complex_files_paths), 4)]
                        prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                        node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                        plt.figure(figsize=(8, 6), dpi=600)
                        nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold",
                                node_size=500, width=3)
                        caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                        plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18,
                                    family='sans-serif')
                        st.pyplot(plt)
                        plt.clf()
                        st.markdown(
                            f"<p style='text-align: center; color: red; font-size: 24px'>PDB Code: {pdb_code}<br>{prediction[0]}<br>Experimental Binding Affinity (μM): {round(experimental_ba, 3)}</p>",
                            unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"This PDB code is not available in the {dataset}, please choose the other set.")
                else:
                    try:
                        experimental_log_ba = refined_index_df.loc[refined_index_df['PDB Code'] == pdb_code, '-logKd/Ki'].iloc[0]
                        experimental_ba = 10 ** (-1 * experimental_log_ba) * (10 ** 6)
                        complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in
                                               range(0, len(complex_files_paths), 4)]
                        prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                        node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                        plt.figure(figsize=(8, 6), dpi=600)
                        nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold",
                                node_size=500, width=3)
                        caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                        plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18,
                                    family='sans-serif')
                        st.pyplot(plt)
                        plt.clf()
                        st.markdown(
                            f"<p style='text-align: center; color: red; font-size: 24px'>PDB Code: {pdb_code}<br>{prediction[0]}<br>Experimental Binding Affinity (μM): {round(experimental_ba, 3)}</p>",
                            unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"This PDB code is not available in the {dataset}, please choose the other set.")


# Second expander (Docked set demo)
with st.expander("Pre-Docked Files Demo"):
    st.markdown('<div style="text-align: center; font-size: 14px"><a href="https://github.com/mvsamudrala/PLAIG/tree/main/example_docked_files" target="_blank">Click here '
                'to download pre-docked protein-ligand complexes for demo testing.</a>', unsafe_allow_html=True)
    form2 = st.form(key="Options2")
    complex_files = form2.file_uploader("Choose your pre-docked files", accept_multiple_files=True)
    submitted2 = form2.form_submit_button("Submit Files")
    complex_files_paths = []
    count = 1
    if complex_files:
        if len(complex_files) > 4:
            st.warning("You can only upload up to 4 files.")
        else:
            for file in complex_files:
                form2.write(f"File name: {count}. {file.name}")
                new_file_path = os.path.join(current_directory, file.name)
                with open(new_file_path, 'wb') as f:
                    f.write(file.getbuffer())
                complex_files_paths.append(new_file_path)
                count += 1
            if submitted2:
                complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
                prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                plt.figure(figsize=(8, 6), dpi=600)
                nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
                caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18, family='sans-serif')
                st.pyplot(plt)
                plt.clf()
                st.markdown(
                    f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}.</p>",
                    unsafe_allow_html=True)

# Third expander (User testing demo)
with st.expander("User Testing"):
    st.markdown(f"<p style='text-align: center; font-size: 16px'>Submit your docked protein and ligand files in this "
                f"order:<br>1. protein_name.pdb<br>2. protein_name.pdbqt<br>3. ligand_name.pdb<br> 4. "
                f"ligand_name.pdbqt</p>", unsafe_allow_html=True)
    form3 = st.form(key="Options3")
    complex_files = form3.file_uploader("Choose your docked protein and ligand files", accept_multiple_files=True)
    submitted3 = form3.form_submit_button("Submit Files")
    complex_files_paths = []
    count = 1
    if complex_files:
        if len(complex_files) > 4:
            st.warning("You can only upload up to 4 files.")
        else:
            for file in complex_files:
                form3.write(f"File name: {count}. {file.name}")
                new_file_path = os.path.join(current_directory, file.name)
                with open(new_file_path, 'wb') as f:
                    f.write(file.getbuffer())
                complex_files_paths.append(new_file_path)
                count += 1
            if submitted3:
                complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
                prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
                node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
                plt.figure(figsize=(8, 6), dpi=600)
                nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
                caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
                plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18, family='sans-serif')
                st.pyplot(plt)
                plt.clf()
                st.markdown(
                    f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}.</p>",
                    unsafe_allow_html=True)









# gen_ref_demo, docked_demo, user_testing = st.columns(3)
#
# with gen_ref_demo:
#     general_index_file = "pages/pdb_key_general"
#     with open(general_index_file, 'r') as general_file:
#         general_text = general_file.read()
#         general_text = general_text[general_text.find("3zzf"):]
#         general_index_df = pd.read_csv(StringIO(general_text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
#         general_index_df = general_index_df[general_index_df["Kd/Ki"].str.contains("Kd|Ki")].reset_index(drop=True)
#         print(general_index_df)
#
#     refined_index_file = "pages/pdb_key_refined"
#     with open(refined_index_file, 'r') as refined_file:
#         refined_text = refined_file.read()
#         refined_text = refined_text[refined_text.find("2r58"):]
#         refined_index_df = pd.read_csv(StringIO(refined_text), sep='\s+', header=None, names=["PDB Code", "resolution", "release year", "-logKd/Ki", "Kd/Ki", "slash", "reference", "ligand name"], index_col=False)
#         print(refined_index_df)
#
#     st.markdown(f"<p style='text-align: center; font-size: 16px'>Would you like to demo protein-ligand complexes "
#                 f"from the PDBbindv.2020 general or refined set?.</p>", unsafe_allow_html=True)
#     dataset = st.radio("", ["General Set", "Refined Set"])
#     st.markdown(f"<p style='text-align: center; font-size: 16px'>You selected the {dataset}.</p>", unsafe_allow_html=True)
#
#     st.markdown(f"<p style='text-align: center; font-size: 16px'>Submit general or refined set files in this "
#                 f"order:<br>1. xxxx_hydrogenated_pocket.pdb<br>2. xxxx_pocket.pdbqt<br>3. "
#                 f"xxxx_hydrogenated_ligand.pdb<br> 4. xxxx_ligand.pdbqt</p>", unsafe_allow_html=True)
#     st.markdown('<div style="text-align: center; font-size: 14px"><a href="https://github.com/mvsamudrala/PLAIG/tree/main/refined_general_files" target="_blank">Click here '
#                 'to download files from the general or refined set for demo testing.</a>', unsafe_allow_html=True)
#     form1 = st.form(key="Options1")
#     complex_files = form1.file_uploader("Choose your general or refined set files", accept_multiple_files=True)
#     complex_files_paths = []
#     count = 1
#     if complex_files:
#         if len(complex_files) > 4:
#             st.warning("You can only upload up to 4 files.")
#         else:
#             for file in complex_files:
#                 form1.write(f"File name: {count}. {file.name}")
#                 new_file_path = os.path.join(current_directory, file.name)
#                 with open(new_file_path, 'wb') as f:
#                     f.write(file.getbuffer())
#                 complex_files_paths.append(new_file_path)
#                 count += 1
#     submitted1 = form1.form_submit_button("Submit Files")
#     if submitted1:
#         complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
#         prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
#         node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
#         plt.figure(figsize=(8, 6), dpi=600)
#         nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
#         caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
#         plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18, family='sans-serif')
#         st.pyplot(plt)
#         plt.clf()
#         st.markdown(
#             f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}.</p>",
#             unsafe_allow_html=True)
#
# with docked_demo:
#     st.markdown('<div style="text-align: center; font-size: 14px"><a href="https://github.com/mvsamudrala/PLAIG/tree/main/example_docked_files" target="_blank">Click here '
#                 'to download pre-docked protein-ligand complexes for demo testing.</a>', unsafe_allow_html=True)
#     form2 = st.form(key="Options2")
#     complex_files = form2.file_uploader("Choose your pre-docked files", accept_multiple_files=True)
#     complex_files_paths = []
#     count = 1
#     if complex_files:
#         if len(complex_files) > 4:
#             st.warning("You can only upload up to 4 files.")
#         else:
#             for file in complex_files:
#                 form2.write(f"File name: {count}. {file.name}")
#                 new_file_path = os.path.join(current_directory, file.name)
#                 with open(new_file_path, 'wb') as f:
#                     f.write(file.getbuffer())
#                 complex_files_paths.append(new_file_path)
#                 count += 1
#     submitted2 = form2.form_submit_button("Submit Files")
#     if submitted2:
#         complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
#         prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
#         node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
#         plt.figure(figsize=(8, 6), dpi=600)
#         nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
#         caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
#         plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18, family='sans-serif')
#         st.pyplot(plt)
#         plt.clf()
#         st.markdown(
#             f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}.</p>",
#             unsafe_allow_html=True)
#
# with user_testing:
#     st.markdown(f"<p style='text-align: center; font-size: 16px'>Submit your docked protein and ligand files in this "
#                 f"order:<br>1. protein_name.pdb<br>2. protein_name.pdbqt<br>3. ligand_name.pdb<br> 4. "
#                 f"ligand_name.pdbqt</p>", unsafe_allow_html=True)
#     form3 = st.form(key="Options3")
#     complex_files = form3.file_uploader("Choose your docked protein and ligand files", accept_multiple_files=True)
#     complex_files_paths = []
#     count = 1
#     if complex_files:
#         if len(complex_files) > 4:
#             st.warning("You can only upload up to 4 files.")
#         else:
#             for file in complex_files:
#                 form3.write(f"File name: {count}. {file.name}")
#                 new_file_path = os.path.join(current_directory, file.name)
#                 with open(new_file_path, 'wb') as f:
#                     f.write(file.getbuffer())
#                 complex_files_paths.append(new_file_path)
#                 count += 1
#     submitted3 = form3.form_submit_button("Submit Files")
#     if submitted3:
#         complex_files_paths = [tuple(complex_files_paths[i:i + 4]) for i in range(0, len(complex_files_paths), 4)]
#         prediction, graph, color_cutoff = PLAIG_Run.run_model(complex_files_paths)
#         node_colors = ["lightblue" if node < color_cutoff else "pink" for node in graph.nodes()]
#         plt.figure(figsize=(8, 6), dpi=600)
#         nx.draw(graph, with_labels=True, node_color=node_colors, edge_color="black", font_weight="bold", node_size=500, width=3)
#         caption = "Visualization of protein-ligand graph with ligand atoms in blue and protein atoms in pink"
#         plt.figtext(0.5, -0.10, caption, wrap=True, horizontalalignment='center', fontsize=18, family='sans-serif')
#         st.pyplot(plt)
#         plt.clf()
#         st.markdown(
#             f"<p style='text-align: center; color: red; font-size: 24px'>{prediction[0]}.</p>",
#             unsafe_allow_html=True)



