import streamlit as st
from extracting_git import extract_url
import os
from pathlib import Path
from langauge_extract import  parse_codebase

# Title
st.title("Git Repository Cloner")

# Get absolute path to repo directory (works in Streamlit cloud/local)
current_dir = Path(__file__).parent
repo_dir = current_dir / "repo"

# User input
repo_url = st.text_input("Enter Git repository URL:", 
                        placeholder="https://github.com/user/repo.git")

if st.button("Clone Repository"):
    if not repo_url:
        st.warning("Please enter a repository URL")
    else:
        with st.spinner("Processing repository..."):
            success = extract_url(repo_url, repo_dir)
            
            if success:
                st.balloons()
                st.success("Repository processed successfully!")
                st.session_state.repo_dir = str(repo_dir)
               
            else:
                st.error("Failed to process repository")

if st.button("Analyze Repository"):
    repo_dir = st.session_state.get("repo_dir", None)
    if repo_dir:
        with st.spinner("Analyzing repository..."):
            code_blocks = parse_codebase(repo_dir)
            st.write("Extraction completed")
            
    else:
        st.warning("No repository directory found. Please clone a repository first.")