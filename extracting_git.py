import pygit2
import os
import streamlit as st
from pathlib import Path

def extract_url(repo_url, local_path):
    try:
        # Convert to absolute path and ensure parent exists
        local_path = Path(local_path).absolute()
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Check if directory is empty (not a full repo check)
        if (local_path / ".git").exists():
            st.info(f"Found existing repo at {local_path}")
            repo = pygit2.Repository(str(local_path))
            
            # Try to pull latest if it's a valid repo
            try:
                origin = repo.remotes["origin"]
                origin.fetch()
                st.success("Successfully fetched latest changes")
            except Exception as fetch_error:
                st.warning(f"Couldn't fetch updates: {fetch_error}")
        else:
            st.write(f"Cloning {repo_url} to {local_path}")
            repo = pygit2.clone_repository(repo_url, str(local_path))
            st.success("Clone successful!")
            
        # Verify repo state
        commit = repo[repo.head.target]
        st.write(f"Latest commit: {commit.message[:50]}...")
        return True
        
    except pygit2.GitError as git_error:
        st.error(f"Git operation failed: {git_error}")
    except Exception as e:
        st.error(f"Unexpected error: {type(e).__name__} - {str(e)}")
    
    return False