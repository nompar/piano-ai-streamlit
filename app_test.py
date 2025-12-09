import streamlit as st
import subprocess
import sys

st.title("Test Piano AI")

if st.button("Run test"):
    result = subprocess.run(
        [sys.executable, "test_direct.py"],
        capture_output=True,
        text=True,
        cwd="/Users/hadriendecaumont/code/nompar/piano-ai-streamlit"
    )
    st.write("STDOUT:")
    st.code(result.stdout)
    st.write("STDERR:")
    st.code(result.stderr)
