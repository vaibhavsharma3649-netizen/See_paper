import streamlit as st
import requests

st.set_page_config(page_title="Research Paper Visualizer", page_icon="📄", layout="wide")

st.title("📄 Research Paper Visualizer")
st.markdown("Search for a topic to find papers, then visualize any paper with a beginner-friendly explanation.")

# ── 1. Search for papers ─────────────────────────────────────────────────────
search_query = st.text_input("🔍 Search papers by topic", placeholder="e.g. transformers, deep learning, GANs")

if st.button("Search", disabled=not search_query):
    with st.spinner("Searching for related papers…"):
        try:
            response = requests.post(
                "http://localhost:8000/recommend",
                json={"query": search_query},
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state["search_results"] = data.get("recommendations", [])
            else:
                st.error(f"Search error ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is the FastAPI server running on port 8000?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── 2. Selectbox with results ────────────────────────────────────────────────
results = st.session_state.get("search_results", [])

if results:
    # Build display labels: "Title  (arxiv_id)"
    options = [r['title'] for r in results]
    selected = st.selectbox("📚 Select a paper", options)

    selected_paper = results[options.index(selected)]
    arxiv_id = selected_paper["arxiv_id"]

    st.caption(f"**Category:** {selected_paper.get('category_code', 'N/A')}  •  **ArXiv ID:** `{arxiv_id}`  •  [View on ArXiv ↗](https://arxiv.org/abs/{arxiv_id})")

    # ── 3. Visualize the selected paper ──────────────────────────────────────
    if st.button("🔬 Visualize Paper"):
        with st.spinner("Downloading paper, building RAG context, generating visualization… This may take a minute."):
            try:
                viz_response = requests.post(
                    "http://localhost:8000/visualize",
                    json={"arxiv_id": arxiv_id},
                    timeout=300
                )
                if viz_response.status_code == 200:
                    viz_data = viz_response.json()
                    st.session_state["viz_title"]  = viz_data.get("title", arxiv_id)
                    st.session_state["viz_result"] = viz_data.get("explanation", "")
                else:
                    st.error(f"Server Error ({viz_response.status_code}): {viz_response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── 4. Display Visualization ────────────────────────────────────────────────
if st.session_state.get("viz_result"):
    st.success(f"✅ **{st.session_state['viz_title']}**")
    st.markdown("---")
    st.markdown(st.session_state["viz_result"])