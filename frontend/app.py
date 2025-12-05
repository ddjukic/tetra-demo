"""
Tetra v1 - Scientific Knowledge Graph Agent Frontend

Streamlit application for interacting with the knowledge graph agent.
Features:
- Build knowledge graphs from seed proteins via multi-agent pipeline
- Interactive chat interface with GraphRAG Q&A agent
- Knowledge graph visualization using PyVis
- Graph statistics and entity exploration

Usage:
    uv run streamlit run frontend/app.py --server.port 8501
"""

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(_project_root / ".env")


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Tetra v1 - Scientific Knowledge Graph Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Async Helper
# =============================================================================

def run_async(coro):
    """
    Run async function in Streamlit.

    Uses asyncio.run() which properly handles task cleanup before closing
    the event loop, preventing "Task was destroyed but it is pending" warnings.
    """
    return asyncio.run(coro)


# =============================================================================
# Resource Initialization (Cached)
# =============================================================================

@st.cache_resource(show_spinner="Loading link predictor model...")
def load_link_predictor():
    """Load the pre-trained link predictor model (PyG or legacy)."""
    # Try PyG model first (preferred - GPU accelerated)
    # Requires torch and torch-geometric: uv sync --extra gpu
    pyg_model_path = _project_root / "models" / "pyg_link_predictor.pkl"

    if pyg_model_path.exists():
        try:
            from ml.pyg_link_predictor import PyGLinkPredictor
            predictor = PyGLinkPredictor.load(str(pyg_model_path), device="auto")
            return predictor
        except ImportError:
            # torch/torch-geometric not installed - fall through to legacy
            pass
        except Exception as e:
            st.warning(f"Failed to load PyG link predictor: {e}")

    # Fallback to legacy gensim model
    legacy_model_path = _project_root / "models" / "gensim_link_predictor.pkl"

    if legacy_model_path.exists():
        try:
            from ml.link_predictor import LinkPredictor
            predictor = LinkPredictor.load(str(legacy_model_path))
            return predictor
        except Exception as e:
            st.warning(f"Failed to load legacy link predictor: {e}")

    # No model found
    return None


@st.cache_resource(show_spinner="Initializing demo knowledge graph...")
def get_demo_knowledge_graph():
    """Get or create a demo knowledge graph instance."""
    from models.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph()


# =============================================================================
# Graph Visualization
# =============================================================================

def render_graph(graph_data: dict[str, Any], height: str = "600px") -> str:
    """
    Render knowledge graph as interactive HTML using PyVis.

    Args:
        graph_data: Dictionary with 'entities' and 'relationships' keys
        height: Height of the visualization

    Returns:
        HTML string for rendering
    """
    from pyvis.network import Network

    # Create network
    net = Network(
        height=height,
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.04,
                "damping": 0.09
            }
        },
        "nodes": {
            "font": {"size": 14}
        },
        "edges": {
            "font": {"size": 10},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200
        }
    }
    """)

    # Entity type colors
    type_colors = {
        "protein": "#4CAF50",
        "gene": "#2196F3",
        "disease": "#F44336",
        "chemical": "#FF9800",
        "pathway": "#9C27B0",
        "unknown": "#9E9E9E"
    }

    # Add nodes
    entities = graph_data.get("entities", {})
    for entity_id, entity_data in entities.items():
        entity_type = entity_data.get("type", "unknown")
        name = entity_data.get("name", entity_id)
        color = type_colors.get(entity_type, type_colors["unknown"])

        net.add_node(
            entity_id,
            label=name,
            title=f"{name}\nType: {entity_type}",
            color=color,
            size=20
        )

    # Add edges
    relationships = graph_data.get("relationships", {})
    for rel_key, rel_data in relationships.items():
        # Parse the key (source|target|type)
        parts = rel_key.split("|")
        if len(parts) >= 2:
            source = parts[0]
            target = parts[1]
            rel_type = parts[2] if len(parts) > 2 else "associated_with"

            ml_score = rel_data.get("ml_score")
            evidence_count = len(rel_data.get("evidence", []))

            # Edge title for hover
            title = f"{rel_type}"
            if ml_score:
                title += f"\nML Score: {ml_score:.2f}"
            if evidence_count:
                title += f"\nEvidence: {evidence_count} sources"

            # Edge color based on evidence
            if ml_score and not evidence_count:
                edge_color = "#FFC107"  # Predicted (yellow)
            elif evidence_count > 0:
                edge_color = "#4CAF50"  # Literature-backed (green)
            else:
                edge_color = "#9E9E9E"  # Unknown (gray)

            net.add_edge(
                source,
                target,
                title=title,
                label=rel_type[:10] if rel_type else "",
                color=edge_color
            )

    # Generate HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as f:
        net.save_graph(f.name)
        with open(f.name, "r") as html_file:
            html_content = html_file.read()
        os.unlink(f.name)
        return html_content


# =============================================================================
# Session State Initialization
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("Tetra v1")
    st.markdown("**Scientific Knowledge Graph Agent**")

    st.divider()

    # System Status
    st.subheader("System Status")

    # Check API keys
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        st.success("Google API Key: Configured")
    else:
        st.error("Google API Key: Missing")
        st.info("Set GOOGLE_API_KEY in your .env file")

    pubmed_api_key = os.environ.get("PUBMED_API_KEY")
    if pubmed_api_key:
        st.success("PubMed API Key: Configured")
    else:
        st.warning("PubMed API Key: Optional (recommended)")

    # Load resources
    link_predictor = load_link_predictor()
    if link_predictor:
        st.success("Link Predictor: Loaded")
        st.caption(f"Proteins: {len(link_predictor.gene_to_string_id):,}")
    else:
        st.warning("Link Predictor: Not loaded")

    st.divider()

    # Graph Statistics
    st.subheader("Knowledge Graph")

    if "knowledge_graph" in st.session_state and st.session_state.knowledge_graph:
        graph = st.session_state.knowledge_graph
        summary = graph.to_summary()
        col1, col2 = st.columns(2)
        col1.metric("Nodes", summary.get("node_count", 0))
        col2.metric("Edges", summary.get("relationship_count", 0))

        # Entity types
        entity_types = summary.get("entity_types", {})
        if entity_types:
            st.caption("Entity Types:")
            for etype, count in entity_types.items():
                st.text(f"  {etype}: {count}")

        # ML predictions
        ml_edges = summary.get("ml_predicted_edges", 0)
        novel = summary.get("novel_predictions", 0)
        if ml_edges > 0:
            st.metric("ML-Predicted Edges", ml_edges)
            st.metric("Novel Predictions", novel)
    else:
        st.info("No graph built yet. Use the Research tab to build one.")

    st.divider()

    # Quick Actions
    st.subheader("Quick Actions")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        if "agent_manager" in st.session_state:
            del st.session_state.agent_manager
        st.rerun()

    if st.button("Reset Knowledge Graph", use_container_width=True):
        if "knowledge_graph" in st.session_state:
            del st.session_state.knowledge_graph
        if "pipeline_report" in st.session_state:
            del st.session_state.pipeline_report
        if "agent_manager" in st.session_state:
            del st.session_state.agent_manager
        # Clear graph cache
        from agent.graph_agent import clear_graph_cache
        clear_graph_cache()
        st.rerun()

    st.divider()

    # Legend
    st.subheader("Legend")
    st.markdown("""
    **Node Colors:**
    - Green: Protein
    - Blue: Gene
    - Red: Disease
    - Orange: Chemical
    - Purple: Pathway

    **Edge Colors:**
    - Green: Literature-backed
    - Yellow: ML-predicted
    - Gray: Unknown
    """)


# =============================================================================
# Main Content
# =============================================================================

st.title("Tetra v1 - Scientific Knowledge Graph Agent")

# Create tabs
tab_research, tab_chat, tab_graph, tab_explore = st.tabs(["Research", "Chat", "Graph View", "Explore"])


# =============================================================================
# Tab 0: Research - Build Knowledge Graph
# =============================================================================

with tab_research:
    st.subheader("What would you like to investigate today?")

    # Single unified query input - use key for state persistence across tabs
    user_query = st.text_area(
        "Research Query",
        placeholder="BRCA1 cancer",
        help="Enter proteins/genes and research context. Examples: 'HCRTR1 HCRTR2 sleep disorders', 'TP53 DNA damage response', 'insulin diabetes signaling'",
        height=80,
        label_visibility="collapsed",
        key="research_query"
    )

    # Max papers - use key for state persistence
    max_papers = st.number_input(
        "Max Papers",
        min_value=100,
        max_value=1000,
        value=200,
        step=100,
        help="Number of papers to fetch from PubMed",
        key="max_papers"
    )

    # Build button
    build_disabled = not user_query.strip() or not google_api_key
    if not google_api_key:
        st.warning("Google API Key required for pipeline execution. Set GOOGLE_API_KEY in .env file.")

    if st.button("Build Knowledge Graph", type="primary", use_container_width=True, disabled=build_disabled):
        query_text = user_query.strip()
        st.info(f"Building knowledge graph for: {query_text}")

        try:
            # Use the orchestrator which handles everything via LLM agents
            from pipeline.orchestrator import KGOrchestrator
            from pipeline.config import PipelineConfig
            from agent.graph_agent import set_active_graph

            config = PipelineConfig(
                pubmed_max_results=max_papers,
                langfuse_session_id=str(uuid.uuid4()),
            )

            async def build_kg_with_cleanup():
                """Build KG using async context manager for proper client cleanup."""
                async with KGOrchestrator(config=config) as orchestrator:
                    return await orchestrator.build(
                        user_query=query_text,
                        max_articles=max_papers,
                    )

            with st.spinner("Building knowledge graph...", show_time=True):
                graph, pipeline_input = run_async(build_kg_with_cleanup())

                # Set graph for Q&A agent
                set_active_graph(graph)

            # Store in session state
            st.session_state.knowledge_graph = graph
            st.session_state.pipeline_input = pipeline_input  # Store for persistent display

            # Clear agent manager to reinitialize with new graph
            if "agent_manager" in st.session_state:
                del st.session_state.agent_manager

            # Show success message
            summary = graph.to_summary()
            st.success(f"Knowledge graph built: {summary.get('node_count', 0)} nodes, {summary.get('edge_count', 0)} edges")
            st.info("Navigate to the **Chat** tab to query your knowledge graph!")

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    # ==========================================================================
    # Build Details - displayed persistently from session state
    # ==========================================================================
    if "knowledge_graph" in st.session_state and st.session_state.knowledge_graph:
        graph = st.session_state.knowledge_graph
        summary = graph.to_summary()
        pipeline_input = st.session_state.get("pipeline_input")

        with st.expander("Build Details", expanded=False):
            # 1. STRING Network
            st.subheader("1. STRING Network")
            if pipeline_input and pipeline_input.metadata and "string_extension" in pipeline_input.metadata:
                ext = pipeline_input.metadata["string_extension"]
                st.write(f"**Seed Proteins:** {', '.join(ext.get('original_seeds', []))}")
                st.write(f"**Network Extension:** {ext.get('extend_network', 0)} hops")
                st.write(f"**Proteins Found:** {ext.get('expanded_proteins', 0)}")
                st.write(f"**Interactions Found:** {ext.get('total_interactions', 0)}")
            elif pipeline_input and pipeline_input.seed_proteins:
                st.write(f"**Seed Proteins:** {', '.join(pipeline_input.seed_proteins)}")
                st.write(f"**Interactions Found:** {len(pipeline_input.string_interactions)}")
            else:
                st.write("No STRING network data available")

            st.divider()

            # 2. PubMed Search
            st.subheader("2. PubMed Search")
            if pipeline_input and pipeline_input.pubmed_query:
                st.code(pipeline_input.pubmed_query, language=None)
            else:
                st.write("No PubMed query available")

            papers_count = len(pipeline_input.articles) if pipeline_input else 0
            st.write(f"**Papers Fetched:** {papers_count}")

            # 3. Papers List (collapsible)
            if pipeline_input and pipeline_input.articles:
                with st.expander(f"View {len(pipeline_input.articles)} Papers"):
                    for article in pipeline_input.articles[:20]:  # Show first 20
                        pmid = article.get('pmid', '')
                        title = article.get('title') or 'No title'  # Handle None values
                        year = article.get('year', '')
                        title_display = title[:80] + "..." if len(title) > 80 else title
                        st.markdown(f"- **[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})** ({year}): {title_display}")
                    if len(pipeline_input.articles) > 20:
                        st.caption(f"... and {len(pipeline_input.articles) - 20} more papers")

            st.divider()

            # 4. Graph Statistics
            st.subheader("3. Graph Statistics")

            # Entity types breakdown
            entity_types = summary.get("entity_types", {})
            if entity_types:
                st.markdown("**Entity Types:**")
                for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
                    st.markdown(f"- {etype}: {count}")

            # Relationship types breakdown
            rel_types = summary.get("relationship_types", {})
            if rel_types:
                st.markdown("**Relationship Types:**")
                for rtype, count in sorted(rel_types.items(), key=lambda x: -x[1]):
                    st.markdown(f"- {rtype}: {count}")

            st.divider()

            # Evidence and ML stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Nodes", summary.get("node_count", 0))
            col2.metric("Total Edges", summary.get("edge_count", 0))
            col3.metric("ML Predictions", summary.get("ml_predicted_edges", 0))

            # Evidence sources
            evidence_sources = summary.get("evidence_sources", {})
            if evidence_sources:
                st.markdown("**Evidence Sources:**")
                for source, count in sorted(evidence_sources.items(), key=lambda x: -x[1]):
                    st.markdown(f"- {source}: {count}")

            # Novel predictions
            novel = summary.get("novel_predictions", 0)
            if novel > 0:
                st.info(f"Novel predictions (ML-only, no literature evidence): {novel}")



# =============================================================================
# Tab 1: Chat Interface
# =============================================================================


@st.fragment
def render_chat_interface():
    """
    Chat interface fragment for isolated reruns.

    Uses @st.fragment decorator to prevent full page reruns when messages
    are added, which fixes the text input box position shifting issue.
    """
    st.markdown("""
    Ask questions about your knowledge graph. The agent can:
    - Explore graph structure and find paths
    - Compute centrality and detect communities
    - Generate hypotheses for predicted interactions
    """)

    # Display chat history in a container
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show tool calls if available
            if message.get("tool_calls"):
                with st.expander("Tools Used"):
                    for tc in message["tool_calls"]:
                        st.code(f"{tc['name']}({tc.get('args', {})})")

    # Chat input - only manage state, never render inline (let loop handle display)
    if prompt := st.chat_input("Ask about your knowledge graph..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Query backend and add response to state
        with st.spinner("Thinking..."):
            try:
                result = run_async(
                    st.session_state.agent_manager.query(
                        user_id="streamlit_user",
                        session_id=st.session_state.session_id,
                        query=prompt,
                    )
                )

                # Store assistant message with tool calls
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "tool_calls": result.get("tool_calls", []),
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })

        # Rerun fragment - loop above will render all messages in correct order
        st.rerun(scope="fragment")


with tab_chat:
    if "knowledge_graph" not in st.session_state:
        st.info("Build a knowledge graph first using the Research tab.")
        st.markdown("""
        **Example workflows:**

        1. Go to the **Research** tab
        2. Enter seed proteins (e.g., `BRCA1, TP53, MDM2`)
        3. Optionally specify a research focus (e.g., `DNA damage response`)
        4. Click **Build Knowledge Graph**
        5. Return here to query your graph!

        **Sample queries you can ask:**
        - "What are the most important proteins in this network?"
        - "Find the path between BRCA1 and TP53"
        - "What evidence supports the BRCA1-MDM2 interaction?"
        - "Detect communities in the network"
        """)
    else:
        # Initialize agent manager if not exists
        if "agent_manager" not in st.session_state:
            from agent.graph_agent import GraphAgentManager
            # Load link predictor and pass to agent manager
            predictor = load_link_predictor()
            st.session_state.agent_manager = GraphAgentManager(
                graph=st.session_state.knowledge_graph,
                model="gemini-2.5-flash",
                link_predictor=predictor,
            )

        # Render chat interface as a fragment (isolated reruns)
        render_chat_interface()


# =============================================================================
# Tab 2: Graph Visualization
# =============================================================================

with tab_graph:
    st.subheader("Knowledge Graph Visualization")

    # Check for pipeline-built graph first, then demo
    active_graph = None
    if "knowledge_graph" in st.session_state and st.session_state.knowledge_graph:
        active_graph = st.session_state.knowledge_graph
        st.info("Showing knowledge graph built from pipeline")
    else:
        demo_graph = get_demo_knowledge_graph()
        if demo_graph.graph.number_of_nodes() > 0:
            active_graph = demo_graph
            st.info("Showing demo graph")

    if active_graph:
        graph_data = active_graph.to_dict()
        node_count = len(graph_data.get("entities", {}))
    else:
        graph_data = {"entities": {}, "relationships": {}}
        node_count = 0

    if node_count == 0:
        st.info("The knowledge graph is empty. Use the Research tab to build a graph or load a demo.")

        # Demo graph option
        if st.button("Load Demo Graph"):
            from models.knowledge_graph import KnowledgeGraph, RelationshipType

            demo_graph = get_demo_knowledge_graph()

            # Add some demo entities
            demo_graph.add_entity("BRCA1", "protein", "BRCA1")
            demo_graph.add_entity("BRCA2", "protein", "BRCA2")
            demo_graph.add_entity("TP53", "protein", "TP53")
            demo_graph.add_entity("breast_cancer", "disease", "Breast Cancer")

            # Add some relationships
            demo_graph.add_relationship("BRCA1", "BRCA2", RelationshipType.INTERACTS_WITH)
            demo_graph.add_relationship("BRCA1", "TP53", RelationshipType.ACTIVATES)
            demo_graph.add_relationship("BRCA1", "breast_cancer", RelationshipType.ASSOCIATED_WITH)
            demo_graph.add_relationship("TP53", "breast_cancer", RelationshipType.ASSOCIATED_WITH)

            st.rerun()
    else:
        # Render graph
        try:
            html_content = render_graph(graph_data, height="600px")
            components.html(html_content, height=620, scrolling=True)
        except Exception as e:
            st.error(f"Error rendering graph: {e}")

        # Graph stats below visualization
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", node_count)
        col2.metric("Relationships", len(graph_data.get("relationships", {})))

        # Get novel predictions from active graph
        active_summary = active_graph.to_summary() if active_graph else {}
        col3.metric("Novel Predictions", active_summary.get("novel_predictions", 0))


# =============================================================================
# Tab 3: Explore
# =============================================================================

with tab_explore:
    st.subheader("Explore Entities")

    # Use pipeline graph if available
    explore_graph = None
    if "knowledge_graph" in st.session_state and st.session_state.knowledge_graph:
        explore_graph = st.session_state.knowledge_graph
    else:
        explore_graph = get_demo_knowledge_graph()

    entities = list(explore_graph.entities.keys()) if explore_graph else []

    if not entities:
        st.info("No entities in the graph yet. Use the Research tab to build a knowledge graph.")
    else:
        selected_entity = st.selectbox("Select an entity to explore", entities)

        if selected_entity:
            entity_data = explore_graph.entities.get(selected_entity, {})

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** {entity_data.get('name', selected_entity)}")
                st.markdown(f"**Type:** {entity_data.get('type', 'unknown')}")

            # Show neighbors
            neighbors = explore_graph.get_neighbors(selected_entity, max_neighbors=20)

            if neighbors:
                st.subheader("Interactions")

                for neighbor_id, rel_data in neighbors:
                    direction = rel_data.get("direction", "unknown")
                    rel_type = rel_data.get("relation_type", "interacts_with")
                    ml_score = rel_data.get("ml_score")
                    evidence_list = rel_data.get("evidence", [])
                    evidence_count = len(evidence_list)

                    if direction == "outgoing":
                        arrow = "→"
                    else:
                        arrow = "←"

                    info_parts = []
                    if ml_score:
                        info_parts.append(f"ML: {ml_score:.2f}")
                    if evidence_count:
                        info_parts.append(f"Evidence: {evidence_count}")

                    info_str = f" ({', '.join(info_parts)})" if info_parts else ""

                    # Create interaction header
                    interaction_label = f"{selected_entity} {arrow} **{neighbor_id}** [{rel_type}]{info_str}"

                    # If there's evidence with text snippets, show in expander
                    evidence_with_text = [ev for ev in evidence_list if ev.get("text_snippet")]

                    if evidence_with_text:
                        with st.expander(interaction_label, expanded=False):
                            # Group evidence by PMID for cleaner display
                            by_pmid: dict[str, list[str]] = {}
                            for ev in evidence_with_text:
                                pmid = ev.get("source_id", "Unknown")
                                text = ev.get("text_snippet", "")
                                if text:
                                    if pmid not in by_pmid:
                                        by_pmid[pmid] = []
                                    by_pmid[pmid].append(text)

                            for pmid, sentences in by_pmid.items():
                                if pmid and pmid != "Unknown":
                                    st.markdown(f"**[PMID:{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})**")
                                else:
                                    st.markdown("**Source: Unknown**")
                                for sentence in sentences:
                                    st.markdown(f"- _{sentence}_")
                    else:
                        # No text snippets, just show the line
                        st.markdown(f"- {interaction_label}")
            else:
                st.info("No interactions found for this entity.")


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Tetra v1 - Scientific Knowledge Graph Agent | Powered by Google ADK (Gemini) & Multi-Agent Pipeline
</div>
""", unsafe_allow_html=True)
