from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv()) # Load environment variables (AIXPLAIN_API_KEY) from .env file

import streamlit as st
import hashlib
import time
import tempfile
from pathlib import Path
import pandas as pd
from aixplain.factories import AgentFactory, TeamAgentFactory, ModelFactory, IndexFactory
from aixplain.modules.model.record import Record

# Set page config
st.set_page_config(
    page_title="Policy Navigator Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    if 'global_rag_manager' not in st.session_state:
        st.session_state.global_rag_manager = None
    if 'file_rag_agent' not in st.session_state:
        st.session_state.file_rag_agent = None
    if 'policy_navigator_agent' not in st.session_state:
        st.session_state.policy_navigator_agent = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False

class GlobalRAGManager:
    """Manages a single global index with files as records for RAG operations"""

    def __init__(self, index_name: str = "GlobalFileRAG"):
        self.index_name = index_name
        self.index = None
        self.indexed_files = {}  # Track indexed files: {filename: record_id}
        self.docling_model = ModelFactory.get("677bee6c6eb56331f9192a91")
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or get existing global index"""
        try:
            # Try to get existing index first
            self.index = IndexFactory.get(self.index_name)
            st.success(f"Using existing global index: {self.index_name}")
        except:
            # Create new index if it doesn't exist
            self.index = IndexFactory.create(
                name=self.index_name,
                description="Global RAG index for all PDF documents"
            )
            st.success(f"Created new global index: {self.index_name}")

    def add_file_to_index(self, file_path: str, original_filename: str) -> str:
        """Add a single PDF file to the global index as a record"""
        try:
            st.info(f"Adding {original_filename} to global index...")

            # Check if file already indexed
            if original_filename in self.indexed_files:
                st.warning(f"File {original_filename} already in index")
                return self.indexed_files[original_filename]

            # Extract text content
            with st.spinner(f"Extracting content from {original_filename}..."):
                text_response = self.docling_model.run(file_path)
                text_content = text_response.data

            if not text_content or text_content.strip() == "":
                raise ValueError(f"No text content extracted from {original_filename}")

            # Create unique record ID
            content_hash = hashlib.md5(text_content.encode()).hexdigest()[:8]
            record_id = f"file_{original_filename.replace('.', '_')}_{content_hash}"

            # Create record
            record = Record(
                id=record_id,
                value=text_content,
                attributes={
                    "filename": original_filename,
                    "document_type": "pdf",
                    "indexed_at": str(int(time.time())),
                    "content_hash": content_hash
                }
            )

            # Upsert record to global index
            self.index.upsert([record])
            self.indexed_files[original_filename] = record_id

            st.success(f"Successfully added {original_filename} to global index")
            return record_id

        except Exception as e:
            st.error(f"Error adding {original_filename} to index: {e}")
            raise

    def get_index_info(self):
        """Get information about the global index"""
        return {
            "index_name": self.index_name,
            "index_id": self.index.id,
            "total_records": self.index.count(),
            "indexed_files": list(self.indexed_files.keys())
        }
        
    def delete_index(self):
        """Delete the global index and clean up resources"""
        try:
            if self.index:
                self.index.delete()
                st.success(f"Successfully deleted index: {self.index_name}")
                self.index = None
                self.indexed_files = {}
                return True
        except Exception as e:
            st.error(f"Error deleting index: {e}")
            return False
    
def create_global_rag_agent(global_rag_manager):
    """Create a single agent that uses the global index"""
    
    if not global_rag_manager.indexed_files:
        st.error("No files indexed yet. Upload and index files first.")
        return None
    
    instructions = f"""
    You are a Global Document RAG Agent that can search and analyze content from multiple PDF documents using a single global index.

    AVAILABLE DOCUMENTS: {list(global_rag_manager.indexed_files.keys())}

    CAPABILITIES:
    - Search across all indexed PDF documents simultaneously
    - Extract specific information from government documents, reports, policies
    - Provide detailed answers with document citations
    - Handle queries about regulations, compliance requirements, legal documents
    - Compare information across multiple documents

    INSTRUCTIONS:
    1. Use the global search tool to find relevant information across all documents
    2. Always cite which specific document(s) your information comes from
    3. If asked about a specific file, you can search for it by including the filename in your query
    4. Provide comprehensive answers with relevant context from the documents
    5. When information spans multiple documents, clearly distinguish between sources

    You excel at:
    - Finding specific regulatory requirements across policy documents
    - Extracting compliance guidelines from multiple government publications
    - Analyzing and comparing legal documents and case files
    - Summarizing key points from technical reports
    - Cross-referencing information between documents
    """

    # Create agent with single global search tool
    agent = AgentFactory.create(
        name="Global Document RAG Agent",
        description="Agent that searches across all indexed PDF documents using a global index",
        instructions=instructions,
        tools=[
            AgentFactory.create_model_tool(
                model=global_rag_manager.index.id,
                name="global_document_search"
            )
        ]
    )

    return agent

def delete_agent(agent, agent_name):
    """Delete an agent and free up resources"""
    try:
        if agent:
            agent.delete()
            st.success(f"Successfully deleted {agent_name}")
            return True
    except Exception as e:
        st.error(f"Error deleting {agent_name}: {e}")
        return False

def cleanup_all_resources():
    """Clean up all created agents and indexes"""
    cleanup_success = True
    
    with st.spinner("Cleaning up resources..."):
        # Delete team agent
        if st.session_state.policy_navigator_agent:
            if delete_agent(st.session_state.policy_navigator_agent, "Policy Navigator Team Agent"):
                st.session_state.policy_navigator_agent = None
            else:
                cleanup_success = False
        
        # Delete document RAG agent
        if st.session_state.file_rag_agent:
            if delete_agent(st.session_state.file_rag_agent, "Document RAG Agent"):
                st.session_state.file_rag_agent = None
            else:
                cleanup_success = False
        
        # Delete global index
        if st.session_state.global_rag_manager:
            if st.session_state.global_rag_manager.delete_index():
                st.session_state.global_rag_manager = None
            else:
                cleanup_success = False
    
    return cleanup_success

def create_team_agent(file_rag_agent):
    """Create the Policy Navigator Team Agent"""
    
    # Pre-created agent IDs
    EPA_AGENT_ID = "689763bd08d90b9df2c186e4" 
    SCRAPER_AGENT_ID = "6897639b686ec479bfbf910c"  
    CASE_LAW_AGENT_ID = "689763c108d90b9df2c186e5"  
    GDPR_AGENT_ID = "68983c2908d90b9df2c1a478"
    SLACK_AGENT_ID = "689763f708d90b9df2c186e7"
    
    instructions = """You are the Policy Navigator Agent - an advanced agentic RAG system for comprehensive government regulation and policy research.

    Agents CAPABILITIES:
    1. EPA_agent: Answers queries related to Environmental Protection Agency regulations and compliance
    2. scraper_agent: Web scraping agent for retrieving information from websites
    3. caseLawAgent: Case law research and legal precedent analysis using CourtListener API
    4. gdpr_agent: Answers queries related to GDPR and data privacy regulations
    5. slack_agent: Sends notifications to Slack channels
    6. Local Document RAG Agent: Analysis of uploaded PDF documents, reports, and local files

    WORKFLOW:
    1. Analyze the user query to determine which agents are most relevant
    2. Route initial research to appropriate specialist agents
    3. Cross-reference findings between agents when beneficial
    4. Synthesize comprehensive responses combining multiple sources
    5. Always cite sources and distinguish between different types of information

    When you find important policy information, compliance requirements, or regulatory changes:
    1. Provide the information to the user
    2. Offer to send Slack notifications about important findings
    3. Use appropriate notification types based on urgency and content type

    SLACK INTEGRATION:
    - Use the Slack Notification Agent for sending alerts and updates
    - Include source information and effective dates when available
    - For queries about specific uploaded documents, route to Local Document RAG Agent
    """
    
    try:
        # Get pre-created agents
        agents = []
        
        # Try to get each agent (comment out if not available)
        try:
            epa_agent = AgentFactory.get(EPA_AGENT_ID)
            agents.append(epa_agent)
        except:
            st.warning("EPA Agent not found - continuing without it")
        
        try:
            scraper_agent = AgentFactory.get(SCRAPER_AGENT_ID)
            agents.append(scraper_agent)
        except:
            st.warning("Scraper Agent not found - continuing without it")
        
        try:
            case_law_agent = AgentFactory.get(CASE_LAW_AGENT_ID)
            agents.append(case_law_agent)
        except:
            st.warning("Case Law Agent not found - continuing without it")
        
        try:
            gdpr_agent = AgentFactory.get(GDPR_AGENT_ID)
            agents.append(gdpr_agent)
        except:
            st.warning("GDPR Agent not found - continuing without it")
        
        try:
            slack_agent = AgentFactory.get(SLACK_AGENT_ID)
            agents.append(slack_agent)
        except:
            st.warning("Slack Agent not found - continuing without it")
        
        # Always include the file RAG agent
        agents.append(file_rag_agent)
        
        team_agent = TeamAgentFactory.create(
            name="Policy Navigator Agent",
            description="Agentic RAG System for Government Regulation Search",
            instructions=instructions,
            agents=agents,
            use_mentalist=True,
        )
        
        return team_agent
    
    except Exception as e:
        st.error(f"Error creating team agent: {e}")
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    return file_path

def main():
    st.title("Policy Navigator Agent")
    st.markdown("---")
    
    initialize_session_state()
    
    # Sidebar for setup and configuration
    with st.sidebar:
        st.header("Setup & Configuration")
        
        # API Key input
        api_key = st.text_input("aiXplain API Key", type="password", help="Enter your aiXplain API key")
        if api_key:
            os.environ["AIXPLAIN_API_KEY"] = api_key
            st.success("API Key set!")
        
        st.markdown("---")
        
        # File upload section
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents to be indexed and searched"
        )
        
        if uploaded_files and not st.session_state.setup_complete:
            if st.button("Process Files & Setup Agents", type="primary"):
                setup_agents(uploaded_files)
        
        # Display indexed files
        if st.session_state.global_rag_manager:
            st.markdown("---")
            st.header("Indexed Documents")
            indexed_files = list(st.session_state.global_rag_manager.indexed_files.keys())
            for file in indexed_files:
                st.text(f"✓ {file}")
        
        # Cleanup and reset buttons
        if st.session_state.setup_complete:
            st.markdown("---")
            st.header("Cleanup & Management")
            
            delete_all_resources()
    
            # Keep the reset system button separate
            if st.button("Reset Chat Only", 
                    help="Clear chat history but keep resources active",
                    type="secondary"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()


    # Main content area
    if not st.session_state.setup_complete:
        show_welcome_screen()
    else:
        show_chat_interface()

def setup_agents(uploaded_files):
    """Setup the RAG manager and agents with uploaded files"""
    try:
        with st.spinner("Setting up your Policy Navigator system..."):
            # Initialize RAG manager
            progress_bar = st.progress(0)
            st.session_state.global_rag_manager = GlobalRAGManager()
            progress_bar.progress(20)
            
            # Process uploaded files
            temp_files = []
            for i, uploaded_file in enumerate(uploaded_files):
                st.info(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Save file temporarily
                temp_path = save_uploaded_file(uploaded_file)
                temp_files.append((temp_path, uploaded_file.name))
                
                # Add to index
                st.session_state.global_rag_manager.add_file_to_index(temp_path, uploaded_file.name)
                
                progress_bar.progress(20 + (50 * (i + 1) // len(uploaded_files)))
            
            # Create document RAG agent
            st.info("Creating Document RAG Agent...")
            st.session_state.file_rag_agent = create_global_rag_agent(st.session_state.global_rag_manager)
            progress_bar.progress(80)
            
            # Create team agent
            st.info("Creating Policy Navigator Team Agent...")
            st.session_state.policy_navigator_agent = create_team_agent(st.session_state.file_rag_agent)
            progress_bar.progress(100)
            
            # Clean up temporary files
            for temp_path, _ in temp_files:
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            st.session_state.setup_complete = True
            st.success("Setup complete! You can now start chatting with your Policy Navigator Agent.")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during setup: {e}")

def show_welcome_screen():
    """Show welcome screen when system is not set up"""
    st.markdown("""
    ## Welcome to Policy Navigator Agent!
    
    This is an advanced agentic RAG system for comprehensive government regulation and policy research.
    
    ### Getting Started:
    
    1. **Enter your aiXplain API Key** in the sidebar
    2. **Upload your PDF documents** using the file uploader in the sidebar
    3. **Click "Process Files & Setup Agents"** to initialize the system
    4. **Start chatting** with your Policy Navigator Agent!
    
    ### What happens during setup:
    
    - Your PDF documents are processed and indexed
    - A specialized Document RAG Agent is created for your files
    - The main Policy Navigator Team Agent is assembled
    - Multiple specialized agents are integrated (EPA, Case Law, GDPR, etc.)
    
    ### What you can do:
    
    - Search across all your uploaded documents
    - Query EPA regulations and environmental policies
    - Research case law and legal precedents
    - Get GDPR compliance information
    - Send notifications to Slack channels
    - Cross-reference information from multiple sources
    
    ---
    **Ready to get started? Upload your files in the sidebar!**
    """)

def show_chat_interface():
    """Show the main chat interface"""
    st.header("Chat with Policy Navigator Agent")
    
    # Display index information
    with st.expander("User indexed files information", expanded=False):
        if st.session_state.global_rag_manager:
            info = st.session_state.global_rag_manager.get_index_info()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", len(info['indexed_files']))
            with col2:
                st.metric("Total Records", info['total_records'])
            with col3:
                st.metric("Index ID", info['index_id'][:8] + "...")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask me about policies, regulations, or your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Display user message
        st.chat_message("user").write(query)
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your query and searching across all sources..."):
                try:
                    if st.session_state.policy_navigator_agent:
                        response = st.session_state.policy_navigator_agent.run(query)
                        answer = response['data']['output']
                    else:
                        # Fallback to document agent only
                        response = st.session_state.file_rag_agent.run(query)
                        answer = response.data.output
                    
                    st.write(answer)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Show execution stats if available
                    if hasattr(response, 'execution_stats'):
                        with st.expander("Execution Details"):
                            stats = response.execution_stats if hasattr(response, 'execution_stats') else response['execution_stats']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Runtime", f"{stats.get('runtime', 0):.2f}s")
                            with col2:
                                st.metric("Credits Used", f"{stats.get('credits', 0):.5f}")
                            with col3:
                                st.metric("API Calls", stats.get('api_calls', 0))
                
                except Exception as e:
                    st.error(f"Error processing your query: {e}")
                    st.info("Try rephrasing your question or check that your agents are properly deployed.")

    # Quick actions at bottom
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Chat History"):
            export_chat_history()
    
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col3:
        if st.button("Show System Status"):
            show_system_status()
            
def export_chat_history():
    """Export chat history as downloadable file"""
    if st.session_state.chat_history:
        chat_data = []
        for msg in st.session_state.chat_history:
            chat_data.append(f"**{msg['role'].upper()}:** {msg['content']}\n")
        
        chat_text = "\n".join(chat_data)
        
        st.download_button(
            label="Download Chat History",
            data=chat_text,
            file_name=f"policy_navigator_chat_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.info("No chat history to export.")
        
def show_system_status():
    """Show detailed system status"""
    with st.expander("Detailed System Status", expanded=True):
        if st.session_state.global_rag_manager:
            info = st.session_state.global_rag_manager.get_index_info()
            
            st.subheader("Index Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Index Name", info['index_name'])
                st.metric("Total Documents", len(info['indexed_files']))
            with col2:
                st.metric("Index ID", info['index_id'])
                st.metric("Total Records", info['total_records'])
            
            st.subheader("Indexed Documents")
            for i, filename in enumerate(info['indexed_files'], 1):
                st.write(f"{i}. {filename}")
        
        st.subheader("Agent Status")
        agents_status = {
            "Document RAG Agent": "Active" if st.session_state.file_rag_agent else "Not Created",
            "Policy Navigator Team Agent": "Active" if st.session_state.policy_navigator_agent else "Not Created"
        }
        
        for agent, status in agents_status.items():
            st.write(f"**{agent}:** {status}")
        
        st.subheader("Chat Statistics")
        total_messages = len(st.session_state.chat_history)
        user_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
        assistant_messages = len([msg for msg in st.session_state.chat_history if msg['role'] == 'assistant'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", total_messages)
        with col2:
            st.metric("User Messages", user_messages)
        with col3:
            st.metric("Assistant Messages", assistant_messages)
         
def delete_all_resources():
    """Delete all resources with confirmation"""
    if st.button("DELETE ALL RESOURCES", type="primary", help="This will permanently delete all agents and indexes"):
        with st.spinner("Deleting all resources..."):
            success = cleanup_all_resources()
            
            if success:
                st.success("All resources successfully deleted!")
                st.info("Your aiXplain account is no longer being charged for these resources.")
                
                # Reset session state completely
                keys_to_reset = ['global_rag_manager', 'file_rag_agent', 'policy_navigator_agent', 
                               'uploaded_files', 'chat_history', 'setup_complete']
                
                for key in keys_to_reset:
                    if key in st.session_state:
                        st.session_state[key] = [] if key in ['uploaded_files', 'chat_history'] else None
                
                st.session_state.setup_complete = False
                st.balloons()  # This should now work
                st.success("System reset complete! You can now start fresh.")
                
                # Force page refresh
                time.sleep(1)  # Give balloons time to show
                st.rerun()
                
            else:
                st.error("Some resources could not be deleted. Please check manually.")

def reset_system():
    """Reset the system without deleting resources"""
    st.warning("Resetting system (keeping resources active)...")
    
    # Clear session state but keep resources alive
    for key in ['uploaded_files', 'chat_history']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.chat_history = []
    st.success("System reset! Resources are still active.")
    st.rerun()

if __name__ == "__main__":
    main()