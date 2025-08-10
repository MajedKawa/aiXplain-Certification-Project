# Policy Navigator Agent
A Multi-Agent RAG System for Government Regulation Search

This repository represents the project needed for aiXplain certification, which is an Agentic RAG system that allows users to query and extract insights from complex government regulations, compliance policies, or public health guidelines.

## Features

Multi-Agent Architecture

* Document RAG Agent: Analyzes your uploaded PDF documents using a global index

* EPA Agent (first data source, a website: https://www.epa.gov/laws-regulations): Environmental Protection Agency regulations and compliance queries

* Case Law Agent: Legal precedent research using CourtListener API

* GDPR Agent (second data source, a dataset: https://www.kaggle.com/datasets/jessemostipak/gdpr-violations): Data privacy and GDPR compliance information

* Web Scraper Agent: Real-time information retrieval from websites

* Slack Integration: Send notifications about important findings

## Quick Start

Installation

1. Clone the repository

```bash
git clone https://github.com/MajedKawa/aiXplain-Certification-Project
cd policy-navigator-agent
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create a .env file
echo "AIXPLAIN_API_KEY=your_api_key_here" > .env
```

4. Run the application
```bash
streamlit run policy_navigator_ui.py
```

## Usage

Getting Started

1. Enter your aiXplain API Key in the sidebar

2. Upload PDF documents you want to analyze (policies, regulations, reports, etc.)

3. Click "Process Files & Setup Agents" to initialize the system

4. Start chatting with your Policy Navigator Agent!

## Integration with Slack

Currently, the agent sends notifications to this workspace (https://join.slack.com/t/policynavigatorbot/shared_invite/zt-3aut2cxw5-TWb2YXTOGbg5ok424mkLXg). This is for testing and will expire soon. However, you can modify this, and any detail you want, by changing the internal agent's or any agent code (Check buildingTheAgents.ipynb notebook to modify the current implementation)

## Example inputs/outputs

#1 Ask the model a question, provided a website URL 

User Prompt: Using this website "Using this website "https://www.who.int/publications/i/item/9789240111110", what are the ways Arboviral diseases are transmitted?”
Response: 
Arboviral diseases are primarily transmitted by Aedes mosquitoes. These diseases include dengue, chikungunya, Zika, and yellow fever. The transmission of these viruses poses an increasing threat to global health, particularly due to factors such as climate change, urbanization, and growing human mobility. If you have any further questions or need more information, feel free to ask!

#2 Asking the model a question related to the dataset (datasource):

One of the data sources in my project is a dataset about GDPR.
Here, the user query is “Provide me with examples of GDPR violations in Romania”
And the team agent selects the needed GDPR agent to answer the question correctly based on its indexed data.

#3 Asking the model a question related to a document

Here, the user asks the agent a question and provides the document name.
The agent in this case will search in the global index trying to finds an answer to the user’s question
For example, the user query “Based on this document: 'who_guidelines.pdf' provide examples of health risks due to poor housing”
The team agent then selects the necessary agent (Global Document RAG Agent) to answer the question correctly, based on the global document indexed.

#4 Ask the model to send notifications on a Slack channel

The user query: Send a notification in Slack with this data "Hello world from Agent!"
The agent uses the Slack notification agent to send the notification in the Slack channel

#5 Testing the agent with its indexed data (from the website datasource (EPA website))

The query: Using your knowledge, What does the Clean Air Act regulate?
And the agent used the correct agent (EPA Search Assistant) and answered the query correctly

#6 Retrieve case laws linked to specific regulations (Using CourtListener API (via the Free Law Project))

The query: “Has Section 230 ever been challenged in court? What was the outcome?”
And the mentalist uses the correct agent (Case Finder Agent) to fulfill the user’s query.

## Suggested Future Enhancements

* Auto-Save Sessions: Persist chat history and uploaded documents between sessions

* Custom Agent Creation: Allow users to create specialized agents for specific domains

* Advanced Search Filters: Filter by document type, date range, or specific policy categories
