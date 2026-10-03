# Revision Repo for Langchain Course by Eden Marco 

Original repo: <a href="https://github.com/emarco177/langchain-course" target="_blank">link</a>



# Role Play Questions:

- Explain what is an agent?
- Explain the core compenents of a langchain React agent?
- Describe the iterative Thoought-Action_observation loopthe drives the ReAct agent reasoning.
- Articulate how the agent uses observations from the tool to refine its thoughts and determine the next action.

# Observed Questions:

#### Lecture 41

- What is function calling?
- How model vendors enable function calling for a model?
- How does function calling differ from ReAct prompt?
- What were the drawbacks of the ReAct prompt that generated the need for function calling?
- What are the advantages of function calling?
- What are the disadvantages of function calling?

#### Lecture 42

- When asking questions over a large document:
    - What are the challenges of stuffing entire documents into the prompt for Q&A?
    - What if we split up a doc into chunks, find the chunk most relevant to a query and stuff only that / those specific chunks in the LLM - does it solve any issues with the previosu approach and what are the challenges of this new approach?
- How do you breakdown the term RAG?

#### Lecture 43

- Provide a high level, step-by-step view of RAG.
- Explain the differences between KV DB, Document DB, Graph DB and vector DB.

#### Lecture 44
- What are the different metrics used to calculate similarity between vectors in a vector DB (coside, euclidean, dotproduct etc.)?
- Draw a block diagram representation of the Data Indexing pahse as well as the Data Retreival & generation pahse of a basic RAG Pipeline.

#### Lecture 46
- Describe the four steps in the used in the Ingestion step of RAG viz Load, Split, Embed and Store.

#### Lecture 48
- What are the disadvantages of building RAG by invoking different langchain components separately? (Hint: Different steps will appear in different traces)

#### Lecture 49
- Explain how itemgetter in operator module works with examples.
- Explain how RunnablePassthrough is used in Langchain




#### General

- Your RAG system is giving incorrect answers. How would you determine whether the problem is retieival or generation?
- What is the use of metadata fileds in a vector store?



