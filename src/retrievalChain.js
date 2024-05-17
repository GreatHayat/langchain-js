import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { StringOutputParser } from "@langchain/core/output_parsers";

const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
});

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1024,
  chunkOverlap: 200,
});

const prompt =
  ChatPromptTemplate.fromTemplate(`You are helpful AI assitant of DaticsAI.
It is your job to answer the user questions from the given context delimited by xml tags.

<context>
{context}
</context>

Question: {question}
Answer: Helpful answer in markdown language.
`);

(async () => {
  // First, we will load our data as text files from data folder
  const loader = new DirectoryLoader("data", {
    ".txt": (path) => new TextLoader(path),
  });

  const documents = await loader.loadAndSplit(textSplitter);

  const vectorStore = await QdrantVectorStore.fromDocuments(
    documents,
    embeddings,
    {
      apiKey: process.env.QDRANT_API_KEY,
      url: process.env.QDRANT_URL,
      collectionName: process.env.QDRANT_COLLECTION,
    }
  );

  const question = "How to contact with DaticsAI?";

  const retriever = vectorStore.similaritySearch(question, 5);
  const ragChain = await createStuffDocumentsChain({
    llm,
    prompt,
    outputParser: new StringOutputParser(),
  });

  const response = await ragChain.invoke({
    question,
    context: retriever,
  });
  console.log(response);
})();
