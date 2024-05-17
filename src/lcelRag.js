import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";

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

  // If you already have a collection, you can use QdrantVectorStore.fromExistingCollection
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

  const retriever = vectorStore.asRetriever(5);

  // LangChain LCEL
  const chain = RunnableSequence.from([
    {
      question: new RunnablePassthrough(),
      context: retriever.pipe(formatDocumentsAsString),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  const response = await chain.invoke(question);
  console.log(response);
})();
