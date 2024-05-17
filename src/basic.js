import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

// initialize your llm model
const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 0.2,
});

// define your prompt template, you can also use fromTemplate method
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are helpful AI assistant to write a short summary for a given topic.",
  ],
  ["human", "{topic}"],
]);

// define your output parser
const outputParser = new StringOutputParser();

(async () => {
  // Now we will create a basic chain, it will combine the prompt with the llm model
  // We will use LangChain LCEL syntax to create the chain, pipe method will be used to create the chain
  // prompt -> llm model -> response
  const chain = prompt.pipe(llm).pipe(outputParser);

  // Please keep in mind that, the name of your key in the invoke method should be same
  // as you use in the prompt within {} braces.
  const response = await chain.invoke({ topic: "AWS S3" });
  console.log(response);
})();
