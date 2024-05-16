import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 0.2,
});

(async () => {
  const response = await llm.invoke("Write 2 liner summary about NodeJS");
  console.log(response.content);
})();
