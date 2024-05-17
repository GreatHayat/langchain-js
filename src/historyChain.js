import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI } from "@langchain/openai";
import { ConversationChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { RedisChatMessageHistory } from "@langchain/community/stores/message/ioredis";

const memory = new BufferMemory({
  chatHistory: new RedisChatMessageHistory({
    url: "127.0.0.1",
    port: 6379,
  }),
});

// initialize your llm model
const llm = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 0,
});

(async () => {
  const chain = new ConversationChain({
    llm,
    memory,
    outputParser: new StringOutputParser(),
  });

  const response = await chain.invoke({
    input: "What is the capital of Pakistan?",
  });

  const response2 = await chain.invoke({
    input: "What was my first question",
  });

  console.log(response);
  console.log(response2);
})();
