import { StateGraph, END, Annotation } from "@langchain/langgraph";
import { AIMessage, BaseMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { similaritySearch } from "../database/vector-store";
import { getSyncStatus } from "../services/sync_manager";
import { getKeys } from "../services/auth";
import { generateEmbedding } from "../services/synthesis/embedding_cache";

// Define State
const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  context: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "",
  }),
  intent: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "unknown",
  }),
});

// Helper for LLM
async function callLLM(systemPrompt: string, userMessage: string): Promise<string> {
    const keys = getKeys();
    
    console.log('[Agent] Checking API key:', { hasKey: !!keys.openaiKey, keyLength: keys.openaiKey?.length });
    
    if (!keys.openaiKey || keys.openaiKey.trim() === '') {
        return "OpenAI API key not configured. Please add your API key in the login form and reconnect.";
    }
    
    try {
        const model = new ChatOpenAI({
            apiKey: keys.openaiKey,
            modelName: "gpt-4o-mini",
            temperature: 0.7,
        });
        
        const response = await model.invoke([
            { role: "system", content: systemPrompt },
            { role: "user", content: userMessage }
        ]);
        
        return response.content as string;
    } catch (error: any) {
        console.error('LLM call failed:', error);
        return `Error calling LLM: ${error.message}`;
    }
}

// Nodes

async function supervisor(state: typeof AgentState.State) {
    const lastMessage = state.messages[state.messages.length - 1];
    const text = typeof lastMessage.content === 'string' ? lastMessage.content : '';
    
    const lower = text.toLowerCase();
    let intent = 'content';
    if (lower.includes('grade') || lower.includes('sync')) {
        intent = 'grades_sync';
    }
    return { intent };
}

async function retriever(state: typeof AgentState.State) {
    try {
        const lastMessage = state.messages[state.messages.length - 1];
        const query = typeof lastMessage.content === 'string' ? lastMessage.content : '';
        
        console.log(`[Retriever] Searching for: "${query}"`);
        
        // Generate embedding using the same model as synthesis (all-MiniLM-L6-v2)
        const queryVector = await generateEmbedding(query);
        console.log(`[Retriever] Generated query embedding (${queryVector.length} dimensions)`);
        
        // Search vector store
        const results = await similaritySearch(queryVector, 10);
        
        // Handle results being an array or iterator
        const resultsArray = Array.isArray(results) ? results : [];
        console.log(`[Retriever] Found ${resultsArray.length} results`);
        
        if (resultsArray.length === 0) {
            return { context: "No assignments found. Please sync your assignments first." };
        }
        
        const context = resultsArray.map((r: any) => r.text).join("\n\n");
        return { context };
    } catch (error: any) {
        console.error('Error searching vector store:', error);
        return { context: "Vector database not initialized. Please sync your assignments first using 'Sync Now'." };
    }
}

async function syncChecker(_state: typeof AgentState.State) {
    const status = getSyncStatus();
    return { context: status };
}

async function generator(state: typeof AgentState.State) {
    const { messages, context } = state;
    const lastMessage = messages[messages.length - 1];
    const text = typeof lastMessage.content === 'string' ? lastMessage.content : '';
    
    const response = await callLLM(
        `You are a helpful assistant. Use the following context to answer: ${context}`,
        text
    );
    
    return { messages: [new AIMessage(response)] };
}

// Graph
const workflow = new StateGraph(AgentState)
    .addNode("supervisor", supervisor)
    .addNode("retriever", retriever)
    .addNode("sync_checker", syncChecker)
    .addNode("generator", generator)
    .addEdge("retriever", "generator")
    .addEdge("sync_checker", "generator")
    .addEdge("generator", END)
    .addConditionalEdges(
        "supervisor",
        (state) => {
            if (state.intent === 'grades_sync') return "sync_checker";
            return "retriever";
        }
    )
    .setEntryPoint("supervisor");

export const agent = workflow.compile();
