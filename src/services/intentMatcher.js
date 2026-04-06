import embeddingService from './embeddingService.js';
import vectorStore from './vectorStore.js';
import dataLoader from './dataLoader.js';

/**
 * Service for matching human text against pre-defined intents using vector similarity.
 */
class IntentMatcher {
    constructor() {
        this.cachePath = './data/knowledge-base.json';
    }

    /**
     * Re-initializes the vector store.
     */
    async reload(options = {}) {
        await vectorStore.init(options);
    }

    /**
     * Matches a query string against the optimized intent variations in the VectorStore.
     * @param {string} query 
     * @param {number} topK 
     */
    async match(query, topK = 5) {
        if (!query) return [];

        // 1. Ensure vectors are loaded
        if (!vectorStore.isLoaded) {
            await this.reload(); 
        }

        // 2. Generate embedding for query
        const [queryEmbeddingRaw] = await embeddingService.getEmbeddings([query]);
        const queryEmbedding = new Float32Array(queryEmbeddingRaw); // Ensure it's a TypedArray

        // 3. Match against VectorStore
        return vectorStore.matchIntents(queryEmbedding, topK);
    }

    /**
     * Logic for generating new intent embeddings from raw data.
     * Used only when the knowledge base needs a full refresh.
     */
    async regenerate(options = {}) {
        console.log('[IntentMatcher] Regenerating intent embeddings from raw data...');
        const rawData = await dataLoader.loadIntents();
        
        // (The logic for regenerating would go here if we wanted to support writing to the JSON cache. 
        // For now, we assume the JSON is generated offline or by a dedicated script).
    }
}

export default new IntentMatcher();
