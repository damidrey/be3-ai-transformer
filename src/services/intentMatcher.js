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

    get intentEmbeddings() {
        return vectorStore.intents.map(item => ({
            intentName: item.intentName,
            text: item.text,
            embedding: Array.from(item.embedding)
        }));
    }

    /**
     * Re-initializes the vector store.
     */
    async reload(options = {}) {
        if (options.skipCache) {
            await this.regenerate(options);
        } else {
            await vectorStore.init(options);
        }
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

    async regenerate(options = {}) {
        console.log('[IntentMatcher] Regenerating intent embeddings from raw data...');
        const rawData = await dataLoader.loadIntents();
        
        const allVariations = [];
        const intentNames = [];

        for (const [intentName, variations] of Object.entries(rawData)) {
            variations.forEach(v => {
                allVariations.push(v);
                intentNames.push(intentName);
            });
        }

        const total = allVariations.length;
        if (total === 0) return;

        console.log(`[IntentMatcher] Generating embeddings for ${total} variations...`);
        const BATCH_SIZE = 50; 
        const allEmbeddings = [];

        for (let i = 0; i < total; i += BATCH_SIZE) {
            const chunk = allVariations.slice(i, i + BATCH_SIZE);
            const embeddings = await embeddingService.getEmbeddings(chunk);
            allEmbeddings.push(...embeddings);
            
            const progress = Math.min(100, ((i + chunk.length) / total * 100)).toFixed(1);
            console.log(`[IntentMatcher] Progress: ${progress}%`);
            await new Promise(resolve => setTimeout(resolve, 10)); // Yield to CPU
        }

        const intents = allVariations.map((text, i) => ({
            intentName: intentNames[i],
            text,
            embedding: new Float32Array(allEmbeddings[i])
        }));

        vectorStore.setKnowledge(intents, vectorStore.entities);
        console.log(`[IntentMatcher] Successfully regenerated ${intents.length} intents.`);
    }
}

export default new IntentMatcher();
