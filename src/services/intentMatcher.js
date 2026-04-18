import embeddingService from './embeddingService.js';
import vectorStore from './vectorStore.js';
import dataLoader from './dataLoader.js';

/**
 * Service for matching human text against pre-defined intents using vector similarity.
 * 
 * Supports two modes:
 *   1. Legacy (flat match) — match() against all intents in the flat pool
 *   2. Hierarchical match — matchByLevel() against a specific level/parent pool
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
     * (Legacy — used by /analyze endpoint)
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

    // ═══════════════════════════════════════════════
    // Hierarchical matching (Phase 2: micarch)
    // ═══════════════════════════════════════════════

    /**
     * Classify a query against a specific hierarchical level.
     * @param {string} query - User text
     * @param {'class'|'intent'|'subintent'} level - Hierarchy level
     * @param {string|null} parent - Parent context (null for class, class name for intent, etc.)
     * @param {number} topK
     * @returns {Object} { level, parent, winner, scores: [{ name, score, bestMatch }] }
     */
    async matchByLevel(query, level, parent = null, topK = 5) {
        if (!query) return { level, parent, winner: null, scores: [] };

        // 1. Ensure hierarchical pools are loaded
        if (!vectorStore.isHierarchicalLoaded) {
            await vectorStore.initHierarchical();
        }

        // 2. Generate embedding for query
        const [queryEmbeddingRaw] = await embeddingService.getEmbeddings([query]);
        const queryEmbedding = new Float32Array(queryEmbeddingRaw);

        // 3. Match against the specific pool
        const scores = await vectorStore.matchPool(queryEmbedding, level, parent, topK);

        return {
            level,
            parent,
            winner: scores.length > 0 ? scores[0].name : null,
            scores
        };
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
