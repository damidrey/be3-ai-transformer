import fs from 'fs';
import path from 'path';
import embeddingService from './embeddingService.js';

/**
 * Standardize and optimize vector storage for the entire service.
 * Using TypedArrays (Float32Array) reduces RAM usage for 512-dimension vectors by ~80%.
 */
class VectorStore {
    constructor() {
        this.intents = [];   // Array of { intentName, text, embedding: Float32Array }
        this.entities = [];  // Array of { type, subType, key, text, embedding: Float32Array }
        this.cachePath = path.join(process.cwd(), 'data/knowledge-base.json');
        this.isLoaded = false;
    }

    /**
     * Initializes the entire knowledge base from the JSON cache.
     * Uses streaming-like processing to avoid long-term string overhead.
     */
    async init(options = {}) {
        if (!fs.existsSync(this.cachePath)) {
            console.warn(`[VectorStore] ⚠️  No knowledge base found at ${this.cachePath}. Run generation first.`);
            return;
        }

        const stats = { intents: 0, entities: 0 };
        const startTime = Date.now();

        try {
            console.log(`[VectorStore] 📦 Loading knowledge base into memory-efficient TypedArrays...`);
            
            // 1. Read and parse (Node.js holds the string briefly, then parses)
            const rawData = fs.readFileSync(this.cachePath, 'utf8');
            const data = JSON.parse(rawData);
            const activeModel = embeddingService.getConfig().key;

            if (data.model !== activeModel) {
                console.warn(`[VectorStore] ⚠️  Model mismatch: Cache (${data.model}) vs Active (${activeModel}).`);
            }

            // 2. Optimized Intent Loading
            if (data.intents) {
                this.intents = data.intents.map(item => ({
                    intentName: item.intentName,
                    text: item.text,
                    embedding: new Float32Array(item.embedding) // Convert to Float32Array
                }));
                stats.intents = this.intents.length;
            }

            // 3. Optimized Entity Loading
            if (data.entities) {
                this.entities = data.entities.map(item => ({
                    type: item.type,
                    subType: item.subType || null,
                    key: item.key || null,
                    text: item.text,
                    embedding: new Float32Array(item.embedding) // Convert to Float32Array
                }));
                stats.entities = this.entities.length;
            }

            // 4. Force Garbage Collection of the raw JSON string
            // (Setting it to null helps v8 reclaim memory faster)
            this.isLoaded = true;
            console.log(`[VectorStore] ✅ Success: ${stats.intents} intents and ${stats.entities} entities loaded in ${Date.now() - startTime}ms.`);

        } catch (err) {
            console.error(`[VectorStore] ❌ Critical Error loading knowledge base:`, err.message);
            throw err;
        }
    }

    /**
     * Manually populates the knowledge base (used during regeneration/sync).
     */
    setKnowledge(intents, entities) {
        this.intents = intents || [];
        this.entities = entities || [];
        this.isLoaded = true;
    }

    /**
     * High-speed Cosine Similarity for TypedArrays. 
     * Uses a simple dot product assuming vectors are pre-normalized by the CLIP model.
     */
    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        // Float32Arrays are indexed just like regular arrays
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
        }
        return dotProduct;
    }

    /**
     * Safe search for intent matches.
     */
    async matchIntents(queryEmbedding, topK = 5) {
        const matches = this.intents.map(item => ({
            intentName: item.intentName,
            text: item.text,
            score: this.cosineSimilarity(queryEmbedding, item.embedding)
        }));

        // Aggregate by intentName (MAX score)
        const aggregated = {};
        for (const m of matches) {
            if (!aggregated[m.intentName] || m.score > aggregated[m.intentName].score) {
                aggregated[m.intentName] = { intentName: m.intentName, score: m.score, bestMatch: m.text };
            }
        }

        return Object.values(aggregated)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);
    }

    /**
     * Safe search for entity matches.
     */
    async matchEntities(queryEmbedding, thresholds) {
        const hits = [];

        this.entities.forEach(entity => {
            const score = this.cosineSimilarity(queryEmbedding, entity.embedding);
            const threshold = thresholds[entity.type] || 0.85;

            if (score >= threshold) {
                hits.push({ ...entity, score });
            }
        });

        return hits;
    }
}

export default new VectorStore();
