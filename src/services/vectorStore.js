import fs from 'fs';
import path from 'path';
import embeddingService from './embeddingService.js';

/**
 * Standardize and optimize vector storage for the entire service.
 * Using TypedArrays (Float32Array) reduces RAM usage for 512-dimension vectors by ~80%.
 * 
 * Supports two modes:
 *   1. Legacy (flat) — single knowledge-base.json for /analyze backward compat
 *   2. Hierarchical (pools) — fragmented KBs for /classify with level+parent
 */
class VectorStore {
    constructor() {
        this.intents = [];   // Array of { intentName, text, embedding: Float32Array }
        this.entities = [];  // Array of { type, subType, key, text, embedding: Float32Array }
        this.cachePath = path.join(process.cwd(), 'data/knowledge-base.json');
        this.isLoaded = false;

        // ── Hierarchical pools (Phase 2: micarch) ──
        // Map of poolKey → Array of { candidateName, text, embedding: Float32Array }
        // Pool keys: 'class', 'intent:Discovery', 'subintent:Cart_Management', etc.
        this.pools = new Map();
        this.hierarchicalCacheDir = path.join(process.cwd(), 'data/hierarchical-cache');
        this.isHierarchicalLoaded = false;
    }

    /**
     * Initializes the legacy (flat) knowledge base from the JSON cache.
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

    // ═══════════════════════════════════════════════
    // Hierarchical Pool Operations (Phase 2: micarch)
    // ═══════════════════════════════════════════════

    /**
     * Build a pool key for hierarchical lookups.
     * @param {'class'|'intent'|'subintent'} level
     * @param {string|null} parent - null for class, class name for intent, intent name for subintent
     * @returns {string} e.g., 'class', 'intent:Discovery', 'subintent:Cart_Management'
     */
    poolKey(level, parent) {
        if (level === 'class') return 'class';
        return `${level}:${parent}`;
    }

    /**
     * Load all hierarchical pools from the cache directory.
     * Each pool is stored as a JSON file with pre-computed embeddings.
     */
    async initHierarchical() {
        if (!fs.existsSync(this.hierarchicalCacheDir)) {
            console.warn(`[VectorStore] ⚠️  No hierarchical cache found at ${this.hierarchicalCacheDir}. Run generation first.`);
            return;
        }

        const startTime = Date.now();
        let poolCount = 0;
        let totalVectors = 0;

        try {
            console.log(`[VectorStore] 🔀 Loading hierarchical pools...`);

            const files = fs.readdirSync(this.hierarchicalCacheDir).filter(f => f.endsWith('.json'));

            for (const file of files) {
                const filePath = path.join(this.hierarchicalCacheDir, file);
                const rawData = fs.readFileSync(filePath, 'utf8');
                const data = JSON.parse(rawData);

                const key = data.poolKey; // e.g., 'class', 'intent:Discovery'
                const vectors = (data.vectors || []).map(item => ({
                    candidateName: item.candidateName,
                    text: item.text,
                    embedding: new Float32Array(item.embedding)
                }));

                this.pools.set(key, vectors);
                poolCount++;
                totalVectors += vectors.length;
            }

            this.isHierarchicalLoaded = true;
            console.log(`[VectorStore] ✅ Hierarchical: ${poolCount} pools, ${totalVectors} total vectors loaded in ${Date.now() - startTime}ms.`);

        } catch (err) {
            console.error(`[VectorStore] ❌ Error loading hierarchical cache:`, err.message);
            throw err;
        }
    }

    /**
     * Match a query embedding against a specific hierarchical pool.
     * @param {Float32Array} queryEmbedding
     * @param {'class'|'intent'|'subintent'} level
     * @param {string|null} parent
     * @param {number} topK
     * @returns {Array} Sorted by score descending, aggregated by MAX per candidate
     */
    async matchPool(queryEmbedding, level, parent, topK = 5) {
        const key = this.poolKey(level, parent);
        const pool = this.pools.get(key);

        if (!pool || pool.length === 0) {
            console.warn(`[VectorStore] ⚠️  Pool not found or empty: ${key}`);
            return [];
        }

        const matches = pool.map(item => ({
            candidateName: item.candidateName,
            text: item.text,
            score: this.cosineSimilarity(queryEmbedding, item.embedding)
        }));

        // Aggregate by candidateName (MAX score)
        const aggregated = {};
        for (const m of matches) {
            if (!aggregated[m.candidateName] || m.score > aggregated[m.candidateName].score) {
                aggregated[m.candidateName] = { name: m.candidateName, score: m.score, bestMatch: m.text };
            }
        }

        return Object.values(aggregated)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);
    }

    /**
     * Set a hierarchical pool programmatically (used during cache generation).
     */
    setPool(level, parent, vectors) {
        const key = this.poolKey(level, parent);
        this.pools.set(key, vectors);
    }

    /**
     * Save all hierarchical pools to the cache directory.
     */
    async saveHierarchicalCache() {
        if (!fs.existsSync(this.hierarchicalCacheDir)) {
            fs.mkdirSync(this.hierarchicalCacheDir, { recursive: true });
        }

        const model = embeddingService.getConfig().key;

        for (const [key, vectors] of this.pools.entries()) {
            const filename = key.replace(/:/g, '_') + '.json';
            const filePath = path.join(this.hierarchicalCacheDir, filename);

            const data = {
                poolKey: key,
                model,
                generatedAt: new Date().toISOString(),
                vectorCount: vectors.length,
                vectors: vectors.map(v => ({
                    candidateName: v.candidateName,
                    text: v.text,
                    embedding: Array.from(v.embedding)
                }))
            };

            fs.writeFileSync(filePath, JSON.stringify(data), 'utf8');
        }

        console.log(`[VectorStore] 💾 Saved ${this.pools.size} hierarchical pools to ${this.hierarchicalCacheDir}`);
    }

    // ═══════════════════════════════════════════════
    // Shared utilities
    // ═══════════════════════════════════════════════

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
     * Safe search for intent matches. (Legacy — used by /analyze)
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
