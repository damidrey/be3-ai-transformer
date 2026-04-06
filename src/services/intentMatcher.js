import fs from 'fs';
import path from 'path';
import embeddingService from './embeddingService.js';
import dataLoader from './dataLoader.js';
import tokenDemasker from './tokenDemasker.js';

class IntentMatcher {
    constructor() {
        this.intentEmbeddings = []; // Array of { intentName, text, embedding }
        this.cachePath = path.join(process.cwd(), 'data/knowledge-base.json');
    }

    async reload(options = {}) {
        // 1. Check for Pre-computed Vector Cache
        if (!options.skipCache && fs.existsSync(this.cachePath)) {
            try {
                const cache = JSON.parse(fs.readFileSync(this.cachePath, 'utf8'));
                const activeModel = embeddingService.getConfig().key;

                if (cache.model === activeModel && cache.intents) {
                    console.log(`[IntentMatcher] 📦 Loaded ${cache.intents.length} embeddings from vector cache (${cache.model}).`);
                    this.intentEmbeddings = cache.intents;
                    return;
                } else if (cache.model !== activeModel) {
                    console.warn(`[IntentMatcher] ⚠️  Cache model (${cache.model}) mismatch with active model (${activeModel}). Regenerating...`);
                }
            } catch (err) {
                console.warn(`[IntentMatcher] ⚠️  Failed to load cache: ${err.message}. Regenerating...`);
            }
        }

        console.log('[IntentMatcher] Regenerating intent embeddings (No valid cache found)...');
        const rawData = await dataLoader.loadIntents();
        let processedData = rawData;

        if (options.prune) {
            const threshold = typeof options.prune === 'number' ? options.prune : 0.15;
            processedData = await this.pruneIntents(rawData, threshold);
        }

        const allVariations = [];
        const intentNames = [];

        for (const [intentName, variations] of Object.entries(processedData)) {
            for (const text of variations) {
                allVariations.push(text);
                intentNames.push(intentName);
            }
        }

        const total = allVariations.length;
        if (total === 0) return;

        console.log(`[IntentMatcher] Generating embeddings for ${total} variations...`);
        const startTime = Date.now();
        const BATCH_SIZE = embeddingService.mode === 'cloud' ? 20 : 50; 
        const allEmbeddings = [];

        for (let i = 0; i < total; i += BATCH_SIZE) {
            const chunk = allVariations.slice(i, i + BATCH_SIZE);
            const chunkEmbeddings = await embeddingService.getEmbeddings(chunk);
            allEmbeddings.push(...chunkEmbeddings);

            if (i % (BATCH_SIZE * 5) === 0 || i + BATCH_SIZE >= total) {
                const progress = Math.min(100, ((i + chunk.length) / total * 100)).toFixed(1);
                console.log(`[IntentMatcher] Progress: ${progress}%`);
            }
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        this.intentEmbeddings = allVariations.map((text, i) => ({
            intentName: intentNames[i],
            text,
            embedding: allEmbeddings[i]
        }));

        console.log(`[IntentMatcher] Generated ${this.intentEmbeddings.length} embeddings in ${Date.now() - startTime}ms.`);
    }


    /**
     * Internal pruning protocol using Cosine Distance
     */
    async pruneIntents(intentData, threshold) {
        const prunedData = {};
        const stats = { total: 0, kept: 0 };

        for (const [intentName, variations] of Object.entries(intentData)) {
            if (variations.length <= 1) {
                prunedData[intentName] = variations;
                stats.total += variations.length;
                stats.kept += variations.length;
                continue;
            }

            // Shuffle variations so prototypes are random
            const shuffled = [...variations];
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }

            // Get embeddings for this intent group
            const embeddings = await embeddingService.getEmbeddings(shuffled);
            const keptVariations = [];
            const keptEmbeddings = [];

            for (let i = 0; i < shuffled.length; i++) {
                let minDistance = Infinity;

                for (const keptEmb of keptEmbeddings) {
                    const dist = this.cosineDistance(embeddings[i], keptEmb);
                    if (dist < minDistance) minDistance = dist;
                }

                if (minDistance >= threshold) {
                    keptVariations.push(shuffled[i]);
                    keptEmbeddings.push(embeddings[i]);
                }
            }

            prunedData[intentName] = keptVariations;
            stats.total += variations.length;
            stats.kept += keptVariations.length;

            const reduction = (((variations.length - keptVariations.length) / variations.length) * 100).toFixed(0);
            if (reduction > 0) {
                console.log(`[Pruner] ${intentName}: ${variations.length} -> ${keptVariations.length} (-${reduction}%)`);
            }
        }

        console.log(`[IntentMatcher] Summary: ${stats.total} total -> ${stats.kept} pruned variations (-${(((stats.total - stats.kept) / stats.total) * 100).toFixed(1)}%)`);
        return prunedData;
    }

    /**
     * Calculates cosine distance between two vectors.
     * Normalized vectors: distance = 1 - dotProduct
     */
    cosineDistance(vecA, vecB) {
        let dotProduct = 0;
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
        }
        return 1 - dotProduct;
    }

    /**
     * Calculates cosine similarity between two vectors.
     */
    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Matches a query string against all intent variations.
     * @param {string} query 
     * @param {number} topK 
     */
    async match(query, topK = 5) {
        if (!query) return [];
        if (this.intentEmbeddings.length === 0) await this.reload();

        const [queryEmbedding] = await embeddingService.getEmbeddings([query]);

        const individualMatches = this.intentEmbeddings.map(item => ({
            intentName: item.intentName,
            text: item.text,
            score: this.cosineSimilarity(queryEmbedding, item.embedding)
        }));

        // Aggregate by intentName (take MAX score per intent)
        const aggregated = {};
        for (const match of individualMatches) {
            if (!aggregated[match.intentName] || match.score > aggregated[match.intentName].score) {
                aggregated[match.intentName] = {
                    intentName: match.intentName,
                    score: match.score,
                    bestMatch: match.text
                };
            }
        }

        return Object.values(aggregated)
            .sort((a, b) => b.score - a.score)
            .slice(0, topK);
    }
}

export default new IntentMatcher();
