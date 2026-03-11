import embeddingService from './embeddingService.js';
import dataLoader from './dataLoader.js';

class IntentMatcher {
    constructor() {
        this.intentEmbeddings = []; // Array of { intentName, text, embedding }
    }

    async reload() {
        console.log('[IntentMatcher] Reloading intents and regenerating embeddings...');
        const rawData = await dataLoader.loadIntents();
        const allVariations = [];
        const intentNames = [];

        for (const [intentName, variations] of Object.entries(rawData)) {
            for (const text of variations) {
                allVariations.push(text);
                intentNames.push(intentName);
            }
        }

        if (allVariations.length === 0) {
            console.warn('[IntentMatcher] No variations found to embed.');
            return;
        }

        console.log(`[IntentMatcher] Generating embeddings for ${allVariations.length} variations...`);
        const startTime = Date.now();
        const embeddings = await embeddingService.getEmbeddings(allVariations);
        const duration = Date.now() - startTime;

        this.intentEmbeddings = allVariations.map((text, i) => ({
            intentName: intentNames[i],
            text,
            embedding: embeddings[i]
        }));

        console.log(`[IntentMatcher] Generated ${this.intentEmbeddings.length} embeddings in ${duration}ms.`);
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
