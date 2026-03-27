import embeddingService from './embeddingService.js';
import dataLoader from './dataLoader.js';

class EntityExtractor {
    constructor() {
        this.entityEmbeddings = []; // Array of { type: 'category|vendor|clause|attribute', subType: 'brand|color|...', key, text, embedding }
        this.thresholds = {
            category: 0.55, // Low for conceptual matches like hungry -> food
            vendor: 0.82,   // High for brand/entity precision
            clause: 0.75,
            attribute: 0.75
        };
    }

    async reload() {
        console.log('[EntityExtractor] Reloading knowledge base and regenerating embeddings...');
        const entities = await dataLoader.loadEntities();
        const variationsToEmbed = [];
        const metadata = [];

        // 1. Process Categories
        entities.category.forEach(c => {
            c.variations.forEach(text => {
                variationsToEmbed.push(text);
                metadata.push({ type: 'category', key: c.slug, text });
            });
        });

        // 2. Process Vendors
        entities.vendor.forEach(v => {
            v.variations.forEach(text => {
                variationsToEmbed.push(text);
                metadata.push({ type: 'vendor', key: v.label, text });
            });
        });

        // 3. Process Clauses
        for (const [key, variations] of Object.entries(entities.clause)) {
            variations.forEach(text => {
                variationsToEmbed.push(text);
                metadata.push({ type: 'clause', key, text });
            });
        }

        // 4. Process Attributes
        for (const [subType, variations] of Object.entries(entities.attribute)) {
            variations.forEach(text => {
                variationsToEmbed.push(text);
                metadata.push({ type: 'attribute', subType, text });
            });
        }

        const total = variationsToEmbed.length;
        if (total === 0) {
            console.warn('[EntityExtractor] No entities found to embed.');
            return;
        }

        console.log(`[EntityExtractor] Generating embeddings for ${total} entity variations...`);
        const startTime = Date.now();
        const BATCH_SIZE = 50;
        const allEmbeddings = [];

        for (let i = 0; i < total; i += BATCH_SIZE) {
            const chunk = variationsToEmbed.slice(i, i + BATCH_SIZE);
            const chunkEmbeddings = await embeddingService.getEmbeddings(chunk);
            allEmbeddings.push(...chunkEmbeddings);

            if (i % (BATCH_SIZE * 5) === 0 || i + BATCH_SIZE >= total) {
                const progress = Math.min(100, ((i + chunk.length) / total * 100)).toFixed(1);
                console.log(`[EntityExtractor] Progress: ${i + chunk.length}/${total} entities (${progress}%)`);
            }

            // Yield to event loop
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        const duration = Date.now() - startTime;

        this.entityEmbeddings = variationsToEmbed.map((text, i) => ({
            ...metadata[i],
            embedding: allEmbeddings[i]
        }));

        console.log(`[EntityExtractor] Generated ${this.entityEmbeddings.length} entity embeddings in ${duration}ms.`);
    }

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
     * Extracts entities from text using segmented semantic matching.
     */
    async extract(text) {
        if (!text) return { entities: {}, confidence: {} };
        if (this.entityEmbeddings.length === 0) await this.reload();

        const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
        const segments = [];

        // Generate n-grams (1 to 3 words)
        for (let len = 1; len <= 3; len++) {
            for (let i = 0; i <= words.length - len; i++) {
                segments.push(words.slice(i, i + len).join(' '));
            }
        }

        // Add the whole string as a segment too
        if (words.length > 3) {
            segments.push(text.toLowerCase());
        }

        console.log(`[EntityExtractor] Analyzing ${segments.length} text segments...`);
        const segmentEmbeddings = await embeddingService.getEmbeddings(segments);
        const results = {
            category: new Set(),
            vendor: new Set(),
            clause: new Set(),
            attribute: {} // subType -> [values]
        };
        const confidence = {};

        segments.forEach((segment, segmentIdx) => {
            const segEmbedding = segmentEmbeddings[segmentIdx];

            this.entityEmbeddings.forEach(entity => {
                const score = this.cosineSimilarity(segEmbedding, entity.embedding);
                const threshold = this.thresholds[entity.type] || 0.85;

                // Log only if needed for internal debugging
                // if (score >= 0.5) {
                //     console.log(`[EntityExtractor] Segment: "${segment}" -> Entity: "${entity.text}" (${entity.type}${entity.subType ? ':' + entity.subType : ''}) | Score: ${score.toFixed(4)}`);
                // }

                if (score >= threshold) {
                    const type = entity.type;
                    const key = entity.key || entity.text;
                    const confidenceKey = (type === 'attribute') ? segment : key;

                    if (type === 'category') results.category.add(key);
                    if (type === 'vendor') results.vendor.add(key);
                    if (type === 'clause') results.clause.add(key);
                    if (type === 'attribute') {
                        if (!results.attribute[entity.subType]) results.attribute[entity.subType] = new Set();
                        // Use the actual matched segment from user text, not the training variation
                        results.attribute[entity.subType].add(segment);
                    }

                    // Track highest confidence for each resolved key
                    const uniqueKey = entity.subType ? `${type}:${entity.subType}:${confidenceKey}` : `${type}:${confidenceKey}`;
                    if (!confidence[uniqueKey] || score > confidence[uniqueKey]) {
                        confidence[uniqueKey] = score;
                    }
                }
            });
        });

        // Convert Sets to Arrays for JSON response
        return {
            entities: {
                category: Array.from(results.category),
                vendor: Array.from(results.vendor),
                clause: Array.from(results.clause),
                attribute: Object.fromEntries(
                    Object.entries(results.attribute).map(([k, v]) => [k, Array.from(v)])
                )
            },
            confidence
        };
    }
}

export default new EntityExtractor();
