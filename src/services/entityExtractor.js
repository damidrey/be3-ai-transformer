import fs from 'fs';
import path from 'path';
import embeddingService from './embeddingService.js';
import dataLoader from './dataLoader.js';

class EntityExtractor {
    constructor() {
        this.entityEmbeddings = []; // Array of { type, subType, key, text, embedding }
        this.cachePath = path.join(process.cwd(), 'data/knowledge-base.json');
        this.thresholds = embeddingService.getConfig()?.thresholds || {
            category: 0.55,
            vendor: 0.82,
            clause: 0.75,
            attribute: 0.75
        };
    }

    async reload(options = {}) {
        // 1. Check for Pre-computed Vector Cache
        if (!options.skipCache && fs.existsSync(this.cachePath)) {
            try {
                const cache = JSON.parse(fs.readFileSync(this.cachePath, 'utf8'));
                const activeModel = embeddingService.getConfig().key;

                if (cache.model === activeModel && cache.entities) {
                    console.log(`[EntityExtractor] 📦 Loaded ${cache.entities.length} embeddings from vector cache (${cache.model}).`);
                    this.entityEmbeddings = cache.entities;
                    this.thresholds = embeddingService.getConfig()?.thresholds || this.thresholds;
                    return;
                }
            } catch (err) {
                console.warn(`[EntityExtractor] ⚠️  Failed to load cache: ${err.message}. Regenerating...`);
            }
        }

        console.log('[EntityExtractor] Regenerating entity embeddings (No valid cache found)...');
        this.thresholds = embeddingService.getConfig()?.thresholds || this.thresholds;
        
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
        if (total === 0) return;

        console.log(`[EntityExtractor] Generating embeddings for ${total} entity variations...`);
        const startTime = Date.now();
        const BATCH_SIZE = embeddingService.mode === 'cloud' ? 20 : 50; 
        const allEmbeddings = [];

        for (let i = 0; i < total; i += BATCH_SIZE) {
            const chunk = variationsToEmbed.slice(i, i + BATCH_SIZE);
            const chunkEmbeddings = await embeddingService.getEmbeddings(chunk);
            allEmbeddings.push(...chunkEmbeddings);

            if (i % (BATCH_SIZE * 5) === 0 || i + BATCH_SIZE >= total) {
                const progress = Math.min(100, ((i + chunk.length) / total * 100)).toFixed(1);
                console.log(`[EntityExtractor] Progress: ${progress}%`);
            }
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        this.entityEmbeddings = variationsToEmbed.map((text, i) => ({
            ...metadata[i],
            embedding: allEmbeddings[i]
        }));

        console.log(`[EntityExtractor] Generated ${this.entityEmbeddings.length} entity embeddings in ${Date.now() - startTime}ms.`);
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

                if (score >= threshold) {
                    const type = entity.type;
                    const key = entity.key || entity.text;
                    const confidenceKey = (type === 'attribute') ? segment : key;

                    if (type === 'category') results.category.add(key);
                    if (type === 'vendor') results.vendor.add(key);
                    if (type === 'clause') results.clause.add(key);
                    if (type === 'attribute') {
                        if (!results.attribute[entity.subType]) results.attribute[entity.subType] = new Set();
                        results.attribute[entity.subType].add(segment);
                    }

                    const uniqueKey = entity.subType ? `${type}:${entity.subType}:${confidenceKey}` : `${type}:${confidenceKey}`;
                    if (!confidence[uniqueKey] || score > confidence[uniqueKey]) {
                        confidence[uniqueKey] = score;
                    }
                }
            });
        });

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
