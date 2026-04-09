import embeddingService from './embeddingService.js';
import vectorStore from './vectorStore.js';
import dataLoader from './dataLoader.js';

/**
 * Service for extracting semantic entities (Category, Vendor, Clause, Attribute) 
 * from segmented text using vector similarity.
 */
class EntityExtractor {
    constructor() {
        this.thresholds = embeddingService.getConfig()?.thresholds || {
            category: 0.55,
            vendor: 0.82,
            clause: 0.75,
            attribute: 0.75
        };
    }

    get entityEmbeddings() {
        return vectorStore.entities.map(item => ({
            type: item.type,
            subType: item.subType,
            key: item.key,
            text: item.text,
            embedding: Array.from(item.embedding)
        }));
    }

    /**
     * Initializes the underlying vector store.
     */
    async reload(options = {}) {
        if (options.skipCache) {
            await this.regenerate(options);
        } else {
            await vectorStore.init(options);
        }
    }

    /**
     * Extracts entities from text using segmented semantic matching.
     */
    async extract(text) {
        if (!text) return { entities: {}, confidence: {} };
        
        // 1. Ensure vectors are loaded
        if (!vectorStore.isLoaded) {
            await this.reload(); 
        }

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

        // 2. Generate embeddings for all segments (batched)
        const segmentEmbeddingsRaw = await embeddingService.getEmbeddings(segments);
        const segmentEmbeddings = segmentEmbeddingsRaw.map(v => new Float32Array(v));

        const results = {
            category: new Set(),
            vendor: new Set(),
            clause: new Set(),
            attribute: {} // subType -> [values]
        };
        const confidence = {};

        // 3. Perform matching for each segment
        for (let i = 0; i < segments.length; i++) {
            const segment = segments[i];
            const sigEmbedding = segmentEmbeddings[i];
            
            // Call the memory-optimized matching logic in VectorStore
            const hits = await vectorStore.matchEntities(sigEmbedding, this.thresholds);

            hits.forEach(hit => {
                const type = hit.type;
                const key = hit.key || hit.text;
                const score = hit.score;
                const confidenceKey = (type === 'attribute') ? segment : key;

                if (type === 'category') results.category.add(key);
                if (type === 'vendor') results.vendor.add(key);
                if (type === 'clause') results.clause.add(key);
                if (type === 'attribute') {
                    if (!results.attribute[hit.subType]) results.attribute[hit.subType] = new Set();
                    results.attribute[hit.subType].add(segment);
                }

                const uniqueKey = hit.subType ? `${type}:${hit.subType}:${confidenceKey}` : `${type}:${confidenceKey}`;
                if (!confidence[uniqueKey] || score > confidence[uniqueKey]) {
                    confidence[uniqueKey] = score;
                }
            });
        }

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

    async regenerate(options = {}) {
        console.log('[EntityExtractor] Regenerating entity embeddings from raw data...');
        const rawEntities = await dataLoader.loadEntities();
        
        const allEntities = [];
        
        // 1. Process Categories
        rawEntities.category.forEach(c => {
            c.variations.forEach(v => {
                allEntities.push({ type: 'category', key: c.key, text: v });
            });
        });

        // 2. Process Vendors
        rawEntities.vendor.forEach(v => {
            v.variations.forEach(vari => {
                allEntities.push({ type: 'vendor', key: v.label, text: vari });
            });
        });

        // 3. Process Clauses
        Object.entries(rawEntities.clause).forEach(([key, variations]) => {
            variations.forEach(v => {
                allEntities.push({ type: 'clause', key: key, text: v });
            });
        });

        // 4. Process Attributes
        Object.entries(rawEntities.attribute).forEach(([subType, variations]) => {
            variations.forEach(v => {
                allEntities.push({ type: 'attribute', subType, text: v });
            });
        });

        const total = allEntities.length;
        if (total === 0) return;

        console.log(`[EntityExtractor] Generating embeddings for ${total} entities...`);
        const BATCH_SIZE = 50;
        const allEmbeddings = [];

        for (let i = 0; i < total; i += BATCH_SIZE) {
            const chunk = allEntities.slice(i, i + BATCH_SIZE).map(e => e.text);
            const embeddings = await embeddingService.getEmbeddings(chunk);
            allEmbeddings.push(...embeddings);
            
            const progress = Math.min(100, ((i + chunk.length) / total * 100)).toFixed(1);
            console.log(`[EntityExtractor] Progress: ${progress}%`);
            await new Promise(resolve => setTimeout(resolve, 10)); // Yield to CPU
        }

        const entities = allEntities.map((ent, i) => ({
            ...ent,
            embedding: new Float32Array(allEmbeddings[i])
        }));

        vectorStore.setKnowledge(vectorStore.intents, entities);
        console.log(`[EntityExtractor] Successfully regenerated ${entities.length} entities.`);
    }
}

export default new EntityExtractor();
