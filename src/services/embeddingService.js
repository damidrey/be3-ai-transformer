import { pipeline, env } from '@huggingface/transformers';
import { buildConfigFromEnv, getModelConfig } from './modelRegistry.js';

// Setup environment for explicit caching and better logging
env.cacheDir = './.cache';
env.allowLocalModels = false;
env.useBrowserCache = false;

class EmbeddingService {
    constructor() {
        this.pipe = null;
        this.config = buildConfigFromEnv();
    }

    async init() {
        if (this.pipe) return;
        console.log(`[EmbeddingService] Initializing ${this.config.modelName}...`);
        console.log(`[EmbeddingService] Cache directory: ${env.cacheDir}`);

        try {
            this.pipe = await pipeline(
                'feature-extraction',
                this.config.modelName,
                {
                    progress_callback: (p) => {
                        if (p.status === 'progress') {
                            const percent = (p.loaded / p.total * 100).toFixed(1);
                            console.log(`[EmbeddingService] Downloading ${p.file}: ${percent}% (${(p.loaded / 1024 / 1024).toFixed(1)}MB / ${(p.total / 1024 / 1024).toFixed(1)}MB)`);
                        } else if (p.status === 'initiate') {
                            console.log(`[EmbeddingService] Initiating download: ${p.file}`);
                        } else if (p.status === 'done') {
                            console.log(`[EmbeddingService] Download complete: ${p.file}`);
                        }
                    }
                }
            );
            console.log(`[EmbeddingService] Model loaded successfully.`);
        } catch (err) {
            console.error(`[EmbeddingService] CRITICAL: Failed to load model. This is often due to an interrupted network connection or corrupted cache.`);
            console.error(`[EmbeddingService] Error detail: ${err.message}`);
            if (err.message.includes('Protobuf')) {
                console.error(`[EmbeddingService] HINT: The model file is corrupted. Try deleting the ${env.cacheDir} folder and restarting.`);
            }
            throw err;
        }
    }

    /**
     * Switch the active embedding model (clears cached pipeline).
     * @param {string} modelKey
     */
    async setModel(modelKey) {
        const next = getModelConfig(modelKey);
        if (!next) {
            throw new Error(`Unknown model key: ${modelKey}`);
        }
        this.pipe = null;
        this.config = {
            ...next
        };
        console.log(`[EmbeddingService] Switched model to ${this.config.modelName} (${this.config.key})`);
    }

    /**
     * Returns current model config.
     */
    getConfig() {
        return this.config;
    }

    /**
     * Generates embeddings for the given texts.
     * @param {string|string[]} texts 
     * @param {{ purpose?: 'query'|'passage'|'default' }} options
     * @returns {Promise<number[][]>}
     */
    async getEmbeddings(texts, options = {}) {
        if (!this.pipe) await this.init();

        const purpose = options.purpose || 'default';
        const prepared = this.prepareTexts(texts, purpose);
        const output = await this.pipe(prepared, { pooling: this.config.pooling, normalize: this.config.normalize });
        // output is a Tensor, we convert to nested array
        return output.tolist();
    }

    /**
     * Apply model-specific formatting. For BGE, use query/passage prefixes.
     * @param {string|string[]} texts
     * @param {string} purpose
     * @returns {string|string[]}
     */
    prepareTexts(texts, purpose) {
        const isArray = Array.isArray(texts);
        const arr = isArray ? texts : [texts];

        let prepared = arr.map(t => {
            const text = typeof t === 'string' ? t : String(t);
            return text.substring(0, this.config.maxLength);
        });

        if (this.config.prefixMode === 'bge' && (purpose === 'query' || purpose === 'passage')) {
            const prefix = purpose === 'query' ? this.config.queryPrefix : this.config.passagePrefix;
            prepared = prepared.map(t => `${prefix}${t}`);
        }

        return isArray ? prepared : prepared[0];
    }
}

export default new EmbeddingService();
