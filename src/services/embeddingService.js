import { pipeline, env } from '@huggingface/transformers';

// Setup environment for explicit caching and better logging
env.cacheDir = './.cache';
env.allowLocalModels = false;
env.useBrowserCache = false;

class EmbeddingService {
    constructor() {
        this.pipe = null;
        this.modelName = 'Xenova/all-MiniLM-L6-v2';
    }

    async init() {
        if (this.pipe) return;
        console.log(`[EmbeddingService] Initializing ${this.modelName}...`);
        console.log(`[EmbeddingService] Cache directory: ${env.cacheDir}`);

        try {
            this.pipe = await pipeline(
                'feature-extraction',
                this.modelName,
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
     * Generates embeddings for the given texts.
     * @param {string|string[]} texts 
     * @returns {Promise<number[][]>}
     */
    async getEmbeddings(texts) {
        if (!this.pipe) await this.init();

        const output = await this.pipe(texts, { pooling: 'mean', normalize: true });
        // output is a Tensor, we convert to nested array
        return output.tolist();
    }
}

export default new EmbeddingService();
