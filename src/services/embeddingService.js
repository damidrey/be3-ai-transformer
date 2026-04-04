import { pipeline, env, RawImage } from '@huggingface/transformers';
import { buildConfigFromEnv, getModelConfig } from './modelRegistry.js';

// Setup environment for explicit caching and better logging
env.cacheDir = './.cache';
env.allowLocalModels = false;
env.useBrowserCache = false;

const CLIP_MODEL = 'Xenova/clip-vit-base-patch32';

class EmbeddingService {
    constructor() {
        this.pipe = null;        // Text embedding pipeline (BGE/MiniLM)
        this.clipPipe = null;    // CLIP vision pipeline (lazy-loaded separately)
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
                    quantized: this.config.quantized,
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
     * Lazy-load the CLIP vision pipeline (separate from text pipeline).
     * Uses 'image-feature-extraction' which runs the vision encoder only.
     */
    async initClip() {
        if (this.clipPipe) return;
        console.log(`[EmbeddingService] Initializing CLIP vision encoder (${CLIP_MODEL})...`);
        try {
            this.clipPipe = await pipeline(
                'image-feature-extraction',
                CLIP_MODEL,
                {
                    quantized: true,
                    progress_callback: (p) => {
                        if (p.status === 'progress') {
                            const pct = (p.loaded / p.total * 100).toFixed(1);
                            console.log(`[CLIP] Downloading ${p.file}: ${pct}%`);
                        } else if (p.status === 'done') {
                            console.log(`[CLIP] ${p.file} ready.`);
                        }
                    }
                }
            );
            console.log(`[EmbeddingService] CLIP vision encoder ready (512D).`);
        } catch (err) {
            console.error(`[EmbeddingService] Failed to load CLIP:`, err.message);
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
     * Generates embeddings for the given texts or images.
     * @param {string|string[]} input
     * @param {{ purpose?: 'query'|'passage'|'default', type?: 'image'|'text' }} options
     * @returns {Promise<number[][]>}
     */
    async getEmbeddings(input, options = {}) {
        if (options.type === 'image') {
            return this.getImageEmbeddings(input);
        }

        if (!this.pipe) await this.init();

        const purpose = options.purpose || 'default';
        const prepared = this.prepareTexts(input, purpose);
        const output = await this.pipe(prepared, { pooling: this.config.pooling, normalize: this.config.normalize });
        return output.tolist();
    }

    /**
     * Generates 512D CLIP embeddings for images using the dedicated vision encoder.
     * Handles URLs and base64 data URIs.
     * @param {string|string[]} input - URL or base64 data URI
     * @returns {Promise<number[][]>}
     */
    async getImageEmbeddings(input) {
        await this.initClip(); // Use CLIP pipeline, not text pipeline

        const isArray = Array.isArray(input);
        const inputs = isArray ? input : [input];

        const results = [];
        for (const imageSource of inputs) {
            try {
                let image;

                if (typeof imageSource === 'string' && imageSource.startsWith('data:')) {
                    // base64 data URI — extract the base64 payload and convert to Buffer
                    const base64Data = imageSource.split(',')[1];
                    if (!base64Data) throw new Error('Malformed base64 data URI');
                    const buffer = Buffer.from(base64Data, 'base64');
                    image = await RawImage.fromBlob(new Blob([buffer]));
                } else {
                    // Regular URL or file path
                    image = await RawImage.read(imageSource);
                }

                const output = await this.clipPipe(image, { pooling: 'mean', normalize: true });
                // image-feature-extraction returns shape [1, 512] — grab first row
                const tensor = output;
                const arr = tensor.tolist ? tensor.tolist() : Array.from(tensor.data);
                // If nested (batch), unwrap
                results.push(Array.isArray(arr[0]) ? arr[0] : arr);
            } catch (err) {
                console.error(`[EmbeddingService] Failed to embed image: ${String(imageSource).substring(0, 100)}`, err.message);
                results.push(null);
            }
        }

        return isArray ? results : [results[0]];
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
