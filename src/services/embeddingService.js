import { pipeline, env, RawImage } from '@huggingface/transformers';
import { HfInference } from '@huggingface/inference';
import { buildConfigFromEnv, getModelConfig } from './modelRegistry.js';
import axios from 'axios';

// Setup environment for explicit caching and better logging
env.cacheDir = './.cache';
env.allowLocalModels = false;
env.useBrowserCache = false;

const CLIP_MODEL = 'Xenova/clip-vit-base-patch32';
const CLIP_MODEL_HF = 'sentence-transformers/clip-ViT-B-32';

class EmbeddingService {
    constructor() {
        this.pipe = null;        // Text embedding pipeline (Local)
        this.clipPipe = null;    // CLIP vision pipeline (Local)
        this.hf = null;          // HF Inference Client (Cloud)
        this.config = buildConfigFromEnv();
        this.mode = process.env.EMBEDDING_MODE || 'local'; // 'local' or 'cloud'
        this.clipSpaceUrl = process.env.CLIP_SPACE_URL;
        if (this.clipSpaceUrl) {
            console.log(`[EmbeddingService] 🛰️  CLIP Custom Space: ${this.clipSpaceUrl}`);
        }

        if (this.mode === 'cloud') {
            const token = process.env.HF_TOKEN;
            if (!token) {
                console.warn('[EmbeddingService] Cloud mode active but HF_TOKEN is missing! Falling back to local.');
                this.mode = 'local';
            } else {
                this.hf = new HfInference(token);
                console.log(`[EmbeddingService] ☁️  Cloud Inference Mode active (${this.config.modelName})`);
            }
        } else {
             console.log(`[EmbeddingService] 🏠 Local Transformers.js Mode active (${this.config.modelName})`);
        }
    }

    async init() {
        if (this.mode === 'cloud') return; // No local init needed for cloud mode
        if (this.pipe) return;
        
        console.log(`[EmbeddingService] Initializing ${this.config.modelName} (Local)...`);
        try {
            this.pipe = await pipeline(
                'feature-extraction',
                this.config.modelName, // Xenova slug
                {
                    quantized: this.config.quantized,
                    progress_callback: (p) => {
                        if (p.status === 'progress') {
                            const percent = (p.loaded / p.total * 100).toFixed(1);
                            console.log(`[EmbeddingService] Downloading... ${percent}%`);
                        }
                    }
                }
            );
            console.log(`[EmbeddingService] Local model loaded.`);
        } catch (err) {
            console.error(`[EmbeddingService] Local init failed: ${err.message}`);
            throw err;
        }
    }

    async initClip() {
        if (this.mode === 'cloud') return; // No local init needed
        if (this.clipPipe) return;
        
        console.log(`[EmbeddingService] Initializing CLIP vision encoder (Local)...`);
        try {
            this.clipPipe = await pipeline(
                'image-feature-extraction',
                CLIP_MODEL,
                { quantized: true }
            );
            console.log(`[EmbeddingService] Local CLIP ready.`);
        } catch (err) {
             console.error(`[EmbeddingService] Local CLIP fail: ${err.message}`);
             throw err;
        }
    }

    /**
     * Switch the active embedding model.
     */
    async setModel(modelKey) {
        const next = getModelConfig(modelKey);
        if (!next) throw new Error(`Unknown model key: ${modelKey}`);
        this.pipe = null;
        this.config = { ...next };
        console.log(`[EmbeddingService] Switched model to ${this.config.modelName}`);
    }

    /**
     * Cloud Inference: Get embeddings from HF API
     */
    async getCloudEmbeddings(input, options = {}) {
        const modelId = this.config.hfModelId || this.config.modelName.replace('Xenova/', 'sentence-transformers/');
        
        try {
            if (options.type === 'image') {
                const results = await Promise.all((Array.isArray(input) ? input : [input]).map(async (img) => {
                    // 1. Check if we should use the custom CLIP Space
                    if (this.clipSpaceUrl) {
                        try {
                            const res = await this.callClipSpace(null, img);
                            return res.embedding;
                        } catch (err) {
                            console.warn(`[EmbeddingService] Custom CLIP Space failed, falling back to HF API: ${err.message}`);
                        }
                    }

                    // 2. Fallback to standard HF Inference (may fail for CLIP)
                    let blob;
                    if (typeof img === 'string' && img.startsWith('data:')) {
                        const base64Data = img.split(',')[1];
                        blob = new Blob([Buffer.from(base64Data, 'base64')]);
                    } else if (img.startsWith('http')) {
                        const response = await fetch(img);
                        blob = await response.blob();
                    }
                    
                    return this.hf.featureExtraction({
                        model: CLIP_MODEL_HF,
                        inputs: blob
                    });
                }));
                return results;
            }

            // Text embeddings
            if (options.purpose === 'clip' || modelId.toLowerCase().includes('clip')) {
                 if (this.clipSpaceUrl) {
                     const results = await Promise.all((Array.isArray(input) ? input : [input]).map(async (t) => {
                         const res = await this.callClipSpace(t, null);
                         return res.embedding;
                     }));
                     return Array.isArray(input) ? results : [results[0]];
                 }
            }

            // Text embeddings
            const purpose = options.purpose || 'default';
            const prepared = this.prepareTexts(input, purpose);
            
            const results = await this.hf.featureExtraction({
                model: modelId,
                inputs: prepared,
                provider: "hf-inference"
            });
            
            // HuggingFace returns raw array for single input, or nested for batch
            return Array.isArray(input) ? results : [results];
        } catch (err) {
            console.error(`[EmbeddingService] Cloud extraction failed: ${err.message}`);
            throw err;
        }
    }

    async getEmbeddings(input, options = {}) {
        if (this.mode === 'cloud') {
            return this.getCloudEmbeddings(input, options);
        }

        if (options.type === 'image') {
            return this.getImageEmbeddings(input);
        }

        if (!this.pipe) await this.init();

        const purpose = options.purpose || 'default';
        const prepared = this.prepareTexts(input, purpose);
        const output = await this.pipe(prepared, { pooling: this.config.pooling, normalize: this.config.normalize });
        return output.tolist();
    }

    async getImageEmbeddings(input) {
        await this.initClip();

        const isArray = Array.isArray(input);
        const inputs = isArray ? input : [input];
        const results = [];

        for (const imageSource of inputs) {
            try {
                let image;
                if (typeof imageSource === 'string' && imageSource.startsWith('data:')) {
                    const buffer = Buffer.from(imageSource.split(',')[1], 'base64');
                    image = await RawImage.fromBlob(new Blob([buffer]));
                } else {
                    image = await RawImage.read(imageSource);
                }

                const output = await this.clipPipe(image, { pooling: 'mean', normalize: true });
                const arr = output.tolist ? output.tolist() : Array.from(output.data);
                results.push(Array.isArray(arr[0]) ? arr[0] : arr);
            } catch (err) {
                console.error(`[EmbeddingService] Image embed fail:`, err.message);
                results.push(null);
            }
        }
        return isArray ? results : [results[0]];
    }

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

    /**
     * Call the custom Gradio API hosted on HF Spaces
     */
    async callClipSpace(text, image) {
        if (!this.clipSpaceUrl) throw new Error("CLIP_SPACE_URL not configured");

        const endpoint = `${this.clipSpaceUrl.replace(/\/$/, '')}/embed`;
        console.log(`[EmbeddingService] 📡 Sending request to CLIP Space: ${endpoint}`);

        const payload = {
            text: text || null,
            image_data: image || null
        };

        try {
            const response = await axios.post(endpoint, payload, {
                headers: { 'Content-Type': 'application/json' },
                timeout: 30000 
            });

            if (response.data && response.data.embedding) {
                console.log(`[EmbeddingService] ✅ Received embedding from Space.`);
                return response.data;
            }
            throw new Error("Invalid response format from HF Space");
        } catch (err) {
            const msg = err.response?.data?.detail || err.message;
            console.error(`[EmbeddingService] ❌ HF Space Error:`, msg);
            throw new Error(`CLIP Space Error: ${msg}`);
        }
    }

    getConfig() {
        return this.config;
    }
}

export default new EmbeddingService();

