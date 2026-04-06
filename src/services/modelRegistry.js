// Central registry for embedding model presets and tuning configs.
// Switch globally by setting EMBEDDING_MODEL_KEY in .env or via /model endpoint.

const MODEL_PRESETS = {
    'minilm': {
        key: 'minilm',
        label: 'MiniLM',
        modelName: 'Xenova/all-MiniLM-L6-v2',
        hfModelId: 'sentence-transformers/all-MiniLM-L6-v2',
        dimensions: 384,
        pooling: 'mean',
        normalize: true,
        prefixMode: 'none',
        maxLength: 1000,
        thresholds: {
            category: 0.55,
            vendor: 0.82,
            clause: 0.75,
            attribute: 0.75
        }
    },
    'bge-small': {
        key: 'bge-small',
        label: 'BGE Small',
        modelName: 'Xenova/bge-small-en-v1.5',
        hfModelId: 'BAAI/bge-small-en-v1.5',
        dimensions: 384,
        pooling: 'cls',
        normalize: true,
        prefixMode: 'bge',
        maxLength: 2000,
        queryPrefix: 'query: ',
        passagePrefix: 'passage: ',
        thresholds: {
            category: 0.78,
            vendor: 0.88,
            clause: 0.82,
            attribute: 0.82
        }
    },
    'clip-vit-base-patch32': {
        key: 'clip-vit-base-patch32',
        label: 'CLIP (Visual)',
        modelName: 'Xenova/clip-vit-base-patch32',
        hfModelId: 'openai/clip-vit-base-patch32',
        dimensions: 512,
        pooling: 'none',
        normalize: true,
        quantized: true,
        prefixMode: 'none',
        maxLength: 77,
        thresholds: {
            category: 0.25,
            vendor: 0.35,
            clause: 0.30,
            attribute: 0.30
        }
    }
};

function listModelConfigs() {
    return Object.values(MODEL_PRESETS);
}

function getModelConfig(key) {
    const resolvedKey = (key || '').toLowerCase().trim();
    return MODEL_PRESETS[resolvedKey] || null;
}

function buildConfigFromEnv(env = process.env) {
    const envKey = env.EMBEDDING_MODEL_KEY || '';
    const preset = getModelConfig(envKey) || null;

    // Allow overrides via env (useful for quick tuning tests)
    const override = {
        key: envKey || preset?.key || 'bge-small',
        label: preset?.label || envKey || 'Custom',
        modelName: env.EMBEDDING_MODEL_NAME || preset?.modelName || 'Xenova/bge-small-en-v1.5',
        hfModelId: env.HF_MODEL_ID || preset?.hfModelId || 'BAAI/bge-small-en-v1.5',
        dimensions: env.EMBEDDING_DIMENSIONS ? parseInt(env.EMBEDDING_DIMENSIONS) : (preset?.dimensions || 384),
        pooling: env.EMBEDDING_POOLING || preset?.pooling || 'mean',
        normalize: env.EMBEDDING_NORMALIZE ? env.EMBEDDING_NORMALIZE === 'true' : (preset?.normalize ?? true),
        quantized: env.EMBEDDING_QUANTIZED ? env.EMBEDDING_QUANTIZED === 'true' : (preset?.quantized ?? false),
        prefixMode: env.EMBEDDING_PREFIX_MODE || preset?.prefixMode || 'none',
        maxLength: env.EMBEDDING_MAX_LENGTH ? parseInt(env.EMBEDDING_MAX_LENGTH) : (preset?.maxLength || 1000),
        queryPrefix: env.EMBEDDING_QUERY_PREFIX || preset?.queryPrefix || 'query: ',
        passagePrefix: env.EMBEDDING_PASSAGE_PREFIX || preset?.passagePrefix || 'passage: ',
        thresholds: {
            category: env.ENTITY_THRESHOLD_CATEGORY ? parseFloat(env.ENTITY_THRESHOLD_CATEGORY) : (preset?.thresholds?.category ?? 0.55),
            vendor: env.ENTITY_THRESHOLD_VENDOR ? parseFloat(env.ENTITY_THRESHOLD_VENDOR) : (preset?.thresholds?.vendor ?? 0.82),
            clause: env.ENTITY_THRESHOLD_CLAUSE ? parseFloat(env.ENTITY_THRESHOLD_CLAUSE) : (preset?.thresholds?.clause ?? 0.75),
            attribute: env.ENTITY_THRESHOLD_ATTRIBUTE ? parseFloat(env.ENTITY_THRESHOLD_ATTRIBUTE) : (preset?.thresholds?.attribute ?? 0.75)
        }
    };

    return override;
}


export {
    MODEL_PRESETS,
    listModelConfigs,
    getModelConfig,
    buildConfigFromEnv
};
