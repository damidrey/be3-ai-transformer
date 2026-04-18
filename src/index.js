import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import intentMatcher from './services/intentMatcher.js';
import entityExtractor from './services/entityExtractor.js';
import embeddingService from './services/embeddingService.js';
import vectorStore from './services/vectorStore.js';
import { listModelConfigs } from './services/modelRegistry.js';
import aliasService from './services/aliasService.js';

const app = express();
const PORT = process.env.PORT || 3009;

app.use(cors());
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

// Health Check
app.get('/status', (req, res) => {
    res.json({
        status: 'online',
        model: embeddingService.getConfig().modelName,
        model_key: embeddingService.getConfig().key,
        model_config: embeddingService.getConfig(),
        intentsLoaded: intentMatcher.intentEmbeddings.length,
        entitiesLoaded: entityExtractor.entityEmbeddings.length,
        timestamp: new Date().toISOString()
    });
});

app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

/**
 * Endpoint: Classify Text
 * POST /classify
 * 
 * Legacy mode (flat classification):
 *   Body: { "text": "I want to buy a laptop" }
 * 
 * Hierarchical mode (level-specific classification):
 *   Body: { "text": "I want to buy a laptop", "level": "class" }
 *   Body: { "text": "I want to buy a laptop", "level": "intent", "parent": "Shopping_Management" }
 *   Body: { "text": "I want to buy a laptop", "level": "subintent", "parent": "Cart_Management" }
 */
app.post('/classify', async (req, res) => {
    const { text, level, parent } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text field is required' });
    }

    try {
        const startTime = Date.now();

        const normalizedText = aliasService.normalize(text);

        // Hierarchical mode: level is provided
        if (level) {
            const validLevels = ['class', 'intent', 'subintent'];
            if (!validLevels.includes(level)) {
                return res.status(400).json({ error: `Invalid level. Must be one of: ${validLevels.join(', ')}` });
            }
            if (level !== 'class' && !parent) {
                return res.status(400).json({ error: `'parent' is required for level '${level}'` });
            }

            console.log(`[API] Hierarchical classify: "${normalizedText}" | level=${level} | parent=${parent || 'ROOT'}`);
            const result = await intentMatcher.matchByLevel(normalizedText, level, parent);
            const duration = Date.now() - startTime;

            return res.json({
                text,
                ...result,
                duration: `${duration}ms`
            });
        }

        // Legacy mode: flat classification against all intents
        console.log(`[API] Classifying (legacy): "${normalizedText}"`);
        const results = await intentMatcher.match(normalizedText);
        const duration = Date.now() - startTime;

        res.json({
            text,
            results,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Classification error:', err);
        res.status(500).json({ error: 'Internal server error', details: err.message });
    }
});

/**
 * Endpoint: Extract Entities
 * POST /extract
 * Body: { "text": "Does Dareymi sells cheap iphone 16" }
 */
app.post('/extract', async (req, res) => {
    const { text } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text field is required' });
    }

    try {
        const normalizedText = aliasService.normalize(text);
        console.log(`[API] Extracting entities from: "${normalizedText}"`);
        const startTime = Date.now();
        const results = await entityExtractor.extract(normalizedText);
        const duration = Date.now() - startTime;

        res.json({
            text,
            ...results,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Extraction error:', err);
        res.status(500).json({ error: 'Internal server error', details: err.message });
    }
});



/**
 * Endpoint: Generate Embeddings
 * POST /embed
 * Body: { "text": "laptops" } OR { "texts": ["laptop", "phone"] }
 */
app.post('/embed', async (req, res) => {
    const { text, texts, image, images, purpose, type } = req.body;
    const input = texts || text || images || image;

    if (!input) {
        return res.status(400).json({ error: 'Text or image input is required' });
    }

    try {
        const isBatch = Array.isArray(input);
        const inputType = type || (image || images ? 'image' : 'text');

        if (isBatch) {
            console.log(`[API] Batch Embedding (${inputType}): ${input.length} items`);
        } else {
            const display = typeof input === 'string' ? input.substring(0, 50) : 'Buffer/Binary';
            console.log(`[API] Embedding (${inputType}): "${display}"`);
        }

        const startTime = Date.now();
        const embeddings = await embeddingService.getEmbeddings(input, { purpose, type: inputType });
        const duration = Date.now() - startTime;

        res.json({
            embeddings,
            dimensions: (embeddings && embeddings[0]) ? embeddings[0].length : 0,
            inputType,
            isBatch,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Embedding error:', err);
        res.status(500).json({ error: 'Internal server error', details: err.message });
    }
});

/**
 * Endpoint: Generate Image Embeddings
 * POST /embed-image
 * Body: { "image": "base64..." } OR { "images": ["base64...", "..."] }
 */
app.post('/embed-image', async (req, res) => {
    const { image, images, purpose } = req.body;
    const input = images || image;

    if (!input) {
        return res.status(400).json({ error: 'Image or images field is required' });
    }

    try {
        const isBatch = Array.isArray(input);
        console.log(`[API] Embedding Image(s): ${isBatch ? input.length : 1} items`);

        const startTime = Date.now();
        const embeddings = await embeddingService.getEmbeddings(input, { purpose, type: 'image' });
        const duration = Date.now() - startTime;

        res.json({
            embeddings,
            dimensions: (embeddings && embeddings[0]) ? embeddings[0].length : 0,
            inputType: 'image',
            isBatch,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Image Embedding error:', err);
        res.status(500).json({ error: 'Internal server error', details: err.message });
    }
});

/**
 * Endpoint: Switch Model
 * POST /model
 * Body: { "modelKey": "minilm" }
 */
app.post('/model', async (req, res) => {
    const { modelKey } = req.body;
    if (!modelKey) {
        return res.status(400).json({ error: 'modelKey field is required' });
    }

    try {
        await embeddingService.setModel(modelKey);
        const startTime = Date.now();
        await Promise.all([
            intentMatcher.reload(),
            entityExtractor.reload()
        ]);
        const duration = Date.now() - startTime;

        res.json({
            success: true,
            model: embeddingService.getConfig().modelName,
            model_key: embeddingService.getConfig().key,
            model_config: embeddingService.getConfig(),
            reload_duration: `${duration}ms`
        });
    } catch (err) {
        res.status(500).json({ error: 'Failed to switch model', details: err.message });
    }
});

/**
 * Endpoint: List available model presets
 * GET /models
 */
app.get('/models', (req, res) => {
    res.json({
        models: listModelConfigs()
    });
});

/**
 * Endpoint: Reload Intents
 * POST /reload
 */
app.post('/reload', async (req, res) => {
    try {
        const startTime = Date.now();
        await Promise.all([
            intentMatcher.reload(),
            entityExtractor.reload()
        ]);
        const duration = Date.now() - startTime;

        res.json({
            success: true,
            message: 'Intents and Entities reloaded successfully.',
            stats: {
                intents: intentMatcher.intentEmbeddings.length,
                entities: entityExtractor.entityEmbeddings.length,
                duration: `${duration}ms`
            }
        });
    } catch (err) {
        console.error('[API] Reload error:', err);
        res.status(500).json({ error: 'Failed to reload knowledge base', details: err.message });
    }
});

const server = app.listen(PORT, async () => {
    console.log(`🚀 [be3-ai-transformer] Service running on http://localhost:${PORT}`);

    // Parse command line arguments
    const args = process.argv.slice(2);
    const pruneEnabled = args.includes('--prune');
    const pruneThreshold = args.find(a => a.startsWith('--threshold='))?.split('=')[1] || 0.15;

    // Entity Extractor Memory-Efficient Startup
    try {
        console.log(`\n[Server] 🏁 Starting Entity Store initialization...`);
        await vectorStore.init({ 
            prune: pruneEnabled ? parseFloat(pruneThreshold) : false 
        });
        
        console.log(`[Server] ✅ Entity extraction cache ready (RAM optimized).`);
    } catch (err) {
        console.error('❌ Failed to load Entity extraction cache during startup:', err);
    }

    // Load hierarchical pools (Phase 2: micarch)
    try {
        await vectorStore.initHierarchical();
    } catch (err) {
        console.warn('[Server] ⚠️  Hierarchical pools not loaded (may need generation):', err.message);
    }


    // Set up periodic auto-reload (every 30 minutes)
    const RELOAD_INTERVAL = 30 * 60 * 1000;
    setInterval(async () => {
        console.log(`\n[Auto-Reload] 🔄 Triggering periodic Entity Store update...`);
        try {
            await vectorStore.init({ 
                prune: pruneEnabled ? parseFloat(pruneThreshold) : false 
            });
            console.log(`[Auto-Reload] ✅ Update complete. Next update in 30 minutes.`);
        } catch (err) {
            console.error('[Auto-Reload] ❌ Failed to auto-reload:', err.message);
        }
    }, RELOAD_INTERVAL);
});

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.error(`\n❌ [ERROR] Port ${PORT} is already in use!`);
        console.error(`This usually means another instance of the transformer server is already running in the background.`);
        console.error(`Please kill the existing process or use a different port.\n`);
        process.exit(1);
    } else {
        console.error('\n❌ [ERROR] Server encountered an error:', err);
    }
});

// --- Graceful Shutdown Handlers ---
const handleShutdown = (signal) => {
    console.log(`\n[Server] Received ${signal}. Shutting down gracefully...`);
    server.close(() => {
        console.log('[Server] HTTP server closed.');
        process.exit(0);
    });

    // Force exit after 5s if server.close() hangs
    setTimeout(() => {
        console.error('[Server] Could not close connections in time, forceful exit.');
        process.exit(1);
    }, 5000);
};

process.on('SIGINT', () => handleShutdown('SIGINT'));
process.on('SIGTERM', () => handleShutdown('SIGTERM'));
