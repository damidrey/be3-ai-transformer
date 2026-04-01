import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import intentMatcher from './services/intentMatcher.js';
import entityExtractor from './services/entityExtractor.js';
import embeddingService from './services/embeddingService.js';
import { listModelConfigs } from './services/modelRegistry.js';

const app = express();
const PORT = process.env.PORT || 3009;

app.use(cors());
app.use(bodyParser.json());

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

/**
 * Endpoint: Classify Text
 * POST /classify
 * Body: { "text": "I want to buy a laptop" }
 */
app.post('/classify', async (req, res) => {
    const { text } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text field is required' });
    }

    try {
        console.log(`[API] Classifying: "${text}"`);
        const startTime = Date.now();
        const results = await intentMatcher.match(text);
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
        console.log(`[API] Extracting entities from: "${text}"`);
        const startTime = Date.now();
        const results = await entityExtractor.extract(text);
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
 * Endpoint: Unified Analysis (Classification + Entity Extraction)
 * POST /analyze
 * Body: { "text": "show me cheap samsung phones" }
 * 
 * Runs both classification and entity extraction in parallel.
 * Returns a single structured response for the pipeline to consume early.
 */
app.post('/analyze', async (req, res) => {
    const { text, texts } = req.body;
    const inputTexts = texts || (text ? [text] : null);

    if (!inputTexts || !Array.isArray(inputTexts)) {
        return res.status(400).json({ error: 'Text or texts (array) field is required' });
    }

    try {
        const startTime = Date.now();
        console.log(`[API] Analyzing ${inputTexts.length} statements...`);

        const results = await Promise.all(inputTexts.map(async (t) => {
            const [classificationResults, extractionResults] = await Promise.all([
                intentMatcher.match(t),
                entityExtractor.extract(t)
            ]);
            return {
                text: t,
                classification: classificationResults,
                entities: extractionResults.entities,
                confidence: extractionResults.confidence
            };
        }));

        const duration = Date.now() - startTime;

        res.json({
            results,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Analysis error:', err);
        res.status(500).json({ error: 'Internal server error', details: err.message });
    }
});

/**
 * Endpoint: Generate Embeddings
 * POST /embed
 * Body: { "text": "laptops" } OR { "texts": ["laptop", "phone"] }
 */
app.post('/embed', async (req, res) => {
    const { text, texts, purpose } = req.body;
    const input = texts || text;

    if (!input) {
        return res.status(400).json({ error: 'Text or texts field is required' });
    }

    try {
        const isBatch = Array.isArray(input);
        if (isBatch) {
            input.forEach(text => console.log(`[API] Batch Embedding: "${text}"`));
        } else {
            console.log(`[API] Embedding: "${input}"`);
        }
        
        const startTime = Date.now();
        const embeddings = await embeddingService.getEmbeddings(input, { purpose });
        const duration = Date.now() - startTime;

        res.json({
            text: isBatch ? undefined : input,
            texts: isBatch ? input : undefined,
            embeddings,
            dimensions: embeddings[0]?.length || 0,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Embedding error:', err);
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

    // Warm up the model and load data on startup
    try {
        await Promise.all([
            intentMatcher.reload({ prune: pruneEnabled ? parseFloat(pruneThreshold) : false }),
            entityExtractor.reload()
        ]);
    } catch (err) {
        console.error('❌ Failed to load knowledge base during startup:', err);
    }

    // Set up periodic auto-reload (every 30 minutes) to keep de-masking context fresh
    const RELOAD_INTERVAL = 30 * 60 * 1000; 
    setInterval(async () => {
        console.log(`\n[Auto-Reload] 🔄 Triggering periodic knowledge base update...`);
        try {
            await Promise.all([
                intentMatcher.reload({ prune: pruneEnabled ? parseFloat(pruneThreshold) : false }),
                entityExtractor.reload()
            ]);
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
