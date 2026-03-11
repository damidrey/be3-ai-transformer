import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import intentMatcher from './services/intentMatcher.js';
import entityExtractor from './services/entityExtractor.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3009;

app.use(cors());
app.use(bodyParser.json());

// Health Check
app.get('/status', (req, res) => {
    res.json({
        status: 'online',
        model: 'all-MiniLM-L6-v2',
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
    const { text } = req.body;

    if (!text) {
        return res.status(400).json({ error: 'Text field is required' });
    }

    try {
        console.log(`[API] Analyzing: "${text}"`);
        const startTime = Date.now();

        // Run classification and entity extraction in parallel
        const [classificationResults, extractionResults] = await Promise.all([
            intentMatcher.match(text),
            entityExtractor.extract(text)
        ]);

        const duration = Date.now() - startTime;

        res.json({
            text,
            classification: classificationResults,
            entities: extractionResults.entities,
            confidence: extractionResults.confidence,
            duration: `${duration}ms`
        });
    } catch (err) {
        console.error('[API] Analysis error:', err);
        res.status(500).json({ error: 'Internal server error', details: err.message });
    }
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

app.listen(PORT, async () => {
    console.log(`🚀 [be3-ai-transformer] Service running on http://localhost:${PORT}`);

    // Warm up the model and load data on startup
    try {
        await Promise.all([
            intentMatcher.reload(),
            entityExtractor.reload()
        ]);
    } catch (err) {
        console.error('❌ Failed to load knowledge base during startup:', err.message);
    }
});
