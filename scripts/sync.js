import fs from 'fs';
import path from 'path';
import 'dotenv/config';
import dataLoader from '../src/services/dataLoader.js';
import intentMatcher from '../src/services/intentMatcher.js';
import entityExtractor from '../src/services/entityExtractor.js';
import embeddingService from '../src/services/embeddingService.js';

const DATA_DIR = path.join(process.cwd(), 'data');
const CACHE_FILE = path.join(DATA_DIR, 'knowledge-base.json');

/**
 * Sync script to snapshot data and pre-generate embeddings.
 */
async function sync() {
    console.log('\n🚀 [Sync] Starting Vector Cache Generation...');
    const startTime = Date.now();

    // 1. Ensure Data Directory exists
    if (!fs.existsSync(DATA_DIR)) {
        fs.mkdirSync(DATA_DIR, { recursive: true });
    }

    // 2. Snapshot Raw Files (Intents, Context)
    // This allows the transformer to be portable even without be3_ai access
    console.log('[Sync] Phase 1: Snapping raw data files...');
    try {
        const be3AiIntents = path.join(dataLoader.be3AiBase, 'src/services/intentResolver/semanticLab/intents');
        if (fs.existsSync(be3AiIntents)) {
            const destIntents = path.join(DATA_DIR, 'intents');
            if (fs.existsSync(destIntents)) fs.rmSync(destIntents, { recursive: true });
            fs.mkdirSync(destIntents, { recursive: true });
            
            // Shallow copy of directories containing bench.json
            const items = fs.readdirSync(be3AiIntents);
            for (const item of items) {
                const src = path.join(be3AiIntents, item);
                const bench = path.join(src, 'bench.json');
                if (fs.existsSync(bench)) {
                    const dest = path.join(destIntents, item);
                    fs.mkdirSync(dest, { recursive: true });
                    fs.copyFileSync(bench, path.join(dest, 'bench.json'));
                }
            }
            console.log('✅ Intents snapped.');
        }

        if (fs.existsSync(dataLoader.storeContextPath)) {
            fs.copyFileSync(dataLoader.storeContextPath, path.join(DATA_DIR, 'storeContext.js'));
            console.log('✅ storeContext snapped.');
        }
    } catch (err) {
        console.warn('⚠️  Could not snap some raw files. continuing with existing data...', err.message);
    }

    // 3. Trigger Full Reload (Generates Embeddings)
    console.log('\n[Sync] Phase 2: Generating Vector Embeddings...');
    console.log(`[Sync] Mode: ${embeddingService.mode.toUpperCase()}`);
    
    try {
        await Promise.all([
            intentMatcher.reload({ skipCache: true }),
            entityExtractor.reload({ skipCache: true })
        ]);

        // 4. Save to Cache File
        const cacheData = {
            version: '1.0.0',
            generatedAt: new Date().toISOString(),
            model: embeddingService.getConfig().key,
            modelName: embeddingService.getConfig().modelName,
            intents: intentMatcher.intentEmbeddings,
            entities: entityExtractor.entityEmbeddings
        };

        fs.writeFileSync(CACHE_FILE, JSON.stringify(cacheData, null, 2));
        
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`\n✨ [Sync] SUCCESS!`);
        console.log(`- Intents: ${cacheData.intents.length}`);
        console.log(`- Entities: ${cacheData.entities.length}`);
        console.log(`- Model: ${cacheData.modelName}`);
        console.log(`- Cache saved to: ${CACHE_FILE}`);
        console.log(`- Total Duration: ${duration}s\n`);

    } catch (err) {
        console.error('\n❌ [Sync] FAILED:', err.message);
        process.exit(1);
    }
}

sync();
