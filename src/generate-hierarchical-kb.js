import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import embeddingService from './services/embeddingService.js';
import vectorStore from './services/vectorStore.js';

/**
 * Hierarchical KB Generator
 * 
 * Reads all variation files from data/hierarchical/ and generates
 * pre-computed embeddings stored in data/hierarchical-cache/.
 * 
 * Usage: node src/generate-hierarchical-kb.js
 */

const HIERARCHICAL_DIR = path.join(process.cwd(), 'data/hierarchical');
const BATCH_SIZE = 50;

async function generatePool(poolKey, candidates) {
    const allTexts = [];
    const allNames = [];

    for (const [candidateName, data] of Object.entries(candidates)) {
        const variations = data.variations || [];
        for (const text of variations) {
            allTexts.push(text);
            allNames.push(candidateName);
        }
    }

    if (allTexts.length === 0) {
        console.warn(`  ⚠️  Pool ${poolKey}: no variations, skipping.`);
        return;
    }

    console.log(`  📐 Pool ${poolKey}: embedding ${allTexts.length} variations...`);

    const allEmbeddings = [];
    for (let i = 0; i < allTexts.length; i += BATCH_SIZE) {
        const chunk = allTexts.slice(i, i + BATCH_SIZE);
        const embeddings = await embeddingService.getEmbeddings(chunk);
        allEmbeddings.push(...embeddings);
    }

    const vectors = allTexts.map((text, i) => ({
        candidateName: allNames[i],
        text,
        embedding: new Float32Array(allEmbeddings[i])
    }));

    vectorStore.setPool(
        poolKey.includes(':') ? poolKey.split(':')[0] : poolKey,
        poolKey.includes(':') ? poolKey.split(':')[1] : null,
        vectors
    );

    console.log(`  ✅ Pool ${poolKey}: ${vectors.length} vectors ready.`);
}

async function main() {
    console.log(`\n🔀 [Hierarchical KB Generator] Starting...\n`);
    const startTime = Date.now();

    // 1. Process class_kb.json
    const classKbPath = path.join(HIERARCHICAL_DIR, 'class_kb.json');
    if (fs.existsSync(classKbPath)) {
        const data = JSON.parse(fs.readFileSync(classKbPath, 'utf8'));
        await generatePool('class', data.candidates);
    }

    // 2. Process intent KBs
    const intentDir = path.join(HIERARCHICAL_DIR, 'intent_kb');
    if (fs.existsSync(intentDir)) {
        const files = fs.readdirSync(intentDir).filter(f => f.endsWith('.json'));
        for (const file of files) {
            const data = JSON.parse(fs.readFileSync(path.join(intentDir, file), 'utf8'));
            const parent = data.parent; // e.g., 'Discovery'
            await generatePool(`intent:${parent}`, data.candidates);
        }
    }

    // 3. Process sub-intent KBs
    const subintentDir = path.join(HIERARCHICAL_DIR, 'subintent_kb');
    if (fs.existsSync(subintentDir)) {
        const files = fs.readdirSync(subintentDir).filter(f => f.endsWith('.json'));
        for (const file of files) {
            const data = JSON.parse(fs.readFileSync(path.join(subintentDir, file), 'utf8'));
            const parent = data.parent; // e.g., 'Cart_Management'
            await generatePool(`subintent:${parent}`, data.candidates);
        }
    }

    // 4. Save all pools to cache
    await vectorStore.saveHierarchicalCache();

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\n✅ [Hierarchical KB Generator] Complete in ${duration}s.`);
    console.log(`   Pools: ${vectorStore.pools.size}`);
    console.log(`   Cache: data/hierarchical-cache/\n`);

    process.exit(0);
}

main().catch(err => {
    console.error('❌ Generation failed:', err);
    process.exit(1);
});
