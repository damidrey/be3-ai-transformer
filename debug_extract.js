import entityExtractor from './src/services/entityExtractor.js';

async function debug() {
    await entityExtractor.reload();
    const texts = [
        'I am hungry',
        'Does Dareymi sells cheap iphone 16',
        'do you have a red airpods pro'
    ];
    for (const text of texts) {
        console.log(`\nTEST: "${text}"`);
        const result = await entityExtractor.extract(text);
        console.log(JSON.stringify(result.entities, null, 2));
    }
}
debug().catch(console.error);
