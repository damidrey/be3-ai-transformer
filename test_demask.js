import tokenDemasker from './src/services/tokenDemasker.js';

const testVariations = [
    "I need to buy [product] again, I've run out.",
    "Show me [attributes] [product] from [vendor].",
    "Where can I find [product] in Nigeria?",
    "I want to see [product] with long-lasting battery."
];

const processed = testVariations.map(v => tokenDemasker.demask(v));

console.log('--- Original ---');
testVariations.forEach(v => console.log(v));

console.log('\n--- Demasked ---');
processed.forEach(v => console.log(v));

console.log(`\nReplacement count: ${tokenDemasker.demaskCount}`);
