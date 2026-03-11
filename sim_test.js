import embeddingService from './src/services/embeddingService.js';

async function verify() {
    const [v1, v2, v3] = await embeddingService.getEmbeddings(['airpods pro', 'Phone Accessories', 'Sound Gadget']);

    const cosineSimilarity = (a, b) => {
        let dot = 0, ma = 0, mb = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            ma += a[i] * a[i];
            mb += b[i] * b[i];
        }
        return dot / (Math.sqrt(ma) * Math.sqrt(mb));
    };

    console.log('Airpods vs Phone Accessories:', cosineSimilarity(v1, v2));
    console.log('Airpods vs Sound Gadget:', cosineSimilarity(v1, v3));
}
verify().catch(console.error);
