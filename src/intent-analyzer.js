import readline from 'readline';
import dataLoader from './services/dataLoader.js';
import embeddingService from './services/embeddingService.js';

const C = {
    reset: '\x1b[0m',
    bold: '\x1b[1m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    red: '\x1b[31m',
    dim: '\x1b[2m',
    magenta: '\x1b[35m',
    white: '\x1b[37m'
};

/**
 * K-Means Clustering implementation for embeddings
 */
class KMeans {
    constructor(k, data, maxIterations = 50) {
        this.k = k;
        this.data = data; // Array of embeddings (number[][])
        this.maxIterations = maxIterations;
        this.centroids = [];
        this.assignments = [];
        this.inertia = 0;
    }

    initCentroids() {
        // Randomly pick K unique points as initial centroids
        const indices = new Set();
        while (indices.size < Math.min(this.k, this.data.length)) {
            indices.add(Math.floor(Math.random() * this.data.length));
        }
        this.centroids = Array.from(indices).map(i => [...this.data[i]]);
    }

    cosineDistance(a, b) {
        // Since embeddings are normalized (mean pooling + normalize: true in service),
        // Cosine similarity is just the dot product.
        // Distance = 1 - similarity
        let dot = 0;
        for (let i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return 1 - dot;
    }

    /**
     * Fit the model multiple times and keep the best result (lowest inertia)
     */
    fit(n_init = 5) {
        let bestInertia = Infinity;
        let bestCentroids = [];
        let bestAssignments = [];

        for (let i = 0; i < n_init; i++) {
            this.runOnce();
            if (this.inertia < bestInertia) {
                bestInertia = this.inertia;
                bestCentroids = [...this.centroids.map(c => [...c])];
                bestAssignments = [...this.assignments];
            }
        }

        this.inertia = bestInertia;
        this.centroids = bestCentroids;
        this.assignments = bestAssignments;
    }

    runOnce() {
        this.initCentroids();
        let changed = true;
        let iter = 0;
        this.assignments = [];

        while (changed && iter < this.maxIterations) {
            changed = false;
            iter++;

            // Assignment step
            const newAssignments = [];
            for (let i = 0; i < this.data.length; i++) {
                let minDist = Infinity;
                let clusterIndex = 0;
                for (let j = 0; j < this.centroids.length; j++) {
                    const dist = this.cosineDistance(this.data[i], this.centroids[j]);
                    if (dist < minDist) {
                        minDist = dist;
                        clusterIndex = j;
                    }
                }
                newAssignments.push(clusterIndex);
            }

            if (JSON.stringify(newAssignments) !== JSON.stringify(this.assignments)) {
                changed = true;
                this.assignments = newAssignments;
            } else {
                changed = false;
            }

            // Update step (calculate new centroids)
            const newCentroids = Array.from({ length: this.centroids.length }, () => new Array(this.data[0].length).fill(0));
            const counts = new Array(this.centroids.length).fill(0);

            for (let i = 0; i < this.data.length; i++) {
                const c = this.assignments[i];
                counts[c]++;
                for (let j = 0; j < this.data[0].length; j++) {
                    newCentroids[c][j] += this.data[i][j];
                }
            }

            for (let i = 0; i < this.centroids.length; i++) {
                if (counts[i] > 0) {
                    // Average and RE-NORMALIZE (Spherical K-means)
                    let magnitude = 0;
                    for (let j = 0; j < this.data[0].length; j++) {
                        newCentroids[i][j] /= counts[i];
                        magnitude += newCentroids[i][j] * newCentroids[i][j];
                    }
                    magnitude = Math.sqrt(magnitude);
                    if (magnitude > 0) {
                        for (let j = 0; j < this.data[0].length; j++) {
                            newCentroids[i][j] /= magnitude;
                        }
                    }
                    this.centroids[i] = newCentroids[i];
                }
            }
        }

        // Calculate Inertia (Sum of cosine distances)
        this.inertia = 0;
        for (let i = 0; i < this.data.length; i++) {
            this.inertia += this.cosineDistance(this.data[i], this.centroids[this.assignments[i]]);
        }
    }
}

async function suggestOptimalK(data) {
    console.log(`${C.dim}Calculating stability-enhanced cluster count (Elbow Method)...${C.reset}`);
    const maxK = Math.min(12, Math.floor(data.length / 2) + 1);
    const inertias = [];
    
    for (let k = 1; k <= maxK; k++) {
        const kmeans = new KMeans(k, data);
        kmeans.fit(8); // Use 8 restarts per K for high stability
        inertias.push(kmeans.inertia);
    }

    // Find the "elbow" (point of maximum curvature)
    // Simple heuristic: normalize inertias and finds the k where the drop slows significantly
    let bestK = 1;
    let maxDrop = 0;
    for (let i = 1; i < inertias.length - 1; i++) {
        const drop = (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1] || 1);
        if (drop > maxDrop) {
            maxDrop = drop;
            bestK = i + 1;
        }
    }

    return bestK;
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const question = (query) => new Promise((resolve) => rl.question(query, resolve));

async function main() {
    console.log(`
${C.bold}${C.magenta}╔══════════════════════════════════════════════╗
║      Intent Variation Analyzer & Pruner      ║
╚══════════════════════════════════════════════╝${C.reset}
`);

    await embeddingService.init();
    const allIntents = await dataLoader.loadIntents();
    const intentNames = Object.keys(allIntents);

    while (true) {
        console.log(`\n${C.bold}Available Intents:${C.reset}`);
        intentNames.forEach((name, i) => console.log(`  ${C.cyan}${i + 1}. ${name}${C.reset} ${C.dim}(${allIntents[name].length} variations)${C.reset}`));
        
        const answer = await question(`\nSelect intent ID or Name (or ':q' to quit): `);
        
        if (answer === ':q' || answer === 'exit') break;

        let selectedIntent = intentNames.find(n => n === answer);
        if (!selectedIntent && !isNaN(parseInt(answer))) {
            selectedIntent = intentNames[parseInt(answer) - 1];
        }

        if (!selectedIntent) {
            console.log(`${C.red}Invalid selection.${C.reset}`);
            continue;
        }

        const variations = allIntents[selectedIntent];
        // Shuffle variations so the "prototypes" (first items) are random each run
        for (let i = variations.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [variations[i], variations[j]] = [variations[j], variations[i]];
        }

        console.log(`\nAnalyzing ${C.bold}${selectedIntent}${C.reset} with ${variations.length} variations...`);

        console.log(`\n${C.bold}Select Analysis Mode:${C.reset}`);
        console.log(`  ${C.cyan}1. K-Means Clustering${C.reset} (Group by similarity)`);
        console.log(`  ${C.cyan}2. Distance-based Pruning${C.reset} (Identify redundant items)`);
        
        const mode = await question(`\nSelect mode (1/2): `);

        const embeddings = await embeddingService.getEmbeddings(variations);

        if (mode === '2') {
            // --- DISTANCE-BASED PRUNING ---
            const thresholdStr = await question(`Enter cosine distance threshold (default 0.05, 0=equal): `);
            const threshold = parseFloat(thresholdStr) || 0.05;

            console.log(`${C.dim}Calculating redundancy using Cosine distance...${C.reset}`);
            
            const kept = [];
            const pruned = [];

            for (let i = 0; i < variations.length; i++) {
                let closestMatch = null;
                let minDistance = Infinity;

                // Compare with all variations ALREADY KEPT
                for (const keptItem of kept) {
                    const dist = cosineDistance(embeddings[i], keptItem.embedding);
                    if (dist < minDistance) {
                        minDistance = dist;
                        closestMatch = keptItem.text;
                    }
                }

                if (minDistance < threshold) {
                    pruned.push({ text: variations[i], distance: minDistance, closest: closestMatch });
                } else {
                    kept.push({ text: variations[i], embedding: embeddings[i] });
                }
            }

            console.log(`\n${C.bold}${C.green}--- KEPT VARIATIONS (${kept.length}) ---${C.reset}`);
            kept.forEach(k => console.log(`  ${C.green}• ${C.reset}${k.text}`));

            console.log(`\n${C.bold}${C.red}--- PRUNED VARIATIONS (${pruned.length}) ---${C.reset}`);
            pruned.forEach(p => {
                console.log(`  ${C.red}✗ ${C.reset}${p.text} ${C.dim}(close to: "${p.closest}", cos-dist: ${p.distance.toFixed(4)})${C.reset}`);
            });

            console.log(`\n${C.bold}${C.magenta}═══ Summary of Pruning (Cosine) ═══${C.reset}`);
            console.log(`${C.white}  Total variations:  ${variations.length}${C.reset}`);
            console.log(`${C.green}  Unique kept:       ${kept.length}${C.reset}`);
            console.log(`${C.red}  Redundant pruned:  ${pruned.length}${C.reset}`);
            console.log(`${C.yellow}  Reduction:         ${((pruned.length / variations.length) * 100).toFixed(1)}%${C.reset}`);
            console.log(`${C.dim}  Threshold used:    ${threshold}${C.reset}`);

        } else {
            // --- K-MEANS CLUSTERING ---
            const suggestedK = await suggestOptimalK(embeddings);

            console.log(`\n${C.green}Suggested Clusters: ${C.bold}${suggestedK}${C.reset} ${C.dim}(based on semantic diversity)${C.reset}`);
            
            const kStr = await question(`How many clusters do you want to see? (Enter for suggested ${suggestedK}): `);
            const k = parseInt(kStr) || suggestedK;

            console.log(`${C.dim}Running Spherical K-Means (Cosine) for K=${k}...${C.reset}`);
            const kmeans = new KMeans(k, embeddings);
            kmeans.fit(8);

            // Organize variations by cluster
            const clusters = Array.from({ length: k }, () => []);
            kmeans.assignments.forEach((assignment, i) => {
                clusters[assignment].push({
                    text: variations[i],
                    distToCentroid: cosineDistance(embeddings[i], kmeans.centroids[assignment])
                });
            });

            // Display results
            clusters.forEach((cluster, i) => {
                // Sort by distance to centroid (most representative first)
                cluster.sort((a, b) => a.distToCentroid - b.distToCentroid);
                
                console.log(`\n${C.bold}${C.yellow}Cluster #${i + 1}${C.reset} ${C.dim}(${cluster.length} items)${C.reset}`);
                console.log(`${C.dim}Representative: "${C.reset}${C.bold}${cluster[0].text}${C.dim}"${C.reset}`);
                
                // Show all items in cluster
                cluster.forEach((item, j) => {
                    const color = j === 0 ? C.green : (item.distToCentroid < 0.05 ? C.dim : C.white);
                    console.log(`  ${color}• ${item.text}${C.reset} ${C.dim}(cos-dist: ${item.distToCentroid.toFixed(4)})${C.reset}`);
                });
            });

            console.log(`\n${C.magenta}${C.bold}Pruning Advice:${C.reset}`);
            console.log(`- Items with ${C.green}cosine-dist < 0.05${C.reset} are highly redundant.`);
            console.log(`- You can likely keep only the first (green) item from each cluster.`);
        }

        await question(`\nPress Enter to continue...`);
    }

    console.log(`${C.dim}Goodbye!${C.reset}`);
    process.exit(0);
}

// Global utility for unit-vector cosine distance
function cosineDistance(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
    }
    return 1 - dot;
}

main().catch(err => {
    console.error(`\n${C.red}FATAL ERROR: ${err.message}${C.reset}`);
    process.exit(1);
});
