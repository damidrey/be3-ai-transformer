import readline from 'readline';
import axios from 'axios';

const TRANSFORMER_PORT = 3009;
const BACKEND_PORT = 3000;
const TRANSFORMER_URL = `http://localhost:${TRANSFORMER_PORT}`;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;
const TENANT_ID = 'cbe1df05-45ed-455a-9ce6-156b0bd45713';

const C = {
    reset: '\x1b[0m',
    bold: '\x1b[1m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    red: '\x1b[31m',
    dim: '\x1b[2m',
    magenta: '\x1b[35m',
    white: '\x1b[37m',
    blue: '\x1b[34m'
};

/**
 * Perform a search via the backend.
 * Uses /vector/search or /vector/hybrid-search depending on mode.
 */
async function performSearch(queryText, limit = 20, isHybrid = false) {
    try {
        const startTime = Date.now();
        const endpoint = isHybrid ? '/vector/hybrid-search' : '/vector/search';
        
        const response = await axios.get(`${BACKEND_URL}${endpoint}`, {
            params: { query: queryText, limit },
            headers: { 'x-tenant-id': TENANT_ID }
        });
        const duration = Date.now() - startTime;
        const { data } = response.data;

        const modeLabel = isHybrid ? 'Hybrid (Text + Vector)' : 'Vector (Semantic)';
        console.log(`\n${C.bold}${C.blue}── ${modeLabel} Search Results (${duration}ms) ──${C.reset}`);
        console.log(`${C.dim}   Query: "${queryText}"  |  Found: ${data.total} products${C.reset}\n`);

        if (!data.results || data.results.length === 0) {
            console.log(`${C.yellow}   No results found. Make sure embeddings have been generated.${C.reset}`);
            console.log(`${C.dim}   Run: node modules/vector/scripts/generateEmbeddings.js${C.reset}\n`);
            return;
        }

        data.results.forEach((product, i) => {
            const rank = String(i + 1).padStart(2, ' ');
            
            // In hybrid mode, we show the hybrid score and the break-down
            if (isHybrid) {
                const score = (product.hybrid_score || 0).toFixed(4);
                const vSim = (product.vector_similarity || 0).toFixed(1);
                const tRank = (product.text_rank || 0).toFixed(3);

                console.log(`${C.white}${C.bold}   ${rank}. ${product.name}${C.reset} ${C.dim}[ID: ${product.id}]${C.reset}`);
                console.log(`       ${C.cyan}Score: ${score}${C.reset} ${C.dim}(Vector: ${vSim}%, Text: ${tRank})${C.reset}`);
            } else {
                const sim = product.similarity.toFixed(1);
                
                // Color the similarity score
                let simColor = C.red;
                if (product.similarity >= 80) simColor = C.green;
                else if (product.similarity >= 60) simColor = C.cyan;
                else if (product.similarity >= 40) simColor = C.yellow;

                // Build the similarity bar
                const barLength = 20;
                const filled = Math.round((product.similarity / 100) * barLength);
                const bar = '█'.repeat(filled) + '░'.repeat(barLength - filled);

                console.log(`${C.white}${C.bold}   ${rank}. ${product.name}${C.reset} ${C.dim}[ID: ${product.id}]${C.reset}`);
                console.log(`       ${simColor}${bar} ${sim}%${C.reset}`);
            }

            // Show price if available
            if (product.price) {
                console.log(`       ${C.dim}Price: ₦${Number(product.price).toLocaleString()}${C.reset}`);
            }

            // Show tags if available
            if (product.tags && product.tags.length > 0) {
                const tagsRaw = Array.isArray(product.tags) ? product.tags : [];
                const tags = tagsRaw.slice(0, 4).join(', ');
                if (tags) console.log(`       ${C.dim}Tags: ${tags}${C.reset}`);
            }

            console.log('');
        });

    } catch (err) {
        if (err.code === 'ECONNREFUSED') {
            const failedUrl = err.config?.url || '';
            if (failedUrl.includes(BACKEND_PORT.toString())) {
                console.log(`${C.red}${C.bold}   ✗ Backend server is not running at ${BACKEND_URL}${C.reset}`);
                console.log(`${C.dim}     Start it with: cd be3_backend && npm run dev${C.reset}\n`);
            } else {
                console.log(`${C.red}${C.bold}   ✗ Cannot connect to ${failedUrl}${C.reset}\n`);
            }
        } else if (err.response?.status === 400) {
            console.log(`${C.red}   ✗ Bad request: ${err.response.data?.error || err.message}${C.reset}\n`);
        } else {
            console.log(`${C.red}   ✗ Error: ${err.response?.data?.error || err.message}${C.reset}\n`);
        }
    }
}

/**
 * Find similar products via the backend's /vector/similar/:id endpoint.
 */
async function findSimilar(productId, limit = 5) {
    try {
        const response = await axios.get(`${BACKEND_URL}/vector/similar/${productId}`, {
            params: { limit },
            headers: { 'x-tenant-id': TENANT_ID }
        });
        const products = response.data.data;

        console.log(`\n${C.bold}${C.magenta}── Similar Products ──${C.reset}\n`);

        if (!products || products.length === 0) {
            console.log(`${C.yellow}   No similar products found.${C.reset}\n`);
            return;
        }

        products.forEach((p, i) => {
            const sim = p.similarity.toFixed(1);
            let simColor = p.similarity >= 70 ? C.green : p.similarity >= 50 ? C.cyan : C.yellow;
            console.log(`   ${C.white}${C.bold}${i + 1}. ${p.name}${C.reset} ${simColor}(${sim}%)${C.reset}${p.price ? ` ${C.dim}₦${Number(p.price).toLocaleString()}${C.reset}` : ''}`);
        });
        console.log('');

    } catch (err) {
        console.log(`${C.red}   ✗ Error: ${err.response?.data?.error || err.message}${C.reset}\n`);
    }
}

/**
 * Show vector coverage stats via /vector/status
 */
async function showStats() {
    try {
        const response = await axios.get(`${BACKEND_URL}/vector/status`, {
            headers: { 'x-tenant-id': TENANT_ID }
        });
        const stats = response.data;

        console.log(`\n${C.bold}${C.cyan}── Vector Stats ──${C.reset}`);
        console.log(`   ${C.white}Model:${C.reset}     ${stats.model} (${stats.dimensions}D)`);
        console.log(`   ${C.white}Products:${C.reset}  ${stats.embedded_products} / ${stats.total_products}`);
        console.log(`   ${C.white}Coverage:${C.reset}  ${stats.coverage}`);
        console.log(`   ${C.white}Last run:${C.reset}  ${stats.last_embedded_at || 'never'}\n`);

    } catch (err) {
        console.log(`${C.red}   ✗ Error: ${err.response?.data?.error || err.message}${C.reset}\n`);
    }
}

/**
 * Embed a single product via /vector/embed/product/:id
 */
async function embedProduct(productId) {
    try {
        console.log(`${C.dim}   Embedding product ${productId}...${C.reset}`);
        const response = await axios.post(`${BACKEND_URL}/vector/embed/product/${productId}`, {}, {
            headers: { 'x-tenant-id': TENANT_ID }
        });
        console.log(`${C.green}   ✓ Embedded: "${response.data.text?.substring(0, 80)}..."${C.reset}\n`);
    } catch (err) {
        console.log(`${C.red}   ✗ Error: ${err.response?.data?.error || err.message}${C.reset}\n`);
    }
}

async function main() {
    console.log(`
${C.bold}${C.blue}╔══════════════════════════════════════════════╗
║        be3 Vector Search REPL                ║
╚══════════════════════════════════════════════╝${C.reset}
${C.dim}Backend:     ${C.reset}${C.cyan}${BACKEND_URL}${C.reset}
${C.dim}Transformer: ${C.reset}${C.cyan}${TRANSFORMER_URL}${C.reset}
${C.dim}Tenant:      ${C.reset}${C.cyan}${TENANT_ID}${C.reset}

${C.dim}Commands:${C.reset}
  ${C.yellow}:hybrid${C.reset}             ${C.bold}${C.green}Toggle Hybrid Search (ON/OFF)${C.reset}
  ${C.yellow}:stats${C.reset}              ${C.dim}Show embedding coverage${C.reset}
  ${C.yellow}:similar <id>${C.reset}       ${C.dim}Find similar products${C.reset}
  ${C.yellow}:embed <id>${C.reset}         ${C.dim}Embed a single product${C.reset}
  ${C.yellow}:limit <n>${C.reset}          ${C.dim}Set result limit (default 20)${C.reset}
  ${C.yellow}:q${C.reset}                  ${C.dim}Quit${C.reset}
  ${C.dim}Anything else → search${C.reset}
`);

    let limit = 20;
    let isHybrid = false;

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: `${C.blue}🔍 ${C.reset}`
    });

    rl.prompt();

    rl.on('line', async (line) => {
        const input = line.trim();

        if (!input) {
            rl.prompt();
            return;
        }

        if (input === ':q' || input === ':quit' || input === 'exit') {
            rl.close();
            return;
        }

        if (input === ':hybrid') {
            isHybrid = !isHybrid;
            console.log(`${C.green}   ✓ Hybrid Search mode: ${C.bold}${isHybrid ? 'ON' : 'OFF'}${C.reset}`);
            console.log(`     ${C.dim}${isHybrid ? 'Combining text rank + vector similarity' : 'Using raw vector similarity'}${C.reset}\n`);
            rl.prompt();
            return;
        }

        if (input === ':stats') {
            await showStats();
            rl.prompt();
            return;
        }

        if (input.startsWith(':similar ')) {
            const productId = input.replace(':similar ', '').trim();
            await findSimilar(productId);
            rl.prompt();
            return;
        }

        if (input.startsWith(':embed ')) {
            const productId = input.replace(':embed ', '').trim();
            await embedProduct(productId);
            rl.prompt();
            return;
        }

        if (input.startsWith(':limit ')) {
            const newLimit = parseInt(input.replace(':limit ', '').trim());
            if (isNaN(newLimit) || newLimit < 1) {
                console.log(`${C.red}   Invalid limit. Use a positive number.${C.reset}\n`);
            } else {
                limit = newLimit;
                console.log(`${C.green}   ✓ Limit set to ${limit}${C.reset}\n`);
            }
            rl.prompt();
            return;
        }

        // Default: perform search
        await performSearch(input, limit, isHybrid);
        rl.prompt();
    });

    rl.on('close', () => {
        console.log(`${C.dim}Goodbye!${C.reset}`);
        process.exit(0);
    });
}

main();
