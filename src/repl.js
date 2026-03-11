import readline from 'readline';
import axios from 'axios';

const PORT = 3009;
const API_URL = `http://localhost:${PORT}`;

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

async function classify(text) {
    try {
        const [classResponse, extractResponse] = await Promise.all([
            axios.post(`${API_URL}/classify`, { text }),
            axios.post(`${API_URL}/extract`, { text })
        ]);

        const { results, duration: classDuration } = classResponse.data;
        const { entities, confidence: entityScores, duration: extractDuration } = extractResponse.data;

        // --- 1. DISPLAY INTENTS ---
        console.log(`\n${C.bold}${C.magenta}--- Classification (${classDuration}) ---${C.reset}`);
        if (results.length === 0) {
            console.log(`${C.red}  ✗ No intents found.${C.reset}`);
        } else {
            results.forEach((res, i) => {
                const color = i === 0 ? C.green : (res.score > 0.7 ? C.cyan : C.yellow);
                const scorePercent = (res.score * 100).toFixed(1);
                console.log(`${color}${C.bold}  ${i + 1}. ${res.intentName}${C.reset} ${C.dim}(${scorePercent}%)${C.reset}`);
                console.log(`${C.dim}     Match: "${C.reset}${C.magenta}${res.bestMatch}${C.dim}"${C.reset}`);
            });
        }

        // --- 2. DISPLAY ENTITIES ---
        console.log(`\n${C.bold}${C.cyan}--- Entities (${extractDuration}) ---${C.reset}`);
        let entityFound = false;

        const printEntity = (label, list, color) => {
            if (list && list.length > 0) {
                entityFound = true;
                console.log(`${C.white}${C.bold}  ${label}:${C.reset}`);
                list.forEach(val => {
                    const scoreKey = label === 'Attribute' ? `attribute:${val.key}:${val.value}` : `${label.toLowerCase()}:${val}`;
                    // Find matching confidence
                    let score = 0;
                    for (const [k, v] of Object.entries(entityScores)) {
                        if (k.includes(val)) score = v;
                    }
                    const scoreText = score ? `${C.dim}${(score * 100).toFixed(1)}%${C.reset}` : '';
                    console.log(`    ${color}• ${val}${C.reset} ${scoreText}`);
                });
            }
        };

        const getScore = (key) => {
            const score = entityScores[key] || 0;
            return score ? `${C.dim}(${(score * 100).toFixed(1)}%)${C.reset}` : '';
        };

        if (entities.category?.length > 0) {
            entityFound = true;
            console.log(`${C.white}${C.bold}  Categories:${C.reset}`);
            entities.category.forEach(c => {
                console.log(`    ${C.green}• ${c}${C.reset} ${getScore(`category:${c}`)}`);
            });
        }
        if (entities.vendor?.length > 0) {
            entityFound = true;
            console.log(`${C.white}${C.bold}  Vendors:${C.reset}`);
            entities.vendor.forEach(v => {
                console.log(`    ${C.yellow}• ${v}${C.reset} ${getScore(`vendor:${v}`)}`);
            });
        }
        if (entities.clause?.length > 0) {
            entityFound = true;
            console.log(`${C.white}${C.bold}  Clauses:${C.reset}`);
            entities.clause.forEach(cl => {
                console.log(`    ${C.magenta}• ${cl}${C.reset} ${getScore(`clause:${cl}`)}`);
            });
        }
        if (entities.attribute && Object.keys(entities.attribute).length > 0) {
            entityFound = true;
            console.log(`${C.white}${C.bold}  Attributes:${C.reset}`);
            for (const [type, vals] of Object.entries(entities.attribute)) {
                vals.forEach(val => {
                    console.log(`    ${C.cyan}• ${type}: ${val}${C.reset} ${getScore(`attribute:${type}:${val}`)}`);
                });
            }
        }

        if (!entityFound) {
            console.log(`${C.dim}  (No entities extracted)${C.reset}`);
        }
        console.log('');

    } catch (err) {
        if (err.code === 'ECONNREFUSED') {
            console.log(`${C.red}${C.bold}  ✗ Error: Could not connect to transformer server at ${API_URL}.${C.reset}`);
            console.log(`${C.dim}    Please start the server first with 'npm start' or 'node src/index.js'${C.reset}\n`);
        } else {
            console.log(`${C.red}  ✗ Error: ${err.message}${C.reset}\n`);
            if (err.response) console.log(err.response.data);
        }
    }
}

async function reload() {
    try {
        console.log(`${C.dim}Reloading knowledge base...${C.reset}`);
        const response = await axios.post(`${API_URL}/reload`);
        const { stats } = response.data;
        console.log(`${C.green}${C.bold}  ✓ Reloaded successfully in ${stats.duration}${C.reset}`);
        console.log(`${C.dim}    Intents: ${stats.intents}, Entities: ${stats.entities}${C.reset}\n`);
    } catch (err) {
        console.log(`${C.red}  ✗ Failed to reload: ${err.message}${C.reset}\n`);
    }
}

async function main() {
    console.log(`
${C.bold}${C.magenta}╔══════════════════════════════════════════════╗
║      be3-ai-transformer Semantic REPL        ║
╚══════════════════════════════════════════════╝${C.reset}
${C.dim}Connected to: ${C.reset}${C.cyan}${API_URL}${C.reset}
${C.dim}Commands: ${C.reset}${C.yellow}:reload${C.reset}${C.dim} (sync intents), ${C.reset}${C.yellow}:q${C.reset}${C.dim} (quit)${C.reset}
`);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: `${C.magenta}❯ ${C.reset}`
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

        if (input === ':reload') {
            await reload();
            rl.prompt();
            return;
        }

        await classify(input);
        rl.prompt();
    });

    rl.on('close', () => {
        console.log(`${C.dim}Goodbye!${C.reset}`);
        process.exit(0);
    });
}

main();
