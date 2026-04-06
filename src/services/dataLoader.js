import fs from 'fs';
import path from 'path';
import tokenDemasker from './tokenDemasker.js';

class DataLoader {
    constructor() {
        // 1. Determine Base Directories
        // Default to absolute paths for local dev, but allow override for Render
        this.be3AiBase = process.env.BE3_AI_PATH || 'C:\\Users\\chatz\\Downloads\\eCommerce\\be3_ai';
        this.localDataDir = path.join(process.cwd(), 'data');

        // 2. Resolve Paths (Priority: Env -> Absolute -> Local)
        this.intentsBaseDir = this.resolvePath(
            process.env.INTENTS_DIR, 
            path.join(this.be3AiBase, 'src/services/intentResolver/semanticLab/intents'),
            path.join(this.localDataDir, 'intents')
        );

        this.storeContextPath = this.resolvePath(
            process.env.STORE_CONTEXT_PATH,
            path.join(this.be3AiBase, 'src/context/storeContext.js'),
            path.join(this.localDataDir, 'storeContext.js')
        );

        this.clausesBenchPath = this.resolvePath(
            process.env.CLAUSES_BENCH_PATH,
            path.join(this.be3AiBase, 'src/services/intentResolver/semanticLab/clauses/clause_bench.json'),
            path.join(this.localDataDir, 'clause_bench.json')
        );

        this.facetsBenchPath = this.resolvePath(
            process.env.FACETS_BENCH_PATH,
            path.join(this.be3AiBase, 'src/services/intentResolver/semanticLab/facets/facet_bench.json'),
            path.join(this.localDataDir, 'facet_bench.json')
        );
    }

    /**
     * Resolves a path by checking several options in order.
     */
    resolvePath(envVar, absolutePath, localPath) {
        if (envVar && fs.existsSync(envVar)) return envVar;
        if (fs.existsSync(absolutePath)) return absolutePath;
        if (fs.existsSync(localPath)) return localPath;
        
        // Fallback to localPath even if it doesn't exist yet (for sync script)
        return localPath;
    }

    /**
     * Recursively find all bench.json files and extract variations.
     * @returns {Object} { intentName: [variations] }
     */
    async loadIntents() {
        console.log(`[DataLoader] Scanning for intents in: ${this.intentsBaseDir}`);
        const intentData = {};

        if (!fs.existsSync(this.intentsBaseDir)) {
            console.warn(`[DataLoader] Intents directory not found: ${this.intentsBaseDir}`);
            return {};
        }

        const items = fs.readdirSync(this.intentsBaseDir);

        for (const item of items) {
            const itemPath = path.join(this.intentsBaseDir, item);
            if (!fs.statSync(itemPath).isDirectory()) continue;

            // Suspend deprecated/suspended intents
            if (item === 'check_availability' || item === 'browse_collection') {
                continue;
            }

            const benchPath = path.join(itemPath, 'bench.json');

            if (fs.existsSync(benchPath)) {
                try {
                    const content = fs.readFileSync(benchPath, 'utf8');
                    const bench = JSON.parse(content);

                    for (const [intentName, data] of Object.entries(bench)) {
                        if (data.variations && Array.isArray(data.variations)) {
                            if (!intentData[intentName]) {
                                intentData[intentName] = [];
                            }
                            intentData[intentName].push(...data.variations);
                        }
                    }
                } catch (err) {
                    console.error(`[DataLoader] Error reading ${benchPath}:`, err.message);
                }
            }
        }
        
        return tokenDemasker.demaskVariations(intentData);
    }

    /**
     * Load knowledge entities for semantic extraction.
     */
    async loadEntities() {
        console.log(`[DataLoader] Loading entities...`);
        const entities = {
            category: [],
            vendor: [],
            clause: {},
            attribute: {}
        };

        // 1. Load Categories, Vendors, and Attributes from storeContext.js
        if (fs.existsSync(this.storeContextPath)) {
            try {
                const content = fs.readFileSync(this.storeContextPath, 'utf8');

                // Improved extractor that handles ES modules and commonjs
                const extractObject = (varName) => {
                    const regex = new RegExp(`(?:const|let|var|export const) ${varName} = ({[\\s\\S]*?});`, 'm');
                    const match = content.match(regex);
                    if (match && match[1]) {
                        try {
                            // Try clean JSON first
                            return JSON.parse(match[1].replace(/(\w+):/g, '"$1":').replace(/'/g, '"'));
                        } catch (e) {
                            // Fallback to eval-like logic (safe here since we're local)
                            try {
                                return new Function(`return ${match[1]}`)();
                            } catch (e2) {
                                console.error(`[DataLoader] Failed to parse ${varName}:`, e2.message);
                                return null;
                            }
                        }
                    }
                    return null;
                };

                const categories = extractObject('CATEGORIES');
                if (categories) {
                    Object.entries(categories).forEach(([key, c]) => {
                        const variations = [c.label, c.slug.replace(/-/g, ' ')];
                        if (c.description) variations.push(c.description);

                        if (c.children) {
                            c.children.forEach(childKey => {
                                const child = categories[childKey];
                                if (child) variations.push(child.label);
                            });
                        }

                        // Conceptual augmentation
                        if (key === 'phone_accessories') variations.push('airpods', 'headphones', 'earbuds', 'chargers', 'cases');
                        if (key === 'food') variations.push('hungry', 'eat', 'meal', 'snacks');
                        if (key === 'sound_gadget') variations.push('airpods', 'headphones', 'music', 'sound');

                        entities.category.push({
                            label: c.label,
                            slug: c.slug,
                            key: key,
                            variations: [...new Set(variations)].filter(Boolean)
                        });
                    });
                }

                const vendors = extractObject('VENDORS');
                if (vendors) {
                    Object.values(vendors).forEach(v => {
                        entities.vendor.push({
                            label: v.business_name,
                            variations: [v.business_name, v.tag].filter(Boolean)
                        });
                    });
                }

                const attributes = extractObject('ATTRIBUTES');
                if (attributes) {
                    Object.entries(attributes).forEach(([attrKey, attr]) => {
                        if (!entities.attribute[attrKey]) entities.attribute[attrKey] = [];
                        if (attr.label) entities.attribute[attrKey].push(attr.label);

                        if (attr.predefined_values) {
                            const values = attr.predefined_values.map(v => typeof v === 'object' ? v.label : v);
                            entities.attribute[attrKey].push(...values);
                        }

                        if (attr.clauses) {
                            attr.clauses.forEach(cl => {
                                if (cl.matches) {
                                    const matches = cl.matches.map(m => typeof m === 'object' ? m.label : m);
                                    entities.attribute[attrKey].push(...matches);
                                }
                            });
                        }
                    });
                }
            } catch (err) {
                console.error('[DataLoader] Error parsing storeContext.js:', err.message);
            }
        }

        // 2. Load Clause Variations
        if (fs.existsSync(this.clausesBenchPath)) {
            try {
                const content = fs.readFileSync(this.clausesBenchPath, 'utf8');
                const clauses = JSON.parse(content);
                for (const [name, data] of Object.entries(clauses)) {
                    entities.clause[name] = data.variations || [];
                }
            } catch (err) {
                console.error('[DataLoader] Error parsing clause_bench.json:', err.message);
            }
        }

        // 3. Load Facet (Attribute) Variations
        if (fs.existsSync(this.facetsBenchPath)) {
            try {
                const content = fs.readFileSync(this.facetsBenchPath, 'utf8');
                const facets = JSON.parse(content);
                for (const [name, data] of Object.entries(facets)) {
                    if (!entities.attribute[name]) entities.attribute[name] = [];
                    entities.attribute[name].push(...(data.variations || []));
                }
            } catch (err) {
                console.error('[DataLoader] Error parsing facet_bench.json:', err.message);
            }
        }

        return entities;
    }
}

export default new DataLoader();

