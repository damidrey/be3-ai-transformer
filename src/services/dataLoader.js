import fs from 'fs';
import path from 'path';
import tokenDemasker from './tokenDemasker.js';

class DataLoader {
    constructor() {
        // Paths to the be3_ai project
        this.be3AiBase = 'C:\\Users\\chatz\\Downloads\\eCommerce\\be3_ai';
        this.intentsBaseDir = path.join(this.be3AiBase, 'src\\services\\intentResolver\\semanticLab\\intents');
        this.storeContextPath = path.join(this.be3AiBase, 'src\\context\\storeContext.js');
        this.clausesBenchPath = path.join(this.be3AiBase, 'src\\services\\intentResolver\\semanticLab\\clauses\\clause_bench.json');
        this.facetsBenchPath = path.join(this.be3AiBase, 'src\\services\\intentResolver\\semanticLab\\facets\\facet_bench.json');
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

        const folders = fs.readdirSync(this.intentsBaseDir);

        for (const folder of folders) {
            // Suspend deprecated/suspended intents
            if (folder === 'check_availability' || folder === 'browse_collection') {
                console.log(`[DataLoader] Skipping suspended intent: ${folder}`);
                continue;
            }

            const benchPath = path.join(this.intentsBaseDir, folder, 'bench.json');

            if (fs.existsSync(benchPath)) {
                try {
                    const content = fs.readFileSync(benchPath, 'utf8');
                    const bench = JSON.parse(content);

                    // bench.json structure: { "intent_name": { "variations": [...] } }
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
        
        // 1. Fetch Dynamic Context from be3_ai
        let dynamicContext = {};
        try {
            const entities = await this.loadEntities();
            dynamicContext = {
                vendors: entities.vendor.map(v => v.label),
                categories: entities.category.map(c => c.label),
                // Flatten all possible attribute values into one pool for [attributes]
                attributes: [
                    ...Object.values(entities.attribute).flat(),
                    ...Object.values(entities.clause).flat()
                ]
            };
        } catch (err) {
            console.warn('[DataLoader] Could not load dynamic context for de-masking:', err.message);
        }

        // 2. Apply De-masking Layer (now simple structural de-masking)
        return tokenDemasker.demaskVariations(intentData);
    }

    /**
     * Load knowledge entities for semantic extraction.
     */
    async loadEntities() {
        console.log('[DataLoader] Loading entities from be3_ai...');
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

                // Helper to extract JSON-like object from a JS file
                const extractObject = (varName) => {
                    const regex = new RegExp(`const ${varName} = ({[\\s\\S]*?});`, 'm');
                    const match = content.match(regex);
                    if (match && match[1]) {
                        try {
                            return JSON.parse(match[1]);
                        } catch (e) {
                            return eval(`(${match[1]})`);
                        }
                    }
                    return null;
                };

                const categories = extractObject('CATEGORIES');
                if (categories) {
                    Object.entries(categories).forEach(([key, c]) => {
                        const variations = [c.label, c.slug.replace(/-/g, ' ')];
                        if (c.description) variations.push(c.description);

                        // Add children labels as variations (helps match parent)
                        if (c.children && c.children.length > 0) {
                            c.children.forEach(childKey => {
                                const child = categories[childKey];
                                if (child) variations.push(child.label);
                            });
                        }

                        // Conceptual augmentation for common but distinct terms
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

        console.log(`[DataLoader] Summary: ${entities.category.length} categories, ${entities.vendor.length} vendors, ${Object.keys(entities.clause).length} clause types, ${Object.keys(entities.attribute).length} attribute types.`);
        return entities;
    }
}

export default new DataLoader();
