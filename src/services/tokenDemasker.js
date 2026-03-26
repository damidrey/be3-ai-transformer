import { PRODUCT_WORDS } from '../utils/productWords.js';
import { DATE_WORDS } from '../utils/dateWords.js';

class TokenDemasker {
    constructor() {
        this.demaskCount = 0;
        this.activeMasks = ['[product]', '[vendor]', '[category]', '[attributes]', '[date]'];
    }

    /**
     * Replaces masked tokens with random words from the dictionary or dynamic context.
     * @param {string} text 
     * @param {Object} dynamicContext { vendors, categories, attributes }
     * @returns {string}
     */
    demask(text, dynamicContext = {}) {
        if (!text || typeof text !== 'string') return text;

        let demaskedText = text;

        const masks = {
            '\\[product\\]': PRODUCT_WORDS,
            '\\[date\\]': DATE_WORDS,
            '\\[vendor\\]': dynamicContext.vendors?.length ? dynamicContext.vendors : ['Generic Vendor', 'Local Store'],
            '\\[category\\]': dynamicContext.categories?.length ? dynamicContext.categories : ['General Items', 'Products'],
            '\\[attributes\\]': dynamicContext.attributes?.length ? dynamicContext.attributes : ['Standard', 'New Style']
        };

        for (const [pattern, words] of Object.entries(masks)) {
            const regex = new RegExp(pattern, 'g');
            if (regex.test(demaskedText)) {
                demaskedText = demaskedText.replace(regex, () => {
                    const randomWord = words[Math.floor(Math.random() * words.length)];
                    this.demaskCount++;
                    return randomWord;
                });
            }
        }

        return demaskedText;
    }

    /**
     * Processes a full intent data object.
     * @param {Object} intentData { intentName: [variations] }
     * @param {Object} dynamicContext { vendors, categories, attributes }
     * @returns {Object}
     */
    demaskVariations(intentData, dynamicContext = {}) {
        console.log('[TokenDemasker] 🎭 Starting multi-mask de-masking layer...');
        this.demaskCount = 0;
        const startTime = Date.now();

        const demaskedData = {};

        for (const [intentName, variations] of Object.entries(intentData)) {
            demaskedData[intentName] = variations.map(v => this.demask(v, dynamicContext));
        }

        const duration = Date.now() - startTime;
        console.log(`[TokenDemasker] ✅ Multi-mask de-masking complete. Tokens replaced: ${this.demaskCount}. Duration: ${duration}ms.`);
        
        return demaskedData;
    }
}

export default new TokenDemasker();
