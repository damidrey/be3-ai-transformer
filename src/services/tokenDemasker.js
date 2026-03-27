import { PRODUCT_WORDS } from '../utils/productWords.js';
import { DATE_WORDS } from '../utils/dateWords.js';

class TokenDemasker {
    constructor() {
        this.demaskCount = 0;
        this.activeMasks = ['[product]', '[vendor]', '[category]', '[attributes]', '[date]'];
    }

    /**
     * Replaces masked tokens with their name (stripping brackets).
     * e.g. [product] -> product
     * @param {string} text 
     * @returns {string}
     */
    demask(text) {
        if (!text || typeof text !== 'string') return text;

        let demaskedText = text;

        for (const mask of this.activeMasks) {
            // Escape brackets for the regex
            const pattern = mask.replace(/\[/g, '\\[').replace(/\]/g, '\\]');
            const replacement = mask.replace(/\[|\]/g, ''); // [product] -> product
            
            const regex = new RegExp(pattern, 'g');
            if (regex.test(demaskedText)) {
                demaskedText = demaskedText.replace(regex, () => {
                    this.demaskCount++;
                    return replacement;
                });
            }
        }

        return demaskedText;
    }

    /**
     * Processes a full intent data object by stripping brackets from all masks.
     * @param {Object} intentData { intentName: [variations] }
     * @returns {Object}
     */
    demaskVariations(intentData) {
        console.log('[TokenDemasker] 🛠️  Applying simple structural de-masking ([mask] -> mask)...');
        this.demaskCount = 0;
        const startTime = Date.now();

        const demaskedData = {};

        for (const [intentName, variations] of Object.entries(intentData)) {
            demaskedData[intentName] = variations.map(v => this.demask(v));
        }

        const duration = Date.now() - startTime;
        console.log(`[TokenDemasker] ✅ Stripping complete. Tokens modified: ${this.demaskCount}. Duration: ${duration}ms.`);
        
        return demaskedData;
    }
}

export default new TokenDemasker();
