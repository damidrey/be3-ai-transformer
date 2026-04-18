import { ALIAS_MAP } from '../config/aliasMap.js';

/**
 * Alias Service
 * 
 * Manages semantic normalization for the transformer service.
 * Map common ambiguous terms to high-precision alternatives.
 */
class AliasService {
    constructor() {
        // Strict word-boundary mapping
        this.ALIAS_MAP = ALIAS_MAP;

        // Cache regexes for performance
        this.regexes = {};
        for (const alias of Object.keys(this.ALIAS_MAP)) {
            this.regexes[alias] = new RegExp(`\\b${alias}\\b`, 'gi');
        }
    }

    /**
     * Normalize text using the alias map.
     * Uses word boundaries to ensure we don't accidentally modify 
     * compound words like "headphones" or "megaphone".
     * 
     * @param {string} text 
     * @returns {string}
     */
    normalize(text) {
        if (!text || typeof text !== 'string') return text;

        let normalized = text;
        for (const [alias, target] of Object.entries(this.ALIAS_MAP)) {
            normalized = normalized.replace(this.regexes[alias], target);
        }

        if (normalized !== text) {
            console.log(`[Alias] Normalized: "${text}" -> "${normalized}"`);
        }

        return normalized;
    }
}

export default new AliasService();
