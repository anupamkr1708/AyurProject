import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata

class SanskritSpellChecker:
    """
    High-performance Sanskrit spell checker with fuzzy matching
    Optimized for large vocabulary (100K+ terms)
    """
    
    def __init__(self, vocab_file: Optional[str] = None):
        self.vocab = set()
        self.vocab_lower = set()
        self.trigram_index = defaultdict(set)
        self.word_frequencies = {}
        self.cache = {}
        self.max_cache_size = 10000
        
        # Load vocabulary
        if vocab_file and Path(vocab_file).exists():
            self.load_vocabulary(vocab_file)
        else:
            self.initialize_default_vocabulary()
        
        # Build trigram index for fast fuzzy matching
        self._build_trigram_index()
    
    def initialize_default_vocabulary(self):
        """Initialize with comprehensive Ayurvedic Sanskrit vocabulary"""
        print("Initializing default Sanskrit vocabulary...")
        
        # Core Ayurvedic terms
        base_vocab = {
            # Doshas and sub-doshas
            'vāta', 'pitta', 'kapha', 'tridoṣa', 'doṣa',
            'prāṇa', 'udāna', 'samāna', 'vyāna', 'apāna',
            'pācaka', 'rañjaka', 'sādhaka', 'ālochaka', 'bhrājaka',
            'avalambaka', 'kledaka', 'bodhaka', 'tarpaka', 'śleṣmaka',
            
            # Dhatus
            'rasa', 'rakta', 'māṃsa', 'medas', 'asthi', 'majjā', 'śukra',
            
            # Malas
            'purīṣa', 'mūtra', 'sveda',
            
            # Agni
            'agni', 'jāṭharāgni', 'bhūtāgni', 'dhātvāgni',
            'tīkṣṇāgni', 'mandāgni', 'samāgni', 'viṣamāgni',
            
            # Gunas
            'guru', 'laghu', 'snigdha', 'rūkṣa', 'śīta', 'uṣṇa',
            'sthira', 'sara', 'mṛdu', 'kaṭhina', 'viśada', 'picchila',
            'ślakṣṇa', 'khara', 'sthūla', 'sūkṣma', 'sandra', 'drava',
            
            # Rasas
            'madhura', 'amla', 'lavaṇa', 'kaṭu', 'tikta', 'kaṣāya',
            
            # Vipaka
            'vipāka', 'madhura', 'amla', 'kaṭu',
            
            # Virya
            'vīrya', 'śīta', 'uṣṇa',
            
            # Prabhava
            'prabhāva',
            
            # Srotas
            'srotas', 'prāṇavaha', 'annavaha', 'udakavaha', 'rasavaha',
            'raktavaha', 'māṃsavaha', 'medovaha', 'asthivaha', 'majjāvaha',
            'śukravaha', 'mūtravaha', 'purīṣavaha', 'svedavaha',
            'stanyavaha', 'artavavaha', 'manovaha',
            
            # Diseases
            'roga', 'vyādhi', 'āmaya', 'vikāra',
            'jvara', 'kāsa', 'śvāsa', 'hikka', 'chardi', 'atīsāra',
            'grahaṇī', 'arśa', 'bhagandara', 'prameha', 'madhumeha',
            'kṣaya', 'rajayakṣmā', 'pāṇḍu', 'kāmalā', 'śotha',
            'gulma', 'udara', 'plīhā', 'yakṛt', 'hṛdroga',
            'ardhāvabhedaka', 'śiraḥśūla', 'netraroga',
            
            # Treatments
            'cikitsā', 'śamana', 'śodhana', 'laṅghana', 'bṛṃhaṇa',
            'snehana', 'svedana', 'vamana', 'virecana', 'vasti',
            'nasya', 'raktamokṣaṇa', 'pañcakarma',
            
            # Drugs and preparations
            'auṣadha', 'dravya', 'yoga', 'kalpa', 'rasāyana',
            'vājīkaraṇa', 'cūrṇa', 'kvātha', 'phāṇṭa', 'hima',
            'taila', 'ghṛta', 'arka', 'āsava', 'ariṣṭa', 'guṭikā',
            'vaṭī', 'modaka', 'avaleha', 'leha', 'pāka',
            
            # Anatomy
            'śarīra', 'aṅga', 'pratyaṅga', 'upāṅga', 'koṣṭha',
            'marma', 'nāḍī', 'sirā', 'dhamanī', 'snayu', 'kandara',
            'asthi', 'sandhi', 'śiras', 'hṛdaya', 'kloma', 'yakṛt',
            'plīhā', 'vṛkka', 'basti', 'guda', 'garbhāśaya',
            
            # Classical texts
            'samhitā', 'sūtra', 'śārīra', 'nidāna', 'cikitsā',
            'kalpa', 'uttara', 'indriya', 'tantra',
            'caraka', 'suśruta', 'vāgbhaṭa', 'aṣṭāṅga', 'hṛdaya',
            'saṃgraha', 'nighaṇṭu', 'bhāvaprakāśa', 'śārṅgadhara',
            
            # Time and seasons
            'kāla', 'ṛtu', 'vasanta', 'grīṣma', 'varṣā', 'śarat',
            'hemanta', 'śiśira', 'dinacarya', 'ṛtucarya',
            
            # Constitution
            'prakṛti', 'vikṛti', 'sama', 'sattva', 'rajas', 'tamas',
            
            # Diagnostic terms
            'nidāna', 'pūrvarūpa', 'rūpa', 'upśaya', 'samprapti',
            'darśana', 'sparśana', 'praśna', 'parīkṣā',
            
            # Common Sanskrit terms
            'yoga', 'prayoga', 'vidhi', 'doṣa', 'guṇa', 'karma',
            'nāma', 'rūpa', 'svabhāva', 'lakṣaṇa', 'vidhāna',
            'anupāna', 'pathya', 'apathya', 'mānā', 'pramāṇa',
        }
        
        # Add common variations and compounds
        expanded_vocab = set()
        for word in base_vocab:
            expanded_vocab.add(word)
            expanded_vocab.add(word.lower())
            # Remove diacritics version
            expanded_vocab.add(self._remove_diacritics(word))
            expanded_vocab.add(self._remove_diacritics(word).lower())
        
        self.vocab = expanded_vocab
        self.vocab_lower = {w.lower() for w in expanded_vocab}
        
        # Set default frequencies (can be updated with actual corpus frequencies)
        for word in self.vocab:
            self.word_frequencies[word] = 1
        
        print(f"✓ Initialized with {len(self.vocab)} Sanskrit terms")
    
    def load_vocabulary(self, vocab_file: str):
        """Load vocabulary from file"""
        vocab_path = Path(vocab_file)
        
        if vocab_path.suffix == '.txt':
            with open(vocab_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    word = parts[0]
                    freq = int(parts[1]) if len(parts) > 1 else 1
                    
                    self.vocab.add(word)
                    self.vocab_lower.add(word.lower())
                    self.word_frequencies[word] = freq
        
        elif vocab_path.suffix == '.json':
            with open(vocab_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vocab = set(data.get('vocab', []))
                self.vocab_lower = {w.lower() for w in self.vocab}
                self.word_frequencies = data.get('frequencies', {})
        
        print(f"✓ Loaded {len(self.vocab)} Sanskrit terms from {vocab_file}")
    
    def save_vocabulary(self, output_file: str):
        """Save vocabulary for future use"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': list(self.vocab),
            'frequencies': self.word_frequencies
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Vocabulary saved to {output_file}")
    
    def _remove_diacritics(self, text: str) -> str:
        """Remove diacritical marks from text"""
        # Normalize to NFD (decomposed form)
        nfd = unicodedata.normalize('NFD', text)
        # Filter out combining characters (diacritics)
        return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    def _build_trigram_index(self):
        """Build trigram index for fast candidate retrieval"""
        print("Building trigram index for fast fuzzy matching...")
        
        for word in self.vocab:
            trigrams = self._get_trigrams(word.lower())
            for trigram in trigrams:
                self.trigram_index[trigram].add(word)
        
        print(f"✓ Built index with {len(self.trigram_index)} trigrams")
    
    def _get_trigrams(self, word: str) -> set:
        """Generate trigrams for a word"""
        word = f"#{word}#"  # Add boundaries
        return {word[i:i+3] for i in range(len(word) - 2)}
    
    def _get_candidates_fast(self, word: str, max_candidates: int = 50) -> set:
        """Fast candidate retrieval using trigram index"""
        trigrams = self._get_trigrams(word.lower())
        
        # Count trigram matches for each vocabulary word
        candidate_scores = defaultdict(int)
        
        for trigram in trigrams:
            for candidate in self.trigram_index.get(trigram, []):
                candidate_scores[candidate] += 1
        
        # Sort by trigram match count and get top candidates
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {cand for cand, _ in sorted_candidates[:max_candidates]}
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = range(len(s1) + 1)
        
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(1 + min((
                        distances[i1],      # substitution
                        distances[i1 + 1],  # insertion
                        new_distances[-1]   # deletion
                    )))
            distances = new_distances
        
        return distances[-1]
    
    def _similarity_score(self, word1: str, word2: str) -> float:
        """Calculate similarity score (0-1) between two words"""
        # Normalize to lowercase
        w1 = word1.lower()
        w2 = word2.lower()
        
        # Exact match
        if w1 == w2:
            return 1.0
        
        # Check without diacritics
        w1_plain = self._remove_diacritics(w1)
        w2_plain = self._remove_diacritics(w2)
        
        if w1_plain == w2_plain:
            return 0.95
        
        # Use SequenceMatcher for ratio
        ratio = SequenceMatcher(None, w1, w2).ratio()
        
        # Adjust for length difference
        len_diff = abs(len(w1) - len(w2))
        len_penalty = len_diff / max(len(w1), len(w2))
        
        final_score = ratio * (1 - len_penalty * 0.3)
        
        return final_score
    
    def correct_word(self, word: str, threshold: float = 0.75, 
                    max_candidates: int = 10) -> Optional[str]:
        """
        Correct a single Sanskrit word
        
        Args:
            word: Word to correct
            threshold: Minimum similarity threshold (0-1)
            max_candidates: Maximum number of candidates to consider
            
        Returns:
            Corrected word or None if no good match found
        """
        # Check cache
        cache_key = (word, threshold)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Clean word
        word_clean = word.strip('.,;:!?"\'()[]{}')
        
        # Exact match
        if word_clean in self.vocab:
            return word_clean
        
        # Case-insensitive match
        if word_clean.lower() in self.vocab_lower:
            matches = [w for w in self.vocab if w.lower() == word_clean.lower()]
            if matches:
                result = matches[0]
                self._update_cache(cache_key, result)
                return result
        
        # Skip very short words
        if len(word_clean) < 3:
            return None
        
        # Get candidates using trigram index
        candidates = self._get_candidates_fast(word_clean, max_candidates=max_candidates)
        
        if not candidates:
            return None
        
        # Score candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._similarity_score(word_clean, candidate)
            if score >= threshold:
                freq = self.word_frequencies.get(candidate, 1)
                scored_candidates.append((candidate, score, freq))
        
        if not scored_candidates:
            return None
        
        # Sort by score, then by frequency
        scored_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Return best match
        best_match = scored_candidates[0][0]
        
        # Update cache
        self._update_cache(cache_key, best_match)
        
        return best_match
    
    def _update_cache(self, key, value):
        """Update cache with size limit"""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            items_to_remove = len(self.cache) // 4
            for _ in range(items_to_remove):
                self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = value
    
    def correct_text(self, text: str, threshold: float = 0.75) -> str:
        """
        Correct all Sanskrit words in text
        
        Args:
            text: Text to correct
            threshold: Minimum similarity threshold
            
        Returns:
            Corrected text
        """
        # Split into words, preserving punctuation
        words = re.findall(r'\b[\w\u0900-\u097F]+\b|[^\w\s]', text)
        
        corrected_words = []
        for word in words:
            if re.match(r'[^\w\s]', word):
                # Keep punctuation as is
                corrected_words.append(word)
            else:
                # Try to correct word
                corrected = self.correct_word(word, threshold=threshold)
                corrected_words.append(corrected if corrected else word)
        
        return ''.join(corrected_words)
    
    def batch_correct(self, texts: List[str], threshold: float = 0.75,
                     show_progress: bool = True) -> List[str]:
        """
        Correct multiple texts efficiently
        
        Args:
            texts: List of texts to correct
            threshold: Minimum similarity threshold
            show_progress: Show progress bar
            
        Returns:
            List of corrected texts
        """
        corrected = []
        
        for i, text in enumerate(texts):
            corrected_text = self.correct_text(text, threshold=threshold)
            corrected.append(corrected_text)
            
            if show_progress and (i + 1) % 100 == 0:
                print(f"Corrected {i + 1}/{len(texts)} texts...")
        
        return corrected
    
    def add_words(self, words: List[str], frequencies: Optional[Dict[str, int]] = None):
        """Add new words to vocabulary"""
        for word in words:
            self.vocab.add(word)
            self.vocab_lower.add(word.lower())
            
            if frequencies and word in frequencies:
                self.word_frequencies[word] = frequencies[word]
            else:
                self.word_frequencies[word] = 1
            
            # Update trigram index
            trigrams = self._get_trigrams(word.lower())
            for trigram in trigrams:
                self.trigram_index[trigram].add(word)
        
        print(f"✓ Added {len(words)} new words to vocabulary")
    
    def get_vocabulary_stats(self) -> Dict:
        """Get vocabulary statistics"""
        return {
            'total_words': len(self.vocab),
            'unique_lowercase': len(self.vocab_lower),
            'trigrams': len(self.trigram_index),
            'cache_size': len(self.cache),
            'avg_frequency': np.mean(list(self.word_frequencies.values()))
        }


# Create singleton instance
_sanskrit_checker = None

def get_sanskrit_checker(vocab_file: Optional[str] = None) -> SanskritSpellChecker:
    """Get or create Sanskrit spell checker instance"""
    global _sanskrit_checker
    
    if _sanskrit_checker is None:
        _sanskrit_checker = SanskritSpellChecker(vocab_file)
    
    return _sanskrit_checker