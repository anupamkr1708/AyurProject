

import re
import unicodedata
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict

class IntegratedTextCleaner:
    """Integrated cleaner using your CharCNN classifier and Sanskrit dataset"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Load Sanskrit vocabulary from your dataset
        self.sanskrit_vocab = self._load_sanskrit_vocab()
        self.ayurveda_entities = self._load_ayurveda_entities()
        
        # Load CharCNN classifier
        self.classifier = self._load_classifier()
        
        # Compile patterns
        self.ocr_patterns = self._compile_ocr_patterns()
        
        self.stats = {
            'total_processed': 0,
            'ocr_fixes': 0,
            'sanskrit_preserved': 0,
            'junk_removed': 0
        }
    
    def _default_config(self) -> Dict:
        return {
            'sanskrit_vocab_path': 'sanskrit_terms_collection/final_language_dataset.csv',
            'classifier_path': 'artifacts/best_charcnn_model.keras',
            'quality_threshold': 50.0,
            'enable_ocr_fix': True,
            'preserve_entities': True
        }
    
    def _load_sanskrit_vocab(self) -> set:
        """Load Sanskrit terms from your final_language_dataset.csv"""
        vocab_path = Path(self.config['sanskrit_vocab_path'])
        if not vocab_path.exists():
            print(f" Sanskrit vocab not found: {vocab_path}")
            return set()
        
        df = pd.read_csv(vocab_path)
        # Filter Sanskrit terms (label=1)
        sanskrit_terms = df[df['label'] == 1]['ASCII'].str.lower().tolist()
        print(f"✓ Loaded {len(sanskrit_terms)} Sanskrit terms")
        return set(sanskrit_terms)
    
    def _load_ayurveda_entities(self) -> set:
        """Load protected Ayurvedic entities"""
        import json
        entities_path = Path("resources/ayurveda_terms.json")
        if entities_path.exists():
            with open(entities_path, 'r', encoding='utf-8') as f:
                terms = json.load(f)
                return set(k.lower() for k in terms.keys())
        return set()
    
    def _load_classifier(self):
        """Load your CharCNN model"""
        import tensorflow as tf
        model_path = Path(self.config['classifier_path'])
        
        if model_path.exists():
            try:
                model = tf.keras.models.load_model(str(model_path))
                print(f"✓ Loaded CharCNN classifier from {model_path}")
                return model
            except Exception as e:
                print(f" Failed to load classifier: {e}")
        return None
    
    def _compile_ocr_patterns(self) -> Dict:
        """Compile OCR error patterns"""
        return {
            'char_spacing': re.compile(r'\b([a-zA-Zāīūṛṝḷḹēōṃḥṅñṭḍṇśṣ])\s+(?=[a-zA-Zāīūṛṝḷḹēōṃḥṅñṭḍṇśṣ]\s)'),
            'devanagari': re.compile(r'[\u0900-\u097F]+'),
            'page_artifacts': re.compile(r'\b\d+\s*\|\s*Page\b|\[Page\s+\d+\]|^[\d\s\|]+$', re.MULTILINE),
            'multi_space': re.compile(r'\s+'),
            'broken_words': re.compile(r'(\w+)-\s*\n\s*(\w+)'),
        }
    
    def clean_text(self, text: str, metadata: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Main cleaning pipeline"""
        if not text or len(text.strip()) < 10:
            return "", {'skipped': 'too_short'}
        
        original_text = text
        cleaning_log = {'steps': []}
        
        # Step 1: Fix OCR artifacts
        if self.config['enable_ocr_fix']:
            text = self._fix_ocr_artifacts(text)
            cleaning_log['steps'].append('ocr_fixed')
        
        # Step 2: Remove Devanagari script
        text = self.ocr_patterns['devanagari'].sub(' ', text)
        
        # Step 3: Normalize Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Step 4: Remove page artifacts
        text = self.ocr_patterns['page_artifacts'].sub('', text)
        
        # Step 5: Fix broken words
        text = self.ocr_patterns['broken_words'].sub(r'\1\2', text)
        
        # Step 6: Preserve Sanskrit/Ayurveda terms
        if self.config['preserve_entities']:
            text = self._preserve_sanskrit_terms(text)
        
        # Step 7: Remove repeated lines if provided
        if metadata and 'repeated_lines' in metadata:
            text = self._remove_repeated_lines(text, metadata['repeated_lines'])
        
        # Step 8: Clean spacing
        text = self.ocr_patterns['multi_space'].sub(' ', text)
        text = text.strip()
        
        self.stats['total_processed'] += 1
        
        return text, cleaning_log
    
    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR errors"""
        # Fix spaced letters: "A y u r v e d a" → "Ayurveda"
        text = self.ocr_patterns['char_spacing'].sub(
            lambda m: m.group(0).replace(' ', ''),
            text
        )
        
        # Fix ligatures
        ligatures = {
            "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"
        }
        for old, new in ligatures.items():
            text = text.replace(old, new)
        
        self.stats['ocr_fixes'] += 1
        return text
    
    def _preserve_sanskrit_terms(self, text: str) -> str:
        """Preserve Sanskrit terms from corruption"""
        words = text.split()
        preserved = []
        
        for word in words:
            word_lower = word.lower().strip('.,;:!?()')
            
            # Check if it's a known Sanskrit term or Ayurveda entity
            if (word_lower in self.sanskrit_vocab or 
                word_lower in self.ayurveda_entities):
                preserved.append(word)
                self.stats['sanskrit_preserved'] += 1
            else:
                preserved.append(word)
        
        return ' '.join(preserved)
    
    def _remove_repeated_lines(self, text: str, repeated_lines: set) -> str:
        """Remove headers/footers while preserving Sanskrit terms"""
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Keep if it's not a repeated line OR contains Sanskrit/Ayurveda terms
            if (line_lower not in repeated_lines or
                any(term in line_lower for term in self.ayurveda_entities)):
                cleaned.append(line)
        
        return '\n'.join(cleaned)
    
    def get_stats(self) -> Dict:
        """Return cleaning statistics"""
        return self.stats