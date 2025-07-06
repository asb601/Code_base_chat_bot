import os
import json
import mmap
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Generator
from concurrent.futures import ThreadPoolExecutor
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language
from pygments.lexers import guess_lexer_for_filename
from pygments.util import ClassNotFound
import tokenize
from io import BytesIO
from collections import defaultdict
import datetime  
import streamlit as st
# ======================
# CORE DATA MODELS
# ======================

class ChunkType(Enum):
    FUNCTION = auto()
    CLASS = auto()
    RULE_SET = auto()
    MEDIA_QUERY = auto()
    IMPORT = auto()
    GENERIC = auto()
data_path = Path(__file__).parent / "data"
@dataclass
class CodeChunk:
    file_path: Path
    language: str
    chunk_type: ChunkType
    content: str
    start_line: int
    end_line: int
    metadata: Dict = None

    def to_dict(self):
        base = {
            'file': str(self.file_path),
            'language': self.language,
            'type': self.chunk_type.name,
            'content': self.content,
            'start_line': self.start_line,
            'end_line': self.end_line
        }
        if self.metadata:
            base.update(self.metadata)
        return base

# ======================
# CONFIGURATION
# ======================

class ParserConfig:
    def __init__(self):
        self.max_chunk_size = 1500
        self.min_chunk_size = 50
        self.exclude_patterns = ['.*', 'node_modules/*', 'venv/*']
        self.include_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.h', '.css', '.html']
        self.language_parsers = {
            'python': {'enabled': True, 'features': ['functions', 'classes', 'decorators']},
            'javascript': {'enabled': True, 'features': ['functions', 'classes']},
            'css': {'enabled': True, 'features': ['rulesets', 'media_queries']}
        }

# ======================
# LANGUAGE PARSER REGISTRY
# ======================

class LanguageParserRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.parsers = {}
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        if self.initialized:
            return
            
        config = ParserConfig()
        for lang, settings in config.language_parsers.items():
            if settings['enabled']:
                try:
                    language = get_language(lang)
                    if language:
                        parser = Parser()
                        parser.set_language(language)
                        self.parsers[lang] = {
                            'parser': parser,
                            'features': settings['features']
                        }
                except Exception as e:
                    logging.warning(f"Failed to initialize {lang} parser: {str(e)}")
        
        self.initialized = True
    
    def get_parser(self, language):
        return self.parsers.get(language.lower(), {}).get('parser')

    def get_features(self, language):
        return self.parsers.get(language.lower(), {}).get('features', [])

# ======================
# CORE PROCESSING
# ======================

class CodebaseProcessor:
    def __init__(self, config=None):
        self.config = config or ParserConfig()
        self.parser_registry = LanguageParserRegistry()
        self.parser_registry.initialize()
        self._setup_logging()
        self.total_token_count = 0  # Track total tokens across all chunks
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('code_parser.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def should_process_file(self, filepath):
        """Config-driven file inclusion/exclusion"""
        if filepath.name.startswith('.'):
            return False
            
        if not any(filepath.suffix.lower() == ext for ext in self.config.include_extensions):
            return False
            
        for pattern in self.config.exclude_patterns:
            if filepath.match(pattern):
                return False
                
        return True
    
    def detect_language(self, filepath):
        """Enhanced language detection with confidence scoring"""
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(8192).decode('utf-8', errors='ignore')
            
            # First try: Pygments with confidence threshold
            try:
                lexer = guess_lexer_for_filename(filepath.name, sample)
                if lexer.analyse_text(sample) > 0.2:  # Confidence threshold
                    return lexer.name
            except ClassNotFound:
                pass
            
            # Second try: Content heuristics with pattern matching
            content_indicators = [
                (r'@media\s', 'CSS'),
                (r'def\s+\w+\s*\(', 'Python'),
                (r'function\s+\w+\s*\(', 'JavaScript'),
                (r'class\s+\w+', 'Java'),
                (r'#include\s+[<"]', 'C++')
            ]
            
            for pattern, lang in content_indicators:
                if re.search(pattern, sample):
                    return lang
                    
            # Fallback to extension mapping
            ext_map = {
                '.py': 'Python',
                '.js': 'JavaScript',
                '.ts': 'TypeScript',
                '.java': 'Java',
                '.cpp': 'C++',
                '.h': 'C++',
                '.css': 'CSS',
                '.html': 'HTML'
            }
            return ext_map.get(filepath.suffix.lower(), 'Unknown')
            
        except Exception as e:
            self.logger.error(f"Language detection failed for {filepath}: {str(e)}")
            return 'Unknown'
    
    def _extract_tokens(self, content, language):
        """Count ALL token occurrences (including duplicates)"""
        token_count = 0
        
        try:
            if language.lower() == 'python':
                # Count ALL Python tokens
                for tok in tokenize.tokenize(BytesIO(content.encode('utf-8')).readline):
                    if tok.type == tokenize.NAME:  # Count all identifiers
                        token_count += 1
            else:
                # Count ALL tokens for other languages
                token_count = len(re.findall(r'[a-zA-Z_]\w*', content))
            
            # Update global count
            self.total_token_count += token_count
            
            return token_count
            
        except Exception as e:
            self.logger.warning(f"Token counting failed: {str(e)}")
            return 0
    
    def parse_file(self, filepath):
        """AST-aware parsing with language-specific rules"""
        language = self.detect_language(filepath)
        if language == 'Unknown':
            return []
            
        parser = self.parser_registry.get_parser(language)
        features = self.parser_registry.get_features(language)
        
        try:
            with open(filepath, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as code_bytes:
                    if parser:
                        return self._parse_with_ast(filepath, language, parser, code_bytes, features)
                    return self._parse_with_fallback(filepath, language, code_bytes)
        except Exception as e:
            self.logger.error(f"Failed to parse {filepath}: {str(e)}")
            return []
    
    def _parse_with_ast(self, filepath, language, parser, code_bytes, features):
        """Advanced AST-based parsing"""
        tree = parser.parse(code_bytes)
        chunks = []
        
        # Language-specific parsing rules
        if language.lower() == 'python':
            if 'functions' in features or 'classes' in features:
                query = []
                if 'functions' in features:
                    query.append("(function_definition) @function")
                if 'classes' in features:
                    query.append("(class_definition) @class")
                if 'decorators' in features:
                    query.append("(decorated_definition) @decorated")
                
                for capture in tree.query("\n".join(query)).captures(tree.root_node):
                    chunk_type = ChunkType.FUNCTION if 'function' in capture[1] else ChunkType.CLASS
                    chunks.append(self._create_chunk(
                        filepath, language, chunk_type,
                        code_bytes, capture[0]
                    ))
        
        elif language.lower() == 'css':
            query = []
            if 'rulesets' in features:
                query.append("(rule_set) @rule")
            if 'media_queries' in features:
                query.append("(media_statement) @media")
            
            for capture in tree.query("\n".join(query)).captures(tree.root_node):
                chunk_type = ChunkType.RULE_SET if 'rule' in capture[1] else ChunkType.MEDIA_QUERY
                chunks.append(self._create_chunk(
                    filepath, language, chunk_type,
                    code_bytes, capture[0]
                ))
        
        return chunks
    
    def _create_chunk(self, filepath, language, chunk_type, code_bytes, node):
        """Create a standardized chunk from AST node"""
        content = code_bytes[node.start_byte:node.end_byte].decode()
        token_count = self._extract_tokens(content, language)
        
        return CodeChunk(
            filepath, language, chunk_type,
            content, node.start_point[0], node.end_point[0],
            {
                'byte_range': (node.start_byte, node.end_byte),
                'token_count': token_count
            }
        )
    
    def _parse_with_fallback(self, filepath, language, code_bytes):
        """Fallback parsing with token counting"""
        content = code_bytes.read().decode('utf-8', errors='ignore')
        token_count = self._extract_tokens(content, language)
        
        return [CodeChunk(
            filepath, language, ChunkType.GENERIC,
            content[i:i+self.config.max_chunk_size],
            0, 0,
            {
                'fallback': True,
                'token_count': token_count
            }
        ) for i in range(0, len(content), self.config.max_chunk_size)]
  
    def process_repository(self, repo_path, incremental=False):
        """Main processing method with optional incremental updates"""
        repo_path = Path(repo_path)
        self.total_token_count = 0  # Reset counter
        
        if not incremental:
            chunks = self._full_parse(repo_path)
        else:
            chunks = self._incremental_parse(repo_path)
        
        # Write final count to file
        with open("total_tokens.txt", 'w') as f:
            f.write(str(self.total_token_count))
        
        # Prepare chunks with proper string escaping
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            # Ensure content is properly escaped
            chunk_dict['content'] = chunk_dict['content'].encode('unicode_escape').decode('utf-8')
            chunk_dicts.append(chunk_dict)
        
        # Create output directory if it doesn't exist
        data_path.mkdir(exist_ok=True)
        output_path = data_path / "code_metadata.json"
        
        try:
            # Validate JSON before writing
            json.dumps(chunk_dicts)  # Test serialization
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({
                    "metadata": {
                        "repo": str(repo_path),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "parser_version": "1.0",
                        "total_chunks": len(chunks),
                        "total_tokens": self.total_token_count,
                        "file_types": {
                            ext: sum(1 for c in chunks if str(c.file_path).endswith(ext))
                            for ext in self.config.include_extensions
                        }
                    },
                    "chunks": chunk_dicts
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(chunks)} chunks to {output_path}")
            st.write("Metadata saved to code_metadata.json")
        except Exception as e:
            self.logger.error(f"JSON validation failed: {str(e)}")
            st.error(f"Failed to save metadata: {str(e)}")
            raise
        
        return chunks

            
       
    
    def _full_parse(self, repo_path):
        """Process entire repository"""
        with ThreadPoolExecutor() as executor:
            futures = []
            for filepath in self._walk_repository(repo_path):
                futures.append(executor.submit(self.parse_file, filepath))
            
            chunks = []
            for future in futures:
                chunks.extend(future.result())
            return chunks
    
    def _incremental_parse(self, repo_path):
        """Process only changed files (stub for implementation)"""
        # Would integrate with git diff in real implementation
        return self._full_parse(repo_path)
    
    def _walk_repository(self, repo_path):
        """Config-driven repository traversal"""
        for root, dirs, files in os.walk(repo_path):
            # Apply exclude patterns to directories
            dirs[:] = [d for d in dirs if not any(
                Path(root, d).match(pattern) for pattern in self.config.exclude_patterns
            )]
            
            for file in files:
                filepath = Path(root, file)
                if self.should_process_file(filepath):
                    yield filepath

# ======================
# STREAMLIT INTEGRATION
# ======================


def parse_codebase(repo_path):
    """Streamlit-compatible interface"""
    processor = CodebaseProcessor()
    chunks = processor.process_repository(repo_path)
    return [chunk.to_dict() for chunk in chunks]