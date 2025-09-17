import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

CONFIG_PATH = os.path.join("documents", "settings.json")
DEFAULT_CONFIG = {
    "OLLAMA_HOST": "http://localhost:11434",
    "LLM_MODEL": "llama3.1:8b",
    "EMBEDDING_MODEL": "nomic-embed-text:latest",
    "CHUNK_SIZE": 200,
    "OVERLAP_SIZE": 20,
    "TOP_K_RESULTS": 5,
    "SIMILARITY_THRESHOLD": 0.25,
    "SYSTEM_PROMPT": "You are an AI assistant. Answer questions only using the provided documents. Be precise. If the documents don't contain the information, say clearly that you cannot answer due to lack of information.",
    "DOCUMENTS_DIR": "documents",
    "STORAGE_FILE": "documents/chunk_database.pkl",
    "REQUEST_TIMEOUT": 300.0,
    "TEMPERATURE": 0.0,
    "SEARCH_MODE": "natural"
}

def load_config():
    if not os.path.exists("documents"):
        os.makedirs("documents")
    if not os.path.exists(CONFIG_PATH):
        print("Configuration file not found. Creating a default one.")
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config file: {e}. Using default settings.")
        return DEFAULT_CONFIG

def save_config(config):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration saved successfully.")
    except Exception as e:
        print(f"Error saving config file: {e}")

class OllamaRAG:
    def __init__(self, config):
        self.config = config
        self.fixed_chunks = []
        self.natural_chunks = []
        self.fixed_embeddings = np.array([])
        self.natural_embeddings = np.array([])
        self.storage_file = self.config['STORAGE_FILE']
        
        if not os.path.exists(self.config['DOCUMENTS_DIR']):
            os.makedirs(self.config['DOCUMENTS_DIR'])
    
    def load_documents(self):
        dir_path = self.config['DOCUMENTS_DIR']
        file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.txt')]
        
        if not file_paths:
            print(f"Warning: No .txt files found in the '{dir_path}' directory.")
            return {}

        documents = {}
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents[file_path] = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        return documents
    
    def create_fixed_chunks(self, documents):
        all_chunks = []
        chunk_size = self.config['CHUNK_SIZE']
        overlap_size = self.config['OVERLAP_SIZE']
        step = max(1, chunk_size - overlap_size)
        
        for file_path, text in documents.items():
            words = text.split()
            for i in range(0, len(words), step):
                chunk_text = " ".join(words[i:i + chunk_size])
                if chunk_text.strip():
                    all_chunks.append({
                        'text': chunk_text,
                        'file': os.path.basename(file_path),
                        'chunk_id': len(all_chunks),
                        'type': 'fixed'
                    })
        return all_chunks
    
    def create_natural_chunks(self, documents):
        all_chunks = []
        for file_path, text in documents.items():
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for line in lines:
                all_chunks.append({
                    'text': line,
                    'file': os.path.basename(file_path),
                    'chunk_id': len(all_chunks),
                    'type': 'natural'
                })
        return all_chunks
    
    def get_embedding(self, text):
        try:
            response = requests.post(
                f"{self.config['OLLAMA_HOST']}/api/embeddings",
                json={"model": self.config['EMBEDDING_MODEL'], "prompt": text},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"\nError getting embedding: {e}")
            return None

    def _process_embeddings_parallel(self, chunks):
        embeddings = [None] * len(chunks)
        
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(self.get_embedding, chunk['text']): i for i, chunk in enumerate(chunks)}
            
            for future in tqdm(as_completed(future_to_index), total=len(chunks), desc="Processing Embeddings"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        embeddings[index] = result
                except Exception as e:
                    print(f"Chunk {index} failed: {e}")

        successful_embeddings = [emb for emb in embeddings if emb is not None]
        successful_chunks = [chunk for i, chunk in enumerate(chunks) if embeddings[i] is not None]

        return successful_chunks, np.array(successful_embeddings)

    def build_index(self):
        documents = self.load_documents()
        if not documents:
            return

        print("Creating fixed and natural chunks...")
        fixed_chunks_initial = self.create_fixed_chunks(documents)
        natural_chunks_initial = self.create_natural_chunks(documents)
        
        print(f"Building embeddings for {len(fixed_chunks_initial)} fixed chunks...")
        self.fixed_chunks, self.fixed_embeddings = self._process_embeddings_parallel(fixed_chunks_initial)
        
        print(f"Building embeddings for {len(natural_chunks_initial)} natural chunks...")
        self.natural_chunks, self.natural_embeddings = self._process_embeddings_parallel(natural_chunks_initial)
        
        self.save_index()
        print("Index built and saved successfully.")

    def save_index(self):
        data = {
            'fixed_chunks': self.fixed_chunks,
            'natural_chunks': self.natural_chunks,
            'fixed_embeddings': self.fixed_embeddings.tolist(),
            'natural_embeddings': self.natural_embeddings.tolist()
        }
        with open(self.storage_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved with {len(self.fixed_chunks)} fixed chunks and {len(self.natural_chunks)} natural chunks.")
    
    def load_index(self):
        if not os.path.exists(self.storage_file):
            return False
        try:
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
            self.fixed_chunks = data['fixed_chunks']
            self.natural_chunks = data['natural_chunks']
            self.fixed_embeddings = np.array(data['fixed_embeddings'])
            self.natural_embeddings = np.array(data['natural_embeddings'])
            print(f"Index loaded with {len(self.fixed_chunks)} fixed chunks and {len(self.natural_chunks)} natural chunks.")
            return True
        except (pickle.UnpicklingError, KeyError, TypeError) as e:
            print(f"Error loading index file (it might be corrupted or outdated): {e}")
            return False

    def search_similar(self, query, mode):
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            print("Could not get embedding for query.")
            return []
        
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        all_scored_chunks = []
        threshold = self.config['SIMILARITY_THRESHOLD']

        def find_similarities(chunks, embeddings, chunk_type):
            if embeddings.shape[0] > 0:
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                for i, score in enumerate(similarities):
                    if score >= threshold:
                        all_scored_chunks.append({
                            'score': score,
                            'chunk': chunks[i]['text'],
                            'file': chunks[i]['file'],
                            'chunk_id': chunks[i]['chunk_id'],
                            'type': chunk_type
                        })

        if mode == 'fixed':
            find_similarities(self.fixed_chunks, self.fixed_embeddings, 'fixed')
        elif mode == 'natural':
            find_similarities(self.natural_chunks, self.natural_embeddings, 'natural')
        
        unique_chunks = {}
        for item in all_scored_chunks:
            key = (item['chunk'], item['file'], item['type'])
            if key not in unique_chunks or unique_chunks[key]['score'] < item['score']:
                unique_chunks[key] = item
        
        sorted_unique_chunks = sorted(unique_chunks.values(), key=lambda x: x['score'], reverse=True)

        top_k = self.config['TOP_K_RESULTS']
        top_results = sorted_unique_chunks[:top_k]
        
        print(f"\nTop {len(top_results)} most relevant chunks found:")
        for i, result in enumerate(top_results):
            preview = (result['chunk'][:80] + "...") if len(result['chunk']) > 80 else result['chunk']
            print(f"{i+1}. [{result['score']:.3f}] ({result['type']}) (File: {result['file']}, ID: {result['chunk_id']}) {preview}")

        return [f"File: {r['file']}\nContent: {r['chunk']}" for r in top_results]
    
    def generate_response(self, query, context_chunks):
        if not context_chunks:
            return "No relevant info found in the provided documents."
            
        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"Using the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        try:
            response = requests.post(
                f"{self.config['OLLAMA_HOST']}/api/generate", 
                json={
                    "model": self.config['LLM_MODEL'],
                    "prompt": prompt,
                    "system": self.config['SYSTEM_PROMPT'],
                    "stream": True,
                    "options": {
                        "temperature": self.config['TEMPERATURE']
                    }
                },
                timeout=self.config['REQUEST_TIMEOUT'],
                stream=True
            )
            response.raise_for_status()
            
            full_response = ""
            print("\nAnswer: ", end="", flush=True)
            
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8'))
                        if 'response' in json_response:
                            chunk = json_response['response']
                            print(chunk, end="", flush=True)
                            full_response += chunk
                        if json_response.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
            
            print()
            return full_response
            
        except requests.exceptions.Timeout:
            return f"The request timed out after {self.config['REQUEST_TIMEOUT']} seconds. The server may be busy."
        except requests.exceptions.RequestException as e:
            return f"An error occurred while contacting the Ollama server: {e}"
    
    def query(self, question, mode):
        relevant_chunks = self.search_similar(question, mode)
        response = self.generate_response(question, relevant_chunks)
        return response

class CommandLineInterface:
    def __init__(self):
        self.config = load_config()
        self.rag = OllamaRAG(self.config)
        self.temp_config = self.config.copy()
        self.commands = {
            "quit": self._handle_quit,
            "exit": self._handle_quit,
            "help": self._handle_help,
            "settings": self._handle_settings,
            "temp_settings": self._handle_temp_settings,
            "view": self._handle_view,
            "rebuild": self._handle_rebuild,
        }

    def run(self):
        if self.rag.load_index():
            print("Loaded existing index.")
            rebuild = input("Rebuild index? (y/N): ").lower().strip()
            if rebuild == 'y':
                self.rag.build_index()
        else:
            print("No index found. Building index...")
            self.rag.build_index()
        
        print("\nWelcome to the RAG CLI. Type 'help' for commands.")
        
        while True:
            prompt = f"\n[Mode: {self.temp_config['SEARCH_MODE']} | Temp: {self.temp_config['TEMPERATURE']}] Enter your question or command: "
            user_input = input(prompt).strip()
            if not user_input:
                continue

            parts = user_input.split()
            command_key = parts[0].lower()

            if len(parts) == 1 and command_key in self.commands:
                command_handler = self.commands[command_key]
                command_handler([])
            else:
                self._handle_query(user_input)
    
    def _handle_query(self, query):
        print("\nThinking...")
        old_config = self.rag.config.copy()
        self.rag.config.update(self.temp_config)
        
        response = self.rag.query(query, mode=self.temp_config['SEARCH_MODE'])
        
        self.rag.config = old_config

    def _handle_quit(self, args):
        print("Exiting. Goodbye!")
        exit()

    def _handle_help(self, args):
        print("\n--- RAG CLI Commands ---")
        print("   quit/exit:       Exit the program.")
        print("   help:            Show this help message.")
        print("   rebuild:         Rebuild the document index from the source files.")
        print("   settings:        View and permanently change configuration.")
        print("   temp_settings:   View and temporarily change settings for this session.")
        print("   view:            View chunks from the vector databases.")
        print("\nAnything else will be treated as a question.")

    def _handle_rebuild(self, args):
        print("Rebuilding index...")
        self.rag.build_index()

    def _handle_settings(self, args):
        print("\n--- Current Permanent Settings ---")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print("------------------------")
        
        key_to_change = input("Which setting do you want to change? (Enter to cancel): ").strip()
        if not key_to_change:
            return
        
        if key_to_change not in self.config:
            print("Invalid setting name.")
            return

        new_value_str = input(f"Enter the new value for '{key_to_change}': ").strip()
        current_value = self.config[key_to_change]
        
        try:
            if isinstance(current_value, bool):
                new_value = new_value_str.lower() in ['true', '1', 'y']
            elif isinstance(current_value, int):
                new_value = int(new_value_str)
            elif isinstance(current_value, float):
                new_value = float(new_value_str)
            else:
                new_value = new_value_str
            
            self.config[key_to_change] = new_value
            self.temp_config[key_to_change] = new_value
            save_config(self.config)
            self.rag.config[key_to_change] = new_value
            print("Settings updated permanently. A 'rebuild' may be required for some changes to take effect.")
        except ValueError:
            print("Invalid value type. Could not update setting.")

    def _handle_temp_settings(self, args):
        print("\n--- Current Temporary Settings ---")
        for key, value in self.temp_config.items():
            permanent_marker = " (same as permanent)" if value == self.config[key] else " (temporary override)"
            print(f"{key}: {value}{permanent_marker}")
        print("------------------------")
        
        key_to_change = input("Enter the setting to change temporarily (or press Enter to cancel): ").strip()
        if not key_to_change:
            return
        
        if key_to_change not in self.temp_config:
            print("Invalid setting name.")
            return

        new_value_str = input(f"Enter the new temporary value for '{key_to_change}': ").strip()
        current_value = self.temp_config[key_to_change]
        
        try:
            if isinstance(current_value, bool):
                new_value = new_value_str.lower() in ['true', '1', 'y']
            elif isinstance(current_value, int):
                new_value = int(new_value_str)
            elif isinstance(current_value, float):
                new_value = float(new_value_str)
            else:
                new_value = new_value_str
            
            self.temp_config[key_to_change] = new_value
            print(f"Temporary setting updated for this session. Use 'settings' to make it permanent.")
        except ValueError:
            print("Invalid value type. Could not update setting.")

    def _handle_view(self, args):
        db_choice = input("Which database to view? (fixed/natural): ").lower().strip()
        if db_choice not in ['fixed', 'natural']:
            print("Invalid choice.")
            return
        
        chunks = self.rag.fixed_chunks if db_choice == 'fixed' else self.rag.natural_chunks
        embeddings = self.rag.fixed_embeddings if db_choice == 'fixed' else self.rag.natural_embeddings
        
        if not chunks:
            print(f"The '{db_choice}' database is empty.")
            return
            
        try:
           count = int(input(f"How many chunks to show from the top? (total {len(chunks)}): ").strip())
        except ValueError:
           print("Invalid number. Showing first 5.")
           count = 5
           
        for i, chunk_info in enumerate(chunks[:count]):
           print(f"\n--- Chunk {i+1} ---")
           print(f"File: {chunk_info['file']}, Chunk ID: {chunk_info['chunk_id']}")
           print(f"Text: {chunk_info['text']}")

if __name__ == "__main__":
    cli = CommandLineInterface()
    cli.run()