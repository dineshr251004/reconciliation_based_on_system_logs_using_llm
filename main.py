import os
import re
import json
import chromadb
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import jellyfish 

class PaymentLogEntry(BaseModel):
    timestamp: str
    correlation_id: str
    application: str
    process_name: str
    log_level: str
    message: str
    details: str
    user_id: str = None
    session_id: str = None
    request_id: str = None
    host: str = None


class ProcessMatcher:
    """
    Enhanced process matching system that combines multiple similarity metrics
    for better matching accuracy without hardcoding process names.
    """
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = self.preprocess_text(text).split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    def calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate lexical similarity using multiple string matching techniques"""
        text1_clean = self.preprocess_text(text1)
        text2_clean = self.preprocess_text(text2)
        
        # 1. Sequence Matcher (overall similarity)
        seq_sim = SequenceMatcher(None, text1_clean, text2_clean).ratio()
        
        # 2. Jaro-Winkler similarity (good for typos and variations)
        jaro_sim = jellyfish.jaro_winkler_similarity(text1_clean, text2_clean)
        
        # 3. Levenshtein distance normalized
        max_len = max(len(text1_clean), len(text2_clean))
        if max_len == 0:
            leven_sim = 1.0
        else:
            leven_sim = 1 - (jellyfish.levenshtein_distance(text1_clean, text2_clean) / max_len)
        
        # 4. Keyword overlap
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))
        
        if not keywords1 and not keywords2:
            keyword_sim = 1.0
        elif not keywords1 or not keywords2:
            keyword_sim = 0.0
        else:
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            keyword_sim = intersection / union if union > 0 else 0.0
        
        combined_score = (
            seq_sim * 0.3 +
            jaro_sim * 0.3 +
            leven_sim * 0.2 +
            keyword_sim * 0.2
        )
        
        return combined_score
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            embedding1 = self.embedding_model.encode([self.preprocess_text(text1)])
            embedding2 = self.embedding_model.encode([self.preprocess_text(text2)])
            return cosine_similarity(embedding1, embedding2)[0][0]
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def find_best_matches(self, user_input: str, available_processes: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Find best matching processes using combined similarity metrics
        
        Returns:
            List of tuples (process_name, combined_score, individual_scores)
        """
        results = []
        
        for process in available_processes:
            lexical_sim = self.calculate_lexical_similarity(user_input, process)
            semantic_sim = self.calculate_semantic_similarity(user_input, process)
            
            combined_score = (
                lexical_sim * 0.6 +
                semantic_sim * 0.4 
            )
            
            individual_scores = {
                'lexical': lexical_sim,
                'semantic': semantic_sim,
                'combined': combined_score
            }
            
            results.append((process, combined_score, individual_scores))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:1]


class LogAnalysis:
    def __init__(self, chroma_db_path="./payment_chroma_db", model_name="all-MiniLM-L6-v2"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.embedding_model = SentenceTransformer(model_name)
        
        self.process_matcher = ProcessMatcher(self.embedding_model)
        
        try:
            self.collection = self.chroma_client.get_collection(name="payment_logs")
            print("Loaded existing ChromaDB collection")
        except:
            self.collection = self.chroma_client.create_collection(
                name="payment_logs",
                metadata={"description": "Payment workflow logs for analysis"}
            )
            print("Created new ChromaDB collection")

        self.applications = []

        self.llm = OllamaLLM(model="mistral:7b")

    def load_payment_logs(self, logs_directory="synthetic_logs"):
        """Load payment logs from individual application log files"""
        print("Loading payment logs from individual files into ChromaDB...")
        self.collection.delete(where={"correlation_id": {"$ne": "___"}}) 
        
        if not os.path.exists(logs_directory):
            print(f"Error: Directory {logs_directory} not found")
            return
        
        total_logs = 0
        logs_by_correlation = defaultdict(list)
        
        # Get all .jsonl files in the directory
        log_files = [f for f in os.listdir(logs_directory) if f.endswith('.jsonl')]
        
        if not log_files:
            print(f"No .jsonl files found in {logs_directory}")
            return
        
        print(f"Found {len(log_files)} log files to process:")
        for file in log_files:
            print(f"  - {file}")
        
        # Process each individual log file
        for log_file in log_files:
            file_path = os.path.join(logs_directory, log_file)
            print(f"\nProcessing: {log_file}")
            
            file_log_count = 0
            
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            raw = json.loads(line.strip())
                            log_entry = PaymentLogEntry(**raw)
                            logs_by_correlation[log_entry.correlation_id].append(log_entry)
                            file_log_count += 1
                            total_logs += 1
                        except (json.JSONDecodeError, ValidationError) as e:
                            print(f"Error in {log_file} line {line_num}: {e}")
                
                print(f"  Loaded {file_log_count} logs from {log_file}")
                
            except Exception as e:
                print(f"Error reading file {log_file}: {e}")
        
        # Store logs grouped by correlation ID for better retrieval
        self._store_payment_logs_in_chroma(logs_by_correlation)
        self._update_applications()
        print(f"\nTotal logs loaded: {total_logs}")

    def _update_applications(self):
        """Update the list of available applications from the database"""
        try:
            all_results = self.collection.get(include=["metadatas"])
            app_names = set()
        
            for metadata in all_results['metadatas']:
                if 'application' in metadata and metadata['application']:
                    app_names.add(metadata['application'])
        
            self.applications = sorted(list(app_names))
        except Exception as e:
            print(f"Error updating applications: {e}")
            self.applications = []    

    def _store_payment_logs_in_chroma(self, logs_by_correlation: Dict[str, List[PaymentLogEntry]]):
        """Store logs in ChromaDB with process context"""
        documents = []
        metadatas = []
        ids = []
        
        for correlation_id, logs in logs_by_correlation.items():
            logs.sort(key=lambda x: x.timestamp)
            
            for i, log in enumerate(logs):
                doc_text = f"""
                            Process: {log.process_name}
                            Application: {log.application}
                            Level: {log.log_level}
                            Message: {log.message}
                            Details: {log.details}
                            Timestamp: {log.timestamp}
                            Sequence: {i+1}/{len(logs)}
                            """.strip()
                
                documents.append(doc_text)
                metadatas.append({
                    "application": log.application,
                    "correlation_id": log.correlation_id,
                    "process_name": log.process_name,
                    "log_level": log.log_level,
                    "timestamp": log.timestamp,
                    "message": log.message,
                    "details": log.details,
                    "user_id": log.user_id,
                    "session_id": log.session_id,
                    "request_id": log.request_id,
                    "host": log.host,
                    "sequence_order": i
                })
                ids.append(f"{correlation_id}_{log.application}_{i}")
        
        if documents:
            batch_size = 1000
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            print(f"Stored {len(documents)} log entries in ChromaDB")

    def find_process_name_matches(self, user_input: str) -> List[str]:
        """
        Find matching process names using enhanced similarity matching
        """

        base_processes = self.list_available_processes(limit=100)     
        matches = self.process_matcher.find_best_matches(user_input, base_processes)

        if matches:
            best_match = matches[0][0]
            print(f"\nSelected best match: {best_match}")
            
            # Get all full process names that start with the best match
            all_results = self.collection.get(include=["metadatas"])
            matching_full_processes = list(set(
                metadata['process_name'] for metadata in all_results['metadatas']
                if 'process_name' in metadata and metadata['process_name'].startswith(best_match)
            ))
            
            return sorted(matching_full_processes)
        
        return []

    def retrieve_logs_by_process_name(self, process_names: List[str]) -> Dict[str, Dict[str, List[Dict]]]:
        """Retrieve all logs for given process names, grouped by correlation ID and application"""
        print(f"Retrieving logs for process names: {process_names}")
        
        all_correlation_logs = defaultdict(lambda: defaultdict(list))
        
        for process_name in process_names:
            results = self.collection.get(
                where={"process_name": process_name},
                include=["documents", "metadatas"]
            )
            
            # Group logs by correlation ID and application
            for doc, metadata in zip(results['documents'], results['metadatas']):
                correlation_id = metadata['correlation_id']
                app_name = metadata['application']
                
                all_correlation_logs[correlation_id][app_name].append({
                    'document': doc,
                    'metadata': metadata
                })
        
        # Sort logs within each application by sequence order
        for correlation_id in all_correlation_logs:
            for app in all_correlation_logs[correlation_id]:
                all_correlation_logs[correlation_id][app].sort(
                    key=lambda x: x['metadata'].get('sequence_order', 0)
                )
        
        print(f"Found logs for {len(all_correlation_logs)} correlation IDs")
        return dict(all_correlation_logs)

    def analyze_logs_batch_with_llm(self, batch_data: Dict[str, Dict[str, List[Dict]]], 
                                   process_name: str, batch_size: int = 5) -> Dict[str, Dict[str, str]]:
        """
        Analyze multiple correlation IDs in a single LLM call for efficiency
        
        Args:
            batch_data: Dict with correlation_id as key and app_logs as value
            process_name: The process being analyzed
            batch_size: Number of correlation IDs to process in one batch
            
        Returns:
            Dict[correlation_id, Dict[app_name, status]]
        """
        batch_results = {}
        correlation_ids = list(batch_data.keys())
    
        for i in range(0, len(correlation_ids), batch_size):
            batch_correlation_ids = correlation_ids[i:i + batch_size]
            batch_context = self._prepare_batch_context_for_llm(
                {cid: batch_data[cid] for cid in batch_correlation_ids}
            )
            
            print(f"Processing batch {i//batch_size + 1}: {len(batch_correlation_ids)} correlation IDs")
            
            # Generate batch prompt
            prompt = self._create_batch_prompt(batch_correlation_ids, batch_context, process_name)
            
            try:
                response = self.llm.invoke(prompt).strip()
                batch_parsed_results = self._parse_batch_response(response, batch_correlation_ids)
                batch_results.update(batch_parsed_results)
                
            except Exception as e:
                print(f"Error analyzing batch {i//batch_size + 1}: {e}")
                for cid in batch_correlation_ids:
                    batch_results[cid] = {app: "NO_LOGS" for app in self.applications}
        
        return batch_results

    def _prepare_batch_context_for_llm(self, batch_data: Dict[str, Dict[str, List[Dict]]]) -> str:
        """Prepare context for multiple correlation IDs"""
        context_sections = []
        
        for correlation_id, app_logs in batch_data.items():
            if not app_logs:
                context_sections.append(f"\n--- CORRELATION_ID: {correlation_id} ---")
                context_sections.append("NO LOGS FOUND")
                continue
                
            # Combine all logs for this correlation ID
            combined_logs = []
            for logs in app_logs.values():
                combined_logs.extend(logs)
            combined_logs.sort(key=lambda x: x['metadata'].get('sequence_order', 0))

            context_sections.append(f"\n--- CORRELATION_ID: {correlation_id} ---")
            
            for log_data in combined_logs:
                metadata = log_data['metadata']
                details = metadata['details']
                if len(details) > 100:  
                    details = details[:100] + "..."
                
                signals = []
                message_lower = metadata['message'].lower()
                details_lower = metadata['details'].lower() if metadata['details'] else ""

                combined_text = f"{message_lower} {details_lower}"

                success_signals = ["accepted", "completed", "success", "authorized", "initiated"]
                failure_signals = ["failed", "rejected", "declined", "not authorized", "timeout", "exception", "error"]

                for word in success_signals:
                    if word in combined_text:
                        signals.append(f"success {word}")

                for word in failure_signals:
                    if word in combined_text:
                        signals.append(f"failure/warn/incomplete {word}")

                # log line with signals
                log_line = f"[{metadata['application']}] [{metadata['timestamp']}] {metadata['log_level']}: {metadata['message']}"
                if details:
                    log_line += f" | {details}"

                if signals:
                    log_line += f" | SIGNALS: {'; '.join(signals)}"

                
                context_sections.append(log_line)
        
        return "\n".join(context_sections)

    def _create_batch_prompt(self, correlation_ids: List[str], context: str, process_name: str) -> str:
        """Create prompt for batch analysis"""
        correlation_list = ", ".join(correlation_ids)

        prompt = f"""
                You are a highly accurate system log analysis engine.

                Your task is to analyze the following end-to-end logs for the process: **{process_name}**.

                You are analyzing multiple correlation IDs: {correlation_list}

                ---
                Each workflow has logs from multiple applications (like AuthService, PaymentGateway, etc).

                For each correlation ID, examine the logs and assign a **status** to each application involved in that workflow.

                Use these **ONLY**: `SUCCESS`, `FAILURE`, `INCOMPLETE`, `NO_LOGS`

                **SUCCESS**: If the logs indicate the application completed its step or processed without errors.
                **FAILURE**: If there are errors, exceptions, or failed operations logged.
                **INCOMPLETE**: If logs are present but the execution does not seem to finish properly.
                **NO_LOGS**: If logs from that application are completely missing.

                ---
                Your response format MUST strictly be:

                CORRELATION_ID_1:
                AuthService: SUCCESS
                PaymentGateway: FAILURE
                ...

                CORRELATION_ID_2:
                NotificationService: INCOMPLETE
                ...

                If no logs are present for a correlation ID, output:

                CORRELATION_ID_3:
                NO_LOGS

                ---
                LOGS:
                {context}
                """
        return prompt


    def _parse_batch_response(self, response: str, correlation_ids: List[str]) -> Dict[str, Dict[str, str]]:
        """Parse LLM response for batch analysis"""
        results = {}
        
        # Initialize all correlation IDs with NO_LOGS for all apps
        for cid in correlation_ids:
            results[cid] = {app: "NO_LOGS" for app in self.applications}
        
        lines = response.split('\n')
        current_correlation_id = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a correlation ID header
            found_correlation = None
            for cid in correlation_ids:
                if cid in line and ':' in line and line.endswith(':'):
                    found_correlation = cid
                    break
            
            if found_correlation:
                current_correlation_id = found_correlation
                continue
            
            # Parse application status if we have a current correlation ID
            if current_correlation_id and ':' in line:
                for app in self.applications:
                    if app.lower() in line.lower():
                        status_match = re.search(rf"{re.escape(app)}[:\- ]+\s*(SUCCESS|FAILURE|INCOMPLETE|NO_LOGS)", 
                                               line, re.IGNORECASE)
                        if status_match:
                            status = status_match.group(1).upper()
                            results[current_correlation_id][app] = status
                            break
        
        return results

    def analyze_process_by_name_batch(self, user_input: str, batch_size: int = 5) -> pd.DataFrame:
        """
        Analyze process using batch processing for better efficiency
        
        Args:
            user_input: Process name to search for
            batch_size: Number of correlation IDs to process in one LLM call
        """
        print(f"\n{'='*80}")
        print(f"BATCH ANALYZING PROCESS: {user_input.upper()} (Batch Size: {batch_size})")
        print(f"{'='*80}")
    
        # Find matching process names 
        matching_processes = self.find_process_name_matches(user_input)
    
        if not matching_processes:
            print(f"No processes found matching: {user_input}")
            return pd.DataFrame()
    
        print(f"\nFound matching processes:")
        for i, process in enumerate(matching_processes, 1):
            print(f"  {i}. {process}")
    
        # Retrieve logs for all matching processes
        all_correlation_logs = self.retrieve_logs_by_process_name(matching_processes)
    
        if not all_correlation_logs:
            print("No logs found for matching processes")
            return pd.DataFrame()
    
        # Determine involved applications
        involved_apps = set()
        for app_logs in all_correlation_logs.values():
            involved_apps.update(app_logs.keys())
        involved_apps = sorted(involved_apps)

        print(f"Involved applications: {involved_apps}")
        print(f"Processing {len(all_correlation_logs)} workflow instances in batches of {batch_size}...")
    
        # Get process name for analysis
        first_correlation_id = next(iter(all_correlation_logs.keys()))
        first_log = next(iter(all_correlation_logs[first_correlation_id].values()))[0]['metadata']
        process_name = first_log['process_name'].split('_')[0]
        
        # Analysis using LLM
        batch_results = self.analyze_logs_batch_with_llm(all_correlation_logs, process_name, batch_size)
        
        # Convert results to DataFrame format
        all_results = []
        for correlation_id in all_correlation_logs.keys():
            row = {"correlation_id": correlation_id}
            
            correlation_results = batch_results.get(correlation_id, {})
            
            for app_name in self.applications:
                row[app_name] = correlation_results.get(app_name, "NO_LOGS")
            
            all_results.append(row)
    
        df = pd.DataFrame(all_results)
        df.attrs["involved_apps"] = involved_apps  # Store for later use in summary and CSV
        return df

    def generate_process_report_batch(self, user_input: str, batch_size: int = 5, output_file: str = None) -> pd.DataFrame:
        """Generate batch analysis report for a process"""
        df = self.analyze_process_by_name_batch(user_input, batch_size)
        
        if df.empty:
            return df
        
        if output_file is None:
            safe_name = re.sub(r'[^\w\s-]', '', user_input).strip()[:20]
            output_file = f"{safe_name}_batch_analysis.csv"
        
        involved_apps = getattr(df, 'attrs', {}).get('involved_apps', [])
        
        if involved_apps:
            columns_to_save = ['correlation_id'] + involved_apps
            df_filtered = df[columns_to_save].copy()
        else:
            df_filtered = df.copy()
        
        df_filtered.to_csv(output_file, index=False)
        print(f"\nBatch analysis report saved to: {output_file}")
        
        return df

    def list_available_processes(self, limit: int = 30) -> List[str]:
        """List unique base process prefixes in the database"""
        all_results = self.collection.get(include=["metadatas"])
        base_process_names = set()
    
        for metadata in all_results['metadatas']:
            if 'process_name' in metadata:
                full_name = metadata['process_name']
                base_name = full_name.split('_')[0]  # Get prefix before first underscore
                base_process_names.add(base_name)
    
        return sorted(list(base_process_names))[:limit]


def main():
    analyzer = LogAnalysis()
    
    analyzer.load_payment_logs("synthetic_logs")
    
    print("\n" + "="*80)
    print("LOG ANALYSIS SYSTEM")
    print("="*80)
    
    while True:
        try:
            print("\nOptions:")
            print("1. Analyze a specific process")
            print("2. List available processes")
            print("3. Quit")
            
            choice = input("\nEnter your choice: ").strip()
            
            if choice == '1':
                process_input = input("\nEnter the process name to analyze: ").strip()
                
                if not process_input:
                    print("Process name cannot be empty.")
                    continue
                try:
                    batch_size_input = input("Enter batch size: ").strip()
                    batch_size = int(batch_size_input) if batch_size_input else 5
                    if batch_size <= 0:
                        batch_size = 5
                        print("Invalid batch size, using default: 5")
                except ValueError:
                    batch_size = 5
                    print("Invalid batch size, using default: 5")
                
                print(f"\nAnalyzing process: {process_input} with batch size: {batch_size}")
                
                # Generate batch analysis report
                df = analyzer.generate_process_report_batch(process_input, batch_size)
                
                if not df.empty:
                    pd.set_option('display.max_columns', None)
                    pd.set_option('display.width', None)
                    pd.set_option('display.max_colwidth', 15)
                    
                    print(f"\nSummary:")
                    print(f"Total workflows analyzed: {len(df)}")
                    involved_apps = getattr(df, 'attrs', {}).get('involved_apps', [])
                    if involved_apps:
                        for app in involved_apps:
                            app_counts = df[app].value_counts()
                            print(f"{app}: {app_counts.to_dict()}")
                
            elif choice == '2':
                processes = analyzer.list_available_processes()
                print(f"\nAvailable processes (showing first 30):")
                for i, process in enumerate(processes, 1):
                    print(f"{i:2d}. {process}")
                
            elif choice == '3':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()