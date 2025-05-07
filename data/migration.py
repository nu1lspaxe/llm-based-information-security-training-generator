import os
import json
import pandas as pd
from typing import List, Dict, Optional
import logging
import chardet
from tqdm import tqdm
from scapy.all import PcapReader, Raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.http import HTTP
from scapy.layers.dns import DNS
from datetime import datetime
import pytz
import ijson
import pathlib
import tarfile
import shutil
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_encoding(file_path: str, sample_size: int = 50000) -> str:
    """Detect file encoding using chardet with a larger sample size."""
    try:
        path = pathlib.Path(file_path).resolve()
        with open(path, 'rb') as f:
            raw_data = f.read(sample_size)
            if not raw_data:
                return 'utf-8'
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
            confidence = result['confidence']
            if confidence < 0.7:
                logger.warning(f"Low confidence ({confidence}) for {path}, using utf-8")
                return 'utf-8'
            return encoding
    except Exception as e:
        logger.warning(f"Encoding detection failed for {path}: {e}, using utf-8")
        return 'utf-8'


def is_valid_data_file(file_path: str) -> bool:
    """Check if the file is a valid data file (not macOS metadata or empty)."""
    path = pathlib.Path(file_path)
    if '__MACOSX' in path.parts or path.name.startswith('._'):
        return False
    try:
        if path.stat().st_size < 10:
            logger.warning(f"Skipping {file_path}: File is too small")
            return False
    except OSError:
        logger.warning(f"Skipping {file_path}: Cannot access file")
        return False
    return True

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten a nested dictionary into a single level with dot-separated keys."""
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        elif isinstance(value, list):
            for i, v in enumerate(value):
                items.extend(flatten_dict({f"{key}_{i}": v}, parent_key, sep).items())
        else:
            if value is not None and not isinstance(value, (bytes, bytearray)):
                items.append((new_key, value))
    return dict(items)

def process_json_file(file_path: str) -> List[Dict]:
    """Process a JSON or JSONL file and convert to prompt-response pairs."""
    path = pathlib.Path(file_path)
    if not is_valid_data_file(str(path)):
        logger.info(f"Skipping {path}: Not a valid data file")
        return []
    
    try:
        encoding = detect_encoding(file_path)
        results = []
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            first_lines = [f.readline().strip() for _ in range(3)]
            is_jsonl = any('\n' in line for line in first_lines if line) or len([l for l in first_lines if l.startswith('{')]) > 1
        
        if is_jsonl:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if line.endswith(','):
                            line = line[:-1]
                        if not line.endswith('}') and not line.endswith(']'):
                            line = line + '}'
                        item = json.loads(line)
                        if isinstance(item, dict):
                            results.extend(create_response(item))
                        elif isinstance(item, list):
                            for sub_item in item:
                                results.extend(create_response(sub_item))
                        else:
                            logger.warning(f"Skipping line {line_number} in {file_path}: Not a dictionary or list")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_number} in {file_path}: {e}. Line content: {line[:100]}...")
                        continue
        else:
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                parser = ijson.parse(f)
                current_item = {}
                in_item = False
                key = None
                for prefix, event, value in parser:
                    if prefix == '' and event == 'start_map':
                        current_item = {}
                        in_item = True
                    elif prefix == '' and event == 'end_map' and in_item:
                        results.extend(create_response(current_item))
                        current_item = {}
                        in_item = False
                    elif in_item and event in ('string', 'number', 'boolean', 'null'):
                        key = prefix.split('.')[-1] if prefix else None
                        if key:
                            current_item[key] = value
                    elif in_item and event == 'start_map':
                        key = prefix.split('.')[-1] if prefix else None
                        current_item[key] = {}
                    elif in_item and event == 'end_map':
                        key = None
                    elif in_item and event == 'start_array':
                        key = prefix.split('.')[-1] if prefix else None
                        current_item[key] = []
                    elif in_item and event == 'end_array':
                        key = None
                    elif in_item and event in ('string', 'number', 'boolean', 'null') and key:
                        if isinstance(current_item[key], list):
                            current_item[key].append(value)
                        else:
                            current_item[key] = value
                if current_item:
                    results.extend(create_response(current_item))
        
        return results
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

def process_csv_file(file_path: str) -> List[Dict]:
    """Process a CSV file and convert to prompt-response pairs."""
    if not is_valid_data_file(file_path):
        logger.info(f"Skipping {file_path}: Not a valid data file")
        return []
    
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding, encoding_errors='replace')
        results = []
        for _, row in df.iterrows():
            item = {col: row[col] for col in df.columns if pd.notna(row[col])}
            for key, value in item.items():
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        item[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            results.extend(create_response(item))
        return results
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

def process_cap_file(file_path: str, max_packets: int = 100) -> List[Dict]:
    """Process a .cap (PCAP) file and convert packets to prompt-response pairs."""
    if not is_valid_data_file(file_path):
        logger.info(f"Skipping {file_path}: Not a valid data file")
        return []

    try:
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            valid_magic = [b'\xa1\xb2\xc3\xd4', b'\xd4\xc3\xb2\xa1', b'\xa1\xb2\x3c\x4d', b'\x4d\x3c\xb2\xa1']
            if magic not in valid_magic:
                logger.warning(f"Skipping {file_path}: Invalid PCAP magic number")
                return []

        results = []
        packet_count = 0

        with PcapReader(file_path) as packets:
            for pkt in packets:
                if packet_count >= max_packets:
                    break
                try:
                    pkt_data = {}

                    # Timestamp
                    try:
                        timestamp = float(pkt.time)
                        pkt_data['timestamp'] = datetime.fromtimestamp(timestamp, tz=pytz.UTC).isoformat()
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Skipping packet in {file_path}: Invalid timestamp ({e})")
                        continue

                    if IP in pkt:
                        pkt_data['protocol'] = pkt[IP].proto
                        pkt_data['source_ip'] = pkt[IP].src
                        pkt_data['destination_ip'] = pkt[IP].dst
                        proto_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
                        pkt_data['protocol_name'] = proto_map.get(pkt_data['protocol'], f"Unknown ({pkt_data['protocol']})")

                    if TCP in pkt:
                        pkt_data['source_port'] = pkt[TCP].sport
                        pkt_data['destination_port'] = pkt[TCP].dport
                        pkt_data['flags'] = str(pkt[TCP].flags)
                        pkt_data['description'] = f"TCP packet with flags {pkt[TCP].flags}"
                    elif UDP in pkt:
                        pkt_data['source_port'] = pkt[UDP].sport
                        pkt_data['destination_port'] = pkt[UDP].dport
                        pkt_data['description'] = "UDP packet"
                    else:
                        pkt_data['description'] = f"{pkt_data.get('protocol_name', 'Unknown')} packet"

                    if pkt.haslayer(HTTP):
                        try:
                            http_layer = pkt[HTTP]
                            pkt_data['http_method'] = getattr(http_layer, 'Method', b'').decode('utf-8', errors='ignore') if hasattr(http_layer, 'Method') else None
                            pkt_data['http_path'] = getattr(http_layer, 'Path', b'').decode('utf-8', errors='ignore') if hasattr(http_layer, 'Path') else None
                            pkt_data['description'] = f"HTTP {pkt_data.get('http_method', '')} request to {pkt_data.get('http_path', '')}"
                        except Exception as e:
                            logger.warning(f"Skipping HTTP fields in {file_path}: {e}")

                    if DNS in pkt:
                        try:
                            qname = getattr(pkt[DNS].qd, 'qname', None)
                            if qname:
                                pkt_data['dns_query'] = qname.decode('utf-8', errors='ignore')
                            
                            if pkt[DNS].ancount > 0 and hasattr(pkt[DNS], 'an'):
                                if hasattr(pkt[DNS].an, 'rdata'):
                                    pkt_data['dns_response'] = pkt[DNS].an.rdata

                            pkt_data['description'] = f"DNS query for {pkt_data.get('dns_query', '')} => {pkt_data.get('dns_response', 'N/A')}"
                            
                        except Exception as e:
                            logger.warning(f"Skipping DNS fields in {file_path}: {e}")


                    if pkt.haslayer(Raw):
                        try:
                            payload = pkt[Raw].load.decode('utf-8', errors='ignore')
                            if payload.strip():
                                pkt_data['payload'] = payload[:100] + '...' if len(payload) > 100 else payload
                        except UnicodeDecodeError:
                            pkt_data['payload'] = 'Binary data'

                    results.extend(create_response(pkt_data))
                    packet_count += 1

                except Exception as e:
                    logger.warning(f"Skipping packet in {file_path}: {e}")
                    continue

        return results

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

def process_tar_file(file_path: str, max_packets: int = 100) -> List[Dict]:
    """Extract a .tar file and process contained JSON/CSV/CAP files."""
    if not is_valid_data_file(file_path):
        logger.info(f"Skipping {file_path}: Not a valid data file")
        return []
    
    try:
        temp_dir = tempfile.mkdtemp()
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        results = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                extracted_file = os.path.join(root, file)
                if not is_valid_data_file(extracted_file):
                    continue
                if file.endswith('.json'):
                    results.extend(process_json_file(extracted_file))
                elif file.endswith('.csv'):
                    results.extend(process_csv_file(extracted_file))
                elif file.endswith('.cap'):
                    results.extend(process_cap_file(extracted_file, max_packets=max_packets))
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        logger.info(f"Processed {file_path}, extracted to {temp_dir}")
        return results
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

def create_response(item: Dict) -> List[Dict]:
    """Create response pair from a single log or packet item."""
    flat_item = flatten_dict(item)
    
    event_type_keys = ['EventID', 'event_id', 'eventName', 'type', 'Task', 'protocol_name']
    timestamp_keys = ['@timestamp', 'TimeCreated', 'EventTime', 'UtcTime', 'ts', 'timestamp']
    source_keys = ['SourceName', 'eventSource', 'Image', 'Application', 'ProcessName', 'source_ip']
    message_keys = ['Message', 'msg', 'Details', 'description']
    
    def get_first_available(keys: List[str], default: str) -> str:
        for key in keys:
            value = flat_item.get(key)
            if value and str(value).strip():
                return str(value)
        return default
    
    event_type = get_first_available(event_type_keys, 'Unknown')
    timestamp = get_first_available(timestamp_keys, 'Unknown')
    source = get_first_available(source_keys, 'Unknown')
    message = get_first_available(message_keys, 'No description available')
    
    if message.lower() in ['no description available', 'binary data']:
        return []
    
    prompt = (
        f"Event Type: {event_type}\n"
        f"Timestamp: {timestamp}\n"
        f"Source: {source}\n"
        f"Description: {message}\n"
    )
    
    response_lines = []
    if message and message != 'No description available':
        response_lines.append(f"Description: {message}")
    response_lines.extend(
        f"{key.capitalize()}: {value}"
        for key, value in flat_item.items()
        if value is not None and str(value).strip() and key not in message_keys
    )
    response = "\n".join(response_lines)
    
    if not response.strip() or len(response.splitlines()) < 3:
        return []
    
    return [{"prompt": prompt, "response": response}]

def collect_dataset(dataset_dir: str, output_file: str = "security_dataset.jsonl", max_packets: int = 100) -> List[Dict]:
    """Walk through dataset directory and collect JSON/CSV/CAP/TAR data."""
    dataset_dir = pathlib.Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        logger.error(f"Directory {dataset_dir} does not exist or is not a directory")
        return []
    
    dataset = []
    files = [
        str(path)
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix in ('.json', '.csv', '.cap', '.tar') and is_valid_data_file(str(path))
    ]
    
    for file_path in tqdm(files, desc="Processing files"):
        if file_path.endswith('.json'):
            dataset.extend(process_json_file(file_path))
        elif file_path.endswith('.csv'):
            dataset.extend(process_csv_file(file_path))
        elif file_path.endswith('.cap'):
            dataset.extend(process_cap_file(file_path, max_packets=max_packets))
        elif file_path.endswith('.tar'):
            dataset.extend(process_tar_file(file_path, max_packets=max_packets))
    
    dataset = [item for item in dataset if item["prompt"].strip() and item["response"].strip()]
    
    output_path = pathlib.Path(output_file)
    with output_path.open('w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    logger.info(f"Saved {len(dataset)} examples to {output_file}")
    return dataset

if __name__ == "__main__":
    dataset_dir = pathlib.Path("./security_datasets").resolve()
    collect_dataset(str(dataset_dir))