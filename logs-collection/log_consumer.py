import json
import re
from confluent_kafka import Consumer

# Initialize Kafka Consumer
consumer = Consumer({
    'bootstrap.servers': '192.168.37.132:9092',
    'group.id': 'log-consumer-group',
    'auto.offset.reset': 'latest'
})

consumer.subscribe(['server-logs'])

# Regular expressions for parsing Nginx and auth logs
nginx_pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>[A-Z]+) (?P<url>.*?) HTTP/1.1" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'

auth_pattern = r'(?P<timestamp>\w+ \d+ \d+:\d+:\d+:\d+) (?P<hostname>[\w\-]+) (?P<process>[\w\-]+)\[(?P<pid>\d+)\]: (?P<message>.*)'

def parse_nginx_log(log):
    """
    Parse Nginx log using regex pattern.
    """
    match = re.match(nginx_pattern, log)
    if match:
        return match.groupdict()
    else:
        return None

def parse_auth_log(log):
    """
    Parse auth log using regex pattern for traditional logs.
    """
    match = re.match(auth_pattern, log)
    if match:
        return match.groupdict()
    else:
        return None

def process_json_log(log_json):
    """
    Process structured JSON logs (e.g., from Filebeat) and extract relevant fields.
    """
    try:
        log_data = json.loads(log_json)  # Parse the JSON string

        # Extract the message field and other metadata
        message = log_data.get("message", "No message available")
        timestamp = log_data.get("@timestamp", "No timestamp")
        host = log_data.get("host", {}).get("hostname", "No hostname")
        file_path = log_data.get("log", {}).get("file", {}).get("path", "No file path")

        parsed_log = {
            "timestamp": timestamp,
            "host": host,
            "file_path": file_path,
            "message": message
        }

        print("Parsed Structured Log:", json.dumps(parsed_log, indent=4))

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Invalid log: {log_json}")

def process_log(log):
    """
    Process the log, identify the type, and parse it.
    """
    # First, check if it's a JSON log (Filebeat log, structured log)
    try:
        log_data = json.loads(log)
        process_json_log(log)
        return
    except json.JSONDecodeError:
        pass  # Not a JSON log, so continue checking other formats

    # Check if it's an Nginx log
    nginx_parsed = parse_nginx_log(log)
    if nginx_parsed:
        print("Parsed Nginx Log:", json.dumps(nginx_parsed, indent=4))
        return
    
    # If not Nginx, check if it's an auth log
    auth_parsed = parse_auth_log(log)
    if auth_parsed:
        print("Parsed Auth Log:", json.dumps(auth_parsed, indent=4))
        return
    
    # If it's neither, just print the raw log
    print("Unrecognized Log:", log)

# Main loop for consuming logs
print("Listening for logs...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("Error:", msg.error())
            continue

        log = msg.value().decode('utf-8')
        process_log(log)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    consumer.close()