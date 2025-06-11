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

# Regex patterns
nginx_pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>[A-Z]+) (?P<url>.*?) HTTP/1.1" (?P<status>\d+) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
auth_pattern = r'(?P<timestamp>\w+ \d+ \d+:\d+:\d+) (?P<hostname>[\w\-.]+) (?P<process>[\w\-]+)(\[(?P<pid>\d+)\])?: (?P<message>.*)'
syslog_pattern = r'(?P<timestamp>\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (?P<hostname>[\w\-.]+) (?P<program>[\w\-.]+)(?:\[(?P<pid>\d+)\])?: (?P<message>.+)'
kern_pattern = r'(?P<timestamp>\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (?P<hostname>[\w\-.]+) kernel: \[(?P<uptime>[\d\.]+)\] (?P<message>.+)'

# Parsing functions
def parse_nginx_log(log):
    match = re.match(nginx_pattern, log)
    return match.groupdict() if match else None

def parse_auth_log(log):
    match = re.match(auth_pattern, log)
    return match.groupdict() if match else None

def parse_syslog_log(log):
    match = re.match(syslog_pattern, log)
    return match.groupdict() if match else None

def parse_kern_log(log):
    match = re.match(kern_pattern, log)
    return match.groupdict() if match else None

def should_drop_log(parsed_log):
    return parsed_log.get("program") == "filebeat" or parsed_log.get("process") == "filebeat"

def process_json_log(log_json):
    try:
        log_data = json.loads(log_json)

        message = log_data.get("message", "No message available")
        timestamp = log_data.get("@timestamp", "No timestamp")
        host = log_data.get("host", {}).get("hostname", "No hostname")
        file_path = log_data.get("log", {}).get("file", {}).get("path", "unknown")

        for parser, label in [
            (parse_nginx_log, "Parsed Nginx Log (from JSON)"),
            (parse_auth_log, "Parsed System Log (from JSON)"),
            (parse_syslog_log, "Parsed System Log (from JSON)"),
            (parse_kern_log, "Parsed Kernel Log (from JSON)")
        ]:
            result = parser(message)
            if result:
                if should_drop_log(result):
                    return
                result["source_file"] = file_path
                print(f"{label}:", json.dumps(result, indent=4))
                return

        # Generic structured log fallback
        print("Parsed Structured Log (Generic):", json.dumps({
            "timestamp": timestamp,
            "host": host,
            "file_path": file_path,
            "message": message
        }, indent=4))

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Invalid log: {log_json}")

def process_log(log):
    try:
        json.loads(log)
        process_json_log(log)
        return
    except json.JSONDecodeError:
        pass

    # For raw logs, file path is unknown
    file_path = "unknown"

    nginx_parsed = parse_nginx_log(log)
    if nginx_parsed:
        nginx_parsed["source_file"] = file_path
        print("Parsed Nginx Log:", json.dumps(nginx_parsed, indent=4))
        return

    auth_parsed = parse_auth_log(log)
    if auth_parsed:
        if should_drop_log(auth_parsed):
            return
        auth_parsed["source_file"] = file_path
        print("Parsed Auth Log:", json.dumps(auth_parsed, indent=4))
        return

    syslog_parsed = parse_syslog_log(log)
    if syslog_parsed:
        if should_drop_log(syslog_parsed):
            return
        syslog_parsed["source_file"] = file_path
        print("Parsed Syslog Log:", json.dumps(syslog_parsed, indent=4))
        return

    kern_parsed = parse_kern_log(log)
    if kern_parsed:
        kern_parsed["source_file"] = file_path
        print("Parsed Kernel Log:", json.dumps(kern_parsed, indent=4))
        return

    print("Unrecognized Log:", log)

# Main loop
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
