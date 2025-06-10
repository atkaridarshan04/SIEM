from confluent_kafka import Consumer

consumer = Consumer({
    'bootstrap.servers': '192.168.37.132:9092',
    'group.id': 'log-consumer-group',
    'auto.offset.reset': 'earliest'
})

consumer.subscribe(['system-logs'])

print("Listening for logs...")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("Error:", msg.error())
            continue

        print("Log Received:", msg.value().decode('utf-8'))

except KeyboardInterrupt:
    print("Exiting...")
finally:
    consumer.close()