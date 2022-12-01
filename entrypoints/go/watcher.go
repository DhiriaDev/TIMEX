package main

import (
	"fmt"
	"github.com/confluentinc/confluent-kafka-go/kafka"
    "os"
    "os/exec"
    "time"
    "github.com/joho/godotenv"
)

type Data struct {
    output []byte
    error  error
}

func runCommand(message string) {
    boostrapServers := os.Args[1]
    role := os.Args[2]

    cmd := exec.Command("python", os.Getenv("PYTHON_MODULES_DIR") + role + ".py", boostrapServers, message)
    data, err := cmd.CombinedOutput()
    fmt.Printf("Data: %s", data)

	if err != nil {
        fmt.Printf("Err: %s", err)
	}
}

func main() {
    boostrapServers := os.Args[1]
    role := os.Args[2]

    // Load env variables.
    err := godotenv.Load(".env")

    if err != nil {
        panic(err)
    }

    fmt.Printf("Hello, I am the watcher for %s.\n", role)

    var c *kafka.Consumer
    if os.Getenv("KAFKA_SASL") == "true" {  // On the cluster
        c, err = kafka.NewConsumer(&kafka.ConfigMap{
            "bootstrap.servers": boostrapServers,
            "group.id":          role,
            "auto.offset.reset": "earliest",
            "max.in.flight.requests.per.connection": 5,
            "receive.message.max.bytes" : 2000000000,
            "security.protocol" : "sasl_ssl",
            "ssl.ca.location" : os.Getenv("SSL.CA.LOCATION"),
            "sasl.username" : os.Getenv("SASL.USERNAME"),
            "sasl.password" : os.Getenv("SASL.PASSWORD"),
            "sasl.mechanisms" : "SCRAM-SHA-256",
        })
    } else {  // In local
        c, err = kafka.NewConsumer(&kafka.ConfigMap{
            "bootstrap.servers": boostrapServers,
            "group.id":          role,
            "auto.offset.reset": "earliest",
            "max.in.flight.requests.per.connection": 5,
            "receive.message.max.bytes" : 2000000000,
        })
    }

	if err != nil {
		panic(err)
	}

	c.SubscribeTopics([]string{"control_topic"}, nil)

	for {
		msg, err := c.ReadMessage(-1)
		if err == nil {
            fmt.Printf("Message received!")
            // This will work in background
            go runCommand(string(msg.Value))

		} else {
			// The client will automatically try to recover from all errors.
			fmt.Printf("Consumer error: %v (%v)\n", err, msg)
            time.Sleep(1 * time.Second)
		}
	}
}

